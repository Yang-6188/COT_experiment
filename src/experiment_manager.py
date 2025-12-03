"""实验管理器 - 处理单个样本的生成和检测"""
import torch
import time
from typing import Dict, Any, Optional, Tuple
from collections import Counter

from .data_structures import GenerationState
from .answer_extractor import AnswerExtractor
from .early_stopping import SmartHaltDecisionMaker
from .probe_system import SmartProbeSystem
from .generation_manager import GenerationManager
from .sentence_detector import SentenceBasedCheckpointManager


class ExperimentManager:
    """
    实验管理器 - 负责单个样本的实验执行
    从ExperimentRunner中分离出来，专注于生成逻辑 [[0]](#__0)
    """
    
    def __init__(self, config: dict, model, tokenizer):
        """
        初始化实验管理器
        
        Args:
            config: 配置字典
            model: 语言模型
            tokenizer: 分词器
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # 初始化各个管理器
        self.generation_manager = GenerationManager(config)
        self.halt_decision_maker = SmartHaltDecisionMaker(config)
        
        # 阶段控制配置
        stage_config = config.get('stage_control', {})
        self.use_smart_stage_detection = stage_config.get('use_smart_detection', True)
        self.fixed_check_stages = stage_config.get('fixed_check_stages', ['reasoning', 'calculation'])
        
        # 句子边界检测器（用于固定模式）[[3]](#__3)
        early_stop_config = config.get('early_stopping', {})
        self.sentence_checkpoint_manager = SentenceBasedCheckpointManager(
            min_tokens=early_stop_config.get('min_tokens_before_check', 100),
            cooldown_tokens=early_stop_config.get('cooldown_tokens', 40)
        )
        
        self.debug_mode = config.get('experiment', {}).get('debug_probe', False)
        
        print(f"🔧 实验管理器初始化: 智能检测={self.use_smart_stage_detection}, "
              f"句子边界检测={not self.use_smart_stage_detection}")
    
    def run_single_sample(
        self, 
        question: str, 
        ground_truth: str, 
        sample_id: int
    ) -> Dict[str, Any]:
        """
        运行单个样本的实验
        
        Args:
            question: 问题文本
            ground_truth: 标准答案
            sample_id: 样本ID
            
        Returns:
            实验结果字典
        """
        prompt = self._create_prompt(question)
        
        print(f"\n📝 样本 {sample_id + 1}: {question[:80]}...")
        start_time = time.time()
        
        try:
            # 初始化状态
            state = GenerationState()
            probe_system = SmartProbeSystem(self.model, self.tokenizer, debug=self.debug_mode)
            self.halt_decision_maker.reset()
            self.sentence_checkpoint_manager.reset()
            
            # 准备输入
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(self.model.device)
            attention_mask = inputs['attention_mask'].to(self.model.device)
            
            state.full_sequence_ids = input_ids.clone()
            
            # 执行生成循环
            result_data = self._generation_loop(
                state, probe_system, input_ids, attention_mask
            )
            
            # 计算最终结果
            generation_time = time.time() - start_time
            is_correct = self._check_correctness(state.predicted_answer, ground_truth)
            
            # 构建结果
            result = {
                "sample_id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": state.predicted_answer,
                "correct": is_correct,
                "generation_time": generation_time,
                "tokens_used": state.tokens_used,
                "response": result_data['clean_response'],
                "early_stopped": state.early_stopped,
                "halt_reason": state.halt_reason,
                "avg_entropy": result_data['avg_entropy'],
                "entropy_history": result_data['entropy_values'][:10],
                "stage_distribution": result_data['stage_distribution'],
                "stage_detection_mode": "smart" if self.use_smart_stage_detection else "sentence_based",
                "checkpoint_stats": self.sentence_checkpoint_manager.get_statistics()
            }
            
            self._print_sample_result(result)
            return result
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(sample_id, question, ground_truth, str(e))
    
    def _generation_loop(
        self,
        state: GenerationState,
        probe_system: SmartProbeSystem,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        主生成循环
        
        Args:
            state: 生成状态
            probe_system: 探针系统
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            
        Returns:
            生成结果数据
        """
        exp_config = self.config.get('experiment', {})
        max_new_tokens = exp_config.get('max_new_tokens', 512)
        temperature = exp_config.get('temperature', 0.7)
        do_sample = exp_config.get('do_sample', False)
        
        past_key_values = None
        current_input_ids = input_ids
        entropy_values = []
        stage_history = []
        
        while state.tokens_used < max_new_tokens and not state.early_stopped:
            # 模型前向传播
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    attention_mask=attention_mask if past_key_values is None else None
                )
            
            # 选择下一个token
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            if do_sample:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # 更新状态
            state.full_sequence_ids = torch.cat([state.full_sequence_ids, next_token], dim=-1)
            state.tokens_used += 1
            current_input_ids = next_token
            
            # 解码新token
            new_token_id = next_token.item()
            new_text = self.tokenizer.decode([new_token_id], skip_special_tokens=True)
            state.full_response += new_text
            
            # 检查自然停止
            should_stop, stop_reason = self.generation_manager.should_stop_naturally(
                state.full_response, new_token_id, self.tokenizer
            )
            if should_stop:
                print(f"   🛑 自然停止: {stop_reason}")
                break
            
            # 智能检查点逻辑
            should_check, current_stage = self._should_check_now(state)
            
            if should_check and current_stage:
                probe_result = probe_system.probe_answer(
                    state.full_sequence_ids,
                    state.full_response,
                    current_stage
                )
                
                if probe_result.answer:
                    self._print_checkpoint_info(state, current_stage, probe_result)
                    entropy_values.append(probe_result.entropy)
                    stage_history.append(current_stage)
                
                decision = self.halt_decision_maker.make_decision(probe_result, current_stage)
                
                if decision.should_halt:
                    state.early_stopped = True
                    state.halt_reason = decision.halt_reason
                    state.predicted_answer = decision.answer
                    print(f"   🛑 [早停] {decision.halt_reason} | 答案: {decision.answer}")
                    break
        
        # 清理响应
        clean_response = self._clean_response(state.full_response)
        
        # 提取最终答案
        if not state.predicted_answer:
            state.predicted_answer = AnswerExtractor.extract_answer(clean_response, strict=False)
        
        avg_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
        
        return {
            'clean_response': clean_response,
            'avg_entropy': avg_entropy,
            'entropy_values': entropy_values,
            'stage_distribution': dict(Counter(stage_history))
        }
    
    def _should_check_now(self, state: GenerationState) -> Tuple[bool, Optional[str]]:
        """
        判断是否应该进行检查
        
        Returns:
            (是否检查, 当前阶段)
        """
        if self.use_smart_stage_detection:
            # 智能阶段检测模式 [[7]](#__7)
            from .probe_system import SmartProbeSystem
            temp_probe = SmartProbeSystem(self.model, self.tokenizer, debug=False)
            current_stage = temp_probe.identify_reasoning_stage(state.full_response)
            
            cooldown = self.config.get('early_stopping', {}).get('cooldown_tokens', 40)
            should_check = self.halt_decision_maker.should_check_now(
                state.full_response, 
                state.tokens_used, 
                current_stage,
                cooldown
            )
            
            if should_check:
                self.halt_decision_maker.update_check_state(state.tokens_used, current_stage)
            
            return should_check, current_stage if should_check else None
        else:
            # 句子边界检测模式 [[6]](#__6)
            should_check, reason = self.sentence_checkpoint_manager.should_check_now(
                state.full_response,
                state.tokens_used
            )
            
            if should_check:
                # 根据当前内容推断阶段
                current_stage = self._infer_stage_from_content(state.full_response)
                if self.debug_mode:
                    print(f"      🔍 句子检查点: {reason} | 阶段: {current_stage}")
                return True, current_stage
            
            return False, None
    
    def _infer_stage_from_content(self, text: str) -> str:
        """从内容推断当前阶段"""
        text_lower = text.lower()
        last_100 = text_lower[-100:]
        
        if any(marker in last_100 for marker in ['answer is', '####', 'therefore']):
            return 'conclusion'
        elif any(marker in last_100 for marker in ['=', 'equals', 'total']):
            return 'calculation'
        else:
            return 'reasoning'
    
    def _create_prompt(self, question: str) -> str:
        """创建提示"""
        return f"""Question: {question}

Please solve this step by step and provide your final answer.

Answer:"""
    
    def _clean_response(self, response: str) -> str:
        """清理响应文本"""
        clean_response = response
        for stop_word in self.generation_manager.stop_words:
            if stop_word in clean_response:
                clean_response = clean_response.split(stop_word)[0].strip()
        return clean_response
    
    def _check_correctness(self, predicted: Optional[str], ground_truth: str) -> bool:
        """检查答案正确性"""
        if not predicted or not ground_truth:
            return False
        
        try:
            clean_pred = str(predicted).replace(',', '')
            clean_gt = str(ground_truth).replace(',', '')
            return float(clean_pred) == float(clean_gt)
        except ValueError:
            return str(predicted).strip() == str(ground_truth).strip()
    
    def _print_checkpoint_info(self, state: GenerationState, stage: str, probe_result):
        """打印检查点信息"""
        clean_context = state.full_response[-100:].replace('\n', '⏎')
        mode = "智能" if self.use_smart_stage_detection else "句子"
        print(f"   🔎 [{mode}检查点@{stage}] Tokens: {state.tokens_used}")
        print(f"      📄 上下文: ...{clean_context}")
        print(f"      🧪 探针: '{probe_result.answer}' | 熵: {probe_result.entropy:.4f}")
    
    def _print_sample_result(self, result: Dict[str, Any]):
        """打印单个样本结果"""
        status = "✅ 正确" if result['correct'] else "❌ 错误"
        halt_info = f"| 早停: {result['halt_reason']}" if result['early_stopped'] else ""
        
        print(f"   {status} | 预测: {result['predicted_answer']} | "
              f"实际: {result['ground_truth']} | "
              f"用时: {result['generation_time']:.1f}s | "
              f"Tokens: {result['tokens_used']} {halt_info}")
        
        if self.config['experiment'].get('verbose', False):
            preview = result['response'][:150].replace('\n', ' ')
            print(f"   回答预览: {preview}...")
    
    def _create_error_result(self, sample_id: int, question: str, 
                            ground_truth: str, error: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": None,
            "correct": False,
            "generation_time": 0,
            "tokens_used": 0,
            "response": f"Error: {error}",
            "error": error,
            "early_stopped": False,
            "halt_reason": None,
            "avg_entropy": 0.0,
            "entropy_history": [],
            "stage_detection_mode": "smart" if self.use_smart_stage_detection else "sentence_based"
        }
