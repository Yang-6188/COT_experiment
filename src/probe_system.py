"""智能探针系统"""
import torch
from typing import Optional

from .data_structures import CheckpointResult
from .answer_extractor import AnswerExtractor


class SmartProbeSystem:
    """智能探针系统 - 识别推理阶段并选择性探测"""
    
    def __init__(self, model, tokenizer, debug: bool = False):
        """
        初始化探针系统
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            debug: 是否开启调试模式
        """
        self.model = model
        self.tokenizer = tokenizer
        self.debug = debug
        
        # 推理阶段标记词
        self.reasoning_markers = {
            'calculation': ['=', 'equals', 'total', 'sum', 'result'],
            'conclusion': ['therefore', 'thus', 'so', 'hence', 'finally'],
            'intermediate': ['step', 'first', 'next', 'then', 'now'],
            'answer_signal': ['answer is', 'answer:', '####', '\\boxed', 'final answer']
        }
    
    def identify_reasoning_stage(self, text: str) -> str:
        """
        识别当前推理阶段
        
        Args:
            text: 当前生成的文本
            
        Returns:
            推理阶段标识
        """
        text_lower = text.lower()
        last_200_chars = text_lower[-200:]
        
        # 优先级：答案信号 > 结论 > 计算 > 中间步骤
        for marker in self.reasoning_markers['answer_signal']:
            if marker in last_200_chars:
                return 'answer_signal'
        
        for marker in self.reasoning_markers['conclusion']:
            if marker in last_200_chars:
                return 'conclusion'
        
        for marker in self.reasoning_markers['calculation']:
            if marker in last_200_chars:
                return 'calculation'
        
        for marker in self.reasoning_markers['intermediate']:
            if marker in last_200_chars:
                return 'intermediate'
        
        return 'unknown'
    
    def should_probe_at_stage(self, stage: str) -> bool:
        """
        判断该阶段是否应该探测
        
        Args:
            stage: 推理阶段
            
        Returns:
            是否应该探测
        """
        if stage == 'answer_signal':
            return True
        if stage == 'conclusion':
            return True
        if stage == 'calculation':
            return True
        if stage == 'intermediate':
            return False
        return False
    
    def detect_answer_in_context(self, text: str) -> Optional[str]:
        """
        直接从上下文检测答案（无需探针）
        
        Args:
            text: 当前文本
            
        Returns:
            检测到的答案
        """
        if '####' in text:
            import re
            match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
            if match:
                return match.group(1).replace(',', '')
        
        if '\\boxed{' in text:
            import re
            match = re.search(r'\\boxed\{(-?\d+(?:,\d+)*(?:\.\d+)?)\}', text)
            if match:
                return match.group(1).replace(',', '')
        
        return None
    
    def create_probe_prompt(self, stage: str) -> str:
        """
        根据推理阶段创建合适的探针提示
        
        Args:
            stage: 推理阶段
            
        Returns:
            探针提示文本
        """
        prompts = {
            'answer_signal': "\n#### ",
            'conclusion': "\n\nTherefore, the final answer is: ",
            'calculation': "\n\nThe result of this calculation is: ",
            'intermediate': "\n\nThe current value is: "
        }
        return prompts.get(stage, "\n#### ")
    
    def probe_answer(
        self, 
        full_sequence_ids: torch.Tensor, 
        current_text: str,
        stage: Optional[str] = None
    ) -> CheckpointResult:
        """
        智能探针 - 根据推理阶段调整策略
        
        Args:
            full_sequence_ids: 完整的token序列
            current_text: 当前文本
            stage: 推理阶段（可选）
            
        Returns:
            检查点结果
        """
        # 1. 先尝试直接从上下文提取
        context_answer = self.detect_answer_in_context(current_text)
        if context_answer:
            if self.debug:
                print(f"      ✓ 直接提取: {context_answer}")
            return CheckpointResult(
                answer=context_answer,
                entropy=0.1,
                confidence=0.95
            )
        
        # 2. 识别推理阶段
        if stage is None:
            stage = self.identify_reasoning_stage(current_text)
        
        if self.debug:
            print(f"      🎯 推理阶段: {stage}")
        
        # 3. 根据阶段决定是否探测
        if not self.should_probe_at_stage(stage):
            return CheckpointResult(
                should_halt=False,
                answer=None,
                entropy=100.0
            )
        
        # 4. 执行探针
        try:
            probe_text = self.create_probe_prompt(stage)
            probe_tokens = self.tokenizer.encode(
                probe_text, 
                return_tensors='pt', 
                add_special_tokens=False
            ).to(self.model.device)
            
            probe_input_ids = torch.cat([full_sequence_ids, probe_tokens], dim=-1)
            
            max_new_tokens = 30 if stage == 'answer_signal' else 20
            
            with torch.no_grad():
                gen_output = self.model.generate(
                    probe_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 5. 解析结果
            if gen_output.sequences.shape[1] > probe_input_ids.shape[1]:
                answer_tokens = gen_output.sequences[0][probe_input_ids.shape[1]:]
                probe_response = self.tokenizer.decode(
                    answer_tokens, 
                    skip_special_tokens=True
                ).strip()
                
                if self.debug:
                    print(f"      📝 探针响应: '{probe_response.replace(chr(10), '  ')}'")
                
                answer = AnswerExtractor.extract_answer(probe_response)
                
                if not answer:
                    answer = self._extract_number_fallback(probe_response)
                
                if self.debug:
                    print(f"      🔢 提取答案: '{answer}'")
                
                # 计算熵
                entropy = 100.0
                confidence = 0.0
                
                if gen_output.scores and len(gen_output.scores) > 0:
                    first_token_logits = gen_output.scores[0][0]
                    probs = torch.softmax(first_token_logits, dim=-1)
                    log_probs = torch.log_softmax(first_token_logits, dim=-1)
                    entropy = -torch.sum(probs * log_probs).item()
                    confidence = 1.0 / (1.0 + entropy)
                
                return CheckpointResult(
                    should_halt=False,
                    answer=answer,
                    entropy=entropy,
                    confidence=confidence
                )
        
        except Exception as e:
            if self.debug:
                print(f"      ❌ 探针错误: {e}")
        
        return CheckpointResult(entropy=100.0)
    
    def _extract_number_fallback(self, text: str) -> Optional[str]:
        """后备数字提取方案"""
        import re
        match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if match:
            return match.group(0).replace(',', '')
        
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            return match.group(0)
        
        return None
