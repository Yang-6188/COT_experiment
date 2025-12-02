"""实验运行器"""
import torch
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import ConfigManager
from .data_structures import GenerationState
from .answer_extractor import AnswerExtractor
from .early_stopping import SmartHaltDecisionMaker
from .probe_system import SmartProbeSystem
from .generation_manager import GenerationManager


class ExperimentRunner:
    """使用智能探针的实验运行器"""
    
    def __init__(self, base_dir: Path = None):
        """
        初始化实验运行器
        
        Args:
            base_dir: 基础目录路径
        """
        if base_dir is None:
            base_dir = Path("/root/autodl-tmp")
        
        self.base_dir = base_dir
        self.config_dir = base_dir / "config"
        self.data_dir = base_dir / "data"
        self.results_dir = base_dir / "results"
        
        self.results_dir.mkdir(exist_ok=True)
        
        # 加载配置
        config_file = self.config_dir / "config.json"
        self.config = ConfigManager.load_config(config_file)
        
        # 初始化管理器
        self.generation_manager = GenerationManager(self.config)
        self.halt_decision_maker = SmartHaltDecisionMaker(self.config)
        
        self.debug_mode = self.config.get('experiment', {}).get('debug_probe', False)
        
        print(f"🔧 智能探针配置: 答案一致性={self.halt_decision_maker.use_consistency}, "
              f"熵检测={self.halt_decision_maker.use_entropy}, "
              f"调试模式={self.debug_mode}")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        test_file = Path(self.config['paths']['test_data'])
        if not test_file.is_absolute():
            test_file = self.base_dir / test_file
        
        if not test_file.exists():
            raise FileNotFoundError(f"测试数据不存在: {test_file}")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sample_size = self.config['experiment']['sample_size']
        if len(data) > sample_size:
            data = data[:sample_size]
        
        print(f"✅ 已加载 {len(data)} 条测试数据")
        return data
    
    def load_model(self):
        """加载模型和分词器"""
        model_key = self.config['active_model']
        model_name = self.config['model_configs'][model_key]['name']
        
        print(f"🤖 加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        print(f"✅ 模型已加载到: {model.device}")
        return tokenizer, model
    
    def get_ground_truth(self, item: Dict[str, Any]) -> Optional[str]:
        """获取标准答案"""
        if 'numerical_answer' in item and item['numerical_answer']:
            return str(item['numerical_answer'])
        
        if 'answer' in item:
            return AnswerExtractor.extract_answer(item['answer'])
        
        return None
    
    def run_single_experiment(
        self, 
        tokenizer, 
        model, 
        question: str, 
        ground_truth: str, 
        sample_id: int
    ) -> Dict[str, Any]:
        """运行单个实验"""
        prompt = f"""Question: {question}

Please solve this step by step and provide your final answer.

Answer:"""
        
        print(f"\n📝 样本 {sample_id + 1}: {question[:80]}...")
        start_time = time.time()
        
        # 初始化
        state = GenerationState()
        probe_system = SmartProbeSystem(model, tokenizer, debug=self.debug_mode)
        self.halt_decision_maker.reset()
        
        # 准备输入
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        
        state.full_sequence_ids = input_ids.clone()
        
        # 生成参数
        exp_config = self.config.get('experiment', {})
        max_new_tokens = exp_config.get('max_new_tokens', 512)
        temperature = exp_config.get('temperature', 0.7)
        do_sample = exp_config.get('do_sample', False)
        
        try:
            # 主生成循环
            past_key_values = None
            current_input_ids = input_ids
            entropy_values = []
            stage_history = []
            
            while state.tokens_used < max_new_tokens and not state.early_stopped:
                # 模型前向传播
                with torch.no_grad():
                    outputs = model(
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
                new_text = tokenizer.decode([new_token_id], skip_special_tokens=True)
                state.full_response += new_text
                
                # 检查自然停止
                should_stop, stop_reason = self.generation_manager.should_stop_naturally(
                    state.full_response, new_token_id, tokenizer
                )
                if should_stop:
                    print(f"   🛑 自然停止: {stop_reason}")
                    break
                
                # 智能检查点
                min_tokens = self.config.get('early_stopping', {}).get('min_tokens_before_check', 100)
                
                if state.tokens_used >= min_tokens:
                    current_stage = probe_system.identify_reasoning_stage(state.full_response)
                    stage_history.append(current_stage)
                    
                    cooldown = self.config.get('early_stopping', {}).get('cooldown_tokens', 40)
                    
                    if self.halt_decision_maker.should_check_now(
                        state.full_response, 
                        state.tokens_used, 
                        current_stage,
                        cooldown
                    ):
                        probe_result = probe_system.probe_answer(
                            state.full_sequence_ids,
                            state.full_response,
                            current_stage
                        )
                        
                        self.halt_decision_maker.update_check_state(state.tokens_used, current_stage)
                        
                        if probe_result.answer:
                            clean_context = state.full_response[-100:].replace('\n', '⏎')
                            print(f"   🔎 [检查点@{current_stage}] Tokens: {state.tokens_used}")
                            print(f"      📄 上下文: ...{clean_context}")
                            print(f"      🧪 探针: '{probe_result.answer}' | 熵: {probe_result.entropy:.4f}")
                            
                            entropy_values.append(probe_result.entropy)
                        
                        decision = self.halt_decision_maker.make_decision(probe_result, current_stage)
                        
                        if decision.should_halt:
                            state.early_stopped = True
                            state.halt_reason = decision.halt_reason
                            state.predicted_answer = decision.answer
                            print(f"   🛑 [早停] {decision.halt_reason} | 答案: {decision.answer}")
                            break
            
            # 清理响应
            clean_response = state.full_response
            for stop_word in self.generation_manager.stop_words:
                if stop_word in clean_response:
                    clean_response = clean_response.split(stop_word)[0].strip()
            
            # 提取最终答案
            if not state.predicted_answer:
                state.predicted_answer = AnswerExtractor.extract_answer(clean_response, strict=False)
            
            generation_time = time.time() - start_time
            avg_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
            
            # 判断正确性
            is_correct = self._check_correctness(state.predicted_answer, ground_truth)
            
            from collections import Counter
            
            # 构建结果
            result = {
                "sample_id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": state.predicted_answer,
                "correct": is_correct,
                "generation_time": generation_time,
                "tokens_used": state.tokens_used,
                "response": clean_response,
                "early_stopped": state.early_stopped,
                "halt_reason": state.halt_reason,
                "avg_entropy": avg_entropy,
                "entropy_history": entropy_values[:10],
                "stage_distribution": dict(Counter(stage_history))
            }
            
            self._print_sample_result(result)
            return result
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(sample_id, question, ground_truth, str(e))
    
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
            "entropy_history": []
        }
    
    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算统计信息"""
        if not results:
            return {}
        
        from collections import Counter
        
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r['correct'])
        total_time = sum(r['generation_time'] for r in results)
        total_tokens = sum(r['tokens_used'] for r in results)
        early_stops = sum(1 for r in results if r.get('early_stopped', False))
        
        halt_reasons = Counter(r.get('halt_reason') for r in results if r.get('early_stopped', False))
        avg_entropy = sum(r.get('avg_entropy', 0) for r in results) / total_samples
        token_counts = [r['tokens_used'] for r in results]
        
        return {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "accuracy": correct_samples / total_samples,
            "total_time": total_time,
            "avg_time_per_sample": total_time / total_samples,
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": total_tokens / total_samples,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "early_stops": early_stops,
            "early_stop_rate": early_stops / total_samples,
            "halt_reasons": dict(halt_reasons),
            "avg_entropy": avg_entropy
        }
    
    def print_statistics(self, stats: Dict[str, Any]):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("📊 HALT-CoT 实验统计结果")
        print("=" * 60)
        print(f"🤖 模型: {self.config['model_configs'][self.config['active_model']]['name']}")
        print(f"📝 总样本数: {stats['total_samples']}")
        print(f"✅ 正确样本: {stats['correct_samples']}")
        print(f"🎯 准确率: {stats['accuracy']:.2%}")
        print(f"🛑 早停率: {stats['early_stop_rate']:.2%} ({stats['early_stops']}/{stats['total_samples']})")
        print(f"⏱️  平均用时: {stats['avg_time_per_sample']:.1f}秒/样本")
        print(f"💬 平均Token: {stats['avg_tokens_per_sample']:.1f}个/样本")
        print(f"📊 Token范围: {stats['min_tokens']} - {stats['max_tokens']}")
        print(f"📉 平均熵: {stats['avg_entropy']:.3f}")
        print(f"🕐 总用时: {stats['total_time']//60:.0f}分{stats['total_time']%60:.0f}秒")
        
        if stats.get('halt_reasons'):
            print("\n🔍 早停原因分布:")
            for reason, count in stats['halt_reasons'].items():
                if reason:
                    print(f"   - {reason}: {count}次")
        
        print("=" * 60)
    
    def save_results(self, results: List[Dict[str, Any]], 
                    stats: Dict[str, Any]) -> Optional[Path]:
        """保存实验结果"""
        if not self.config['experiment']['save_results']:
            print("⚠️  结果保存已禁用")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_key = self.config['active_model']
        sample_size = len(results)
        
        filename = f"halt_cot_{model_key}_{sample_size}samples_{timestamp}.json"
        results_file = self.results_dir / filename
        
        # 构建统计摘要
        summary_text = f"""
{'=' * 60}
📊 HALT-CoT 实验统计结果
{'=' * 60}
🤖 模型: {self.config['model_configs'][model_key]['name']}
📝 总样本数: {stats['total_samples']}
✅ 正确样本: {stats['correct_samples']}
🎯 准确率: {stats['accuracy']:.2%}
🛑 早停率: {stats['early_stop_rate']:.2%} ({stats['early_stops']}/{stats['total_samples']})
⏱️  平均用时: {stats['avg_time_per_sample']:.1f}秒/样本
💬 平均Token: {stats['avg_tokens_per_sample']:.1f}个/样本
📊 Token范围: {stats['min_tokens']} - {stats['max_tokens']}
📉 平均熵: {stats['avg_entropy']:.3f}
🕐 总用时: {stats['total_time']//60:.0f}分{stats['total_time']%60:.0f}秒
"""
        
        if stats.get('halt_reasons'):
            summary_text += "\n🔍 早停原因分布:\n"
            for reason, count in stats['halt_reasons'].items():
                if reason:
                    summary_text += f"   - {reason}: {count}次\n"
        
        summary_text += f"{'=' * 60}\n"
        
        save_data = {
            "experiment_info": {
                "timestamp": timestamp,
                "model": self.config['model_configs'][model_key]['name'],
                "model_key": model_key,
                "sample_size": sample_size,
                "config": self.config['experiment'],
                "early_stopping_config": self.config.get('early_stopping', {})
            },
            "statistics": stats,
            "summary": summary_text.strip(),
            "results": results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        # 保存纯文本摘要
        summary_file = self.results_dir / filename.replace('.json', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"💾 结果已保存: {results_file}")
        print(f"📄 摘要已保存: {summary_file}")
        return results_file
    
    def run_experiment(self):
        """运行完整实验"""
        print("🧪 开始HALT-CoT增强实验")
        print("=" * 60)
        
        try:
            # 加载数据和模型
            test_data = self.load_test_data()
            tokenizer, model = self.load_model()
            
            # 运行实验
            results = []
            experiment_start = time.time()
            
            for idx, item in enumerate(test_data):
                question = item['question']
                ground_truth = self.get_ground_truth(item)
                
                if ground_truth is None:
                    print(f"⚠️  样本 {idx + 1} 没有有效答案，跳过")
                    continue
                
                result = self.run_single_experiment(tokenizer, model, question, ground_truth, idx)
                results.append(result)
                
                if (idx + 1) % 5 == 0:
                    self._print_progress_report(results, idx + 1, len(test_data))
            
            total_time = time.time() - experiment_start
            
            # 计算统计
            stats = self.calculate_statistics(results)
            stats['total_experiment_time'] = total_time
            
            self.print_statistics(stats)
            
            # 保存结果
            results_file = self.save_results(results, stats)
            
            print(f"\n🎉 实验完成!")
            if results_file:
                print(f"📁 结果文件: {results_file}")
            
            return results, stats
            
        except Exception as e:
            print(f"❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            return [], {}
    
    def _print_progress_report(self, results: List[Dict], current: int, total: int):
        """打印进度报告"""
        current_accuracy = sum(1 for r in results if r['correct']) / len(results)
        early_stop_rate = sum(1 for r in results if r.get('early_stopped', False)) / len(results)
        avg_tokens = sum(r['tokens_used'] for r in results) / len(results)
        
        print(f"📊 进度: {current}/{total}, "
              f"准确率: {current_accuracy:.2%}, "
              f"早停率: {early_stop_rate:.2%}, "
              f"平均tokens: {avg_tokens:.1f}")