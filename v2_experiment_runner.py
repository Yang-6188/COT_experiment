#!/usr/bin/env python3
"""
增强型HALT-CoT实验运行器 - 重构版
融合Liu & Wang (2025)的答案收敛检测和Laaouach的熵基早停方法
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
import re
import numpy as np
import nltk
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from collections import Counter
from dataclasses import dataclass

# ============================================================================
# 配置和路径常量
# ============================================================================
BASE_DIR = Path("/root/autodl-tmp")
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# 数据结构定义
# ============================================================================
@dataclass
class GenerationState:
    """生成状态管理"""
    tokens_used: int = 0
    full_response: str = ""
    full_sequence_ids: Optional[torch.Tensor] = None
    early_stopped: bool = False
    halt_reason: Optional[str] = None
    predicted_answer: Optional[str] = None
    
@dataclass 
class CheckpointResult:
    """检查点结果"""
    should_halt: bool = False
    halt_reason: Optional[str] = None
    answer: Optional[str] = None
    entropy: float = 0.0
    confidence: float = 0.0

# ============================================================================
# 早停检测器
# ============================================================================
class AnswerConsistencyDetector:
    """答案一致性检测器（基于Liu & Wang 2025）"""
    
    def __init__(self, k: int = 3):
        self.k = k
        self.answer_history = []
    
    def add_answer(self, answer: Optional[str]) -> bool:
        """添加答案并检查收敛性"""
        self.answer_history.append(answer)
        
        if len(self.answer_history) < self.k or answer is None:
            return False
        
        recent_answers = self.answer_history[-self.k:]
        return len(set(recent_answers)) == 1
    
    def reset(self):
        self.answer_history = []

class EntropyHaltDetector:
    """基于熵的早停检测器（基于Laaouach HALT-CoT）"""
    
    def __init__(self, threshold: float = 0.6, consecutive_steps: int = 2):
        self.threshold = threshold
        self.consecutive_steps = consecutive_steps
        self.entropy_history = []
        self.low_entropy_count = 0
    
    def should_halt(self, entropy: float) -> Tuple[bool, float]:
        """判断是否应该停止"""
        self.entropy_history.append(entropy)
        
        if entropy < self.threshold:
            self.low_entropy_count += 1
        else:
            self.low_entropy_count = 0
        
        should_stop = self.low_entropy_count >= self.consecutive_steps
        return should_stop, entropy
    
    def reset(self):
        self.entropy_history = []
        self.low_entropy_count = 0

# ============================================================================
# 答案提取器
# ============================================================================
class AnswerExtractor:
    """改进的答案提取器"""
    
    @staticmethod
    def extract_answer(text: str, strict: bool = False) -> Optional[str]:
        """改进的答案提取 - 使用更鲁棒的策略"""
        
        # 第一层：最高优先级格式
        high_confidence_patterns = [
            r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',  # GSM8K标准
            r'\\boxed\{([^}]+)\}',  # 完整的boxed（注意这里改为非贪婪匹配单层大括号内容）
        ]
        
        for pattern in high_confidence_patterns:
            matches = list(re.finditer(pattern, text, re.I))
            if matches:
                answer = matches[-1].group(1).replace(',', '').strip()
                if answer:
                    # 处理 LaTeX 分数
                    frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', answer)
                    if frac_match:
                        return frac_match.group(1)  # 只返回分子，或者改为 f"{分子}/{分母}"
                    return answer
        
        # ===== 改进：处理不完整的 boxed =====
        if '\\boxed{' in text and not strict:
            last_boxed_pos = text.rfind('\\boxed{')
            content_after = text[last_boxed_pos + 7:]
            
            # 尝试提取到闭括号
            brace_count = 1
            answer_content = ""
            
            for i, char in enumerate(content_after):
                if char == '{':
                    brace_count += 1
                    answer_content += char
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        if answer_content.strip():
                            # 处理 LaTeX 分数 \frac{a}{b}
                            frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', answer_content)
                            if frac_match:
                                # 根据需求返回分子或分数
                                return frac_match.group(1)  # 只返回分子
                            
                            # 提取纯数字
                            num_match = re.search(r'-?\d+(?:\.\d+)?', answer_content)
                            if num_match:
                                return num_match.group(0)
                        break
                    answer_content += char
                else:
                    answer_content += char
            
            # 如果没找到闭括号，智能提取
            if brace_count > 0 and answer_content:
                for end_marker in ['\n', '\\text', 'Therefore', 'Thus']:
                    if end_marker in answer_content:
                        answer_content = answer_content[:answer_content.index(end_marker)]
                        break
                
                answer_content = answer_content.strip()
                
                # 处理 LaTeX 分数
                frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', answer_content)
                if frac_match:
                    return frac_match.group(1)
                
                # 提取纯数字
                if len(answer_content) < 50:
                    num_match = re.search(r'-?\d+(?:\.\d+)?', answer_content)
                    if num_match:
                        return num_match.group(0)
        
        # 严格模式下只信任高优先级格式
        if strict:
            return None
        
        # 第二层：带有"final answer"的明确声明
        final_answer_patterns = [
            r'final answer is[:\s]+\$?(-?\d+(?:,\d+)*(?:\.\d+)?)',
            r'answer is[:\s]+\$?(-?\d+(?:,\d+)*(?:\.\d+)?)',
            r'Therefore,?\s+the answer is[:\s]+\$?(-?\d+(?:,\d+)*(?:\.\d+)?)',
        ]
        
        for pattern in final_answer_patterns:
            matches = list(re.finditer(pattern, text, re.I))
            if matches:
                answer = matches[-1].group(1).replace(',', '').strip()
                if answer:
                    return answer
        
        # 第三层：从等式中提取（保留原逻辑）
        return _extract_from_equations(text)


def _extract_from_equations(text: str) -> Optional[str]:
    """从等式中智能提取答案（原逻辑）"""
    lines = text.split('\n')
    keywords = ['total', 'answer', 'result', 'value', 'earnings', 'profit', 'money']
    
    # 倒序遍历，优先找最后的等式
    for line in reversed(lines):
        line_lower = line.lower()
        if any(k in line_lower for k in keywords) and '=' in line:
            rhs = line.split('=')[-1].strip()
            # 确保不包含运算符（不是中间计算）
            if not (re.search(r'[+*/]', rhs) or re.search(r'\s-\s', rhs)):
                num_match = re.search(r'^\$?(-?\d+(?:,\d+)*(?:\.\d+)?)', rhs)
                if num_match:
                    return num_match.group(1).replace(',', '')
    
    return None
    
    @staticmethod
    def _extract_incomplete_boxed(text: str) -> Optional[str]:
        """提取不完整的boxed答案 - 改进版"""
        last_boxed_pos = text.rfind('\\boxed{')
        content_after = text[last_boxed_pos + 7:]
        
        # 尝试提取到闭括号
        brace_count = 1
        answer_content = ""
        
        for i, char in enumerate(content_after):
            if char == '{':
                brace_count += 1
                answer_content += char
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    if answer_content.strip():
                        # 处理 LaTeX 分数 \frac{a}{b}
                        frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', answer_content)
                        if frac_match:
                            # 返回分数形式或小数
                            numerator = int(frac_match.group(1))
                            denominator = int(frac_match.group(2))
                            return f"{numerator}/{denominator}"
                        
                        # 提取纯数字
                        num_match = re.search(r'-?\d+(?:\.\d+)?', answer_content)
                        if num_match:
                            return num_match.group(0)
                    break
                answer_content += char
            else:
                answer_content += char
        
        # 如果没找到闭括号，智能提取
        if brace_count > 0 and answer_content:
            # 提取到第一个不合理的位置
            for end_marker in ['\n', '\\text', 'Therefore', 'Thus']:
                if end_marker in answer_content:
                    answer_content = answer_content[:answer_content.index(end_marker)]
                    break
            
            answer_content = answer_content.strip()
            
            # 处理 LaTeX 分数
            frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', answer_content)
            if frac_match:
                numerator = int(frac_match.group(1))
                denominator = int(frac_match.group(2))
                return f"{numerator}/{denominator}"
            
            # 提取纯数字
            if len(answer_content) < 50:
                num_match = re.search(r'-?\d+(?:\.\d+)?', answer_content)
                if num_match:
                    return num_match.group(0)
        
        return None

# ============================================================================
# 探针系统
# ============================================================================
class SmartProbeSystem:
    """智能探针系统 - 识别推理阶段并选择性探测"""
    
    def __init__(self, model, tokenizer, debug: bool = False):
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
        """识别当前推理阶段"""
        text_lower = text.lower()
        last_200_chars = text_lower[-200:]  # 只看最近的文本
        
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
        """判断该阶段是否应该探测"""
        # 答案信号：必须探测
        if stage == 'answer_signal':
            return True
        
        # 结论阶段：高优先级探测
        if stage == 'conclusion':
            return True
        
        # 计算阶段：中等优先级（可能是中间结果）
        if stage == 'calculation':
            return True
        
        # 中间步骤：低优先级（通常不探测）
        if stage == 'intermediate':
            return False
        
        return False
    
    def detect_answer_in_context(self, text: str) -> Optional[str]:
        """直接从上下文检测答案（无需探针）"""
        # 检查是否已有明确答案格式
        if '####' in text:
            match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
            if match:
                return match.group(1).replace(',', '')
        
        if '\\boxed{' in text:
            match = re.search(r'\\boxed\{(-?\d+(?:,\d+)*(?:\.\d+)?)\}', text)
            if match:
                return match.group(1).replace(',', '')
        
        return None
    
    def create_probe_prompt(self, stage: str) -> str:
        """根据推理阶段创建合适的探针提示"""
        prompts = {
            'answer_signal': "\n#### ",  # GSM8K标准格式
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
        """智能探针 - 根据推理阶段调整策略"""
        
        # 1. 先尝试直接从上下文提取（最快）
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
            
            # 根据阶段调整生成参数
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
                    print(f"      📝 探针响应: '{probe_response.replace('\n', '  ')}'")
                
                # 使用完整的答案提取器
                answer = AnswerExtractor.extract_answer(probe_response)
                
                # 如果提取失败，尝试简单数字提取
                if not answer:
                    answer = self._extract_number_fallback(probe_response)
                
                if self.debug:
                    print(f"      🔢 提取答案: '{answer}'")
                
                # 计算熵（仅用第一个token的熵）
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
        # 尝试提取完整数字（支持逗号分隔）
        match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if match:
            return match.group(0).replace(',', '')
        
        # 尝试提取任何数字序列
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            return match.group(0)
        
        return None

# ============================================================================
# 生成管理器
# ============================================================================
class GenerationManager:
    """改进的生成管理器"""
    
    def __init__(self, config):
        self.config = config
        self.stop_words = ["Human:", "User:", "\n\nHuman", "\n\nUser", "Observation:", "Question:"]
        
    def should_stop_naturally(self, text: str, new_token_id: int, tokenizer) -> Tuple[bool, str]:
        """改进的自然停止检测 - 更保守、更准确"""
        
        # 1. EOS token
        if new_token_id == tokenizer.eos_token_id:
            return True, "eos_token"

        # 2. 停止词检查
        for stop_word in self.stop_words:
            if stop_word in text:
                return True, f"stop_word_{stop_word}"

        # 3. ===== 改进的 Boxed 答案检测 =====
        if "\\boxed{" in text:
            last_boxed_pos = text.rfind("\\boxed{")
            content_after_boxed = text[last_boxed_pos + 7:]
            
            # 计算大括号嵌套
            brace_count = 1
            closed_pos = -1
            
            for i, char in enumerate(content_after_boxed):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        closed_pos = i
                        break
            
            # 如果找到匹配的闭括号
            if closed_pos != -1:
                remaining = content_after_boxed[closed_pos + 1:].strip()
                remaining = remaining.lstrip('.,;:!?\n\r\t ')
                
                # 后面没有实质内容才停止
                if len(remaining) == 0:
                    return True, "boxed_answer_complete"
                
                if len(remaining) < 30:
                    alnum_count = sum(1 for c in remaining if c.isalnum())
                    if alnum_count < 5:
                        return True, "boxed_answer_complete"

        # 4. ===== 改进：只检测明确的最终答案标记 =====
        final_answer_markers = [
            ("#### ", 20),  # GSM8K标准答案格式
            ("The final answer is", 30),
            ("Therefore, the final answer is", 30),
            ("Thus, the final answer is", 30),
        ]
        
        for marker, min_length_after in final_answer_markers:
            if marker in text:
                marker_pos = text.rfind(marker)
                text_after_marker = text[marker_pos:]
                
                # 必须有足够的内容
                if len(text_after_marker) > len(marker) + min_length_after:
                    # 检查是否有 boxed 答案
                    if '\\boxed{' in text_after_marker:
                        boxed_content = text_after_marker[text_after_marker.rfind('\\boxed{') + 7:]
                        if '}' in boxed_content:
                            return True, f"final_marker_with_boxed"
                    
                    # 或者检查是否有明确的数字答案
                    elif re.search(r'\d+', text_after_marker):
                        # 确保答案后有句号或双换行
                        if ('.' in text_after_marker and 
                            not any(word in text_after_marker.lower() for word in ['step', 'then', 'next'])):
                            return True, f"final_marker_with_number"

        # 5. 检测异常模式
        abnormal_patterns = [
            "Human:", "Assistant:", "You are an AI", "I am Claude",
        ]
        text_lower = text.lower()
        for pattern in abnormal_patterns:
            if pattern.lower() in text_lower:
                pattern_pos = text_lower.rfind(pattern.lower())
                if pattern_pos > len(text) * 0.5:
                    return True, f"abnormal_pattern"

        # 6. 检测重复内容
        if len(text) > 300:
            last_200 = text[-200:]
            prev_200 = text[-400:-200] if len(text) > 400 else ""
            if prev_200 and last_200 == prev_200:
                return True, "exact_repetition"

        # 7. 检测过长生成（安全措施）
        if len(text) > 2500:
            return True, "max_length_safety"

        return False, ""
    def _is_likely_final_answer(self, text: str) -> bool:
        """判断是否可能是最终答案 - 更严格的标准"""
        
        # 获取最后300个字符
        tail = text[-300:] if len(text) > 300 else text
        
        # 必须同时满足以下条件：
        # 1. 包含 "answer" 或 "####" 或 "\boxed"
        has_answer_signal = any(marker in tail.lower() for marker in 
                                ['#### ', 'final answer', 'the answer is', '\\boxed{'])
        
        if not has_answer_signal:
            return False
        
        # 2. 答案信号后面有数字
        lines = tail.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(marker in line_lower for marker in ['answer', '####', '\\boxed']):
                # 检查这一行或后续几行是否有数字
                remaining_lines = '\n'.join(lines[i:i+3])
                if re.search(r'\d+', remaining_lines):
                    # 3. 确保不是在计算过程中
                    # 如果后面还有 "Step", "Next", "Then" 等词，说明还在推理
                    if not any(word in remaining_lines.lower() for word in 
                              ['step', 'next', 'then', 'now let', 'we need to']):
                        return True
        
        return False
# ============================================================================
# 早停决策器
# ============================================================================
class SmartHaltDecisionMaker:
    """智能早停决策器 - 结合推理阶段"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_consistency = config.get('early_stopping', {}).get('use_answer_consistency', True)
        self.use_entropy = config.get('early_stopping', {}).get('use_entropy_halt', True)
        
        # 初始化检测器
        self.consistency_detector = AnswerConsistencyDetector(
            k=config.get('early_stopping', {}).get('consistency_k', 3)
        )
        self.entropy_detector = EntropyHaltDetector(
            threshold=config.get('early_stopping', {}).get('entropy_threshold', 0.6),
            consecutive_steps=config.get('early_stopping', {}).get('entropy_consecutive_steps', 2)
        )
        
        self.last_check_token_count = 0
        self.last_stage = None
        self.stage_check_counts = Counter()
    
    def should_check_now(
        self, 
        full_text: str, 
        tokens_used: int, 
        stage: str,
        cooldown: int = 40
    ) -> bool:
        """智能检查判断 - 考虑推理阶段"""
        
        # 如果早停功能完全禁用，不需要检查
        if not self.use_consistency and not self.use_entropy:
            return False
        
        # 如果在答案信号阶段,立即检查
        if stage == 'answer_signal':
            return True
        
        # 如果在结论阶段,且距离上次检查超过20个token
        if stage == 'conclusion' and (tokens_used - self.last_check_token_count) >= 20:
            return True
        
        # 如果在计算阶段,使用正常冷却
        if stage == 'calculation':
            if (tokens_used - self.last_check_token_count) >= cooldown:
                # 限制同一阶段的检查次数
                if self.stage_check_counts.get(stage, 0) < 3:
                    return True
        
        # 中间步骤阶段,更长的冷却时间
        if stage == 'intermediate':
            return (tokens_used - self.last_check_token_count) >= cooldown * 2
        
        return False
    
    def update_check_state(self, tokens_used: int, stage: str):
        """更新检查状态"""
        self.last_check_token_count = tokens_used
        if stage != self.last_stage:
            self.stage_check_counts[stage] = 0
            self.last_stage = stage
        self.stage_check_counts[stage] += 1
    
    def make_decision(
        self, 
        probe_result: CheckpointResult, 
        stage: str
    ) -> CheckpointResult:
        """智能决策 - 考虑推理阶段和配置"""
        
        # 如果早停功能完全禁用,直接返回不停止
        if not self.use_consistency and not self.use_entropy:
            return CheckpointResult(
                should_halt=False,
                halt_reason=None,
                answer=probe_result.answer,
                entropy=probe_result.entropy,
                confidence=probe_result.confidence
            )
        
        if not probe_result.answer:
            return probe_result
        
        # 答案信号阶段的决策更激进
        if stage == 'answer_signal':
            # 只有在启用熵检测时才使用熵判断
            if self.use_entropy and probe_result.entropy < 0.5:
                return CheckpointResult(
                    should_halt=True,
                    halt_reason=f"answer_signal_high_confidence",
                    answer=probe_result.answer,
                    entropy=probe_result.entropy,
                    confidence=probe_result.confidence
                )
        
        # 结论阶段 - 中等激进
        if stage == 'conclusion':
            # 一致性检测
            if self.use_consistency:
                is_consistent = self.consistency_detector.add_answer(probe_result.answer)
                if is_consistent and (not self.use_entropy or probe_result.entropy < 0.8):
                    return CheckpointResult(
                        should_halt=True,
                        halt_reason=f"conclusion_consistency",
                        answer=probe_result.answer,
                        entropy=probe_result.entropy,
                        confidence=probe_result.confidence
                    )
            
            # 极低熵也可以在结论阶段停止(仅当启用熵检测)
            if self.use_entropy and probe_result.entropy < 0.3:
                return CheckpointResult(
                    should_halt=True,
                    halt_reason=f"conclusion_high_confidence",
                    answer=probe_result.answer,
                    entropy=probe_result.entropy,
                    confidence=probe_result.confidence
                )
        
        # 计算阶段 - 保守策略
        if stage == 'calculation':
            # 需要同时满足熵检测和一致性检测(如果都启用)
            if self.use_entropy and probe_result.entropy < 0.15:
                # 如果启用了一致性检测,需要同时满足
                if self.use_consistency:
                    is_consistent = self.consistency_detector.add_answer(probe_result.answer)
                    if is_consistent:
                        return CheckpointResult(
                            should_halt=True,
                            halt_reason=f"calculation_high_confidence_consistent",
                            answer=probe_result.answer,
                            entropy=probe_result.entropy,
                            confidence=probe_result.confidence
                        )
                else:
                    # 如果只启用熵检测,直接停止
                    return CheckpointResult(
                        should_halt=True,
                        halt_reason=f"calculation_high_confidence",
                        answer=probe_result.answer,
                        entropy=probe_result.entropy,
                        confidence=probe_result.confidence
                    )
        
        # 记录答案但不停止(仅当启用一致性检测)
        if self.use_consistency:
            self.consistency_detector.add_answer(probe_result.answer)
        
        return CheckpointResult(
            should_halt=False,
            halt_reason=None,
            answer=probe_result.answer,
            entropy=probe_result.entropy,
            confidence=probe_result.confidence
        )
    
    def reset(self):
        """重置状态"""
        self.consistency_detector.reset()
        self.entropy_detector.reset()
        self.last_check_token_count = 0
        self.last_stage = None
        self.stage_check_counts.clear()


# ============================================================================
# 主实验运行器
# ============================================================================
class ExperimentRunner:
    """使用智能探针的实验运行器"""
    
    def __init__(self):
        self.config = self.load_config()
        self.generation_manager = GenerationManager(self.config)
        self.halt_decision_maker = SmartHaltDecisionMaker(self.config)
        
        # 添加调试模式配置
        self.debug_mode = self.config.get('experiment', {}).get('debug_probe', False)
        
        print(f"🔧 智能探针配置: 答案一致性={self.halt_decision_maker.use_consistency}, "
              f"熵检测={self.halt_decision_maker.use_entropy}, "
              f"调试模式={self.debug_mode}")
    
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config_file = CONFIG_DIR / "config.json"
        
        if not config_file.exists():
            default_config = {
                "active_model": "qwen",
                "model_configs": {
                    "qwen": {"name": "Qwen/Qwen2.5-7B-Instruct"}
                },
                "paths": {
                    "test_data": str(DATA_DIR / "gsm8k_test.json")
                },
                "experiment": {
                    "sample_size": 10,
                    "max_new_tokens": 512,
                    "do_sample": False,
                    "temperature": 0.7,
                    "save_results": True,
                    "verbose": False
                },
                "early_stopping": {
                    "use_answer_consistency": True,
                    "use_entropy_halt": True,
                    "consistency_k": 3,
                    "entropy_threshold": 0.6,
                    "entropy_consecutive_steps": 2,
                    "min_tokens_before_check": 100,
                    "cooldown_tokens": 40
                }
            }
            config_file.parent.mkdir(exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            return default_config
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        test_file = Path(self.config['paths']['test_data'])
        if not test_file.exists():
            test_file = DATA_DIR / "gsm8k_test.json"
            if not test_file.exists():
                raise FileNotFoundError("测试数据不存在")
        
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
        """运行单个实验 - 使用智能探针"""
        
        prompt = f"""Question: {question}

Please solve this step by step and provide your final answer.

Answer:"""
        
        print(f"\n📝 样本 {sample_id + 1}: {question[:80]}...")
        start_time = time.time()
        
        # 初始化状态和系统
        state = GenerationState()
        probe_system = SmartProbeSystem(model, tokenizer, debug=self.debug_mode)  # 使用智能探针
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
        
        gen_kwargs = {
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        
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
                
                # 检查自然停止条件
                should_stop, stop_reason = self.generation_manager.should_stop_naturally(
                    state.full_response, new_token_id, tokenizer
                )
                if should_stop:
                    print(f"   🛑 自然停止: {stop_reason}")
                    break
                
                # 智能检查点判断
                min_tokens = self.config.get('early_stopping', {}).get('min_tokens_before_check', 100)
                
                if state.tokens_used >= min_tokens:
                    # 识别当前推理阶段
                    current_stage = probe_system.identify_reasoning_stage(state.full_response)
                    stage_history.append(current_stage)
                    
                    # 判断是否应该检查
                    cooldown = self.config.get('early_stopping', {}).get('cooldown_tokens', 40)
                    
                    if self.halt_decision_maker.should_check_now(
                        state.full_response, 
                        state.tokens_used, 
                        current_stage,
                        cooldown
                    ):
                        # 执行智能探针
                        probe_result = probe_system.probe_answer(
                            state.full_sequence_ids,
                            state.full_response,
                            current_stage
                        )
                        
                        # 更新检查状态
                        self.halt_decision_maker.update_check_state(state.tokens_used, current_stage)
                        
                        if probe_result.answer:
                            clean_context = state.full_response[-100:].replace('\n', '⏎')
                            print(f"   🔎 [检查点@{current_stage}] Tokens: {state.tokens_used}")
                            print(f"      📄 上下文: ...{clean_context}")
                            print(f"      🧪 探针: '{probe_result.answer}' | 熵: {probe_result.entropy:.4f}")
                            
                            entropy_values.append(probe_result.entropy)
                        
                        # 智能决策
                        decision = self.halt_decision_maker.make_decision(probe_result, current_stage)
                        
                        if decision.should_halt:
                            state.early_stopped = True
                            state.halt_reason = decision.halt_reason
                            state.predicted_answer = decision.answer
                            print(f"   🛑 [早停] {decision.halt_reason} | 答案: {decision.answer}")
                            break
            
            # 清理响应文本
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
                "stage_distribution": dict(Counter(stage_history))  # 新增：阶段分布统计
            }
            
            # 打印结果
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
    
    def _create_error_result(self, sample_id: int, question: str, ground_truth: str, error: str) -> Dict[str, Any]:
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
        
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r['correct'])
        total_time = sum(r['generation_time'] for r in results)
        total_tokens = sum(r['tokens_used'] for r in results)
        early_stops = sum(1 for r in results if r.get('early_stopped', False))
        
        # 统计早停原因
        halt_reasons = Counter(r.get('halt_reason') for r in results if r.get('early_stopped', False))
        
        # 平均熵
        avg_entropy = sum(r.get('avg_entropy', 0) for r in results) / total_samples
        
        # Token统计
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
    
    def save_results(self, results: List[Dict[str, Any]], stats: Dict[str, Any]) -> Optional[Path]:
        """保存实验结果"""
        if not self.config['experiment']['save_results']:
            print("⚠️  结果保存已禁用")
            return None
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_key = self.config['active_model']
        sample_size = len(results)
    
        filename = f"halt_cot_{model_key}_{sample_size}samples_{timestamp}.json"
        results_file = RESULTS_DIR / filename
    
        # 构建统计摘要文本
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
    
        # 添加早停原因分布
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
        "summary": summary_text.strip(),  # 添加文本摘要
        "results": results
        }
    
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
    
        # 同时保存一个纯文本的摘要文件
        summary_file = RESULTS_DIR / filename.replace('.json', '_summary.txt')
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
            # 1. 加载数据和模型
            test_data = self.load_test_data()
            tokenizer, model = self.load_model()
            
            # 2. 运行实验
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
                
                # 实时进度报告
                if (idx + 1) % 5 == 0:
                    self._print_progress_report(results, idx + 1, len(test_data))
            
            total_time = time.time() - experiment_start
            
            # 3. 计算和显示统计
            stats = self.calculate_statistics(results)
            stats['total_experiment_time'] = total_time
            
            self.print_statistics(stats)
            
            # 4. 保存结果
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

# ============================================================================
# 主函数
# ============================================================================
def main():
    """主函数"""
    try:
        runner = ExperimentRunner()
        results, stats = runner.run_experiment()
        
        if results:
            print(f"\n📈 实验成功完成，处理了 {len(results)} 个样本")
            print(f"💡 融合了两种早停方法：")
            print(f"   1. 答案一致性检测（Liu & Wang 2025）")
            print(f"   2. 熵基早停（Laaouach HALT-CoT）")
        else:
            print(f"\n💥 实验失败或没有有效结果")
            
    except KeyboardInterrupt:
        print(f"\n❌ 实验被用户中断")
    except Exception as e:
        print(f"❌ 实验运行失败: {e}")

if __name__ == "__main__":
    main()