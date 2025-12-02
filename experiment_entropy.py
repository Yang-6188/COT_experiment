#!/usr/bin/env python3
"""
增强型HALT-CoT实验运行器 - 熵值分析与可视化版 (修复版v3)
- 修复过度截断问题
- 在所有探测点标注答案
- 优化标注布局避免重叠
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
import re
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field, asdict

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("⚠️ 未检测到 matplotlib/seaborn，将跳过绘图功能。建议 pip install matplotlib seaborn")

# ============================================================================
# 配置和路径常量
# ============================================================================
BASE_DIR = Path("/root/autodl-tmp")
CONFIG_DIR = BASE_DIR / "config_entropy"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results_entropy"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "plots").mkdir(exist_ok=True)

# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class GenerationState:
    tokens_used: int = 0
    full_response: str = ""
    full_sequence_ids: Optional[torch.Tensor] = None
    early_stopped: bool = False
    halt_reason: Optional[str] = None
    predicted_answer: Optional[str] = None

@dataclass 
class CheckpointResult:
    should_halt: bool = False
    halt_reason: Optional[str] = None
    answer: Optional[str] = None
    entropy: float = 100.0
    confidence: float = 0.0

@dataclass
class ProbeRecord:
    """记录单次探测的详细信息"""
    step: int
    stage: str
    answer: Optional[str]
    entropy: float
    text_segment: str = ""
    
    def to_dict(self):
        """转换为字典以便JSON序列化"""
        return {
            'step': self.step,
            'stage': self.stage,
            'answer': self.answer,
            'entropy': self.entropy,
            'text_segment': self.text_segment
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建实例"""
        return cls(
            step=data.get('step', 0),
            stage=data.get('stage', 'unknown'),
            answer=data.get('answer'),
            entropy=data.get('entropy', 0.0),
            text_segment=data.get('text_segment', '')
        )

# ============================================================================
# 文本清理工具
# ============================================================================
class TextCleaner:
    """用于清理和截断异常输出"""
    
    # 需要截断的异常模式（移除了 ### 因为它是GSM8K的答案标记）
    STOP_PATTERNS = [
        "You are an AI assistant",
        "You are a helpful assistant",
        "I am an AI",
        "As an AI",
        "Human:",
        "Assistant:",
        "User:",
        "<|im_start|>",
        "<|im_end|>",
    ]
    
    @staticmethod
    def clean_response(text: str, verbose: bool = False) -> str:
        """清理响应文本，移除异常输出"""
        if not text:
            return text
        
        # 查找第一个异常模式的位置
        min_pos = len(text)
        found_pattern = None
        
        for pattern in TextCleaner.STOP_PATTERNS:
            pos = text.find(pattern)
            if pos != -1 and pos < min_pos:
                min_pos = pos
                found_pattern = pattern
        
        # 如果找到异常模式，截断到该位置
        if found_pattern:
            text = text[:min_pos].strip()
            if verbose:
                print(f"   ⚠️ Truncated at pattern: '{found_pattern}'")
        
        return text
    
    @staticmethod
    def extract_reasoning_part(text: str) -> str:
        """提取推理部分，移除前后的无关内容"""
        # 移除开头的提示词
        text = re.sub(r'^(Question:|Answer:|Problem:|Solution:)\s*', '', text, flags=re.IGNORECASE)
        
        # 清理异常输出（但不打印警告）
        text = TextCleaner.clean_response(text, verbose=False)
        
        return text.strip()

# ============================================================================
# 增强的答案提取器
# ============================================================================
# ============================================================================
# 增强的答案提取器 (修复版)
# ============================================================================
class AnswerExtractor:
    @staticmethod
    def extract_answer(text: str, strict: bool = False) -> Optional[str]:
        """增强版答案提取器，支持多种格式"""
        if not text:
            return None
        
        # 先清理文本
        text = TextCleaner.extract_reasoning_part(text)
        
        # 1. 标准格式匹配（优先级最高）
        patterns = [
            # GSM8K标准格式
            (r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', 10),
            # LaTeX boxed格式
            (r'\\boxed\{(-?\d+(?:,\d+)*(?:\.\d+)?)\}', 9),
            # "Answer: X" 格式（新增）
            (r'[Aa]nswer:\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', 9),
            # "Therefore, the answer is X" 格式
            (r'[Tt]herefore,?\s+(?:the\s+)?(?:answer|total|result)\s+(?:is|equals?)\s+\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', 8),
            # "The answer is X" 格式
            (r'[Tt]he\s+(?:final\s+)?(?:answer|total|result)\s+(?:is|equals?)\s+\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', 7),
            # "John is X miles from home" 格式（新增）
            (r'(?:is|are)\s+(-?\d+(?:,\d+)*(?:\.\d+)?)\s+(?:miles?|dollars?|units?)', 7),
            # "answer is X" 格式
            (r'(?:answer|result|total)\s+(?:is|equals?|=)\s*\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', 6),
            # 句子结尾的数字（带单位或标点）
            (r'(?:is|equals?|=)\s+(-?\d+(?:,\d+)*(?:\.\d+)?)\s*(?:downloads?|dollars?|miles?|units?|\.|$)', 5),
        ]
        
        best_answer = None
        best_priority = -1
        best_position = -1
        
        for pattern, priority in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                # 取最后一个匹配（通常是最终答案）
                match = matches[-1]
                answer = match.group(1).replace(',', '').strip()
                try:
                    val = float(answer)
                    # 排除明显错误的值
                    if val < 0 or val > 1e10:
                        continue
                    
                    position = match.start()
                    # 优先级相同时，选择位置更靠后的
                    if priority > best_priority or (priority == best_priority and position > best_position):
                        best_answer = answer
                        best_priority = priority
                        best_position = position
                except ValueError:
                    continue
        
        if best_answer:
            return best_answer
        
        # 2. 尝试从最后几行提取
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            # 检查最后5行
            for line in reversed(lines[-5:]):
                # 跳过过长的说明性文本
                if len(line) > 150:
                    continue
                
                # 优先查找包含"Answer:"的行
                if re.search(r'\bAnswer:', line, re.IGNORECASE):
                    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', line)
                    if numbers:
                        answer = numbers[0].replace(',', '')  # 取第一个数字
                        try:
                            val = float(answer)
                            if 0 <= val < 1e10:
                                return answer
                        except ValueError:
                            continue
                
                # 查找包含"总计"、"答案"等关键词的行
                if re.search(r'(total|answer|result|sum|final|is\s+\d+)', line, re.IGNORECASE):
                    # 提取该行中的数字
                    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', line)
                    if numbers:
                        # 取最后一个数字
                        answer = numbers[-1].replace(',', '')
                        try:
                            val = float(answer)
                            if 0 <= val < 1e10:
                                return answer
                        except ValueError:
                            continue
                
                # 查找"X miles/dollars"格式
                match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s+(?:miles?|dollars?|units?)', line, re.IGNORECASE)
                if match:
                    answer = match.group(1).replace(',', '')
                    try:
                        val = float(answer)
                        if 0 <= val < 1e10:
                            return answer
                    except ValueError:
                        continue
        
        # 3. 最后尝试：提取所有数字，返回最后一个合理的
        if not strict:
            all_numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
            if all_numbers:
                # 从后往前检查
                for num in reversed(all_numbers[-10:]):
                    cleaned = num.replace(',', '')
                    try:
                        val = float(cleaned)
                        # 排除过小的数字（可能是步骤编号）和过大的异常值
                        # 同时排除明显的中间计算值（如180, 135等）
                        if 1 <= val < 1e6:  # 调整范围
                            return cleaned
                    except ValueError:
                        continue
        
        return None
    
    @staticmethod
    def extract_from_probe_response(text: str) -> Optional[str]:
        """专门用于提取探针响应中的答案"""
        if not text:
            return None
        
        # 清理文本（不打印警告）
        text = TextCleaner.clean_response(text, verbose=False)
        
        # 探针响应通常更简洁，优先匹配开头的数字
        patterns = [
            r'^\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',  # 开头的数字
            r'[Aa]nswer:\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',  # Answer: X
            r'(?:is|equals?|=)\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',  # 带等号的
            r'(-?\d+(?:,\d+)*(?:\.\d+)?)\s+(?:miles?|dollars?)',  # X miles/dollars
            r'(-?\d+(?:,\d+)*(?:\.\d+)?)',  # 任意数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                answer = match.group(1).replace(',', '').strip()
                try:
                    val = float(answer)
                    if 0 <= val < 1e10:
                        return answer
                except ValueError:
                    continue
        
        return None


# ============================================================================
# 智能探针系统
# ============================================================================
class SmartProbeSystem:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.reasoning_markers = {
            'calculation': ['=', 'equals', 'total', 'sum', 'result', '+', '-', '*', '/', 'calculate'],
            'conclusion': ['therefore', 'thus', 'so', 'hence', 'finally', 'conclude', 'in conclusion'],
            'intermediate': ['step', 'first', 'next', 'then', 'now', 'assume', 'let'],
            'answer_signal': ['answer is', 'answer:', '####', '\\boxed', 'final answer']
        }
    
    def identify_reasoning_stage(self, text: str) -> str:
        text_lower = text.lower()
        last_100 = text_lower[-100:]
        
        for marker in self.reasoning_markers['answer_signal']:
            if marker in last_100: 
                return 'answer_signal'
        
        for marker in self.reasoning_markers['conclusion']:
            if marker in last_100: 
                return 'conclusion'
        
        for marker in self.reasoning_markers['calculation']:
            if marker in last_100: 
                return 'calculation'
        
        return 'intermediate'

    def probe_answer(self, full_sequence_ids: torch.Tensor, current_text: str, stage: str) -> CheckpointResult:
        """执行探针检测"""
        prompts = {
            'answer_signal': "\n\nThe final answer is: ",
            'conclusion': "\n\nTherefore, the answer is: ",
            'calculation': "\n\nThis equals: ",
            'intermediate': "\n\nThe current value is: "
        }
        probe_text = prompts.get(stage, "\n\nThe answer is: ")
        
        try:
            probe_tokens = self.tokenizer.encode(
                probe_text, 
                return_tensors='pt', 
                add_special_tokens=False
            ).to(self.model.device)
            
            probe_input = torch.cat([full_sequence_ids, probe_tokens], dim=-1)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    probe_input,
                    max_new_tokens=20,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            if outputs.sequences.shape[1] > probe_input.shape[1]:
                gen_tokens = outputs.sequences[0][probe_input.shape[1]:]
                gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                
                # 使用专门的探针答案提取器
                answer = AnswerExtractor.extract_from_probe_response(gen_text)
                if not answer:
                    answer = AnswerExtractor.extract_answer(gen_text)
                
                # 计算熵
                if outputs.scores and len(outputs.scores) > 0:
                    logits = outputs.scores[0][0]
                    probs = torch.softmax(logits, dim=-1)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * log_probs).item()
                else:
                    entropy = 0.0

                return CheckpointResult(answer=answer, entropy=entropy)
                
        except Exception as e:
            print(f"⚠️ Probe error: {e}")
            
        return CheckpointResult(entropy=10.0)

# ============================================================================
# 可视化工具 - 增强版
# ============================================================================
class EntropyVisualizer:
    """专门用于绘制熵值变化图"""
    
    STAGE_COLORS = {
        'intermediate': '#3498db',
        'calculation': '#f39c12',
        'conclusion': '#2ecc71',
        'answer_signal': '#e74c3c'
    }
    
    @staticmethod
    def smart_label_placement(steps, entropies, answers, ground_truth):
        """智能计算标注位置，避免重叠"""
        positions = []
        
        for i, (step, ent, ans) in enumerate(zip(steps, entropies, answers)):
            if ans == 'None':
                positions.append(None)
                continue
            
            # 计算基础偏移
            base_offset = 15
            
            # 检查与已有标注的距离
            conflicts = []
            for j, prev_pos in enumerate(positions):
                if prev_pos is None:
                    continue
                prev_step, prev_ent, prev_offset = prev_pos
                
                # 计算水平和垂直距离
                h_dist = abs(step - prev_step)
                v_dist = abs(ent - prev_ent)
                
                if h_dist < 25 and v_dist < 0.5:  # 太近了
                    conflicts.append(prev_offset)
            
            # 根据冲突调整偏移
            if conflicts:
                # 尝试不同的偏移量
                possible_offsets = [15, -20, 25, -30, 35, -40]
                for offset in possible_offsets:
                    if offset not in conflicts:
                        y_offset = offset
                        break
                else:
                    # 如果都冲突，使用交替模式
                    y_offset = 15 if i % 2 == 0 else -20
            else:
                # 无冲突，使用交替模式
                y_offset = 15 if i % 2 == 0 else -20
            
            positions.append((step, ent, y_offset))
        
        return positions
    
    @staticmethod
    def plot_single_entropy(ax, records: List[ProbeRecord], sample_id: int, 
                           ground_truth: str, is_correct: bool, final_answer: str):
        """在单个子图上绘制熵值曲线"""
        if not records:
            ax.text(0.5, 0.5, 'No probe data', ha='center', va='center')
            return

        steps = [r.step for r in records]
        entropies = [r.entropy for r in records]
        stages = [r.stage for r in records]
        answers = [str(r.answer) if r.answer else 'None' for r in records]

        # 绘制主曲线
        ax.plot(steps, entropies, color='gray', alpha=0.4, linestyle='--', 
                linewidth=1.5, zorder=1)

        # 绘制散点
        for i, (step, ent, stage) in enumerate(zip(steps, entropies, stages)):
            color = EntropyVisualizer.STAGE_COLORS.get(stage, 'gray')
            size = 120 if stage == 'answer_signal' else 60
            ax.scatter(step, ent, color=color, s=size, zorder=2, 
                      edgecolors='white', linewidth=1.5, alpha=0.8)

        # 智能标注所有有答案的点
        label_positions = EntropyVisualizer.smart_label_placement(
            steps, entropies, answers, ground_truth
        )
        
        for i, (step, ent, ans) in enumerate(zip(steps, entropies, answers)):
            if ans == 'None' or label_positions[i] is None:
                continue
            
            _, _, y_offset = label_positions[i]
            
            # 判断答案是否正确
            ans_is_correct = (ans == str(ground_truth))
            ans_color = '#27ae60' if ans_is_correct else '#c0392b'
            weight = 'bold' if ans_is_correct else 'normal'
            
            # 调整字体大小和边框
            fontsize = 9 if ans_is_correct else 7
            linewidth = 2 if ans_is_correct else 1
            
            ax.annotate(
                f"{ans}", 
                (step, ent),
                xytext=(0, y_offset), 
                textcoords='offset points',
                ha='center', 
                fontsize=fontsize,
                color=ans_color,
                fontweight=weight,
                bbox=dict(
                    boxstyle="round,pad=0.3", 
                    fc="white", 
                    ec=ans_color, 
                    alpha=0.9, 
                    linewidth=linewidth
                ),
                zorder=10
            )

        # 标题和标签
        title_color = '#27ae60' if is_correct else '#c0392b'
        status = '✓' if is_correct else '✗'
        final_display = final_answer if final_answer else "None"
        
        ax.set_title(
            f"Sample #{sample_id} {status}\nGT: {ground_truth} | Final: {final_display}", 
            fontsize=10, fontweight='bold', color=title_color, pad=10
        )
        ax.set_xlabel("Token Steps", fontsize=9)
        ax.set_ylabel("Entropy", fontsize=9)
        
        # 添加低熵阈值线
        ax.axhline(y=0.6, color='red', linestyle=':', alpha=0.3, linewidth=1, label='Low Entropy')
        
        # 网格和样式
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#fafafa')
        
    @staticmethod
    def plot_combined_entropy(correct_results: List[Dict], wrong_results: List[Dict], 
                             save_path: Path):
        """绘制正确和错误案例的对比图"""
        if not HAS_PLOT:
            return

        n_correct = len(correct_results)
        n_wrong = len(wrong_results)
        total = n_correct + n_wrong
        
        if total == 0:
            print("⚠️ No results to plot")
            return

        cols = 3
        rows = (total + cols - 1) // cols
        
        fig = plt.figure(figsize=(18, 5 * rows))
        gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.25)
        
        # 绘制正确案例
        for idx, result in enumerate(correct_results):
            row = idx // cols
            col = idx % cols
            ax = fig.add_subplot(gs[row, col])
            
            records = [ProbeRecord.from_dict(r) for r in result['probe_history']]
            EntropyVisualizer.plot_single_entropy(
                ax, records,
                result['sample_id'],
                result['ground_truth'],
                True,
                result['final_answer']
            )
        
        # 绘制错误案例
        offset = n_correct
        for idx, result in enumerate(wrong_results):
            plot_idx = offset + idx
            row = plot_idx // cols
            col = plot_idx % cols
            ax = fig.add_subplot(gs[row, col])
            
            records = [ProbeRecord.from_dict(r) for r in result['probe_history']]
            EntropyVisualizer.plot_single_entropy(
                ax, records,
                result['sample_id'],
                result['ground_truth'],
                False,
                result['final_answer']
            )
        
        # 全局图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=10, label=stage)
            for stage, color in EntropyVisualizer.STAGE_COLORS.items()
        ]
        fig.legend(handles=legend_elements, loc='upper center', 
                  ncol=4, frameon=True, fontsize=11, 
                  bbox_to_anchor=(0.5, 0.99))
        
        fig.suptitle(
            f"Entropy Dynamics Analysis: {n_correct} Correct vs {n_wrong} Wrong",
            fontsize=16, fontweight='bold', y=0.997
        )
        
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📊 Combined plot saved to: {save_path}")

    @staticmethod
    def plot_statistics_comparison(correct_results: List[Dict], 
                                   wrong_results: List[Dict], 
                                   save_path: Path):
        """绘制统计对比图"""
        if not HAS_PLOT or (not correct_results and not wrong_results):
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Correct vs Wrong Cases: Statistical Comparison", 
                    fontsize=14, fontweight='bold')

        def extract_stats(results):
            all_entropies = []
            probe_counts = []
            final_entropies = []
            
            for r in results:
                history = r['probe_history']
                if history:
                    entropies = [h['entropy'] for h in history]
                    all_entropies.extend(entropies)
                    probe_counts.append(len(history))
                    final_entropies.append(entropies[-1])
            
            return all_entropies, probe_counts, final_entropies

        correct_ent, correct_counts, correct_final = extract_stats(correct_results)
        wrong_ent, wrong_counts, wrong_final = extract_stats(wrong_results)

        # 1. 熵值分布
        ax1 = axes[0, 0]
        if correct_ent:
            ax1.hist(correct_ent, bins=20, alpha=0.6, color='green', label='Correct', density=True)
        if wrong_ent:
            ax1.hist(wrong_ent, bins=20, alpha=0.6, color='red', label='Wrong', density=True)
        ax1.set_xlabel("Entropy Value")
        ax1.set_ylabel("Density")
        ax1.set_title("Entropy Distribution")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. 探测次数
        ax2 = axes[0, 1]
        data_to_plot = []
        labels = []
        if correct_counts:
            data_to_plot.append(correct_counts)
            labels.append('Correct')
        if wrong_counts:
            data_to_plot.append(wrong_counts)
            labels.append('Wrong')
        if data_to_plot:
            bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
        ax2.set_ylabel("Number of Probes")
        ax2.set_title("Probe Count Comparison")
        ax2.grid(alpha=0.3)

        # 3. 最终熵值
        ax3 = axes[1, 0]
        x_pos = []
        heights = []
        colors = []
        if correct_final:
            x_pos.append(0)
            heights.append(np.mean(correct_final))
            colors.append('green')
        if wrong_final:
            x_pos.append(1)
            heights.append(np.mean(wrong_final))
            colors.append('red')
        if x_pos:
            ax3.bar(x_pos, heights, color=colors, alpha=0.7, width=0.6)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(['Correct', 'Wrong'])
            ax3.set_ylabel("Average Final Entropy")
            ax3.set_title("Final Entropy Comparison")
            ax3.grid(alpha=0.3, axis='y')

        # 4. 准确率
        ax4 = axes[1, 1]
        sizes = [len(correct_results), len(wrong_results)]
        labels_pie = ['Correct', 'Wrong']
        colors_pie = ['#27ae60', '#c0392b']
        if sum(sizes) > 0:
            ax4.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
            ax4.set_title("Overall Accuracy")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Statistics plot saved to: {save_path}")

# ============================================================================
# 实验运行器
# ============================================================================
class EntropyExperimentRunner:
    def __init__(self):
        self.config = self._get_config()
        self.tokenizer = None
        self.model = None
        
    def _get_config(self):
        return {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "data_path": str(DATA_DIR / "gsm8k_test.json"),
            "sample_size": 20,
            "cooldown": 8,
            "max_tokens": 1024,
            "debug": True
        }

    def load_resources(self):
        print(f"🤖 Loading model: {self.config['model_name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'], 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'], 
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        self.probe_system = SmartProbeSystem(self.model, self.tokenizer)

    def is_sentence_boundary(self, text: str) -> bool:
        """检查是否是句子边界"""
        if not text: 
            return False
        return text.strip()[-1] in ['.', '!', '?', ':', '\n']
    
    def should_stop_generation(self, text: str) -> bool:
        """检查是否应该停止生成"""
        # 检查异常模式
        for pattern in TextCleaner.STOP_PATTERNS:
            if pattern in text:
                return True
        return False

    def run_sample(self, sample_id: int, question: str, ground_truth: str):
        print(f"\n{'='*40}\n🧪 Sample {sample_id}: {question[:50]}...")
        
        prompt = f"Question: {question}\nPlease solve this step by step.\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        
        state = GenerationState(full_sequence_ids=input_ids)
        probe_records: List[ProbeRecord] = []
        
        last_probe_step = 0
        current_input_ids = input_ids
        past_key_values = None
        
        while state.tokens_used < self.config['max_tokens']:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                past_key_values = outputs.past_key_values
            
            state.full_sequence_ids = torch.cat([state.full_sequence_ids, next_token], dim=-1)
            current_input_ids = next_token
            state.tokens_used += 1
            
            new_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            state.full_response += new_text
            
            # 检查停止条件
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            if self.should_stop_generation(state.full_response):
                print(f"   ⚠️ Early stop: Detected abnormal pattern")
                state.full_response = TextCleaner.clean_response(state.full_response, verbose=True)
                break

            # 探测逻辑
            is_boundary = self.is_sentence_boundary(state.full_response)
            is_cooldown_ok = (state.tokens_used - last_probe_step) >= self.config['cooldown']
            
            if is_boundary and is_cooldown_ok:
                stage = self.probe_system.identify_reasoning_stage(state.full_response)
                result = self.probe_system.probe_answer(
                    state.full_sequence_ids, 
                    state.full_response, 
                    stage
                )
                
                record = ProbeRecord(
                    step=state.tokens_used,
                    stage=stage,
                    answer=result.answer,
                    entropy=result.entropy,
                    text_segment=state.full_response[-30:].replace('\n', ' ')
                )
                probe_records.append(record)
                last_probe_step = state.tokens_used
                
                print(f"   📍 Step {state.tokens_used:3d} [{stage:12s}] "
                      f"Entropy: {result.entropy:.4f} | Ans: {result.answer}")

        # 清理响应（不打印警告，因为已经在上面打印过了）
        state.full_response = TextCleaner.clean_response(state.full_response, verbose=False)
        
        # 提取最终答案
        final_answer = AnswerExtractor.extract_answer(state.full_response)
        is_correct = (str(final_answer) == str(ground_truth))
        
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        print(f"🏁 {status} | Final: {final_answer} | GT: {ground_truth}")
        
        # 调试信息
        if self.config['debug'] and not final_answer:
            print(f"\n⚠️ Debug - Last 300 chars:\n{state.full_response[-300:]}\n")

        return {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "final_answer": final_answer,
            "correct": is_correct,
            "response": state.full_response,
            "probe_history": [record.to_dict() for record in probe_records]
        }

    def run(self):
        self.load_resources()
        
        with open(self.config['data_path'], 'r') as f:
            data = json.load(f)
        
        valid_data = []
        for item in data:
            ans = AnswerExtractor.extract_answer(item.get('answer', ''))
            if ans:
                item['clean_answer'] = ans
                valid_data.append(item)
        
        test_data = valid_data[:self.config['sample_size']]
        results = []
        
        for i, item in enumerate(test_data):
            res = self.run_sample(i, item['question'], item['clean_answer'])
            results.append(res)
        
        # 分类结果
        correct_results = [r for r in results if r['correct']]
        wrong_results = [r for r in results if not r['correct']]
        
        print(f"\n{'='*60}")
        print(f"📊 Summary: {len(correct_results)} Correct | {len(wrong_results)} Wrong")
        print(f"   Accuracy: {len(correct_results)/len(results)*100:.1f}%")
        
        # 保存JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = RESULTS_DIR / f"entropy_analysis_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "config": self.config,
                "summary": {
                    "total": len(results),
                    "correct": len(correct_results),
                    "wrong": len(wrong_results),
                    "accuracy": len(correct_results) / len(results) if results else 0
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 Data saved to: {json_path}")
        
        # 生成可视化
        if HAS_PLOT:
            combined_path = RESULTS_DIR / "plots" / f"entropy_combined_{timestamp}.png"
            EntropyVisualizer.plot_combined_entropy(
                correct_results, wrong_results, combined_path
            )
            
            stats_path = RESULTS_DIR / "plots" / f"entropy_statistics_{timestamp}.png"
            EntropyVisualizer.plot_statistics_comparison(
                correct_results, wrong_results, stats_path
            )

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    runner = EntropyExperimentRunner()
    runner.run()
