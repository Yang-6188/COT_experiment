"""
熵变化率分析可视化脚本
作者: Assistant
日期: 2025-12-04
用途: 分析HALT-CoT实验中的熵变化趋势和探针准确性
"""

import json
# ===== 关键修改开始 =====
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 提前过滤警告
# ===== 关键修改结束 =====

import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# ============================================================
# 配置区域 - 在这里设置你的文件路径
# ============================================================
JSON_FILE_PATH = "/root/autodl-tmp/results/halt_cot_qwen2.5-7b_smart_50samples_20251204_171920.json"
OUTPUT_DIR = "./output"
SAMPLES_TO_PLOT = "all"  # 改为 "all" 表示所有样本
SAMPLES_PER_FIGURE = 5  # 每张图显示5个样本
PLOT_ALL_STATISTICS = True
DPI = 300
FIGURE_FORMAT = 'png'
# ============================================================


# 在文件开头，setup_chinese_font() 函数之前添加
def clean_latex_text(text):
    """清理文本中的LaTeX符号，避免matplotlib解析错误"""
    import re
    if text is None:
        return ""
    text = str(text)
    
    # 移除美元符号（LaTeX数学模式标记）
    text = text.replace('$', '')
    
    # 移除或替换LaTeX命令
    # 例如: \sqrt, \log, \frac 等
    text = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', text)
    text = re.sub(r'\\log_\{([^}]*)\}', r'log_\1', text)
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # 移除其他LaTeX命令
    
    # 替换花括号
    text = text.replace('{', '(').replace('}', ')')
    
    # 移除下划线和上标符号（如果不在数学模式中）
    # text = text.replace('_', ' ').replace('^', ' ')
    
    return text




def setup_chinese_font():
    """配置中文字体 - 简化版"""
    # 直接设置，不要在函数内部过滤警告（已经在外面过滤了）
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✓ 中文字体配置完成")

def load_data(file_path):
    """加载JSON数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ 成功加载数据文件: {file_path}")
        return data
    except FileNotFoundError:
        print(f"✗ 错误: 找不到文件 {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"✗ 错误: 文件 {file_path} 不是有效的JSON格式")
        sys.exit(1)


def create_output_dir(output_dir):
    """创建输出目录"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ 输出目录: {output_dir}")


def print_experiment_info(data):
    """打印实验基本信息"""
    print("\n" + "=" * 70)
    print("📊 HALT-CoT 实验统计信息")
    print("=" * 70)
    
    exp_info = data['experiment_info']
    stats = data['statistics']
    
    print(f"🤖 模型: {exp_info['model']}")
    print(f"📅 时间戳: {exp_info['timestamp']}")
    print(f"📝 样本数: {stats['total_samples']}")
    print(f"✅ 正确数: {stats['correct_samples']}")
    print(f"🎯 准确率: {stats['accuracy']*100:.2f}%")
    print(f"🛑 早停率: {stats['early_stop_rate']*100:.2f}%")
    print(f"⏱️  平均用时: {stats['avg_time_per_sample']:.2f}秒/样本")
    print(f"💬 平均Token: {stats['avg_tokens_per_sample']:.2f}个/样本")
    print(f"📊 Token范围: {stats['min_tokens']} - {stats['max_tokens']}")
    print(f"📉 平均熵: {stats['avg_entropy']:.4f}")
    
    if stats.get('halt_reasons'):
        print(f"\n🔍 早停原因分布:")
        for reason, count in stats['halt_reasons'].items():
            print(f"   - {reason}: {count}次")
    
    print("=" * 70 + "\n")


def plot_entropy_analysis(data, sample_ids=None, output_dir='./output', dpi=300, fmt='png', samples_per_figure=5):
    """
    绘制熵变化率分析图
    
    参数:
        data: 实验数据字典
        sample_ids: 要绘制的样本ID列表，None表示绘制前5个样本，"all"表示所有样本
        output_dir: 输出目录
        dpi: 图片分辨率
        fmt: 图片格式
        samples_per_figure: 每张图显示多少个样本
    """
    results = data['results']
    
    # 处理 sample_ids
    if sample_ids == "all":
        sample_ids = list(range(len(results)))
        print(f"📈 将绘制所有 {len(results)} 个样本的熵分析图...")
    elif sample_ids is None:
        sample_ids = list(range(min(5, len(results))))
    
    # 过滤掉不存在的样本ID
    sample_ids = [sid for sid in sample_ids if sid < len(results)]
    
    if not sample_ids:
        print("⚠️  警告: 没有有效的样本ID可供绘制")
        return
    
    # 分批绘制
    total_samples = len(sample_ids)
    num_figures = (total_samples + samples_per_figure - 1) // samples_per_figure
    
    print(f"📈 正在绘制 {total_samples} 个样本，分为 {num_figures} 张图...")
    
    setup_chinese_font()
    
    for fig_idx in range(num_figures):
        start_idx = fig_idx * samples_per_figure
        end_idx = min(start_idx + samples_per_figure, total_samples)
        batch_sample_ids = sample_ids[start_idx:end_idx]
        
        print(f"  正在绘制第 {fig_idx + 1}/{num_figures} 张图 (样本 {batch_sample_ids})...")
        
        # 创建子图
        n_samples = len(batch_sample_ids)
        fig, axes = plt.subplots(n_samples, 1, figsize=(16, 5*n_samples))
        
        if n_samples == 1:
            axes = [axes]
        
        for idx, sample_id in enumerate(batch_sample_ids):
            sample = results[sample_id]
            ax = axes[idx]
            
            # 提取数据
            probe_history = sample['probe_history']
            
            if not probe_history:
                print(f"    ⚠️  警告: 样本 {sample_id} 没有探针历史数据")
                continue
            
            token_positions = [p['token_position'] for p in probe_history]
            entropies = [p['entropy'] for p in probe_history]
            probed_answers = [p['probed_answer'] for p in probe_history]
            stages = [p['stage'] for p in probe_history]
            confidences = [p.get('confidence', 0) for p in probe_history]
            
            ground_truth = str(sample['ground_truth'])
            is_correct = [str(ans) == ground_truth for ans in probed_answers]
            
            # 计算熵变化率
            entropy_rates = [0]  # 第一个点的变化率为0
            for i in range(1, len(entropies)):
                token_diff = token_positions[i] - token_positions[i-1]
                if token_diff > 0:
                    rate = (entropies[i] - entropies[i-1]) / token_diff
                    entropy_rates.append(rate)
                else:
                    entropy_rates.append(0)
            
            # 创建双Y轴
            ax2 = ax.twinx()
            
            # 绘制熵值曲线
            line1 = ax.plot(token_positions, entropies, 'b-', linewidth=2.5, 
                           label='熵值', marker='o', markersize=10, markerfacecolor='lightblue',
                           markeredgecolor='blue', markeredgewidth=2)
            
            # 绘制熵变化率
            if len(token_positions) > 1:
                line2 = ax2.plot(token_positions, entropy_rates, 'g--', linewidth=2, 
                               label='熵变化率', marker='s', markersize=7, 
                               markerfacecolor='lightgreen', markeredgecolor='green',
                               markeredgewidth=1.5, alpha=0.8)
            else:
                line2 = []
            
            # 标注探针点
            for i, (pos, ent, ans, correct, stage, conf) in enumerate(
                zip(token_positions, entropies, probed_answers, is_correct, stages, confidences)):
                
                # 根据正确性选择颜色和标记
                color = 'green' if correct else 'red'
                marker = 'o' if correct else 'X'
                
                # 绘制探针标记（更大更明显）
                ax.scatter(pos, ent, s=300, c=color, marker=marker, 
                          edgecolors='black', linewidths=2.5, zorder=5, alpha=0.8)
                
                # 添加答案标注
                y_offset = 20 if i % 2 == 0 else -35
                annotation_text = f'答案: {ans}\n阶段: {stage}\n置信度: {conf:.2f}'
                
                ax.annotate(annotation_text, 
                           xy=(pos, ent), 
                           xytext=(0, y_offset),
                           textcoords='offset points',
                           ha='center',
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.5', 
                                    facecolor=color, 
                                    alpha=0.25,
                                    edgecolor='black',
                                    linewidth=1.5),
                           arrowprops=dict(arrowstyle='->', 
                                         connectionstyle='arc3,rad=0',
                                         color='black',
                                         lw=1.5))
            
            # 设置标题 - 使用清理后的文本
            correct_mark = "✓" if sample['correct'] else "✗"
            
            # 清理所有可能包含LaTeX的文本
            question_clean = clean_latex_text(sample["question"][:100])
            gt_clean = clean_latex_text(str(ground_truth))
            pred_clean = clean_latex_text(str(sample["predicted_answer"]))
            
            title_text = (f'样本 {sample_id} {correct_mark}\n'
                        f'问题: {question_clean}...\n'
                        f'正确答案: {gt_clean} | 预测答案: {pred_clean} | '
                        f'平均熵: {sample["avg_entropy"]:.4f} | Token数: {sample["tokens_used"]}')
            
            ax.set_title(title_text, fontsize=12, pad=15, fontweight='bold')

            
            # 设置标签
            ax.set_xlabel('Token 位置', fontsize=11, fontweight='bold')
            ax.set_ylabel('熵值', fontsize=11, color='b', fontweight='bold')
            ax2.set_ylabel('熵变化率 (Δ熵/ΔToken)', fontsize=11, color='g', fontweight='bold')
            
            ax.tick_params(axis='y', labelcolor='b', labelsize=10)
            ax2.tick_params(axis='y', labelcolor='g', labelsize=10)
            ax.tick_params(axis='x', labelsize=10)
            
            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
            
            # 添加早停标记
            if sample['early_stopped']:
                halt_reason = sample.get('halt_reason', 'unknown')
                ax.axvline(x=token_positions[-1], color='purple', linestyle=':', 
                          linewidth=3, label=f'早停: {halt_reason}', alpha=0.7)
            
            # 添加零线（熵变化率）
            if len(token_positions) > 1:
                ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            
            # 添加自定义图例项
            correct_patch = mpatches.Patch(color='green', label='✓ 正确答案', alpha=0.7)
            incorrect_patch = mpatches.Patch(color='red', label='✗ 错误答案', alpha=0.7)
            
            legend_handles = lines + [correct_patch, incorrect_patch]
            
            if sample['early_stopped']:
                early_stop_line = plt.Line2D([0], [0], color='purple', linewidth=3, 
                                            linestyle=':', label=f'早停: {halt_reason}')
                legend_handles.append(early_stop_line)
            
            ax.legend(handles=legend_handles, loc='upper left', fontsize=10, 
                     framealpha=0.9, edgecolor='black')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = Path(output_dir) / f'entropy_analysis_samples_{start_idx}-{end_idx-1}.{fmt}'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"    ✓ 已保存: {output_path}")
        
        plt.close()
    
    print(f"✓ 所有熵分析图已保存完成")

def plot_entropy_statistics(data, output_dir='./output', dpi=300, fmt='png'):
    """
    绘制整体统计图表
    """
    print(f"📊 正在绘制整体统计图...")
    
    results = data['results']
    setup_chinese_font()
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 平均熵分布（正确 vs 错误）
    ax1 = fig.add_subplot(gs[0, 0])
    correct_entropies = [r['avg_entropy'] for r in results if r['correct']]
    incorrect_entropies = [r['avg_entropy'] for r in results if not r['correct']]
    
    ax1.hist([correct_entropies, incorrect_entropies], 
            bins=20, label=['正确', '错误'], 
            color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax1.set_xlabel('平均熵值', fontsize=11, fontweight='bold')
    ax1.set_ylabel('样本数量', fontsize=11, fontweight='bold')
    ax1.set_title('平均熵分布对比', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    if correct_entropies:
        ax1.axvline(np.mean(correct_entropies), color='green', linestyle='--', 
                   linewidth=2, label=f'正确均值: {np.mean(correct_entropies):.3f}')
    if incorrect_entropies:
        ax1.axvline(np.mean(incorrect_entropies), color='red', linestyle='--', 
                   linewidth=2, label=f'错误均值: {np.mean(incorrect_entropies):.3f}')
    ax1.legend(fontsize=9)
    
    # 2. Token数量 vs 准确性
    ax2 = fig.add_subplot(gs[0, 1])
    token_counts = [r['tokens_used'] for r in results]
    colors = ['green' if r['correct'] else 'red' for r in results]
    
    ax2.scatter(range(len(results)), token_counts, c=colors, alpha=0.6, s=100, edgecolors='black')
    ax2.set_xlabel('样本ID', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Token数量', fontsize=11, fontweight='bold')
    ax2.set_title('Token使用量分布', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    correct_patch = mpatches.Patch(color='green', label='正确', alpha=0.6)
    incorrect_patch = mpatches.Patch(color='red', label='错误', alpha=0.6)
    ax2.legend(handles=[correct_patch, incorrect_patch], fontsize=10)
    
    # 3. 准确率饼图
    ax3 = fig.add_subplot(gs[0, 2])
    correct_count = sum(1 for r in results if r['correct'])
    incorrect_count = len(results) - correct_count
    
    colors_pie = ['green', 'red']
    explode = (0.05, 0.05)
    ax3.pie([correct_count, incorrect_count], labels=['正确', '错误'], 
           autopct='%1.1f%%', startangle=90, colors=colors_pie, explode=explode,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title(f'准确率: {correct_count}/{len(results)}', fontsize=12, fontweight='bold')
    
    # 4. 熵变化趋势（所有样本的平均）
    ax4 = fig.add_subplot(gs[1, :2])
    max_probes = max(len(r['entropy_history']) for r in results if r['entropy_history'])
    entropy_by_position = [[] for _ in range(max_probes)]
    
    for r in results:
        for i, ent in enumerate(r['entropy_history']):
            if i < max_probes:
                entropy_by_position[i].append(ent)
    
    avg_entropies = [np.mean(e) if e else 0 for e in entropy_by_position]
    std_entropies = [np.std(e) if e else 0 for e in entropy_by_position]
    
    x = range(len(avg_entropies))
    ax4.plot(x, avg_entropies, 'b-', linewidth=3, marker='o', markersize=8, label='平均熵')
    ax4.fill_between(x, 
                    np.array(avg_entropies) - np.array(std_entropies),
                    np.array(avg_entropies) + np.array(std_entropies),
                    alpha=0.3, label='标准差范围')
    ax4.set_xlabel('探针位置', fontsize=11, fontweight='bold')
    ax4.set_ylabel('平均熵值', fontsize=11, fontweight='bold')
    ax4.set_title('熵值随探针位置的变化趋势（所有样本）', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. 阶段分布统计
    ax5 = fig.add_subplot(gs[1, 2])
    stage_counts = {}
    for r in results:
        for stage, count in r['stage_distribution'].items():
            stage_counts[stage] = stage_counts.get(stage, 0) + count
    
    stages = list(stage_counts.keys())
    counts = list(stage_counts.values())
    colors_stage = plt.cm.Set3(range(len(stages)))
    
    bars = ax5.bar(stages, counts, color=colors_stage, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('推理阶段', fontsize=11, fontweight='bold')
    ax5.set_ylabel('探针次数', fontsize=11, fontweight='bold')
    ax5.set_title('推理阶段分布', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. 早停统计
    ax6 = fig.add_subplot(gs[2, 0])
    early_stopped = sum(1 for r in results if r['early_stopped'])
    not_stopped = len(results) - early_stopped
    
    ax6.bar(['早停', '未早停'], [early_stopped, not_stopped], 
           color=['purple', 'gray'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('样本数量', fontsize=11, fontweight='bold')
    ax6.set_title(f'早停统计 (早停率: {early_stopped/len(results)*100:.1f}%)', 
                 fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate([early_stopped, not_stopped]):
        ax6.text(i, v, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 7. 生成时间分布
    ax7 = fig.add_subplot(gs[2, 1])
    gen_times = [r['generation_time'] for r in results]
    ax7.hist(gen_times, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax7.axvline(np.mean(gen_times), color='red', linestyle='--', linewidth=2,
               label=f'平均: {np.mean(gen_times):.2f}s')
    ax7.set_xlabel('生成时间 (秒)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('样本数量', fontsize=11, fontweight='bold')
    ax7.set_title('生成时间分布', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # 8. Token数量 vs 熵值散点图
    ax8 = fig.add_subplot(gs[2, 2])
    tokens = [r['tokens_used'] for r in results]
    entropies = [r['avg_entropy'] for r in results]
    colors_scatter = ['green' if r['correct'] else 'red' for r in results]
    
    ax8.scatter(tokens, entropies, c=colors_scatter, alpha=0.6, s=100, edgecolors='black')
    ax8.set_xlabel('Token数量', fontsize=11, fontweight='bold')
    ax8.set_ylabel('平均熵值', fontsize=11, fontweight='bold')
    ax8.set_title('Token数量 vs 平均熵值', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend(handles=[correct_patch, incorrect_patch], fontsize=10)
    
    # 添加总标题
    fig.suptitle('HALT-CoT 实验整体统计分析', fontsize=16, fontweight='bold', y=0.995)
    
    # 保存图片
    output_path = Path(output_dir) / f'entropy_statistics.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ 统计图已保存: {output_path}")
    
    plt.close()


def generate_summary_report(data, output_dir='./output'):
    """生成文本摘要报告"""
    print(f"📝 正在生成摘要报告...")
    
    results = data['results']
    stats = data['statistics']
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HALT-CoT 实验详细报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 基本信息
    report_lines.append("【实验配置】")
    report_lines.append(f"模型: {data['experiment_info']['model']}")
    report_lines.append(f"时间戳: {data['experiment_info']['timestamp']}")
    report_lines.append(f"样本大小: {data['experiment_info']['sample_size']}")
    report_lines.append(f"检测模式: {data['experiment_info']['stage_detection_mode']}")
    report_lines.append("")
    
    # 统计信息
    report_lines.append("【整体统计】")
    report_lines.append(f"总样本数: {stats['total_samples']}")
    report_lines.append(f"正确样本: {stats['correct_samples']}")
    report_lines.append(f"准确率: {stats['accuracy']*100:.2f}%")
    report_lines.append(f"早停率: {stats['early_stop_rate']*100:.2f}%")
    report_lines.append(f"平均用时: {stats['avg_time_per_sample']:.2f}秒/样本")
    report_lines.append(f"总用时: {stats['total_time']:.2f}秒")
    report_lines.append(f"平均Token: {stats['avg_tokens_per_sample']:.2f}个/样本")
    report_lines.append(f"Token范围: {stats['min_tokens']} - {stats['max_tokens']}")
    report_lines.append(f"平均熵: {stats['avg_entropy']:.4f}")
    report_lines.append("")
    
    # 早停原因
    if stats.get('halt_reasons'):
        report_lines.append("【早停原因分布】")
        for reason, count in stats['halt_reasons'].items():
            report_lines.append(f"  - {reason}: {count}次")
        report_lines.append("")
    
    # 样本详情
    report_lines.append("【样本详情】")
    for i, r in enumerate(results[:10]):  # 只显示前10个样本
        status = "✓" if r['correct'] else "✗"
        report_lines.append(f"\n样本 {i} {status}")
        report_lines.append(f"  问题: {r['question'][:80]}...")
        report_lines.append(f"  正确答案: {r['ground_truth']}")
        report_lines.append(f"  预测答案: {r['predicted_answer']}")
        report_lines.append(f"  Token数: {r['tokens_used']}")
        report_lines.append(f"  平均熵: {r['avg_entropy']:.4f}")
        report_lines.append(f"  早停: {'是' if r['early_stopped'] else '否'}")
        if r['early_stopped']:
            report_lines.append(f"  早停原因: {r['halt_reason']}")
    
    if len(results) > 10:
        report_lines.append(f"\n... 还有 {len(results) - 10} 个样本未显示")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_path = Path(output_dir) / 'summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 摘要报告已保存: {report_path}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🚀 HALT-CoT 熵变化率分析脚本")
    print("=" * 70 + "\n")
    
    # 1. 加载数据
    data = load_data(JSON_FILE_PATH)
    
    # 2. 创建输出目录
    create_output_dir(OUTPUT_DIR)
    
    # 3. 打印实验信息
    print_experiment_info(data)
    
    # 4. 绘制详细的熵分析图
    if SAMPLES_TO_PLOT:
        plot_entropy_analysis(data, sample_ids=SAMPLES_TO_PLOT, 
                            output_dir=OUTPUT_DIR, dpi=DPI, fmt=FIGURE_FORMAT,
                            samples_per_figure=SAMPLES_PER_FIGURE)
    
    # 5. 绘制整体统计图
    if PLOT_ALL_STATISTICS:
        plot_entropy_statistics(data, output_dir=OUTPUT_DIR, dpi=DPI, fmt=FIGURE_FORMAT)
    
    # 6. 生成摘要报告
    generate_summary_report(data, output_dir=OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("✅ 所有分析完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 70 + "\n")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
