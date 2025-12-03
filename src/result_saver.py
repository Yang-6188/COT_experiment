"""结果保存器"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class ResultSaver:
    """
    结果保存器
    负责保存实验结果到文件
    """
    
    def __init__(self, results_dir: Path, config: dict):
        """
        初始化结果保存器
        
        Args:
            results_dir: 结果保存目录
            config: 配置信息
        """
        self.results_dir = results_dir
        self.config = config
    
    def save(self, results: List[Dict[str, Any]], 
             stats: Dict[str, Any]) -> Optional[Path]:
        """
        保存实验结果
        
        Args:
            results: 实验结果列表
            stats: 统计信息
            
        Returns:
            保存的文件路径
        """
        if not self.config['experiment']['save_results']:
            print("⚠️  结果保存已禁用")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_key = self.config['active_model']
        sample_size = len(results)
        
        # 确定检测模式
        stage_config = self.config.get('stage_control', {})
        use_smart = stage_config.get('use_smart_detection', True)
        stage_mode = "smart" if use_smart else "sentence"
        
        filename = f"halt_cot_{model_key}_{stage_mode}_{sample_size}samples_{timestamp}.json"
        results_file = self.results_dir / filename
        
        # 构建统计摘要
        summary_text = self._build_summary(stats)
        
        save_data = {
            "experiment_info": {
                "timestamp": timestamp,
                "model": self.config['model_configs'][model_key]['name'],
                "model_key": model_key,
                "sample_size": sample_size,
                "stage_detection_mode": stage_mode,
                "config": self.config['experiment'],
                "early_stopping_config": self.config.get('early_stopping', {}),
                "stage_control_config": self.config.get('stage_control', {})
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
    
    def _build_summary(self, stats: Dict[str, Any]) -> str:
        """构建统计摘要文本"""
        model_key = self.config['active_model']
        stage_config = self.config.get('stage_control', {})
        use_smart = stage_config.get('use_smart_detection', True)
        stage_mode_text = "智能检测" if use_smart else "句子边界检测"
        
        summary = f"""
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
🔍 平均检查点: {stats['avg_checkpoints_per_sample']:.1f}次/样本
🕐 总用时: {stats['total_time']//60:.0f}分{stats['total_time']%60:.0f}秒
🔍 检测模式: {stage_mode_text}
"""
        
        if stats.get('halt_reasons'):
            summary += "\n🔍 早停原因分布:\n"
            for reason, count in stats['halt_reasons'].items():
                if reason:
                    summary += f"   - {reason}: {count}次\n"
        
        summary += f"{'=' * 60}\n"
        return summary
