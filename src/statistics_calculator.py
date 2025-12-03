"""统计计算器"""
from typing import Dict, Any, List
from collections import Counter


class StatisticsCalculator:
    """
    统计计算器
    负责计算和展示实验统计信息
    """
    
    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算统计信息
        
        Args:
            results: 实验结果列表
            
        Returns:
            统计信息字典
        """
        if not results:
            return {}
        
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r['correct'])
        total_time = sum(r['generation_time'] for r in results)
        total_tokens = sum(r['tokens_used'] for r in results)
        early_stops = sum(1 for r in results if r.get('early_stopped', False))
        
        halt_reasons = Counter(r.get('halt_reason') for r in results if r.get('early_stopped', False))
        avg_entropy = sum(r.get('avg_entropy', 0) for r in results) / total_samples
        token_counts = [r['tokens_used'] for r in results]
        
        # 阶段检测模式统计
        stage_modes = Counter(r.get('stage_detection_mode', 'unknown') for r in results)
        
        # 检查点统计
        total_checkpoints = sum(
            r.get('checkpoint_stats', {}).get('total_checks', 0) 
            for r in results
        )
        
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
            "avg_entropy": avg_entropy,
            "stage_detection_modes": dict(stage_modes),
            "total_checkpoints": total_checkpoints,
            "avg_checkpoints_per_sample": total_checkpoints / total_samples if total_samples > 0 else 0
        }
    
    def print_statistics(self, stats: Dict[str, Any], config: dict):
        """
        打印统计信息
        
        Args:
            stats: 统计信息
            config: 配置信息
        """
        print("\n" + "=" * 60)
        print("📊 HALT-CoT 实验统计结果")
        print("=" * 60)
        print(f"🤖 模型: {config['model_configs'][config['active_model']]['name']}")
        print(f"📝 总样本数: {stats['total_samples']}")
        print(f"✅ 正确样本: {stats['correct_samples']}")
        print(f"🎯 准确率: {stats['accuracy']:.2%}")
        print(f"🛑 早停率: {stats['early_stop_rate']:.2%} ({stats['early_stops']}/{stats['total_samples']})")
        print(f"⏱️  平均用时: {stats['avg_time_per_sample']:.1f}秒/样本")
        print(f"💬 平均Token: {stats['avg_tokens_per_sample']:.1f}个/样本")
        print(f"📊 Token范围: {stats['min_tokens']} - {stats['max_tokens']}")
        print(f"📉 平均熵: {stats['avg_entropy']:.3f}")
        print(f"🔍 平均检查点: {stats['avg_checkpoints_per_sample']:.1f}次/样本")
        print(f"🕐 总用时: {stats['total_time']//60:.0f}分{stats['total_time']%60:.0f}秒")
        
        # 阶段检测模式信息
        stage_config = config.get('stage_control', {})
        use_smart = stage_config.get('use_smart_detection', True)
        stage_mode = "智能检测" if use_smart else "句子边界检测"
        print(f"🔍 检测模式: {stage_mode}")
        
        if stats.get('halt_reasons'):
            print("\n🔍 早停原因分布:")
            for reason, count in stats['halt_reasons'].items():
                if reason:
                    print(f"   - {reason}: {count}次")
        
        print("=" * 60)
