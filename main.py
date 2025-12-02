#!/usr/bin/env python3
"""
增强型HALT-CoT实验主入口
融合Liu & Wang (2025)的答案收敛检测和Laaouach的熵基早停方法
"""

from src.experiment_runner import ExperimentRunner


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
