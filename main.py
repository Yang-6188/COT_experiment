#!/usr/bin/env python3
"""
增强型HALT-CoT实验主入口
融合Liu & Wang (2025)的答案收敛检测和Laaouach的熵基早停方法
"""

import sys
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from src.experiment_runner import ExperimentRunner
# from src.config_manager import ConfigManager


@contextmanager
def log_to_file(log_file):
    """上下文管理器：将所有输出重定向到文件"""
    class Tee:
        def __init__(self, *files):
            self.files = files
        
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    # 保存原始输出
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # 打开日志文件
    log_file_obj = open(log_file, 'w', encoding='utf-8')
    
    try:
        # 重定向到文件和控制台
        sys.stdout = Tee(original_stdout, log_file_obj)
        sys.stderr = Tee(original_stderr, log_file_obj)
        yield log_file
    finally:
        # 恢复原始输出
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_obj.close()


def main():
    """主函数"""
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    with log_to_file(log_file):
        try:
            print(f"📝 日志文件: {log_file}")
            print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
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
            import traceback
            traceback.print_exc()
        finally:
            print("=" * 60)
            print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"✅ 日志已保存到: {log_file}")


if __name__ == "__main__":
    main()
