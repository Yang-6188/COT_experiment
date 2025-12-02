#!/usr/bin/env python3
"""
配置管理器 - 交互式菜单系统
支持选择数据集并逐项修改参数
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil


class InteractiveConfigManager:
    """交互式配置管理器"""
    
    # 数据集配置
    DATASETS = {
        "gsm8k": {
            "gsm8k_5": {"name": "GSM8K (5样本)", "file": "gsm8k_test_sample_5.json", "size": 5},
            "gsm8k_50": {"name": "GSM8K (50样本)", "file": "gsm8k_test_sample_50.json", "size": 50},
            "gsm8k_full": {"name": "GSM8K (完整)", "file": "gsm8k_test.json", "size": 1319},
        },
        "math": {
            "math_l1": {"name": "MATH Level 1", "file": "math_test_level_Level_1.json", "size": 50},
            "math_l3": {"name": "MATH Level 3", "file": "math_test_level_Level_3.json", "size": 50},
            "math_l5": {"name": "MATH Level 5", "file": "math_test_level_Level_5.json", "size": 50},
        }
    }
    
    # 默认配置
    DEFAULT_CONFIG = {
        "data": "gsm8k_test_sample_5.json",
        "sample_size": 5,
        "max_tokens": 512,
        "temperature": 0.7,
        "do_sample": False,
        "verbose": False,
        "debug": False,
        "use_early_stop": True,
        "consistency_k": 3,
        "entropy_threshold": 0.6,
        "entropy_steps": 2,
        "min_tokens": 100,
        "cooldown": 40
    }
    
    def __init__(self, base_dir: str = "/root/autodl-tmp"):
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "config"
        self.backup_dir = self.config_dir / "backups"
        self.data_dir = self.base_dir / "data"
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_config_path = self.config_dir / "config.json"
        self.current_config = self.load_current_config()
    
    def load_current_config(self) -> Dict[str, Any]:
        """加载当前配置"""
        if self.current_config_path.exists():
            try:
                with open(self.current_config_path, 'r', encoding='utf-8') as f:
                    full_config = json.load(f)
                    # 提取简化配置
                    exp = full_config.get('experiment', {})
                    es = full_config.get('early_stopping', {})
                    paths = full_config.get('paths', {})
                    
                    return {
                        "data": paths.get('test_data', '').replace('data/', ''),
                        "sample_size": exp.get('sample_size', 5),
                        "max_tokens": exp.get('max_new_tokens', 512),
                        "temperature": exp.get('temperature', 0.7),
                        "do_sample": exp.get('do_sample', False),
                        "verbose": exp.get('verbose', False),
                        "debug": exp.get('debug_probe', False),
                        "use_early_stop": es.get('use_answer_consistency', True),
                        "consistency_k": es.get('consistency_k', 3),
                        "entropy_threshold": es.get('entropy_threshold', 0.6),
                        "entropy_steps": es.get('entropy_consecutive_steps', 2),
                        "min_tokens": es.get('min_tokens_before_check', 100),
                        "cooldown": es.get('cooldown_tokens', 40)
                    }
            except Exception as e:
                print(f"⚠️ 加载配置失败: {e}")
        
        return self.DEFAULT_CONFIG.copy()
    
    def show_header(self, title: str):
        """显示标题"""
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}\n")
    
    def show_config_summary(self, config: Dict[str, Any], show_title: bool = True):
        """显示配置摘要"""
        if show_title:
            print(f"\n{'='*80}")
            print(f"{'当前配置':^80}")
            print(f"{'='*80}")
        
        print(f"  📁 数据集: {config['data']}")
        print(f"  📊 样本数: {config['sample_size']}")
        print(f"  🔢 最大tokens: {config['max_tokens']}")
        print(f"  🌡️  温度: {config['temperature']}")
        print(f"  🎲 采样: {'✓' if config['do_sample'] else '✗'}")
        print(f"  📝 详细输出: {'✓' if config['verbose'] else '✗'}")
        print(f"  🐛 调试模式: {'✓' if config['debug'] else '✗'}")
        print(f"  ⏹️  早停机制: {'✓ 启用' if config['use_early_stop'] else '✗ 禁用'}")
        
        if config['use_early_stop']:
            print(f"     └─ 一致性窗口: {config['consistency_k']}")
            print(f"     └─ 熵值阈值: {config['entropy_threshold']}")
            print(f"     └─ 连续步数: {config['entropy_steps']}")
            print(f"     └─ 最小tokens: {config['min_tokens']}")
            print(f"     └─ 冷却tokens: {config['cooldown']}")
        
        if show_title:
            print(f"{'='*80}\n")
    
    def show_dataset_menu(self) -> Optional[tuple]:
        """显示数据集选择菜单"""
        self.show_header("选择数据集")
        
        index = 1
        index_map = {}
        
        for category, datasets in self.DATASETS.items():
            category_name = "GSM8K数据集" if category == "gsm8k" else "MATH数据集"
            print(f"【{category_name}】")
            for key, info in datasets.items():
                print(f"  {index}. {info['name']:<25} ({info['size']}样本)")
                index_map[str(index)] = (info['file'], info['size'])
                index += 1
            print()
        
        print("  0. 返回主菜单")
        print(f"\n{'='*80}\n")
        
        while True:
            choice = input("请选择数据集 [输入数字]: ").strip()
            
            if choice == '0':
                return None
            
            if choice in index_map:
                return index_map[choice]
            
            print("❌ 无效选择，请重新输入")
    
    def main_menu(self) -> Optional[str]:
        """主菜单"""
        self.show_header("配置管理器")
        self.show_config_summary(self.current_config, show_title=False)
        print(f"{'='*80}\n")
        
        print("【操作菜单】")
        print("  1. 选择数据集")
        print("  2. 修改生成参数 (tokens, 温度等)")
        print("  3. 修改早停参数")
        print("  4. 切换开关选项 (采样/详细输出/调试/早停)")
        print("  5. 重置为默认配置")
        print()
        print("  s. 保存当前配置")
        print("  0. 退出")
        print(f"\n{'='*80}\n")
        
        return input("请选择操作 [输入数字或字母]: ").strip().lower()
    
    def modify_generation_params(self):
        """修改生成参数"""
        self.show_header("修改生成参数")
        
        print(f"当前配置:")
        print(f"  样本数量: {self.current_config['sample_size']}")
        print(f"  最大tokens: {self.current_config['max_tokens']}")
        print(f"  温度参数: {self.current_config['temperature']}")
        print()
        
        try:
            val = input("样本数量 (回车跳过): ").strip()
            if val:
                self.current_config['sample_size'] = int(val)
                print(f"  ✓ 已修改为: {self.current_config['sample_size']}")
            
            val = input("最大tokens (回车跳过): ").strip()
            if val:
                self.current_config['max_tokens'] = int(val)
                print(f"  ✓ 已修改为: {self.current_config['max_tokens']}")
            
            val = input("温度参数 0.0-2.0 (回车跳过): ").strip()
            if val:
                self.current_config['temperature'] = float(val)
                print(f"  ✓ 已修改为: {self.current_config['temperature']}")
            
            print("\n✅ 参数修改完成")
        except ValueError:
            print("\n❌ 输入格式错误，修改已取消")
        
        input("\n按回车继续...")
    
    def modify_early_stop_params(self):
        """修改早停参数"""
        if not self.current_config['use_early_stop']:
            print("\n⚠️ 早停机制未启用，请先在开关选项中启用")
            input("按回车继续...")
            return
        
        self.show_header("修改早停参数")
        
        print(f"当前配置:")
        print(f"  一致性窗口: {self.current_config['consistency_k']}")
        print(f"  熵值阈值: {self.current_config['entropy_threshold']}")
        print(f"  连续步数: {self.current_config['entropy_steps']}")
        print(f"  最小tokens: {self.current_config['min_tokens']}")
        print(f"  冷却tokens: {self.current_config['cooldown']}")
        print()
        
        try:
            val = input("一致性窗口 (回车跳过): ").strip()
            if val:
                self.current_config['consistency_k'] = int(val)
            
            val = input("熵值阈值 (回车跳过): ").strip()
            if val:
                self.current_config['entropy_threshold'] = float(val)
            
            val = input("连续步数 (回车跳过): ").strip()
            if val:
                self.current_config['entropy_steps'] = int(val)
            
            val = input("最小tokens (回车跳过): ").strip()
            if val:
                self.current_config['min_tokens'] = int(val)
            
            val = input("冷却tokens (回车跳过): ").strip()
            if val:
                self.current_config['cooldown'] = int(val)
            
            print("\n✅ 早停参数修改完成")
        except ValueError:
            print("\n❌ 输入格式错误，修改已取消")
        
        input("\n按回车继续...")
    
    def toggle_switches(self):
        """切换开关选项"""
        self.show_header("切换开关选项")
        
        switches = {
            '1': ('do_sample', '采样模式'),
            '2': ('verbose', '详细输出'),
            '3': ('debug', '调试模式'),
            '4': ('use_early_stop', '早停机制'),
        }
        
        while True:
            print(f"\n当前状态:")
            for key, (config_key, name) in switches.items():
                status = '✓ 启用' if self.current_config[config_key] else '✗ 禁用'
                print(f"  {key}. {name:<12} [{status}]")
            
            print("\n  0. 返回主菜单")
            print()
            
            choice = input("选择要切换的选项 (输入数字): ").strip()
            
            if choice == '0':
                break
            
            if choice in switches:
                config_key, name = switches[choice]
                self.current_config[config_key] = not self.current_config[config_key]
                status = '启用' if self.current_config[config_key] else '禁用'
                print(f"  ✓ {name}已{status}")
            else:
                print("  ❌ 无效选择")
    
    def reset_to_default(self):
        """重置为默认配置"""
        confirm = input("\n⚠️ 确认重置为默认配置? (y/n): ").strip().lower()
        if confirm == 'y':
            self.current_config = self.DEFAULT_CONFIG.copy()
            print("✅ 已重置为默认配置")
        else:
            print("❌ 已取消重置")
        input("\n按回车继续...")
    
    def build_full_config(self) -> Dict[str, Any]:
        """构建完整配置"""
        return {
            "active_model": "qwen",
            "model_configs": {
                "qwen": {"name": "Qwen/Qwen2.5-7B-Instruct"}
            },
            "paths": {
                "test_data": f"data/{self.current_config['data']}"
            },
            "experiment": {
                "sample_size": self.current_config['sample_size'],
                "max_new_tokens": self.current_config['max_tokens'],
                "do_sample": self.current_config['do_sample'],
                "temperature": self.current_config['temperature'],
                "save_results": True,
                "verbose": self.current_config['verbose'],
                "debug_probe": self.current_config['debug']
            },
            "early_stopping": {
                "use_answer_consistency": self.current_config['use_early_stop'],
                "use_entropy_halt": self.current_config['use_early_stop'],
                "consistency_k": self.current_config['consistency_k'],
                "entropy_threshold": self.current_config['entropy_threshold'],
                "entropy_consecutive_steps": self.current_config['entropy_steps'],
                "min_tokens_before_check": self.current_config['min_tokens'],
                "cooldown_tokens": self.current_config['cooldown']
            }
        }
    
    def save_config(self):
        """保存配置"""
        # 备份旧配置
        if self.current_config_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"config_{timestamp}.json"
            shutil.copy(self.current_config_path, backup_path)
            print(f"📦 已备份旧配置到: {backup_path.name}")
        
        # 保存新配置
        full_config = self.build_full_config()
        with open(self.current_config_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置已保存到: {self.current_config_path}")
        input("\n按回车继续...")
    
    def run(self):
        """运行交互式配置"""
        while True:
            choice = self.main_menu()
            
            if choice == '0':
                print("\n👋 已退出配置管理器\n")
                break
            
            elif choice == '1':
                # 选择数据集
                result = self.show_dataset_menu()
                if result:
                    data_file, sample_size = result
                    self.current_config['data'] = data_file
                    self.current_config['sample_size'] = sample_size
                    print(f"\n✅ 已选择数据集: {data_file} ({sample_size}样本)")
                    input("按回车继续...")
            
            elif choice == '2':
                # 修改生成参数
                self.modify_generation_params()
            
            elif choice == '3':
                # 修改早停参数
                self.modify_early_stop_params()
            
            elif choice == '4':
                # 切换开关选项
                self.toggle_switches()
            
            elif choice == '5':
                # 重置为默认配置
                self.reset_to_default()
            
            elif choice == 's':
                # 保存配置
                self.save_config()
            
            else:
                print("\n❌ 无效选择")
                input("按回车继续...")


def main():
    """主函数"""
    manager = InteractiveConfigManager()
    manager.run()


if __name__ == "__main__":
    main()
