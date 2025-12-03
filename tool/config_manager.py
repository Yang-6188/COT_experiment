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
    
    # 模型配置
    MODELS = {
        "qwen2.5-1.5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "desc": "轻量级"},
        "qwen2.5-3b": {"name": "Qwen/Qwen2.5-3B-Instruct", "desc": "平衡性能"},
        "qwen2.5-7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "desc": "高性能"},
        "qwen2.5-14b": {"name": "Qwen/Qwen2.5-14B-Instruct", "desc": "最高性能"},
    }
    
    # 默认配置（简化版）
    DEFAULT_CONFIG = {
        # 模型
        "active_model": "qwen2.5-7b",
        
        # 数据集
        "data": "gsm8k_test_sample_5.json",
        "sample_size": 5,
        
        # 生成参数
        "max_tokens": 512,
        "temperature": 0.0,
        "do_sample": False,
        
        # 输出控制
        "verbose": False,
        "debug": False,
        "save_results": True,
        
        # 检测模式
        "use_smart_detection": False,
        "use_sentence_detection": True,
        
        # 早停参数
        "use_answer_consistency": True,
        "use_entropy_halt": True,
        "consistency_k": 3,
        "entropy_threshold": 1.0,
        "entropy_steps": 2,
        
        # 检查点参数
        "min_tokens": 100,
        "cooldown": 40,
        
        # 句子检测参数
        "check_after_complete_sentence": True,
        
        # 探针参数
        "max_probe_tokens": 50,
        "probe_temperature": 0.1
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
                    sc = full_config.get('stage_control', {})
                    sd = full_config.get('sentence_detection', {})
                    ps = full_config.get('probe_system', {})
                    paths = full_config.get('paths', {})
                    
                    return {
                        # 模型
                        "active_model": full_config.get('active_model', 'qwen2.5-7b'),
                        
                        # 数据集
                        "data": paths.get('test_data', '').replace('data/', ''),
                        "sample_size": exp.get('sample_size', 5),
                        
                        # 生成参数
                        "max_tokens": exp.get('max_new_tokens', 512),
                        "temperature": exp.get('temperature', 0.0),
                        "do_sample": exp.get('do_sample', False),
                        
                        # 输出控制
                        "verbose": exp.get('verbose', False),
                        "debug": exp.get('debug_probe', False),
                        "save_results": exp.get('save_results', True),
                        
                        # 检测模式
                        "use_smart_detection": sc.get('use_smart_detection', False),
                        "use_sentence_detection": sd.get('enabled', True),
                        
                        # 早停参数
                        "use_answer_consistency": es.get('use_answer_consistency', True),
                        "use_entropy_halt": es.get('use_entropy_halt', True),
                        "consistency_k": es.get('consistency_k', 3),
                        "entropy_threshold": es.get('entropy_threshold', 1.0),
                        "entropy_steps": es.get('entropy_consecutive_steps', 2),
                        
                        # 检查点参数
                        "min_tokens": es.get('min_tokens_before_check', 100),
                        "cooldown": es.get('cooldown_tokens', 40),
                        
                        # 句子检测参数
                        "check_after_complete_sentence": sd.get('check_after_complete_sentence', True),
                        
                        # 探针参数
                        "max_probe_tokens": ps.get('max_probe_tokens', 50),
                        "probe_temperature": ps.get('probe_temperature', 0.1)
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
        
        # 模型信息
        model_info = self.MODELS.get(config['active_model'], {})
        print(f"  🤖 模型: {config['active_model']} ({model_info.get('desc', '')})")
        
        # 数据集信息
        print(f"  📁 数据集: {config['data']}")
        print(f"  📊 样本数: {config['sample_size']}")
        
        # 生成参数
        print(f"\n  【生成参数】")
        print(f"  🔢 最大tokens: {config['max_tokens']}")
        print(f"  🌡️  温度: {config['temperature']}")
        print(f"  🎲 采样: {'✓' if config['do_sample'] else '✗'}")
        
        # 输出控制
        print(f"\n  【输出控制】")
        print(f"  📝 详细输出: {'✓' if config['verbose'] else '✗'}")
        print(f"  🐛 调试模式: {'✓' if config['debug'] else '✗'}")
        print(f"  💾 保存结果: {'✓' if config['save_results'] else '✗'}")
        
        # 检测模式
        print(f"\n  【检测模式】")
        if config['use_smart_detection']:
            print(f"  🔍 智能阶段检测: ✓ 启用")
        elif config['use_sentence_detection']:
            print(f"  🔍 句子边界检测: ✓ 启用")
        else:
            print(f"  🔍 检测模式: ✗ 禁用")
        
        # 早停机制
        print(f"\n  【早停机制】")
        if config['use_answer_consistency'] or config['use_entropy_halt']:
            if config['use_answer_consistency']:
                print(f"  ✓ 答案一致性检测 (窗口={config['consistency_k']})")
            if config['use_entropy_halt']:
                print(f"  ✓ 熵值检测 (阈值={config['entropy_threshold']}, 步数={config['entropy_steps']})")
        else:
            print(f"  ✗ 早停机制禁用")
        
        # 检查点参数
        print(f"\n  【检查点参数】")
        print(f"  ⏱️  最小tokens: {config['min_tokens']}")
        print(f"  ❄️  冷却tokens: {config['cooldown']}")
        print(f"  📏 完整句子检查: {'✓' if config['check_after_complete_sentence'] else '✗'}")
        
        # 探针参数
        print(f"\n  【探针参数】")
        print(f"  🧪 最大探针tokens: {config['max_probe_tokens']}")
        print(f"  🌡️  探针温度: {config['probe_temperature']}")
        
        if show_title:
            print(f"\n{'='*80}\n")
    
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
    
    def show_model_menu(self) -> Optional[str]:
        """显示模型选择菜单"""
        self.show_header("选择模型")
        
        index = 1
        index_map = {}
        
        for key, info in self.MODELS.items():
            print(f"  {index}. {key:<20} - {info['desc']}")
            index_map[str(index)] = key
            index += 1
        
        print("\n  0. 返回主菜单")
        print(f"\n{'='*80}\n")
        
        while True:
            choice = input("请选择模型 [输入数字]: ").strip()
            
            if choice == '0':
                return None
            
            if choice in index_map:
                return index_map[choice]
            
            print("❌ 无效选择，请重新输入")
    
    def main_menu(self) -> Optional[str]:
        """主菜单"""
        self.show_header("HALT-CoT 配置管理器")
        self.show_config_summary(self.current_config, show_title=False)
        print(f"\n{'='*80}\n")
        
        print("【操作菜单】")
        print("  1. 选择模型")
        print("  2. 选择数据集")
        print("  3. 修改生成参数 (tokens, 温度等)")
        print("  4. 修改检测模式 (智能/句子边界)")
        print("  5. 修改早停参数")
        print("  6. 修改检查点参数")
        print("  7. 修改探针参数")
        print("  8. 切换开关选项")
        print("  9. 重置为默认配置")
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
    
    def modify_detection_mode(self):
        """修改检测模式"""
        self.show_header("修改检测模式")
        
        print("【检测模式选择】")
        print("  1. 智能阶段检测 (根据推理阶段自动检测)")
        print("  2. 句子边界检测 (每完成一个句子后检测) [推荐]")
        print("  3. 禁用检测 (仅用于基线对比)")
        print()
        print("  0. 返回")
        print()
        
        choice = input("请选择检测模式 [输入数字]: ").strip()
        
        if choice == '1':
            self.current_config['use_smart_detection'] = True
            self.current_config['use_sentence_detection'] = False
            print("\n✅ 已切换到智能阶段检测模式")
        elif choice == '2':
            self.current_config['use_smart_detection'] = False
            self.current_config['use_sentence_detection'] = True
            print("\n✅ 已切换到句子边界检测模式")
        elif choice == '3':
            self.current_config['use_smart_detection'] = False
            self.current_config['use_sentence_detection'] = False
            print("\n✅ 已禁用检测模式")
        elif choice == '0':
            return
        else:
            print("\n❌ 无效选择")
        
        input("\n按回车继续...")
    
    def modify_early_stop_params(self):
        """修改早停参数"""
        self.show_header("修改早停参数")
        
        print(f"当前配置:")
        print(f"  答案一致性检测: {'✓' if self.current_config['use_answer_consistency'] else '✗'}")
        print(f"  熵值检测: {'✓' if self.current_config['use_entropy_halt'] else '✗'}")
        print(f"  一致性窗口: {self.current_config['consistency_k']}")
        print(f"  熵值阈值: {self.current_config['entropy_threshold']}")
        print(f"  连续步数: {self.current_config['entropy_steps']}")
        print()
        
        try:
            val = input("一致性窗口 (回车跳过): ").strip()
            if val:
                self.current_config['consistency_k'] = int(val)
                print(f"  ✓ 已修改为: {self.current_config['consistency_k']}")
            
            val = input("熵值阈值 (回车跳过): ").strip()
            if val:
                self.current_config['entropy_threshold'] = float(val)
                print(f"  ✓ 已修改为: {self.current_config['entropy_threshold']}")
            
            val = input("连续步数 (回车跳过): ").strip()
            if val:
                self.current_config['entropy_steps'] = int(val)
                print(f"  ✓ 已修改为: {self.current_config['entropy_steps']}")
            
            print("\n✅ 早停参数修改完成")
        except ValueError:
            print("\n❌ 输入格式错误，修改已取消")
        
        input("\n按回车继续...")
    
    def modify_checkpoint_params(self):
        """修改检查点参数"""
        self.show_header("修改检查点参数")
        
        print(f"当前配置:")
        print(f"  最小tokens: {self.current_config['min_tokens']}")
        print(f"  冷却tokens: {self.current_config['cooldown']}")
        print()
        
        try:
            val = input("最小tokens (开始检测前) (回车跳过): ").strip()
            if val:
                self.current_config['min_tokens'] = int(val)
                print(f"  ✓ 已修改为: {self.current_config['min_tokens']}")
            
            val = input("冷却tokens (两次检测间隔) (回车跳过): ").strip()
            if val:
                self.current_config['cooldown'] = int(val)
                print(f"  ✓ 已修改为: {self.current_config['cooldown']}")
            
            print("\n✅ 检查点参数修改完成")
        except ValueError:
            print("\n❌ 输入格式错误，修改已取消")
        
        input("\n按回车继续...")
    
    def modify_probe_params(self):
        """修改探针参数"""
        self.show_header("修改探针参数")
        
        print(f"当前配置:")
        print(f"  最大探针tokens: {self.current_config['max_probe_tokens']}")
        print(f"  探针温度: {self.current_config['probe_temperature']}")
        print()
        
        try:
            val = input("最大探针tokens (回车跳过): ").strip()
            if val:
                self.current_config['max_probe_tokens'] = int(val)
                print(f"  ✓ 已修改为: {self.current_config['max_probe_tokens']}")
            
            val = input("探针温度 0.0-1.0 (回车跳过): ").strip()
            if val:
                self.current_config['probe_temperature'] = float(val)
                print(f"  ✓ 已修改为: {self.current_config['probe_temperature']}")
            
            print("\n✅ 探针参数修改完成")
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
            '4': ('save_results', '保存结果'),
            '5': ('use_answer_consistency', '答案一致性检测'),
            '6': ('use_entropy_halt', '熵值检测'),
            '7': ('check_after_complete_sentence', '完整句子检查'),
        }
        
        while True:
            print(f"\n当前状态:")
            for key, (config_key, name) in switches.items():
                status = '✓ 启用' if self.current_config[config_key] else '✗ 禁用'
                print(f"  {key}. {name:<20} [{status}]")
            
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
        """构建完整配置（精简版）"""
        # 句子结束标记
        sentence_endings = [".", "!", "?", "。", "！", "？"]
        
        # 已知缩写词
        known_abbreviations = [
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
            "etc", "vs", "i.e", "e.g", "approx", "est",
            "inc", "corp", "ltd", "co", "dept"
        ]
        
        return {
            "model_configs": {
                key: {"name": info["name"], "description": info["desc"]}
                for key, info in self.MODELS.items()
            },
            "active_model": self.current_config['active_model'],
            
            "experiment": {
                "sample_size": self.current_config['sample_size'],
                "verbose": self.current_config['verbose'],
                "save_results": self.current_config['save_results'],
                "max_new_tokens": self.current_config['max_tokens'],
                "temperature": self.current_config['temperature'],
                "do_sample": self.current_config['do_sample'],
                "debug_probe": self.current_config['debug']
            },
            
            "paths": {
                "test_data": f"data/{self.current_config['data']}"
            },
            
            "early_stopping": {
                "use_answer_consistency": self.current_config['use_answer_consistency'],
                "use_entropy_halt": self.current_config['use_entropy_halt'],
                "consistency_k": self.current_config['consistency_k'],
                "entropy_threshold": self.current_config['entropy_threshold'],
                "entropy_consecutive_steps": self.current_config['entropy_steps'],
                "min_tokens_before_check": self.current_config['min_tokens'],
                "cooldown_tokens": self.current_config['cooldown']
            },
            
            "stage_control": {
                "use_smart_detection": self.current_config['use_smart_detection']
            },
            
            "sentence_detection": {
                "enabled": self.current_config['use_sentence_detection'],
                "min_tokens_before_check": self.current_config['min_tokens'],
                "cooldown_tokens": self.current_config['cooldown'],
                "sentence_endings": sentence_endings,
                "known_abbreviations": known_abbreviations,
                "check_after_complete_sentence": self.current_config['check_after_complete_sentence']
            },
            
            "probe_system": {
                "max_probe_tokens": self.current_config['max_probe_tokens'],
                "probe_temperature": self.current_config['probe_temperature'],
                "probe_strategies": {
                    "calculation": "\n\nThe result of this calculation is: ",
                    "conclusion": "\n\nTherefore, the final answer is: ",
                    "answer_signal": "\n#### ",
                    "intermediate": "\n\nThe current value is: ",
                    "reasoning": "\n\nBased on the above, the answer is: "
                }
            },
            
            "reasoning_stage_markers": {
                "calculation": ["=", "equals", "total", "sum", "result", "calculate"],
                "conclusion": ["therefore", "thus", "so", "hence", "finally", "in conclusion"],
                "intermediate": ["step", "first", "next", "then", "now", "let's"],
                "answer_signal": ["answer is", "answer:", "####", "\\boxed", "final answer"],
                "reasoning": ["because", "since", "if", "when", "consider"]
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
        
        # 显示保存的配置摘要
        print(f"\n保存的配置摘要:")
        print(f"  模型: {self.current_config['active_model']}")
        print(f"  数据集: {self.current_config['data']} ({self.current_config['sample_size']}样本)")
        
        if self.current_config['use_smart_detection']:
            print(f"  检测模式: 智能阶段检测")
        elif self.current_config['use_sentence_detection']:
            print(f"  检测模式: 句子边界检测")
        else:
            print(f"  检测模式: 禁用")
        
        early_stop_status = []
        if self.current_config['use_answer_consistency']:
            early_stop_status.append("答案一致性")
        if self.current_config['use_entropy_halt']:
            early_stop_status.append("熵值检测")
        
        if early_stop_status:
            print(f"  早停策略: {', '.join(early_stop_status)}")
        else:
            print(f"  早停策略: 禁用")
        
        input("\n按回车继续...")
    
    def run(self):
        """运行交互式配置"""
        while True:
            choice = self.main_menu()
            
            if choice == '0':
                print("\n👋 已退出配置管理器\n")
                break
            
            elif choice == '1':
                # 选择模型
                result = self.show_model_menu()
                if result:
                    self.current_config['active_model'] = result
                    model_info = self.MODELS[result]
                    print(f"\n✅ 已选择模型: {result} ({model_info['desc']})")
                    input("按回车继续...")
            
            elif choice == '2':
                # 选择数据集
                result = self.show_dataset_menu()
                if result:
                    data_file, sample_size = result
                    self.current_config['data'] = data_file
                    self.current_config['sample_size'] = sample_size
                    print(f"\n✅ 已选择数据集: {data_file} ({sample_size}样本)")
                    input("按回车继续...")
            
            elif choice == '3':
                # 修改生成参数
                self.modify_generation_params()
            
            elif choice == '4':
                # 修改检测模式
                self.modify_detection_mode()
            
            elif choice == '5':
                # 修改早停参数
                self.modify_early_stop_params()
            
            elif choice == '6':
                # 修改检查点参数
                self.modify_checkpoint_params()
            
            elif choice == '7':
                # 修改探针参数
                self.modify_probe_params()
            
            elif choice == '8':
                # 切换开关选项
                self.toggle_switches()
            
            elif choice == '9':
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
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                    HALT-CoT 交互式配置管理器 v2.0                          ║
║                                                                            ║
║                    支持句子边界检测 & 智能阶段检测                         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    manager = InteractiveConfigManager()
    manager.run()


if __name__ == "__main__":
    main()
