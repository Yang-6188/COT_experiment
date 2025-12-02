#!/usr/bin/env python3
"""
HALT-CoT配置工具 - 支持多数据集
支持批量快速修改参数，包括早停策略和调试探针
支持 GSM8K 和 MATH 数据集切换
"""

import json
import os
from pathlib import Path

BASE_DIR = Path("/root/autodl-tmp")
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

def ensure_directories():
    """确保必要的目录存在"""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "results").mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ 创建目录时出错: {e}")
        raise

def detect_available_datasets():
    """检测可用的数据集"""
    available = {}
    
    # 检测 GSM8K
    gsm8k_files = {
        'test': DATA_DIR / "gsm8k_test.json",
        'test_processed': DATA_DIR / "gsm8k_test_processed.json",
        'sample_5': DATA_DIR / "gsm8k_test_sample_5.json",
        'sample_10': DATA_DIR / "gsm8k_test_sample_10.json",
        'sample_20': DATA_DIR / "gsm8k_test_sample_20.json",
        'sample_50': DATA_DIR / "gsm8k_test_sample_50.json",
        'sample_100': DATA_DIR / "gsm8k_test_sample_100.json",
    }
    
    gsm8k_available = any(f.exists() for f in gsm8k_files.values())
    if gsm8k_available:
        available['gsm8k'] = {
            'name': 'GSM8K',
            'description': '小学数学应用题',
            'files': {k: str(v) for k, v in gsm8k_files.items() if v.exists()}
        }
    
    # 检测 MATH
    math_files = {
        'test': DATA_DIR / "math_test.json",
        'test_processed': DATA_DIR / "math_test_processed.json",
        'sample_5': DATA_DIR / "math_test_sample_5.json",
        'sample_10': DATA_DIR / "math_test_sample_10.json",
        'sample_20': DATA_DIR / "math_test_sample_20.json",
        'sample_50': DATA_DIR / "math_test_sample_50.json",
        'sample_100': DATA_DIR / "math_test_sample_100.json",
    }
    
    math_available = any(f.exists() for f in math_files.values())
    if math_available:
        # 检测难度级别子集
        level_files = list(DATA_DIR.glob("math_test_level_*.json"))
        type_files = list(DATA_DIR.glob("math_test_type_*.json"))
        
        available['math'] = {
            'name': 'MATH',
            'description': '竞赛数学题',
            'files': {k: str(v) for k, v in math_files.items() if v.exists()},
            'level_subsets': [f.name for f in level_files],
            'type_subsets': [f.name for f in type_files]
        }
    
    return available

def create_default_config():
    """创建与实际配置文件匹配的默认配置"""
    available_datasets = detect_available_datasets()
    
    # 默认使用 GSM8K，如果不存在则使用 MATH
    default_dataset = 'gsm8k' if 'gsm8k' in available_datasets else 'math'
    default_test_file = f"data/{default_dataset}_test.json"
    
    config = {
        "active_model": "qwen",
        "active_dataset": default_dataset,  # 新增：当前使用的数据集
        "model_configs": {
            "qwen": {
                "name": "Qwen/Qwen2.5-7B-Instruct"
            },
            "qwen-3b": {
                "name": "Qwen/Qwen2.5-3B-Instruct"
            },
            "qwen-14b": {
                "name": "Qwen/Qwen2.5-14B-Instruct"
            }
        },
        "dataset_configs": {  # 新增：数据集配置
            "gsm8k": {
                "name": "GSM8K",
                "test_file": "data/gsm8k_test.json",
                "test_processed": "data/gsm8k_test_processed.json",
                "description": "小学数学应用题"
            },
            "math": {
                "name": "MATH",
                "test_file": "data/math_test.json",
                "test_processed": "data/math_test_processed.json",
                "description": "竞赛数学题 (Level 1-5)"
            }
        },
        "paths": {
            "test_data": default_test_file,
            "data_dir": "data",
            "results_dir": "results"
        },
        "experiment": {
            "sample_size": 10,
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.7,
            "save_results": True,
            "verbose": False,
            "debug_probe": True
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
    return config

def load_config():
    """加载配置文件"""
    config_file = CONFIG_DIR / "config.json"
    
    if not config_file.exists():
        config = create_default_config()
        save_config(config)
        return config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查并升级配置结构
        default = create_default_config()
        needs_update = False
        
        # 添加数据集配置
        if 'active_dataset' not in config:
            print("⚠️  添加数据集配置...")
            config['active_dataset'] = default['active_dataset']
            needs_update = True
        
        if 'dataset_configs' not in config:
            config['dataset_configs'] = default['dataset_configs']
            needs_update = True
        
        # 检查是否缺少 early_stopping 配置
        if 'early_stopping' not in config:
            print("⚠️  检测到缺少 early_stopping 配置，正在添加...")
            config['early_stopping'] = default['early_stopping']
            needs_update = True
        else:
            # 检查是否缺少新的早停参数
            early_stop = config['early_stopping']
            default_early = default['early_stopping']
            
            for key in ['min_tokens_before_check', 'cooldown_tokens']:
                if key not in early_stop:
                    print(f"⚠️  添加新的早停参数: {key}")
                    early_stop[key] = default_early[key]
                    needs_update = True
            
            # 移除已废弃的参数
            if 'chunk_by_sentence' in early_stop:
                print("⚠️  移除已废弃的参数: chunk_by_sentence")
                del early_stop['chunk_by_sentence']
                needs_update = True
        
        # 检查实验配置完整性
        if 'experiment' not in config:
            print("⚠️  重建实验配置...")
            config['experiment'] = default['experiment']
            needs_update = True
        else:
            # 添加 debug_probe 参数
            if 'debug_probe' not in config['experiment']:
                print("⚠️  添加调试探针参数: debug_probe")
                config['experiment']['debug_probe'] = default['experiment']['debug_probe']
                needs_update = True
        
        # 清理多余的 metadata
        if 'metadata' in config:
            print("⚠️  移除 metadata 字段（精简配置）")
            del config['metadata']
            needs_update = True
        
        if needs_update:
            save_config(config)
            
        return config
    except Exception as e:
        print(f"❌ 读取配置出错: {e}，将使用默认配置")
        return create_default_config()

def save_config(config):
    """保存配置"""
    ensure_directories()
    config_file = CONFIG_DIR / "config.json"
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"✅ 配置已保存至: {config_file}")
    except Exception as e:
        print(f"❌ 保存配置失败: {e}")

def show_current_config():
    """显示当前配置摘要"""
    config = load_config()
    exp = config['experiment']
    early_stop = config.get('early_stopping', {})
    available_datasets = detect_available_datasets()
    
    print("\n📋 当前配置状态")
    print("=" * 60)
    
    # 基础配置
    model_name = config['model_configs'][config['active_model']]['name']
    print(f"🤖 模型:         {config['active_model']} ({model_name})")
    
    # 数据集信息
    active_dataset = config.get('active_dataset', 'gsm8k')
    dataset_info = config.get('dataset_configs', {}).get(active_dataset, {})
    print(f"📚 数据集:       {active_dataset.upper()} - {dataset_info.get('description', '')}")
    
    # 显示可用数据集
    if len(available_datasets) > 1:
        other_datasets = [d for d in available_datasets.keys() if d != active_dataset]
        print(f"   可切换至:     {', '.join([d.upper() for d in other_datasets])}")
    
    print(f"📊 样本数量:     {exp['sample_size']}")
    print(f"📏 最大生成:     {exp['max_new_tokens']} tokens")
    print(f"🌡️ 温度:         {exp['temperature']}")
    print(f"🎲 随机采样:     {'开启' if exp['do_sample'] else '关闭'}")
    print(f"📝 详细输出:     {'开启' if exp['verbose'] else '关闭'}")
    print(f"🔍 调试探针:     {'开启' if exp.get('debug_probe', True) else '关闭'}")
    print(f"💾 保存结果:     {'开启' if exp['save_results'] else '关闭'}")
    
    print(f"\n🛑 早停策略配置")
    print("-" * 60)
    print(f"✅ 答案一致性:   {'开启' if early_stop.get('use_answer_consistency') else '关闭'}")
    print(f"📉 熵检测:       {'开启' if early_stop.get('use_entropy_halt') else '关闭'}")
    
    if early_stop.get('use_answer_consistency'):
        print(f"🔁 一致性K值:    {early_stop.get('consistency_k', 3)} 次相同答案触发")
    
    if early_stop.get('use_entropy_halt'):
        print(f"📊 熵阈值:       {early_stop.get('entropy_threshold', 0.6)}")
        print(f"🔄 连续低熵步数: {early_stop.get('entropy_consecutive_steps', 2)}")
    
    print(f"\n⚙️ 检查点控制")
    print("-" * 60)
    print(f"🚦 最小检查间隔: {early_stop.get('min_tokens_before_check', 100)} tokens")
    print(f"❄️ 冷却间隔:     {early_stop.get('cooldown_tokens', 40)} tokens")
    
    print("=" * 60)
    return config

def switch_dataset():
    """切换数据集"""
    config = load_config()
    available_datasets = detect_available_datasets()
    
    if not available_datasets:
        print("❌ 未检测到任何可用数据集！")
        print("💡 请先运行数据准备脚本下载数据集")
        return
    
    print("\n📚 切换数据集")
    print("=" * 60)
    print("可用数据集:")
    
    dataset_list = []
    for i, (key, info) in enumerate(available_datasets.items(), 1):
        current = "👈 [当前]" if key == config.get('active_dataset') else ""
        print(f"{i}. {info['name']:10s} - {info['description']} {current}")
        dataset_list.append(key)
        
        # 显示可用文件
        if 'sample_5' in info['files']:
            print(f"   ✅ 包含采样数据集 (5, 10, 20, 50, 100 样本)")
        
        # MATH 数据集的额外信息
        if key == 'math' and 'level_subsets' in info:
            print(f"   ✅ 包含 {len(info['level_subsets'])} 个难度子集")
            print(f"   ✅ 包含 {len(info['type_subsets'])} 个类别子集")
    
    print("0. 取消")
    print("=" * 60)
    
    choice = input("\n请选择数据集 (0-{}): ".format(len(dataset_list))).strip()
    
    try:
        choice_num = int(choice)
        if choice_num == 0:
            print("已取消")
            return
        
        if 1 <= choice_num <= len(dataset_list):
            selected_dataset = dataset_list[choice_num - 1]
            
            # 更新配置
            config['active_dataset'] = selected_dataset
            dataset_config = config['dataset_configs'][selected_dataset]
            config['paths']['test_data'] = dataset_config['test_file']
            
            save_config(config)
            print(f"\n✅ 已切换到数据集: {selected_dataset.upper()}")
            
            # 如果是 MATH 数据集，询问是否使用子集
            if selected_dataset == 'math':
                use_subset = input("\n是否使用特定子集？(y/n): ").strip().lower()
                if use_subset == 'y':
                    select_math_subset(config)
            
            show_current_config()
        else:
            print("❌ 无效选择")
    except ValueError:
        print("❌ 请输入数字")

def select_math_subset():
    """选择 MATH 数据集的子集"""
    print("\n📊 选择 MATH 子集类型")
    print("=" * 60)
    print("1. 按难度选择 (Level 1-5)")
    print("2. 按类别选择 (代数、几何等)")
    print("3. 使用完整测试集")
    print("0. 取消")
    
    choice = input("\n请选择 (0-3): ").strip()
    
    if choice == '1':
        select_math_level_subset()
    elif choice == '2':
        select_math_type_subset()
    elif choice == '3':
        print("✅ 使用完整测试集")
    elif choice == '0':
        print("已取消")
    else:
        print("❌ 无效选择")

def select_math_level_subset():
    """选择 MATH 难度子集"""
    level_files = list(DATA_DIR.glob("math_test_level_*.json"))
    
    if not level_files:
        print("❌ 未找到难度子集文件")
        return
    
    print("\n📈 可用难度级别:")
    print("-" * 40)
    
    levels = []
    for i, f in enumerate(sorted(level_files), 1):
        level_name = f.stem.replace('math_test_level_', '')
        
        # 读取文件获取题目数量
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                count = len(data)
            print(f"{i}. {level_name:15s} ({count} 题)")
            levels.append((level_name, str(f)))
        except:
            continue
    
    print("0. 取消")
    
    choice = input(f"\n请选择难度 (0-{len(levels)}): ").strip()
    
    try:
        choice_num = int(choice)
        if choice_num == 0:
            return
        
        if 1 <= choice_num <= len(levels):
            level_name, file_path = levels[choice_num - 1]
            
            config = load_config()
            config['paths']['test_data'] = file_path
            save_config(config)
            
            print(f"✅ 已选择难度子集: {level_name}")
        else:
            print("❌ 无效选择")
    except ValueError:
        print("❌ 请输入数字")

def select_math_type_subset():
    """选择 MATH 类别子集"""
    type_files = list(DATA_DIR.glob("math_test_type_*.json"))
    
    if not type_files:
        print("❌ 未找到类别子集文件")
        return
    
    print("\n🎯 可用问题类别:")
    print("-" * 40)
    
    types = []
    for i, f in enumerate(sorted(type_files), 1):
        type_name = f.stem.replace('math_test_type_', '').replace('_', ' ')
        
        # 读取文件获取题目数量
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                count = len(data)
            print(f"{i}. {type_name:30s} ({count} 题)")
            types.append((type_name, str(f)))
        except:
            continue
    
    print("0. 取消")
    
    choice = input(f"\n请选择类别 (0-{len(types)}): ").strip()
    
    try:
        choice_num = int(choice)
        if choice_num == 0:
            return
        
        if 1 <= choice_num <= len(types):
            type_name, file_path = types[choice_num - 1]
            
            config = load_config()
            config['paths']['test_data'] = file_path
            save_config(config)
            
            print(f"✅ 已选择类别子集: {type_name}")
        else:
            print("❌ 无效选择")
    except ValueError:
        print("❌ 请输入数字")

def get_input_with_default(prompt, default_val, validator=None, error_msg="输入无效"):
    """获取输入，直接回车则使用默认值"""
    while True:
        user_input = input(f"{prompt} [当前: {default_val}]: ").strip()
        
        if not user_input:
            return default_val
            
        try:
            if validator:
                validated_val = validator(user_input)
                if validated_val is not None:
                    return validated_val
                else:
                    print(f"❌ {error_msg}")
            else:
                return user_input
        except Exception:
            print(f"❌ {error_msg}")

def batch_modify_basic_config():
    """批量修改基础配置"""
    config = load_config()
    print("\n✏️  修改基础配置 (直接回车 = 保持不变)")
    print("-" * 50)

    # 1. 修改模型
    models = list(config['model_configs'].keys())
    print(f"可用模型: {', '.join(models)}")
    
    def validate_model(val):
        return val if val in models else None
        
    config['active_model'] = get_input_with_default(
        "👉 选择模型",
        config['active_model'], 
        validate_model,
        f"请输入以下之一: {models}"
    )

    # 2. 修改样本数
    def validate_int_range(min_val, max_val):
        def validator(val):
            try:
                v = int(val)
                return v if min_val <= v <= max_val else None
            except: return None
        return validator
        
    config['experiment']['sample_size'] = get_input_with_default(
        "👉 样本数量 (1-1000)",
        config['experiment']['sample_size'],
        validate_int_range(1, 1000),
        "请输入 1-1000 的整数"
    )

    # 3. 修改最大生成长度
    config['experiment']['max_new_tokens'] = get_input_with_default(
        "👉 最大生成tokens (128-2048)",
        config['experiment']['max_new_tokens'],
        validate_int_range(128, 2048),
        "请输入 128-2048 的整数"
    )

    # 4. 修改温度
    def validate_float_range(min_val, max_val):
        def validator(val):
            try:
                v = float(val)
                return v if min_val <= v <= max_val else None
            except: return None
        return validator

    config['experiment']['temperature'] = get_input_with_default(
        "👉 温度 (0.0-2.0)",
        config['experiment']['temperature'],
        validate_float_range(0.0, 2.0),
        "请输入 0.0-2.0 的数字"
    )

    # 5. 修改采样模式
    def validate_bool(val):
        v = val.lower()
        if v in ['y', 'yes', 'true', '1']: return True
        if v in ['n', 'no', 'false', '0']: return False
        return None

    config['experiment']['do_sample'] = get_input_with_default(
        "👉 启用随机采样 (y/n)",
        config['experiment']['do_sample'],
        validate_bool,
        "请输入 y 或 n"
    )

    # 6. 详细输出
    config['experiment']['verbose'] = get_input_with_default(
        "👉 显示详细输出 (y/n)",
        config['experiment']['verbose'],
        validate_bool,
        "请输入 y 或 n"
    )

    # 7. 调试探针
    config['experiment']['debug_probe'] = get_input_with_default(
        "👉 启用调试探针 (y/n)",
        config['experiment'].get('debug_probe', True),
        validate_bool,
        "请输入 y 或 n"
    )

    print("-" * 50)
    save_config(config)
    print("\n✨ 基础配置已更新！")

def modify_early_stopping():
    """修改早停策略配置"""
    config = load_config()
    early_stop = config.get('early_stopping', {})
    
    print("\n🛑 修改早停策略配置 (直接回车 = 保持不变)")
    print("-" * 50)
    
    def validate_bool(val):
        v = val.lower()
        if v in ['y', 'yes', 'true', '1']: return True
        if v in ['n', 'no', 'false', '0']: return False
        return None
    
    def validate_int_range(min_val, max_val):
        def validator(val):
            try:
                v = int(val)
                return v if min_val <= v <= max_val else None
            except: return None
        return validator
    
    def validate_float_range(min_val, max_val):
        def validator(val):
            try:
                v = float(val)
                return v if min_val <= v <= max_val else None
            except: return None
        return validator
    
    # 1. 答案一致性开关
    early_stop['use_answer_consistency'] = get_input_with_default(
        "👉 启用答案一致性检测 (y/n)",
        early_stop.get('use_answer_consistency', True),
        validate_bool,
        "请输入 y 或 n"
    )
    
    # 2. 一致性K值
    if early_stop['use_answer_consistency']:
        early_stop['consistency_k'] = get_input_with_default(
            "👉 答案一致性K值 (2-10)",
            early_stop.get('consistency_k', 3),
            validate_int_range(2, 10),
            "请输入 2-10 的整数"
        )
    
    # 3. 熵检测开关
    early_stop['use_entropy_halt'] = get_input_with_default(
        "👉 启用熵检测 (y/n)",
        early_stop.get('use_entropy_halt', True),
        validate_bool,
        "请输入 y 或 n"
    )
    
    # 4. 熵相关参数
    if early_stop['use_entropy_halt']:
        early_stop['entropy_threshold'] = get_input_with_default(
            "👉 熵阈值 (0.1-2.0, 越小越严格)",
            early_stop.get('entropy_threshold', 0.6),
            validate_float_range(0.1, 2.0),
            "请输入 0.1-2.0 的数字"
        )
        
        early_stop['entropy_consecutive_steps'] = get_input_with_default(
            "👉 连续低熵步数 (1-5)",
            early_stop.get('entropy_consecutive_steps', 2),
            validate_int_range(1, 5),
            "请输入 1-5 的整数"
        )
    
    # 5. 检查点控制参数
    print(f"\n⚙️ 检查点控制参数")
    print("-" * 30)
    
    early_stop['min_tokens_before_check'] = get_input_with_default(
        "👉 最小检查间隔 (50-200 tokens)",
        early_stop.get('min_tokens_before_check', 100),
        validate_int_range(50, 200),
        "请输入 50-200 的整数"
    )
    
    early_stop['cooldown_tokens'] = get_input_with_default(
        "👉 冷却间隔 (20-100 tokens)",
        early_stop.get('cooldown_tokens', 40),
        validate_int_range(20, 100),
        "请输入 20-100 的整数"
    )
    
    config['early_stopping'] = early_stop
    
    print("-" * 50)
    save_config(config)
    print("\n✨ 早停策略配置已更新！")

def create_preset_configs():
    """创建预设配置"""
    base_config = create_default_config()
    presets = {}
    
    # 快速测试预设 - GSM8K
    presets['quick_test_gsm8k'] = {
        'active_model': 'qwen-3b',
        'active_dataset': 'gsm8k',
        'model_configs': base_config['model_configs'],
        'dataset_configs': base_config['dataset_configs'],
        'paths': {
            'test_data': 'data/gsm8k_test_sample_5.json',
            'data_dir': 'data',
            'results_dir': 'results'
        },
        'experiment': {
            'sample_size': 5,
            'max_new_tokens': 256,
            'do_sample': False,
            'temperature': 0.0,
            'save_results': True,
            'verbose': True,
            'debug_probe': True
        },
        'early_stopping': {
            'use_answer_consistency': True,
            'use_entropy_halt': True,
            'consistency_k': 2,
            'entropy_threshold': 0.8,
            'entropy_consecutive_steps': 1,
            'min_tokens_before_check': 50,
            'cooldown_tokens': 20
        }
    }
    
    # 快速测试预设 - MATH
    presets['quick_test_math'] = {
        'active_model': 'qwen-3b',
        'active_dataset': 'math',
        'model_configs': base_config['model_configs'],
        'dataset_configs': base_config['dataset_configs'],
        'paths': {
            'test_data': 'data/math_test_sample_5.json',
            'data_dir': 'data',
            'results_dir': 'results'
        },
        'experiment': {
            'sample_size': 5,
            'max_new_tokens': 512,  # MATH 需要更长的生成
            'do_sample': False,
            'temperature': 0.0,
            'save_results': True,
            'verbose': True,
            'debug_probe': True
        },
        'early_stopping': {
            'use_answer_consistency': True,
            'use_entropy_halt': True,
            'consistency_k': 2,
            'entropy_threshold': 0.7,
            'entropy_consecutive_steps': 2,
            'min_tokens_before_check': 80,
            'cooldown_tokens': 30
        }
    }
    
    # 标准实验预设 - GSM8K
    presets['standard_gsm8k'] = {
        'active_model': 'qwen',
        'active_dataset': 'gsm8k',
        'model_configs': base_config['model_configs'],
        'dataset_configs': base_config['dataset_configs'],
        'paths': {
            'test_data': 'data/gsm8k_test.json',
            'data_dir': 'data',
            'results_dir': 'results'
        },
        'experiment': {
            'sample_size': 50,
            'max_new_tokens': 512,
            'do_sample': False,
            'temperature': 0.0,
            'save_results': True,
            'verbose': False,
            'debug_probe': False
        },
        'early_stopping': base_config['early_stopping']
    }
    
    # 标准实验预设 - MATH
    presets['standard_math'] = {
        'active_model': 'qwen',
        'active_dataset': 'math',
        'model_configs': base_config['model_configs'],
        'dataset_configs': base_config['dataset_configs'],
        'paths': {
            'test_data': 'data/math_test.json',
            'data_dir': 'data',
            'results_dir': 'results'
        },
        'experiment': {
            'sample_size': 50,
            'max_new_tokens': 800,  # MATH 需要更长
            'do_sample': False,
            'temperature': 0.0,
            'save_results': True,
            'verbose': False,
            'debug_probe': False
        },
        'early_stopping': {
            'use_answer_consistency': True,
            'use_entropy_halt': True,
            'consistency_k': 3,
            'entropy_threshold': 0.5,  # MATH 更严格
            'entropy_consecutive_steps': 3,
            'min_tokens_before_check': 120,
            'cooldown_tokens': 50
        }
    }
    
    # 高精度预设 - MATH (竞赛题)
    presets['high_precision_math'] = {
        'active_model': 'qwen-14b',
        'active_dataset': 'math',
        'model_configs': base_config['model_configs'],
        'dataset_configs': base_config['dataset_configs'],
        'paths': {
            'test_data': 'data/math_test.json',
            'data_dir': 'data',
            'results_dir': 'results'
        },
        'experiment': {
            'sample_size': 100,
            'max_new_tokens': 1024,
            'do_sample': False,
            'temperature': 0.0,
            'save_results': True,
            'verbose': False,
            'debug_probe': False
        },
        'early_stopping': {
            'use_answer_consistency': True,
            'use_entropy_halt': True,
            'consistency_k': 4,
            'entropy_threshold': 0.4,
            'entropy_consecutive_steps': 3,
            'min_tokens_before_check': 150,
            'cooldown_tokens': 60
        }
    }
    
    return presets

def apply_preset():
    """应用预设配置"""
    presets = create_preset_configs()
    available_datasets = detect_available_datasets()
    
    print("\n🎛️  可用预设配置:")
    print("=" * 60)
    
    preset_list = []
    i = 1
    
    # GSM8K 预设
    if 'gsm8k' in available_datasets:
        print("\n📚 GSM8K 数据集预设:")
        print(f"{i}. quick_test_gsm8k  - 快速测试 (5样本, 3B模型)")
        preset_list.append('quick_test_gsm8k')
        i += 1
        
        print(f"{i}. standard_gsm8k    - 标准实验 (50样本, 7B模型)")
        preset_list.append('standard_gsm8k')
        i += 1
    
    # MATH 预设
    if 'math' in available_datasets:
        print("\n🎓 MATH 数据集预设:")
        print(f"{i}. quick_test_math   - 快速测试 (5样本, 3B模型)")
        preset_list.append('quick_test_math')
        i += 1
        
        print(f"{i}. standard_math     - 标准实验 (50样本, 7B模型)")
        preset_list.append('standard_math')
        i += 1
        
        print(f"{i}. high_precision_math - 高精度 (100样本, 14B模型)")
        preset_list.append('high_precision_math')
        i += 1
    
    print("\n0. 取消")
    print("=" * 60)
    
    choice = input(f"\n请选择预设 (0-{len(preset_list)}): ").strip()
    
    try:
        choice_num = int(choice)
        if choice_num == 0:
            print("已取消")
            return
        
        if 1 <= choice_num <= len(preset_list):
            preset_name = preset_list[choice_num - 1]
            config = presets[preset_name]
            
            save_config(config)
            print(f"\n✅ 已应用预设: {preset_name}")
            show_current_config()
        else:
            print("❌ 无效选择")
    except ValueError:
        print("❌ 请输入数字")

def main():
    """主入口"""
    ensure_directories()
    
    # 检测可用数据集
    available_datasets = detect_available_datasets()
    
    if not available_datasets:
        print("\n⚠️  警告: 未检测到任何数据集！")
        print("💡 请先运行数据准备脚本:")
        print("   python data_preparation.py --dataset both")
        print()
    
    while True:
        print("\n🔧 HALT-CoT 配置工具")
        print("1. 查看当前配置")
        print("2. 切换数据集")
        print("3. 修改基础配置")
        print("4. 修改早停策略")
        print("5. 应用预设配置")
        print("6. 恢复默认设置")
        print("0. 退出")
        
        choice = input("\n请选择 (0-6): ").strip()
        
        if choice == "1":
            show_current_config()
        elif choice == "2":
            switch_dataset()
        elif choice == "3":
            batch_modify_basic_config()
            show_current_config()
        elif choice == "4":
            modify_early_stopping()
            show_current_config()
        elif choice == "5":
            apply_preset()
        elif choice == "6":
            confirm = input("⚠️  确定要重置所有配置为默认值吗？(y/n): ")
            if confirm.lower() == 'y':
                config = create_default_config()
                save_config(config)
                print("✅ 已恢复默认配置")
                show_current_config()
        elif choice == "0":
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    main()
