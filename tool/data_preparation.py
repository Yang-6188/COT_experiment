"""
数据下载和预处理脚本
支持 GSM8K 和 MATH Dataset
不需要GPU资源,可以在CPU环境下运行


# 下载两个数据集
python data_preparation.py --dataset both

# 只下载 GSM8K
python data_preparation.py --dataset gsm8k

# 只下载 MATH
python data_preparation.py --dataset math

# 跳过下载,仅处理现有数据
python data_preparation.py --skip-download


"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pickle
from datetime import datetime

# 设置 Hugging Face 镜像(重要!)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

# 设置路径
BASE_DIR = Path("/root/autodl-tmp")
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

# 创建目录
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_gsm8k_dataset():
    """下载GSM8K数据集"""
    logger.info("开始下载GSM8K数据集...")
    
    try:
        # 下载训练集和测试集
        train_dataset = load_dataset("gsm8k", "main", split="train")
        test_dataset = load_dataset("gsm8k", "main", split="test")
        
        logger.info(f"GSM8K训练集大小: {len(train_dataset)}")
        logger.info(f"GSM8K测试集大小: {len(test_dataset)}")
        
        # 保存为JSON格式,便于后续处理
        train_data = []
        for item in train_dataset:
            train_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "dataset": "gsm8k"
            })
        
        test_data = []
        for item in test_dataset:
            test_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "dataset": "gsm8k"
            })
        
        # 保存训练集
        train_file = DATA_DIR / "gsm8k_train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        logger.info(f"GSM8K训练集已保存到: {train_file}")
        
        # 保存测试集
        test_file = DATA_DIR / "gsm8k_test.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        logger.info(f"GSM8K测试集已保存到: {test_file}")
        
        return train_file, test_file
        
    except Exception as e:
        logger.error(f"GSM8K数据集下载失败: {e}")
        raise

def download_math_dataset():
    """下载MATH数据集"""
    logger.info("开始下载MATH数据集...")
    
    try:
        # MATH数据集在 Hugging Face 上的名称是 "hendrycks/competition_math"
        # 或者 "lighteval/MATH"
        train_dataset = load_dataset("lighteval/MATH", split="train")
        test_dataset = load_dataset("lighteval/MATH", split="test")
        
        logger.info(f"MATH训练集大小: {len(train_dataset)}")
        logger.info(f"MATH测试集大小: {len(test_dataset)}")
        
        # 统计各个难度和类别
        train_levels = {}
        train_types = {}
        
        # 保存为JSON格式
        train_data = []
        for item in train_dataset:
            # MATH数据集包含: problem, solution, level, type
            entry = {
                "question": item["problem"],
                "answer": item["solution"],
                "level": item.get("level", "unknown"),
                "type": item.get("type", "unknown"),
                "dataset": "math"
            }
            train_data.append(entry)
            
            # 统计
            level = entry["level"]
            prob_type = entry["type"]
            train_levels[level] = train_levels.get(level, 0) + 1
            train_types[prob_type] = train_types.get(prob_type, 0) + 1
        
        test_data = []
        test_levels = {}
        test_types = {}
        
        for item in test_dataset:
            entry = {
                "question": item["problem"],
                "answer": item["solution"],
                "level": item.get("level", "unknown"),
                "type": item.get("type", "unknown"),
                "dataset": "math"
            }
            test_data.append(entry)
            
            # 统计
            level = entry["level"]
            prob_type = entry["type"]
            test_levels[level] = test_levels.get(level, 0) + 1
            test_types[prob_type] = test_types.get(prob_type, 0) + 1
        
        # 保存训练集
        train_file = DATA_DIR / "math_train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        logger.info(f"MATH训练集已保存到: {train_file}")
        
        # 保存测试集
        test_file = DATA_DIR / "math_test.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        logger.info(f"MATH测试集已保存到: {test_file}")
        
        # 打印统计信息
        logger.info("\n" + "="*60)
        logger.info("MATH数据集统计信息:")
        logger.info(f"\n训练集难度分布:")
        for level, count in sorted(train_levels.items()):
            logger.info(f"  {level}: {count}")
        
        logger.info(f"\n训练集类别分布:")
        for prob_type, count in sorted(train_types.items()):
            logger.info(f"  {prob_type}: {count}")
        
        logger.info(f"\n测试集难度分布:")
        for level, count in sorted(test_levels.items()):
            logger.info(f"  {level}: {count}")
        
        logger.info(f"\n测试集类别分布:")
        for prob_type, count in sorted(test_types.items()):
            logger.info(f"  {prob_type}: {count}")
        logger.info("="*60 + "\n")
        
        return train_file, test_file
        
    except Exception as e:
        logger.error(f"MATH数据集下载失败: {e}")
        logger.info("提示: 如果下载失败,请检查网络连接或尝试使用VPN")
        raise

def create_sample_datasets(dataset_name="gsm8k"):
    """创建不同大小的采样数据集用于测试
    
    Args:
        dataset_name: "gsm8k" 或 "math"
    """
    logger.info(f"创建{dataset_name.upper()}采样数据集...")
    
    test_file = DATA_DIR / f"{dataset_name}_test.json"
    if not test_file.exists():
        logger.error(f"{dataset_name.upper()}测试集文件不存在,请先运行下载")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 创建不同大小的采样(包含更小的样本用于快速测试)
    sample_sizes = [5, 10, 20, 50, 100, 200]
    
    for size in sample_sizes:
        if size <= len(test_data):
            sample_data = test_data[:size]
            sample_file = DATA_DIR / f"{dataset_name}_test_sample_{size}.json"
            
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"{dataset_name.upper()}采样数据集 ({size}条) 已保存到: {sample_file}")

def create_math_subset_by_difficulty():
    """根据难度创建MATH数据集的子集"""
    logger.info("创建MATH数据集难度子集...")
    
    test_file = DATA_DIR / "math_test.json"
    if not test_file.exists():
        logger.warning("MATH测试集文件不存在,跳过难度子集创建")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 按难度分组
    levels = {}
    for item in test_data:
        level = item.get("level", "unknown")
        if level not in levels:
            levels[level] = []
        levels[level].append(item)
    
    # 为每个难度创建子集
    for level, items in levels.items():
        level_file = DATA_DIR / f"math_test_level_{level}.json"
        with open(level_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.info(f"MATH难度子集 Level {level} ({len(items)}条) 已保存到: {level_file}")

def create_math_subset_by_type():
    """根据类别创建MATH数据集的子集"""
    logger.info("创建MATH数据集类别子集...")
    
    test_file = DATA_DIR / "math_test.json"
    if not test_file.exists():
        logger.warning("MATH测试集文件不存在,跳过类别子集创建")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 按类别分组
    types = {}
    for item in test_data:
        prob_type = item.get("type", "unknown")
        if prob_type not in types:
            types[prob_type] = []
        types[prob_type].append(item)
    
    # 为每个类别创建子集
    for prob_type, items in types.items():
        # 清理文件名中的特殊字符
        safe_type = prob_type.replace(" ", "_").replace("/", "_")
        type_file = DATA_DIR / f"math_test_type_{safe_type}.json"
        with open(type_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.info(f"MATH类别子集 {prob_type} ({len(items)}条) 已保存到: {type_file}")

def preprocess_answers(dataset_name="gsm8k"):
    """预处理答案,提取数值答案
    
    Args:
        dataset_name: "gsm8k" 或 "math"
    """
    logger.info(f"预处理{dataset_name.upper()}答案...")
    
    import re
    
    def extract_numerical_answer_gsm8k(answer_text: str) -> str:
        """从GSM8K答案文本中提取数值"""
        # 查找 #### 后的数字
        match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', answer_text)
        if match:
            num = float(match.group(1))
            return str(int(num)) if num.is_integer() else str(num)
        return None
    
    def extract_numerical_answer_math(answer_text: str) -> str:
        """从MATH答案文本中提取数值
        MATH数据集的答案通常在 \\boxed{} 中
        """
        # 查找 \boxed{} 中的内容
        match = re.search(r'\\boxed\{([^}]+)\}', answer_text)
        if match:
            answer = match.group(1).strip()
            # 尝试提取纯数字
            num_match = re.search(r'([+-]?\d+(?:\.\d+)?)', answer)
            if num_match:
                num = float(num_match.group(1))
                return str(int(num)) if num.is_integer() else str(num)
            # 如果不是纯数字,返回原始答案
            return answer
        return None
    
    # 选择提取函数
    extract_func = extract_numerical_answer_gsm8k if dataset_name == "gsm8k" else extract_numerical_answer_math
    
    # 处理训练集
    train_file = DATA_DIR / f"{dataset_name}_train.json"
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        for item in train_data:
            item['numerical_answer'] = extract_func(item['answer'])
        
        processed_train_file = DATA_DIR / f"{dataset_name}_train_processed.json"
        with open(processed_train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理后的{dataset_name.upper()}训练集已保存到: {processed_train_file}")
    
    # 处理测试集
    test_file = DATA_DIR / f"{dataset_name}_test.json"
    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        for item in test_data:
            item['numerical_answer'] = extract_func(item['answer'])
        
        processed_test_file = DATA_DIR / f"{dataset_name}_test_processed.json"
        with open(processed_test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理后的{dataset_name.upper()}测试集已保存到: {processed_test_file}")
        
        # 统计信息
        valid_answers = sum(1 for item in test_data if item['numerical_answer'] is not None)
        logger.info(f"{dataset_name.upper()}测试集中有效数值答案: {valid_answers}/{len(test_data)}")

def create_config_file():
    """创建配置文件(与实验脚本兼容的格式)"""
    logger.info("创建配置文件...")
    
    config = {
        "model_configs": {
            "qwen2.5-1.5b": {
                "name": "Qwen/Qwen2.5-1.5B-Instruct",
                "description": "轻量级 - 适合快速测试"
            },
            "qwen2.5-3b": {
                "name": "Qwen/Qwen2.5-3B-Instruct", 
                "description": "平衡性能 - 推荐日常使用"
            },
            "qwen2.5-7b": {
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "description": "高性能 - 较慢但更准确"
            },
            "qwen2.5-14b": {
                "name": "Qwen/Qwen2.5-14B-Instruct",
                "description": "最高性能 - 需要大显存"
            }
        },
        "active_model": "qwen2.5-1.5b",  # 默认使用1.5B模型
        "active_dataset": "gsm8k",  # 默认使用GSM8K数据集,可选: "gsm8k", "math"
        "experiment": {
            "sample_size": 50,           # 默认50个样本,适合实验
            "verbose": False,            # 默认不显示详细输出,避免刷屏
            "save_results": True,        # 默认保存结果
            "max_reasoning_steps": 10,   # 数学题一般不需要太多步骤
            "max_new_tokens": 200,       # 控制生成长度
            "temperature": 0.0,          # 确保输出稳定
            "do_sample": False          # 贪婪解码,确保一致性
        },
        "paths": {
            "data_dir": str(DATA_DIR),
            "results_dir": str(BASE_DIR / "results"),
            # GSM8K路径
            "gsm8k_test_data": str(DATA_DIR / "gsm8k_test_processed.json"),
            "gsm8k_raw_test_data": str(DATA_DIR / "gsm8k_test.json"),
            # MATH路径
            "math_test_data": str(DATA_DIR / "math_test_processed.json"),
            "math_raw_test_data": str(DATA_DIR / "math_test.json"),
            # 采样数据路径
            "sample_data_5": str(DATA_DIR / "gsm8k_test_sample_5.json"),
            "sample_data_10": str(DATA_DIR / "gsm8k_test_sample_10.json"),
            "sample_data_20": str(DATA_DIR / "gsm8k_test_sample_20.json"),
            "sample_data_50": str(DATA_DIR / "gsm8k_test_sample_50.json"),
            "sample_data_100": str(DATA_DIR / "gsm8k_test_sample_100.json"),
            "sample_data_200": str(DATA_DIR / "gsm8k_test_sample_200.json")
        },
        "generation": {
            "max_length": 512,           # 输入最大长度
            "max_new_tokens": 200,       # 生成最大长度
            "do_sample": False,          # 不采样,使用贪婪解码
            "temperature": 0.0,          # 温度为0,确保一致性
            "top_p": 1.0,               # 不使用top-p
            "num_return_sequences": 1    # 只返回一个序列
        },
        "halt_cot": {
            "entropy_threshold_strict": 0.3,
            "entropy_threshold_loose": 0.8,
            "k_strict": 2,
            "k_normal": 4,
            "k_conservative": 8,
            "min_reasoning_steps": 3,
            "max_reasoning_steps": 15,
            "entropy_history_size": 10,
            "confidence_decay": 0.95
        },
        "model": {
            "device": "cuda",
            "torch_dtype": "bfloat16"
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "version": "1.2",
            "purpose": "GSM8K和MATH数据集HALT-CoT实验",
            "supported_datasets": ["gsm8k", "math"]
        }
    }
    
    config_file = CONFIG_DIR / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"配置文件已保存到: {config_file}")
    
    # 打印配置信息
    print("\n" + "="*60)
    print("🤖 可用模型配置:")
    for key, model_config in config['model_configs'].items():
        marker = "👈 [当前]" if key == config["active_model"] else ""
        print(f"  {key}: {model_config['name']} {marker}")
        print(f"    {model_config['description']}")
    
    print("\n📊 支持的数据集:")
    for dataset in config['metadata']['supported_datasets']:
        marker = "👈 [当前]" if dataset == config["active_dataset"] else ""
        print(f"  {dataset.upper()} {marker}")
    
    print("\n⚙️ 实验配置:")
    print(f"  样本数量: {config['experiment']['sample_size']}")
    print(f"  详细输出: {config['experiment']['verbose']}")
    print(f"  最大推理步数: {config['experiment']['max_reasoning_steps']}")
    print("="*60)
    
    return config_file

def download_model_cache():
    """预下载模型缓存(可选)"""
    logger.info("预下载模型缓存...")
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        
        # 下载1.5B模型(默认模型)
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        cache_dir = BASE_DIR / "model_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # 下载tokenizer
        logger.info("下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # 下载模型配置
        logger.info("下载模型配置...")
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        logger.info(f"模型缓存已下载到: {cache_dir}")
        
    except Exception as e:
        logger.warning(f"模型缓存下载失败(可忽略): {e}")

def verify_data_integrity():
    """验证数据完整性"""
    logger.info("验证数据完整性...")
    
    files_to_check = [
        # GSM8K文件
        DATA_DIR / "gsm8k_train.json",
        DATA_DIR / "gsm8k_test.json",
        DATA_DIR / "gsm8k_train_processed.json", 
        DATA_DIR / "gsm8k_test_processed.json",
        DATA_DIR / "gsm8k_test_sample_5.json",
        DATA_DIR / "gsm8k_test_sample_10.json",
        DATA_DIR / "gsm8k_test_sample_20.json",
        DATA_DIR / "gsm8k_test_sample_50.json",
        DATA_DIR / "gsm8k_test_sample_100.json",
        DATA_DIR / "gsm8k_test_sample_200.json",
        # MATH文件
        DATA_DIR / "math_train.json",
        DATA_DIR / "math_test.json",
        DATA_DIR / "math_train_processed.json",
        DATA_DIR / "math_test_processed.json",
        # 配置文件
        CONFIG_DIR / "config.json"
    ]
    
    all_good = True
    for file_path in files_to_check:
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                size_info = len(data) if isinstance(data, list) else 'OK'
                logger.info(f"✅ {file_path.name}: {size_info}")
            except Exception as e:
                logger.error(f"❌ {file_path.name}: 文件损坏 - {e}")
                all_good = False
        else:
            # MATH数据集文件是可选的
            if "math" in file_path.name:
                logger.warning(f"⚠️  {file_path.name}: 文件不存在 (MATH数据集可选)")
            else:
                logger.error(f"❌ {file_path.name}: 文件不存在")
                all_good = False
    
    if all_good:
        logger.info("✅ 所有必需的数据文件验证通过!")
    else:
        logger.error("❌ 部分数据文件有问题,请重新运行数据准备")
    
    return all_good

def update_existing_config():
    """更新现有配置文件以确保兼容性"""
    config_file = CONFIG_DIR / "config.json"
    
    if config_file.exists():
        logger.info("发现现有配置文件,更新以确保兼容性...")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 添加数据集选择
            if 'active_dataset' not in config:
                config['active_dataset'] = 'gsm8k'
                logger.info("添加active_dataset配置: gsm8k")
            
            # 确保experiment配置包含所有必要字段
            if 'experiment' not in config:
                config['experiment'] = {}
            
            experiment_defaults = {
                "sample_size": 50,
                "verbose": False,
                "save_results": True,
                "max_reasoning_steps": 10,
                "max_new_tokens": 200,
                "temperature": 0.0,
                "do_sample": False
            }
            
            for key, value in experiment_defaults.items():
                if key not in config['experiment']:
                    config['experiment'][key] = value
                    logger.info(f"添加缺失的配置项: experiment.{key} = {value}")
            
            # 确保paths配置完整
            if 'paths' not in config:
                config['paths'] = {}
            
            path_defaults = {
                "data_dir": str(DATA_DIR),
                "results_dir": str(BASE_DIR / "results"),
                "gsm8k_test_data": str(DATA_DIR / "gsm8k_test_processed.json"),
                "gsm8k_raw_test_data": str(DATA_DIR / "gsm8k_test.json"),
                "math_test_data": str(DATA_DIR / "math_test_processed.json"),
                "math_raw_test_data": str(DATA_DIR / "math_test.json")
            }
            
            for key, value in path_defaults.items():
                if key not in config['paths']:
                    config['paths'][key] = value
                    logger.info(f"添加缺失的路径配置: paths.{key}")
            
            # 确保generation配置存在
            if 'generation' not in config:
                config['generation'] = {
                    "max_length": 512,
                    "max_new_tokens": 200,
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "num_return_sequences": 1
                }
                logger.info("添加generation配置")
            
            # 更新metadata
            if 'metadata' not in config:
                config['metadata'] = {}
            config['metadata']['updated_at'] = datetime.now().isoformat()
            config['metadata']['version'] = "1.2"
            config['metadata']['supported_datasets'] = ["gsm8k", "math"]
            
            # 保存更新后的配置
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info("配置文件已更新")
            return True
            
        except Exception as e:
            logger.error(f"更新配置文件失败: {e}")
            return False
    
    return False

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始数据准备流程")
    logger.info("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description='数据下载和预处理脚本')
    parser.add_argument('--dataset', type=str, default='both', 
                       choices=['gsm8k', 'math', 'both'],
                       help='选择要下载的数据集: gsm8k, math, 或 both')
    parser.add_argument('--skip-download', action='store_true',
                       help='跳过下载,仅处理现有数据')
    args = parser.parse_args()
    
    try:
        # 0. 首先尝试更新现有配置
        config_updated = update_existing_config()
        
        if not args.skip_download:
            # 1. 下载数据集
            if args.dataset in ['gsm8k', 'both']:
                download_gsm8k_dataset()
                create_sample_datasets('gsm8k')
                preprocess_answers('gsm8k')
            
            if args.dataset in ['math', 'both']:
                try:
                    download_math_dataset()
                    create_sample_datasets('math')
                    create_math_subset_by_difficulty()
                    create_math_subset_by_type()
                    preprocess_answers('math')
                except Exception as e:
                    logger.error(f"MATH数据集处理失败: {e}")
                    logger.info("继续处理其他数据集...")
        
        # 2. 创建/更新配置文件
        if not config_updated:
            create_config_file()
        
        # 3. 预下载模型缓存(可选)
        download_model_cache()
        
        # 4. 验证数据完整性
        verify_data_integrity()
        
        logger.info("=" * 60)
        logger.info("数据准备完成! 可以运行模型代码了")
        logger.info("=" * 60)
        
        # 创建结果目录
        results_dir = BASE_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        
        logger.info(f"数据存储位置: {DATA_DIR}")
        logger.info(f"配置文件位置: {CONFIG_DIR}")
        logger.info(f"结果保存位置: {results_dir}")
        
        print("\n🎉 数据准备完成!")
        print("💡 提示:")
        print("   1. 可以使用 config_manager.py 来调整实验参数")
        print("   2. 运行 experiment_runner.py 开始实验")
        print("   3. 建议先用小样本测试 (sample_size=10)")
        print(f"   4. 当前支持的数据集: GSM8K" + (" 和 MATH" if args.dataset in ['math', 'both'] else ""))
        
    except Exception as e:
        logger.error(f"数据准备过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
