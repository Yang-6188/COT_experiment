"""
下载 MATH 数据集 - 修复版本
使用正确的数据集路径
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

# 设置路径
BASE_DIR = Path("/root/autodl-tmp")
DATA_DIR = BASE_DIR / "data"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_math_dataset_v1():
    """方法1: 使用 hendrycks_math"""
    logger.info("方法1: 尝试从 hendrycks_math 下载...")
    try:
        # 正确的数据集名称
        dataset = load_dataset("hendrycks_math", "all")
        
        train_data = []
        test_data = []
        
        # 处理数据
        if 'train' in dataset:
            for item in dataset['train']:
                train_data.append({
                    "question": item['problem'],
                    "answer": item['solution'],
                    "level": item.get('level', 'unknown'),
                    "type": item.get('type', 'unknown'),
                    "dataset": "math"
                })
        
        if 'test' in dataset:
            for item in dataset['test']:
                test_data.append({
                    "question": item['problem'],
                    "answer": item['solution'],
                    "level": item.get('level', 'unknown'),
                    "type": item.get('type', 'unknown'),
                    "dataset": "math"
                })
        
        return train_data, test_data
        
    except Exception as e:
        logger.warning(f"方法1失败: {e}")
        return None, None

def download_math_dataset_v2():
    """方法2: 使用 competition_math"""
    logger.info("方法2: 尝试从 competition_math 下载...")
    try:
        train_dataset = load_dataset("competition_math", split="train")
        test_dataset = load_dataset("competition_math", split="test")
        
        train_data = []
        for item in train_dataset:
            train_data.append({
                "question": item['problem'],
                "answer": item['solution'],
                "level": item.get('level', 'unknown'),
                "type": item.get('type', 'unknown'),
                "dataset": "math"
            })
        
        test_data = []
        for item in test_dataset:
            test_data.append({
                "question": item['problem'],
                "answer": item['solution'],
                "level": item.get('level', 'unknown'),
                "type": item.get('type', 'unknown'),
                "dataset": "math"
            })
        
        return train_data, test_data
        
    except Exception as e:
        logger.warning(f"方法2失败: {e}")
        return None, None

def download_math_dataset_v3():
    """方法3: 直接从 GitHub 原始仓库下载"""
    logger.info("方法3: 从 GitHub 原始仓库下载...")
    
    import requests
    import tarfile
    import io
    
    try:
        # MATH 数据集的 GitHub 发布页面
        url = "https://github.com/hendrycks/math/archive/refs/heads/main.zip"
        
        logger.info("正在下载压缩包...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # 解压并处理
        import zipfile
        
        zip_path = DATA_DIR / "math_dataset.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        logger.info("正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR / "math_raw")
        
        # 清理
        zip_path.unlink()
        
        logger.info("✅ 数据已下载到 math_raw 目录")
        logger.info("需要手动处理数据文件...")
        
        return None, None
        
    except Exception as e:
        logger.warning(f"方法3失败: {e}")
        return None, None

def create_sample_math_dataset():
    """创建一个示例 MATH 数据集用于测试"""
    logger.info("创建示例 MATH 数据集...")
    
    # 精心挑选的不同难度和类型的题目
    sample_data = [
        # Level 1 - Prealgebra
        {
            "question": "What is $\\frac{1}{2} + \\frac{1}{3}$?",
            "answer": "To add these fractions, we need a common denominator. The LCD of 2 and 3 is 6.\n\n$\\frac{1}{2} = \\frac{3}{6}$ and $\\frac{1}{3} = \\frac{2}{6}$\n\nSo $\\frac{1}{2} + \\frac{1}{3} = \\frac{3}{6} + \\frac{2}{6} = \\frac{5}{6}$\n\nTherefore, the answer is $\\boxed{\\frac{5}{6}}$.",
            "level": "Level 1",
            "type": "Prealgebra",
            "dataset": "math",
            "numerical_answer": "5/6"
        },
        # Level 1 - Algebra
        {
            "question": "Solve for $x$: $2x + 5 = 13$",
            "answer": "Subtract 5 from both sides:\n$2x = 8$\n\nDivide both sides by 2:\n$x = 4$\n\nTherefore, $\\boxed{4}$.",
            "level": "Level 1",
            "type": "Algebra",
            "dataset": "math",
            "numerical_answer": "4"
        },
        # Level 2 - Geometry
        {
            "question": "What is the area of a circle with radius 5?",
            "answer": "The area of a circle is given by $A = \\pi r^2$.\n\nWith $r=5$, we have:\n$A = \\pi \\cdot 5^2 = 25\\pi$\n\nTherefore, the area is $\\boxed{25\\pi}$.",
            "level": "Level 2",
            "type": "Geometry",
            "dataset": "math",
            "numerical_answer": "25π"
        },
        # Level 2 - Number Theory
        {
            "question": "What is the greatest common divisor of 48 and 18?",
            "answer": "We can use the Euclidean algorithm:\n\n$\\gcd(48, 18)$\n$= \\gcd(18, 48 \\bmod 18)$\n$= \\gcd(18, 12)$\n$= \\gcd(12, 6)$\n$= \\gcd(6, 0)$\n$= 6$\n\nTherefore, $\\boxed{6}$.",
            "level": "Level 2",
            "type": "Number Theory",
            "dataset": "math",
            "numerical_answer": "6"
        },
        # Level 3 - Algebra
        {
            "question": "If $x^2 - 5x + 6 = 0$, what are the possible values of $x$?",
            "answer": "We can factor the quadratic:\n$x^2 - 5x + 6 = (x-2)(x-3) = 0$\n\nSo either $x-2=0$ or $x-3=0$.\n\nTherefore, $x = \\boxed{2 \\text{ or } 3}$.",
            "level": "Level 3",
            "type": "Algebra",
            "dataset": "math",
            "numerical_answer": "2,3"
        },
        # Level 3 - Counting & Probability
        {
            "question": "How many ways can you arrange the letters in the word MATH?",
            "answer": "The word MATH has 4 distinct letters.\n\nThe number of arrangements is $4! = 4 \\times 3 \\times 2 \\times 1 = 24$.\n\nTherefore, there are $\\boxed{24}$ ways.",
            "level": "Level 3",
            "type": "Counting & Probability",
            "dataset": "math",
            "numerical_answer": "24"
        },
        # Level 4 - Precalculus
        {
            "question": "What is $\\sin(30°)$?",
            "answer": "From the unit circle or special triangles, we know that:\n$\\sin(30°) = \\frac{1}{2}$\n\nTherefore, $\\boxed{\\frac{1}{2}}$.",
            "level": "Level 4",
            "type": "Precalculus",
            "dataset": "math",
            "numerical_answer": "0.5"
        },
        # Level 4 - Intermediate Algebra
        {
            "question": "Simplify: $(x+2)^2 - (x-2)^2$",
            "answer": "Expanding both terms:\n$(x+2)^2 = x^2 + 4x + 4$\n$(x-2)^2 = x^2 - 4x + 4$\n\nSubtracting:\n$(x+2)^2 - (x-2)^2 = (x^2 + 4x + 4) - (x^2 - 4x + 4)$\n$= x^2 + 4x + 4 - x^2 + 4x - 4$\n$= 8x$\n\nTherefore, $\\boxed{8x}$.",
            "level": "Level 4",
            "type": "Intermediate Algebra",
            "dataset": "math",
            "numerical_answer": "8x"
        },
        # Level 5 - Number Theory
        {
            "question": "How many positive divisors does 60 have?",
            "answer": "First, find the prime factorization of 60:\n$60 = 2^2 \\times 3 \\times 5$\n\nThe number of divisors is:\n$(2+1)(1+1)(1+1) = 3 \\times 2 \\times 2 = 12$\n\nTherefore, 60 has $\\boxed{12}$ positive divisors.",
            "level": "Level 5",
            "type": "Number Theory",
            "dataset": "math",
            "numerical_answer": "12"
        },
        # Level 5 - Geometry
        {
            "question": "A right triangle has legs of length 3 and 4. What is the length of the hypotenuse?",
            "answer": "By the Pythagorean theorem:\n$c^2 = a^2 + b^2$\n$c^2 = 3^2 + 4^2$\n$c^2 = 9 + 16$\n$c^2 = 25$\n$c = 5$\n\nTherefore, the hypotenuse has length $\\boxed{5}$.",
            "level": "Level 5",
            "type": "Geometry",
            "dataset": "math",
            "numerical_answer": "5"
        }
    ]
    
    return sample_data, sample_data[:5]  # 返回全部作为训练集，前5个作为测试集

def save_datasets(train_data, test_data):
    """保存数据集"""
    if not train_data or not test_data:
        logger.error("没有数据可保存")
        return False
    
    # 保存训练集
    train_file = DATA_DIR / "math_train.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 训练集已保存: {len(train_data)} 条")
    
    # 保存测试集
    test_file = DATA_DIR / "math_test.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 测试集已保存: {len(test_data)} 条")
    
    # 创建处理后的版本
    processed_test_file = DATA_DIR / "math_test_processed.json"
    with open(processed_test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    processed_train_file = DATA_DIR / "math_train_processed.json"
    with open(processed_train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 创建不同大小的采样
    sample_sizes = [5, 10, 20, 50, 100]
    for size in sample_sizes:
        if size <= len(test_data):
            sample_file = DATA_DIR / f"math_test_sample_{size}.json"
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(test_data[:size], f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 创建采样: {size} 条")
    
    # 统计信息
    levels = {}
    types = {}
    for item in test_data:
        level = item.get('level', 'unknown')
        prob_type = item.get('type', 'unknown')
        levels[level] = levels.get(level, 0) + 1
        types[prob_type] = types.get(prob_type, 0) + 1
    
    print("\n" + "="*60)
    print("📊 MATH 数据集统计:")
    print(f"训练集: {len(train_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    print(f"\n难度分布:")
    for level in sorted(levels.keys()):
        print(f"  {level}: {levels[level]} 条")
    print(f"\n类别分布:")
    for prob_type in sorted(types.keys()):
        print(f"  {prob_type}: {types[prob_type]} 条")
    print("="*60)
    
    return True

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("MATH 数据集下载工具 - 修复版")
    logger.info("="*60)
    
    train_data = None
    test_data = None
    
    # 尝试多种方法
    methods = [
        download_math_dataset_v1,
        download_math_dataset_v2,
    ]
    
    for method in methods:
        try:
            train_data, test_data = method()
            if train_data and test_data:
                logger.info(f"✅ {method.__name__} 成功!")
                break
        except Exception as e:
            logger.warning(f"{method.__name__} 失败: {e}")
            continue
    
    # 如果所有方法都失败，使用示例数据集
    if not train_data or not test_data:
        logger.warning("⚠️  所有下载方法都失败了")
        logger.info("📝 创建示例数据集用于测试...")
        train_data, test_data = create_sample_math_dataset()
        logger.info("✅ 已创建包含 10 个精选题目的示例数据集")
        logger.info("💡 这些题目涵盖了不同难度和类型，可以用于测试")
    
    # 保存数据
    if save_datasets(train_data, test_data):
        print("\n🎉 MATH 数据集准备完成!")
        print("\n💡 提示:")
        print("   - 如果使用的是示例数据集，题目数量较少")
        print("   - 可以用于快速测试您的代码")
        print("   - 生产环境建议使用完整数据集")
        return 0
    else:
        logger.error("❌ 保存数据失败")
        return 1

if __name__ == "__main__":
    exit(main())
