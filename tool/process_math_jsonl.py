"""
处理 MATH 数据集 - 支持 JSONL 格式
"""

import json
import zipfile
from pathlib import Path
import logging

BASE_DIR = Path("/root/autodl-tmp")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_math_zip_jsonl(zip_path):
    """处理包含 JSONL 文件的 MATH 数据集"""
    
    logger.info("="*60)
    logger.info("处理 MATH 数据集 (JSONL 格式)")
    logger.info("="*60)
    
    zip_path = Path(zip_path)
    if not zip_path.exists():
        logger.error(f"文件不存在: {zip_path}")
        return False
    
    # 解压
    extract_dir = DATA_DIR / "math_extracted"
    extract_dir.mkdir(exist_ok=True)
    
    logger.info(f"正在解压 {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # 查找 data 目录
    data_dirs = list(extract_dir.glob("**/data"))
    if not data_dirs:
        logger.warning("未找到 data 目录，尝试查找 .jsonl 文件...")
        jsonl_files = list(extract_dir.rglob("*.jsonl"))
        if jsonl_files:
            data_dir = jsonl_files[0].parent
        else:
            logger.error("未找到 JSONL 文件")
            return False
    else:
        data_dir = data_dirs[0]
    
    logger.info(f"找到数据目录: {data_dir}")
    
    # 处理数据
    train_data = []
    test_data = []
    
    # 所有类别
    categories = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ]
    
    logger.info("\n处理训练集...")
    for category in categories:
        train_file = data_dir / f"{category}_train.jsonl"
        if train_file.exists():
            count = process_jsonl_file(train_file, train_data, category)
            logger.info(f"  ✅ {category}: {count} 题")
        else:
            logger.warning(f"  ⚠️  未找到: {train_file.name}")
    
    logger.info("\n处理测试集...")
    for category in categories:
        test_file = data_dir / f"{category}_test.jsonl"
        if test_file.exists():
            count = process_jsonl_file(test_file, test_data, category)
            logger.info(f"  ✅ {category}: {count} 题")
        else:
            logger.warning(f"  ⚠️  未找到: {test_file.name}")
    
    if not train_data and not test_data:
        logger.error("没有找到有效数据")
        return False
    
    # 保存数据
    save_datasets(train_data, test_data)
    return True

def process_jsonl_file(jsonl_file, data_list, category):
    """处理单个 JSONL 文件"""
    count = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                entry = {
                    "question": item.get("problem", ""),
                    "answer": item.get("solution", ""),
                    "level": item.get("level", "unknown"),
                    "type": category.replace('_', ' ').title(),
                    "dataset": "math"
                }
                # 提取数值答案
                entry["numerical_answer"] = extract_numerical_answer(entry["answer"])
                data_list.append(entry)
                count += 1
            except Exception as e:
                logger.warning(f"解析行失败: {e}")
                continue
    return count

def extract_numerical_answer(solution_text):
    """从解答中提取数值答案"""
    import re
    
    # 查找 \boxed{} 中的内容
    match = re.search(r'\\boxed\{([^}]+)\}', solution_text)
    if match:
        answer = match.group(1).strip()
        # 尝试提取数字
        num_match = re.search(r'([+-]?\d+(?:\.\d+)?)', answer)
        if num_match:
            try:
                num = float(num_match.group(1))
                return str(int(num)) if num.is_integer() else str(num)
            except:
                pass
        return answer
    
    # 查找最后一个数字
    numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', solution_text)
    if numbers:
        try:
            num = float(numbers[-1])
            return str(int(num)) if num.is_integer() else str(num)
        except:
            pass
    
    return None

def save_datasets(train_data, test_data):
    """保存数据集"""
    logger.info("\n保存数据集...")
    
    files_created = []
    
    # 1. 保存原始数据
    if train_data:
        train_file = DATA_DIR / "math_train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        files_created.append(f"math_train.json ({len(train_data)} 题)")
        
        processed_train = DATA_DIR / "math_train_processed.json"
        with open(processed_train, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        files_created.append("math_train_processed.json")
    
    if test_data:
        test_file = DATA_DIR / "math_test.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        files_created.append(f"math_test.json ({len(test_data)} 题)")
        
        processed_test = DATA_DIR / "math_test_processed.json"
        with open(processed_test, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        files_created.append("math_test_processed.json")
    
    # 2. 创建采样
    if test_data:
        logger.info("创建采样数据集...")
        sample_sizes = [5, 10, 20, 50, 100, 200]
        for size in sample_sizes:
            if size <= len(test_data):
                sample_file = DATA_DIR / f"math_test_sample_{size}.json"
                with open(sample_file, 'w', encoding='utf-8') as f:
                    json.dump(test_data[:size], f, ensure_ascii=False, indent=2)
                files_created.append(f"math_test_sample_{size}.json")
    
    # 3. 按难度分类
    if test_data:
        logger.info("创建难度子集...")
        levels = {}
        for item in test_data:
            level = item.get('level', 'unknown')
            if level not in levels:
                levels[level] = []
            levels[level].append(item)
        
        for level, items in levels.items():
            level_file = DATA_DIR / f"math_test_level_{level.replace(' ', '_')}.json"
            with open(level_file, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            files_created.append(f"math_test_level_{level}.json ({len(items)} 题)")
    
    # 4. 按类别分类
    if test_data:
        logger.info("创建类别子集...")
        types = {}
        for item in test_data:
            prob_type = item.get('type', 'unknown')
            if prob_type not in types:
                types[prob_type] = []
            types[prob_type].append(item)
        
        for prob_type, items in types.items():
            safe_type = prob_type.replace(' ', '_').replace('&', 'And')
            type_file = DATA_DIR / f"math_test_type_{safe_type}.json"
            with open(type_file, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            files_created.append(f"math_test_type_{safe_type}.json ({len(items)} 题)")
    
    # 打印统计
    print_statistics(train_data, test_data, files_created)

def print_statistics(train_data, test_data, files_created):
    """打印统计信息"""
    print("\n" + "="*60)
    print("📊 MATH 数据集统计")
    print("="*60)
    
    print(f"\n✅ 成功创建 {len(files_created)} 个文件:")
    for f in files_created[:10]:  # 只显示前10个
        print(f"   - {f}")
    if len(files_created) > 10:
        print(f"   ... 还有 {len(files_created) - 10} 个文件")
    
    if train_data:
        print(f"\n📚 训练集: {len(train_data)} 题")
        
        # 难度分布
        levels = {}
        for item in train_data:
            level = item.get('level', 'unknown')
            levels[level] = levels.get(level, 0) + 1
        
        print("\n难度分布:")
        for level in sorted(levels.keys()):
            count = levels[level]
            percentage = (count / len(train_data)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {level:15s}: {count:5d} ({percentage:5.1f}%) {bar}")
        
        # 类别分布
        types = {}
        for item in train_data:
            prob_type = item.get('type', 'unknown')
            types[prob_type] = types.get(prob_type, 0) + 1
        
        print("\n类别分布:")
        for prob_type in sorted(types.keys()):
            count = types[prob_type]
            percentage = (count / len(train_data)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {prob_type:30s}: {count:5d} ({percentage:5.1f}%) {bar}")
    
    if test_data:
        print(f"\n📝 测试集: {len(test_data)} 题")
        
        # 难度分布
        levels = {}
        for item in test_data:
            level = item.get('level', 'unknown')
            levels[level] = levels.get(level, 0) + 1
        
        print("\n难度分布:")
        for level in sorted(levels.keys()):
            count = levels[level]
            percentage = (count / len(test_data)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {level:15s}: {count:5d} ({percentage:5.1f}%) {bar}")
        
        # 答案提取成功率
        valid_answers = sum(1 for item in test_data if item.get('numerical_answer'))
        success_rate = (valid_answers / len(test_data)) * 100
        print(f"\n✨ 答案提取成功率: {valid_answers}/{len(test_data)} ({success_rate:.1f}%)")
    
    print("="*60)

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    else:
        zip_path = BASE_DIR / "main.zip"
        
        if not Path(zip_path).exists():
            print("="*60)
            print("MATH 数据集处理工具 (JSONL 格式)")
            print("="*60)
            print("\n使用方法:")
            print(f"  python process_math_jsonl.py <zip文件路径>")
            print("\n或将 zip 文件放到:")
            print(f"  {zip_path}")
            print("\n然后运行:")
            print(f"  python process_math_jsonl.py")
            print("="*60)
            return 1
    
    success = process_math_zip_jsonl(zip_path)
    
    if success:
        print("\n🎉 MATH 数据集处理完成!")
        print("\n💡 提示:")
        print("   - 数据已保存到 /root/autodl-tmp/data/")
        print("   - 可以使用 math_test_sample_*.json 进行快速测试")
        print("   - 可以按难度或类别选择子集进行实验")
        return 0
    else:
        print("\n❌ 处理失败")
        return 1

if __name__ == "__main__":
    exit(main())
