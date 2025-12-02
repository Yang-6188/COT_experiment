"""配置管理模块"""
import json
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG = {
        "active_model": "qwen",  # 当前激活的模型
        
        "model_configs": {
            "qwen": {"name": "Qwen/Qwen2.5-7B-Instruct"}  # Qwen模型配置
        },
        
        "paths": {
            # ===== GSM8K 数据集路径 =====
            "test_data": "data/math_extracted/math_test_level_Level_3.json",  # 默认测试数据
            
            # GSM8K 测试集选项
            "gsm8k_test_full": "data/math_extracted/gsm8k_test.json",  # 完整测试集
            "gsm8k_test_processed": "data/math_extracted/gsm8k_test_processed.json",  # 预处理测试集
            "gsm8k_test_sample_5": "data/math_extracted/gsm8k_test_sample_5.json",  # 5样本测试集
            "gsm8k_test_sample_10": "data/math_extracted/gsm8k_test_sample_10.json",  # 10样本测试集
            "gsm8k_test_sample_20": "data/math_extracted/gsm8k_test_sample_20.json",  # 20样本测试集
            "gsm8k_test_sample_50": "data/math_extracted/gsm8k_test_sample_50.json",  # 50样本测试集
            "gsm8k_test_sample_100": "data/math_extracted/gsm8k_test_sample_100.json",  # 100样本测试集
            "gsm8k_test_sample_200": "data/math_extracted/gsm8k_test_sample_200.json",  # 200样本测试集
            
            # GSM8K 训练集选项
            "gsm8k_train_full": "data/math_extracted/gsm8k_train.json",  # 完整训练集
            "gsm8k_train_processed": "data/math_extracted/gsm8k_train_processed.json",  # 预处理训练集
            
            # ===== MATH 数据集路径 =====
            # MATH 测试集 - 按难度级别
            "math_test_full": "data/math_extracted/math_test.json",  # 完整测试集
            "math_test_processed": "data/math_extracted/math_test_processed.json",  # 预处理测试集
            "math_test_level_1": "data/math_extracted/math_test_level_Level_1.json",  # 难度级别1
            "math_test_level_2": "data/math_extracted/math_test_level_Level_2.json",  # 难度级别2
            "math_test_level_3": "data/math_extracted/math_test_level_Level_3.json",  # 难度级别3
            "math_test_level_4": "data/math_extracted/math_test_level_Level_4.json",  # 难度级别4
            "math_test_level_5": "data/math_extracted/math_test_level_Level_5.json",  # 难度级别5
            
            # MATH 测试集 - 按题目类型
            "math_test_algebra": "data/math_extracted/math_test_type_Algebra.json",  # 代数题
            "math_test_counting": "data/math_extracted/math_test_type_Counting_And_Probability.json",  # 计数与概率
            "math_test_geometry": "data/math_extracted/math_test_type_Geometry.json",  # 几何题
            "math_test_intermediate_algebra": "data/math_extracted/math_test_type_Intermediate_Algebra.json",  # 中级代数
            "math_test_number_theory": "data/math_extracted/math_test_type_Number_Theory.json",  # 数论
            "math_test_prealgebra": "data/math_extracted/math_test_type_Prealgebra.json",  # 预备代数
            "math_test_precalculus": "data/math_extracted/math_test_type_Precalculus.json",  # 预备微积分
            
            # MATH 测试集 - 按样本大小
            "math_test_sample_5": "data/math_extracted/math_test_sample_5.json",  # 5样本测试集
            "math_test_sample_10": "data/math_extracted/math_test_sample_10.json",  # 10样本测试集
            "math_test_sample_20": "data/math_extracted/math_test_sample_20.json",  # 20样本测试集
            "math_test_sample_50": "data/math_extracted/math_test_sample_50.json",  # 50样本测试集
            "math_test_sample_100": "data/math_extracted/math_test_sample_100.json",  # 100样本测试集
            "math_test_sample_200": "data/math_extracted/math_test_sample_200.json",  # 200样本测试集
            
            # MATH 训练集选项
            "math_train_full": "data/math_extracted/math_train.json",  # 完整训练集
            "math_train_processed": "data/math_extracted/math_train_processed.json",  # 预处理训练集
            
            # ===== 输出路径 =====
            "results_dir": "results/",  # 结果输出目录
            "results_entropy_dir": "results_entropy/",  # 熵值结果目录
            "old_results_dir": "old_results/",  # 旧结果备份目录
            "logs_dir": "logs/",  # 日志文件目录
        },
        
        "experiment": {
            "sample_size": 50,  # 实验样本数量
            "max_new_tokens": 512,  # 最大生成token数
            "do_sample": False,  # 是否使用采样生成
            "temperature": 0.7,  # 生成温度参数（控制随机性）
            "save_results": True,  # 是否保存结果
            "verbose": False,  # 是否输出详细信息
            "debug_probe": False,  # 是否启用调试探针
            
            # 数据集选择
            "dataset_type": "gsm8k",  # 数据集类型: "gsm8k" 或 "math"
            "test_subset": "full",  # 测试子集: "full", "sample_5", "sample_10"等
            "math_difficulty": None,  # MATH数据集难度级别: 1-5 或 None
            "math_topic": None,  # MATH数据集题目类型: "Algebra", "Geometry"等 或 None
        },
        
        "early_stopping": {
            "use_answer_consistency": True,  # 是否使用答案一致性检查
            "use_entropy_halt": True,  # 是否使用熵值停止机制
            "consistency_k": 3,  # 一致性检查的窗口大小
            "entropy_threshold": 0.6,  # 熵值阈值（低于此值认为模型确定）
            "entropy_consecutive_steps": 2,  # 连续低熵步数要求
            "min_tokens_before_check": 100,  # 开始检查前的最小token数
            "cooldown_tokens": 40,  # 检查后的冷却token数
        }
    }

    # ===== 快速配置预设 =====
    PRESET_CONFIGS = {
        # GSM8K 快速测试
        "gsm8k_quick": {
            "paths.test_data": "data/math_extracted/gsm8k_test_sample_5.json",
            "experiment.sample_size": 5,
        },
        
        # GSM8K 标准测试
        "gsm8k_standard": {
            "paths.test_data": "data/math_extracted/gsm8k_test_sample_50.json",
            "experiment.sample_size": 50,
        },
        
        # MATH 简单难度测试
        "math_easy": {
            "paths.test_data": "data/math_extracted/math_test_level_Level_1.json",
            "experiment.dataset_type": "math",
            "experiment.math_difficulty": 1,
        },
        
        # MATH 困难难度测试
        "math_hard": {
            "paths.test_data": "data/math_extracted/math_test_level_Level_5.json",
            "experiment.dataset_type": "math",
            "experiment.math_difficulty": 5,
        },
        
        # MATH 代数题测试
        "math_algebra": {
            "paths.test_data": "data/math_extracted/math_test_type_Algebra.json",
            "experiment.dataset_type": "math",
            "experiment.math_topic": "Algebra",
        },
    }


    
    @staticmethod
    def load_config(config_path: Path) -> Dict[str, Any]:
        """加载配置文件"""
        if not config_path.exists():
            ConfigManager._create_default_config(config_path)
            return ConfigManager.DEFAULT_CONFIG.copy()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def _create_default_config(config_path: Path):
        """创建默认配置"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(ConfigManager.DEFAULT_CONFIG, f, 
                     ensure_ascii=False, indent=2)
        print(f"✅ 已创建默认配置文件: {config_path}")
