"""配置管理模块"""
import json
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG = {
        "active_model": "qwen",
        "model_configs": {
            "qwen": {"name": "Qwen/Qwen2.5-7B-Instruct"}
        },
        "paths": {
            "test_data": "data/gsm8k_test.json"
        },
        "experiment": {
            "sample_size": 10,
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.7,
            "save_results": True,
            "verbose": False,
            "debug_probe": False
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
