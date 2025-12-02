#!/usr/bin/env python3
"""
增强型HALT-CoT实验配置文件
配置文件用于管理模型、数据路径、实验参数等
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# ============================================================================
# 基础路径配置
# ============================================================================
BASE_DIR = Path("/root/autodl-tmp")
CONFIG_DIR = BASE_DIR / "config_entropy"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results_entropy"
PLOTS_DIR = RESULTS_DIR / "plots"

# 确保目录存在
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 配置数据类
# ============================================================================
@dataclass
class ModelConfig:
    """模型相关配置"""
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "float16"  # 支持 float16, float32, bfloat16
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_cache: bool = True

@dataclass
class DataConfig:
    """数据相关配置"""
    data_path: str = str(DATA_DIR / "gsm8k_test.json")
    sample_size: int = 12
    max_samples: int = 100  # 数据集最大样本数限制
    shuffle_data: bool = False
    random_seed: int = 42

@dataclass
class GenerationConfig:
    """生成相关配置"""
    max_tokens: int = 512
    temperature: float = 0.0  # 确定性生成
    do_sample: bool = False
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0

@dataclass
class ProbeConfig:
    """探测系统配置"""
    cooldown: int = 8  # 探测间隔（token数）
    min_cooldown: int = 5  # 最小探测间隔
    max_cooldown: int = 20  # 最大探测间隔
    probe_max_tokens: int = 20  # 探测生成的最大token数
    entropy_threshold: float = 0.6  # 低熵阈值
    confidence_threshold: float = 0.8  # 置信度阈值
    enable_dynamic_cooldown: bool = True  # 动态调整探测间隔

@dataclass
class VisualizationConfig:
    """可视化配置"""
    enable_plots: bool = True
    plot_dpi: int = 200
    figure_width: int = 18
    figure_height_per_row: int = 5
    plot_columns: int = 3
    save_individual_plots: bool = False
    color_scheme: Dict[str, str] = None

    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'intermediate': '#3498db',
                'calculation': '#f39c12', 
                'conclusion': '#2ecc71',
                'answer_signal': '#e74c3c'
            }

@dataclass
class ExperimentConfig:
    """实验配置"""
    debug: bool = True
    verbose: bool = True
    save_raw_responses: bool = True
    save_probe_details: bool = True
    enable_text_cleaning: bool = True
    strict_answer_extraction: bool = False

@dataclass
class OutputConfig:
    """输出配置"""
    results_dir: str = str(RESULTS_DIR)
    plots_dir: str = str(PLOTS_DIR)
    save_json: bool = True
    save_csv: bool = False
    json_indent: int = 2
    filename_timestamp: bool = True

@dataclass
class HaltCoTConfig:
    """完整的HALT-CoT实验配置"""
    model: ModelConfig
    data: DataConfig
    generation: GenerationConfig
    probe: ProbeConfig
    visualization: VisualizationConfig
    experiment: ExperimentConfig
    output: OutputConfig

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'generation': asdict(self.generation),
            'probe': asdict(self.probe),
            'visualization': asdict(self.visualization),
            'experiment': asdict(self.experiment),
            'output': asdict(self.output)
        }

    def save_to_file(self, filepath: Optional[Path] = None):
        """保存配置到JSON文件"""
        if filepath is None:
            filepath = CONFIG_DIR / "halt_cot_config.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"✅ 配置已保存到: {filepath}")

    @classmethod
    def load_from_file(cls, filepath: Path) -> 'HaltCoTConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            model=ModelConfig(**data['model']),
            data=DataConfig(**data['data']),
            generation=GenerationConfig(**data['generation']),
            probe=ProbeConfig(**data['probe']),
            visualization=VisualizationConfig(**data['visualization']),
            experiment=ExperimentConfig(**data['experiment']),
            output=OutputConfig(**data['output'])
        )

# ============================================================================
# 预设配置模板
# ============================================================================
class ConfigTemplates:
    """配置模板集合"""
    
    @staticmethod
    def default_config() -> HaltCoTConfig:
        """默认配置"""
        return HaltCoTConfig(
            model=ModelConfig(),
            data=DataConfig(),
            generation=GenerationConfig(),
            probe=ProbeConfig(),
            visualization=VisualizationConfig(),
            experiment=ExperimentConfig(),
            output=OutputConfig()
        )
    
    @staticmethod
    def quick_test_config() -> HaltCoTConfig:
        """快速测试配置"""
        config = ConfigTemplates.default_config()
        config.data.sample_size = 5
        config.generation.max_tokens = 256
        config.probe.cooldown = 5
        config.experiment.debug = True
        config.experiment.verbose = True
        return config
    
    @staticmethod
    def large_scale_config() -> HaltCoTConfig:
        """大规模实验配置"""
        config = ConfigTemplates.default_config()
        config.data.sample_size = 100
        config.generation.max_tokens = 1024
        config.probe.cooldown = 10
        config.visualization.plot_columns = 4
        config.experiment.debug = False
        return config
    
    @staticmethod
    def high_precision_config() -> HaltCoTConfig:
        """高精度分析配置"""
        config = ConfigTemplates.default_config()
        config.generation.torch_dtype = "float32"
        config.probe.cooldown = 3
        config.probe.probe_max_tokens = 30
        config.experiment.strict_answer_extraction = True
        config.visualization.plot_dpi = 300
        return config

# ============================================================================
# 配置验证器
# ============================================================================
class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_config(config: HaltCoTConfig) -> List[str]:
        """验证配置，返回警告或错误信息"""
        warnings = []
        
        # 验证路径
        if not Path(config.data.data_path).exists():
            warnings.append(f"❌ 数据文件不存在: {config.data.data_path}")
        
        # 验证数值范围
        if config.data.sample_size <= 0:
            warnings.append("❌ sample_size 必须大于0")
        
        if config.generation.max_tokens <= 0:
            warnings.append("❌ max_tokens 必须大于0")
        
        if config.probe.cooldown < 1:
            warnings.append("❌ cooldown 必须大于等于1")
        
        # 验证模型兼容性
        if "Qwen" in config.model.name and config.generation.temperature > 0 and not config.generation.do_sample:
            warnings.append("⚠️ 使用temperature>0时建议设置do_sample=True")
        
        # 验证输出目录
        try:
            Path(config.output.results_dir).mkdir(parents=True, exist_ok=True)
            Path(config.output.plots_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warnings.append(f"❌ 无法创建输出目录: {e}")
        
        return warnings

# ============================================================================
# 配置管理器
# ============================================================================
class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def create_default_config_file():
        """创建默认配置文件"""
        config = ConfigTemplates.default_config()
        config.save_to_file()
        return config
    
    @staticmethod
    def create_all_template_configs():
        """创建所有模板配置文件"""
        templates = {
            'default': ConfigTemplates.default_config(),
            'quick_test': ConfigTemplates.quick_test_config(),
            'large_scale': ConfigTemplates.large_scale_config(),
            'high_precision': ConfigTemplates.high_precision_config()
        }
        
        for name, config in templates.items():
            filepath = CONFIG_DIR / f"halt_cot_config_{name}.json"
            config.save_to_file(filepath)
            print(f"📄 已创建配置模板: {name}")
        
        return templates
    
    @staticmethod
    def load_config(config_name: str = "default") -> HaltCoTConfig:
        """加载指定配置"""
        if config_name == "default":
            filepath = CONFIG_DIR / "halt_cot_config.json"
        else:
            filepath = CONFIG_DIR / f"halt_cot_config_{config_name}.json"
        
        if not filepath.exists():
            print(f"⚠️ 配置文件不存在: {filepath}")
            print("🔧 使用默认配置")
            return ConfigTemplates.default_config()
        
        try:
            config = HaltCoTConfig.load_from_file(filepath)
            print(f"✅ 已加载配置: {filepath}")
            
            # 验证配置
            warnings = ConfigValidator.validate_config(config)
            if warnings:
                print("⚠️ 配置验证警告:")
                for warning in warnings:
                    print(f"   {warning}")
            
            return config
            
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")
            print("🔧 使用默认配置")
            return ConfigTemplates.default_config()

# ============================================================================
# 兼容性适配器
# ============================================================================
class LegacyConfigAdapter:
    """用于适配原始代码的配置格式"""
    
    @staticmethod
    def to_legacy_format(config: HaltCoTConfig) -> Dict[str, Any]:
        """转换为原始代码期望的配置格式"""
        return {
            "model_name": config.model.name,
            "data_path": config.data.data_path,
            "sample_size": config.data.sample_size,
            "cooldown": config.probe.cooldown,
            "max_tokens": config.generation.max_tokens,
            "debug": config.experiment.debug
        }

# ============================================================================
# Main - 生成配置文件
# ============================================================================
if __name__ == "__main__":
    print("🔧 HALT-CoT 配置文件生成器")
    print("=" * 50)
    
    # 创建所有模板配置
    ConfigManager.create_all_template_configs()
    
    # 显示配置信息
    config = ConfigTemplates.default_config()
    print(f"\n📋 默认配置概览:")
    print(f"   模型: {config.model.name}")
    print(f"   数据: {config.data.sample_size} 个样本")
    print(f"   最大tokens: {config.generation.max_tokens}")
    print(f"   探测间隔: {config.probe.cooldown}")
    print(f"   结果目录: {config.output.results_dir}")
    
    print(f"\n📁 配置文件已保存到: {CONFIG_DIR}")
    print("💡 使用方法:")
    print("   from config import ConfigManager")
    print("   config = ConfigManager.load_config('default')")
    print("   legacy_config = LegacyConfigAdapter.to_legacy_format(config)")
