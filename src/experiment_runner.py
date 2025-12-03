"""实验运行器 - 精简版"""
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import ConfigManager
from .experiment_manager import ExperimentManager
from .statistics_calculator import StatisticsCalculator
from .result_saver import ResultSaver


class ExperimentRunner:
    """
    精简的实验运行器
    只负责协调各个组件，不处理具体逻辑 [[0]](#__0)
    """
    
    def __init__(self, base_dir: Path = None):
        """
        初始化实验运行器
        
        Args:
            base_dir: 基础目录路径
        """
        if base_dir is None:
            base_dir = Path("/root/autodl-tmp")
        
        self.base_dir = base_dir
        self.config_dir = base_dir / "config"
        self.data_dir = base_dir / "data"
        self.results_dir = base_dir / "results"
        
        self.results_dir.mkdir(exist_ok=True)
        
        # 加载配置
        config_file = self.config_dir / "config.json"
        self.config = ConfigManager.load_config(config_file)
        
        # 初始化辅助类
        self.statistics_calculator = StatisticsCalculator()
        self.result_saver = ResultSaver(self.results_dir, self.config)
        
        print(f"🚀 实验运行器初始化完成")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        test_file = Path(self.config['paths']['test_data'])
        if not test_file.is_absolute():
            test_file = self.base_dir / test_file
        
        if not test_file.exists():
            raise FileNotFoundError(f"测试数据不存在: {test_file}")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sample_size = self.config['experiment']['sample_size']
        if len(data) > sample_size:
            data = data[:sample_size]
        
        print(f"✅ 已加载 {len(data)} 条测试数据")
        return data
    
    def load_model(self):
        """加载模型和分词器"""
        model_key = self.config['active_model']
        model_name = self.config['model_configs'][model_key]['name']
        
        print(f"🤖 加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        print(f"✅ 模型已加载到: {model.device}")
        return tokenizer, model
    
    def get_ground_truth(self, item: Dict[str, Any]) -> Optional[str]:
        """获取标准答案"""
        from .answer_extractor import AnswerExtractor
        
        if 'numerical_answer' in item and item['numerical_answer']:
            return str(item['numerical_answer'])
        
        if 'answer' in item:
            return AnswerExtractor.extract_answer(item['answer'])
        
        return None
    
    def run_experiment(self):
        """运行完整实验"""
        print("🧪 开始HALT-CoT增强实验")
        print("=" * 60)
        
        try:
            # 加载数据和模型
            test_data = self.load_test_data()
            tokenizer, model = self.load_model()
            
            # 创建实验管理器
            experiment_manager = ExperimentManager(self.config, model, tokenizer)
            
            # 运行实验
            results = []
            experiment_start = time.time()
            
            for idx, item in enumerate(test_data):
                question = item['question']
                ground_truth = self.get_ground_truth(item)
                
                if ground_truth is None:
                    print(f"⚠️  样本 {idx + 1} 没有有效答案，跳过")
                    continue
                
                result = experiment_manager.run_single_sample(question, ground_truth, idx)
                results.append(result)
                
                if (idx + 1) % 5 == 0:
                    self._print_progress_report(results, idx + 1, len(test_data))
            
            total_time = time.time() - experiment_start
            
            # 计算统计
            stats = self.statistics_calculator.calculate(results)
            stats['total_experiment_time'] = total_time
            
            self.statistics_calculator.print_statistics(stats, self.config)
            
            # 保存结果
            results_file = self.result_saver.save(results, stats)
            
            print(f"\n🎉 实验完成!")
            if results_file:
                print(f"📁 结果文件: {results_file}")
            
            return results, stats
            
        except Exception as e:
            print(f"❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            return [], {}
    
    def _print_progress_report(self, results: List[Dict], current: int, total: int):
        """打印进度报告"""
        current_accuracy = sum(1 for r in results if r['correct']) / len(results)
        early_stop_rate = sum(1 for r in results if r.get('early_stopped', False)) / len(results)
        avg_tokens = sum(r['tokens_used'] for r in results) / len(results)
        
        print(f"📊 进度: {current}/{total}, "
              f"准确率: {current_accuracy:.2%}, "
              f"早停率: {early_stop_rate:.2%}, "
              f"平均tokens: {avg_tokens:.1f}")
