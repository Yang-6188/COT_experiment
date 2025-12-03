"""熵值早停检测器"""
from typing import List, Optional


class EntropyHaltDetector:
    """基于熵值的早停检测器"""
    
    def __init__(self, threshold: float = 0.6, consecutive_steps: int = 2):
        """
        初始化熵值检测器
        
        Args:
            threshold: 熵值阈值（低于此值认为是高置信度）
            consecutive_steps: 需要连续多少步低熵才触发停止
        """
        self.threshold = threshold
        self.consecutive_steps = consecutive_steps
        self.entropy_history: List[float] = []
        self.low_entropy_count = 0
    
    def add_entropy(self, entropy: float) -> bool:
        """
        添加新的熵值并检查是否应该停止
        
        Args:
            entropy: 当前步骤的熵值
            
        Returns:
            是否应该停止生成
        """
        self.entropy_history.append(entropy)
        
        # 检查是否低于阈值
        if entropy < self.threshold:
            self.low_entropy_count += 1
        else:
            self.low_entropy_count = 0
        
        # 判断是否达到连续步数要求
        return self.low_entropy_count >= self.consecutive_steps
    
    def check_entropy(self, entropy: float) -> bool:
        """
        检查单个熵值是否满足停止条件（不更新历史）
        
        Args:
            entropy: 熵值
            
        Returns:
            是否满足停止条件
        """
        return entropy < self.threshold
    
    def get_average_entropy(self, last_n: Optional[int] = None) -> float:
        """
        获取平均熵值
        
        Args:
            last_n: 最近n步，如果为None则计算全部
            
        Returns:
            平均熵值
        """
        if not self.entropy_history:
            return 0.0
        
        if last_n is None:
            return sum(self.entropy_history) / len(self.entropy_history)
        else:
            recent = self.entropy_history[-last_n:]
            return sum(recent) / len(recent) if recent else 0.0
    
    def get_entropy_trend(self, window_size: int = 5) -> str:
        """
        获取熵值趋势
        
        Args:
            window_size: 窗口大小
            
        Returns:
            趋势描述：'increasing', 'decreasing', 'stable'
        """
        if len(self.entropy_history) < window_size:
            return 'insufficient_data'
        
        recent = self.entropy_history[-window_size:]
        first_half_avg = sum(recent[:window_size//2]) / (window_size//2)
        second_half_avg = sum(recent[window_size//2:]) / (window_size - window_size//2)
        
        diff = second_half_avg - first_half_avg
        
        if abs(diff) < 0.1:
            return 'stable'
        elif diff > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def reset(self):
        """重置检测器状态"""
        self.entropy_history.clear()
        self.low_entropy_count = 0
    
    def __repr__(self) -> str:
        """返回检测器的字符串表示"""
        return f"EntropyHaltDetector(threshold={self.threshold}, consecutive_steps={self.consecutive_steps})"
