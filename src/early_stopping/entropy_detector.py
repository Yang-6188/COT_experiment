"""基于熵的早停检测器（基于Laaouach HALT-CoT）"""
from typing import Tuple


class EntropyHaltDetector:
    """基于熵的早停检测器"""
    
    def __init__(self, threshold: float = 0.6, consecutive_steps: int = 2):
        """
        初始化检测器
        
        Args:
            threshold: 熵阈值
            consecutive_steps: 需要连续低熵的步数
        """
        self.threshold = threshold
        self.consecutive_steps = consecutive_steps
        self.entropy_history = []
        self.low_entropy_count = 0
    
    def should_halt(self, entropy: float) -> Tuple[bool, float]:
        """
        判断是否应该停止
        
        Args:
            entropy: 当前熵值
            
        Returns:
            (是否应该停止, 当前熵值)
        """
        self.entropy_history.append(entropy)
        
        if entropy < self.threshold:
            self.low_entropy_count += 1
        else:
            self.low_entropy_count = 0
        
        should_stop = self.low_entropy_count >= self.consecutive_steps
        return should_stop, entropy
    
    def reset(self):
        """重置检测器状态"""
        self.entropy_history = []
        self.low_entropy_count = 0
