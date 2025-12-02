"""答案一致性检测器（基于Liu & Wang 2025）"""
from typing import Optional


class AnswerConsistencyDetector:
    """答案一致性检测器"""
    
    def __init__(self, k: int = 3):
        """
        初始化检测器
        
        Args:
            k: 需要连续一致的答案数量
        """
        self.k = k
        self.answer_history = []
    
    def add_answer(self, answer: Optional[str]) -> bool:
        """
        添加答案并检查收敛性
        
        Args:
            answer: 新的答案
            
        Returns:
            是否已收敛（连续k个答案相同）
        """
        self.answer_history.append(answer)
        
        if len(self.answer_history) < self.k or answer is None:
            return False
        
        recent_answers = self.answer_history[-self.k:]
        return len(set(recent_answers)) == 1
    
    def reset(self):
        """重置检测器状态"""
        self.answer_history = []
