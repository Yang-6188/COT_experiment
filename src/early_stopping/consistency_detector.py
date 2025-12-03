"""答案一致性检测器"""
from typing import List, Optional
from collections import Counter


class AnswerConsistencyDetector:
    """答案一致性检测器 - 检测连续k次答案是否一致"""
    
    def __init__(self, k: int = 3):
        """
        初始化一致性检测器
        
        Args:
            k: 需要连续一致的答案数量
        """
        self.k = k
        self.answer_history: List[str] = []
        
    def add_answer(self, answer: str) -> bool:
        """
        添加新答案并检查是否达到一致性
        
        Args:
            answer: 新的答案
            
        Returns:
            是否达到k次一致
        """
        if not answer:
            return False
        
        # 标准化答案（去除空格、转小写）
        normalized_answer = self._normalize_answer(answer)
        
        # 添加到历史记录
        self.answer_history.append(normalized_answer)
        
        # 只保留最近的k个答案
        if len(self.answer_history) > self.k:
            self.answer_history = self.answer_history[-self.k:]
        
        # 检查是否有k个答案
        if len(self.answer_history) < self.k:
            return False
        
        # 检查最近k个答案是否完全一致
        return len(set(self.answer_history[-self.k:])) == 1
    
    def _normalize_answer(self, answer: str) -> str:
        """
        标准化答案格式
        
        Args:
            answer: 原始答案
            
        Returns:
            标准化后的答案
        """
        # 去除首尾空格
        answer = answer.strip()
        
        # 转换为小写
        answer = answer.lower()
        
        # 去除常见的答案前缀
        prefixes = ['the answer is', 'answer:', 'final answer:', '答案是', '答案：']
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # 去除特殊字符（保留数字、字母、基本符号）
        answer = ''.join(c for c in answer if c.isalnum() or c in ['.', '-', '/', ' '])
        
        return answer.strip()
    
    def get_most_common_answer(self) -> Optional[str]:
        """
        获取历史记录中最常见的答案
        
        Returns:
            最常见的答案，如果没有历史记录则返回None
        """
        if not self.answer_history:
            return None
        
        counter = Counter(self.answer_history)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else None
    
    def get_consistency_rate(self) -> float:
        """
        计算答案一致性比率
        
        Returns:
            一致性比率（0-1之间）
        """
        if not self.answer_history:
            return 0.0
        
        if len(self.answer_history) == 1:
            return 1.0
        
        most_common_count = Counter(self.answer_history).most_common(1)[0][1]
        return most_common_count / len(self.answer_history)
    
    def reset(self):
        """重置检测器状态"""
        self.answer_history.clear()
    
    def __repr__(self) -> str:
        """返回检测器的字符串表示"""
        return f"AnswerConsistencyDetector(k={self.k}, history_size={len(self.answer_history)})"
