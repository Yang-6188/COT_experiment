"""句子边界检测器"""
import re
from typing import List, Tuple


class SentenceDetector:
    """
    轻量级句子边界检测器
    用于在生成过程中检测句子结束点
    """
    
    def __init__(self):
        """初始化句子检测器"""
        # 句子结束标记（基于规则）[[3]](#__3)
        self.sentence_endings = ['.', '!', '?', '。', '！', '？']
        
        # 常见缩写词，避免误判 [[6]](#__6)
        self.abbreviations = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr',
            'etc', 'vs', 'i.e', 'e.g', 'approx', 'est'
        }
        
        # 句子边界正则表达式
        self.sentence_pattern = re.compile(
            r'([.!?。！？]+[\s\n]+|[.!?。！？]+$)',
            re.MULTILINE
        )
    
    def is_sentence_boundary(self, text: str, position: int) -> bool:
        """
        检查指定位置是否为句子边界
        
        Args:
            text: 完整文本
            position: 检查位置
            
        Returns:
            是否为句子边界
        """
        if position <= 0 or position >= len(text):
            return False
        
        char = text[position - 1]
        
        # 检查是否为句子结束标记
        if char not in self.sentence_endings:
            return False
        
        # 检查是否为缩写词（向前查找）
        if self._is_abbreviation(text, position):
            return False
        
        # 检查后面是否有空格或换行（真正的句子结束）[[7]](#__7)
        if position < len(text):
            next_char = text[position]
            if next_char in [' ', '\n', '\t'] or position == len(text) - 1:
                return True
        
        return False
    
    def _is_abbreviation(self, text: str, position: int) -> bool:
        """检查是否为缩写词"""
        # 向前查找最多20个字符
        start = max(0, position - 20)
        preceding_text = text[start:position].lower()
        
        # 检查是否匹配已知缩写
        for abbr in self.abbreviations:
            if preceding_text.endswith(abbr):
                return True
        
        return False
    
    def count_sentences(self, text: str) -> int:
        """
        统计文本中的句子数量
        
        Args:
            text: 输入文本
            
        Returns:
            句子数量
        """
        sentences = self.sentence_pattern.split(text)
        return len([s for s in sentences if s.strip()])
    
    def get_last_sentence_end_position(self, text: str) -> int:
        """
        获取最后一个完整句子的结束位置
        
        Args:
            text: 输入文本
            
        Returns:
            位置索引，-1表示未找到
        """
        for i in range(len(text) - 1, -1, -1):
            if self.is_sentence_boundary(text, i + 1):
                return i + 1
        return -1
    
    def has_new_sentence_since(self, text: str, last_position: int) -> Tuple[bool, int]:
        """
        检查自上次位置以来是否有新句子
        
        Args:
            text: 完整文本
            last_position: 上次检查的位置
            
        Returns:
            (是否有新句子, 新句子结束位置)
        """
        current_pos = self.get_last_sentence_end_position(text)
        
        if current_pos > last_position:
            return True, current_pos
        
        return False, last_position


class SentenceBasedCheckpointManager:
    """
    基于句子的检查点管理器
    结合句子边界和冷却期进行智能检测 [[0]](#__0)
    """
    
    def __init__(self, min_tokens: int = 100, cooldown_tokens: int = 40):
        """
        初始化检查点管理器
        
        Args:
            min_tokens: 最小token数（开始检测前）
            cooldown_tokens: 冷却期token数
        """
        self.min_tokens = min_tokens
        self.cooldown_tokens = cooldown_tokens
        
        self.sentence_detector = SentenceDetector()
        
        # 状态跟踪
        self.last_check_token = 0
        self.last_sentence_position = 0
        self.check_count = 0
    
    def should_check_now(
        self, 
        current_text: str, 
        tokens_used: int
    ) -> Tuple[bool, str]:
        """
        判断是否应该进行检查
        
        策略：在min_tokens之后，每遇到新句子结束且过了cooldown期就检查
        
        Args:
            current_text: 当前生成的文本
            tokens_used: 已使用的token数
            
        Returns:
            (是否应该检查, 原因说明)
        """
        # 1. 检查是否达到最小token数
        if tokens_used < self.min_tokens:
            return False, "未达到最小token数"
        
        # 2. 检查冷却期
        tokens_since_last_check = tokens_used - self.last_check_token
        if tokens_since_last_check < self.cooldown_tokens:
            return False, f"冷却期中 ({tokens_since_last_check}/{self.cooldown_tokens})"
        
        # 3. 检查是否有新句子完成
        has_new_sentence, new_position = self.sentence_detector.has_new_sentence_since(
            current_text, 
            self.last_sentence_position
        )
        
        if has_new_sentence:
            self.last_check_token = tokens_used
            self.last_sentence_position = new_position
            self.check_count += 1
            return True, f"句子边界检测 (第{self.check_count}次)"
        
        return False, "未检测到新句子边界"
    
    def reset(self):
        """重置状态"""
        self.last_check_token = 0
        self.last_sentence_position = 0
        self.check_count = 0
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "total_checks": self.check_count,
            "last_check_token": self.last_check_token,
            "last_sentence_position": self.last_sentence_position
        }
