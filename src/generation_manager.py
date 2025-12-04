"""生成管理器"""
import re
from typing import Tuple


class GenerationManager:
    """改进的生成管理器"""
    
    def __init__(self, config):
        """
        初始化生成管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.stop_words = ["Human:", "User:", "\n\nHuman", "\n\nUser", "Observation:", "Question:"]
    
    def should_stop_naturally(self, text: str, new_token_id: int, tokenizer) -> Tuple[bool, str]:
        """
        改进的自然停止检测 - 更保守、更准确
        
        Args:
            text: 当前生成的文本
            new_token_id: 新生成的token ID
            tokenizer: 分词器
            
        Returns:
            (是否应该停止, 停止原因)
        """
        # 1. EOS token
        if new_token_id == tokenizer.eos_token_id:
            return True, "eos_token"

        # 2. 停止词检查
        for stop_word in self.stop_words:
            if stop_word in text:
                return True, f"stop_word_{stop_word}"

        # 3. 改进的 Boxed 答案检测
        if "\\boxed{" in text:
            last_boxed_pos = text.rfind("\\boxed{")
            content_after_boxed = text[last_boxed_pos + 7:]
            
            brace_count = 1
            closed_pos = -1
            
            for i, char in enumerate(content_after_boxed):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        closed_pos = i
                        break
            
            if closed_pos != -1:
                remaining = content_after_boxed[closed_pos + 1:].strip()
                remaining = remaining.lstrip('.,;:!?\n\r\t ')
                
                if len(remaining) == 0:
                    return True, "boxed_answer_complete"
                
                if len(remaining) < 30:
                    alnum_count = sum(1 for c in remaining if c.isalnum())
                    if alnum_count < 5:
                        return True, "boxed_answer_complete"

        # 4. 改进：只检测明确的最终答案标记
        final_answer_markers = [
            ("#### ", 20),
            ("The final answer is", 30),
            ("Therefore, the final answer is", 30),
            ("Thus, the final answer is", 30),
        ]
        
        for marker, min_length_after in final_answer_markers:
            if marker in text:
                marker_pos = text.rfind(marker)
                text_after_marker = text[marker_pos:]
                
                if len(text_after_marker) > len(marker) + min_length_after:
                    if '\\boxed{' in text_after_marker:
                        boxed_content = text_after_marker[text_after_marker.rfind('\\boxed{') + 7:]
                        if '}' in boxed_content:
                            return True, f"final_marker_with_boxed"
                    
                    elif re.search(r'\d+', text_after_marker):
                        if ('.' in text_after_marker and 
                            not any(word in text_after_marker.lower() for word in ['step', 'then', 'next'])):
                            return True, f"final_marker_with_number"

        # 5. 检测异常模式
        abnormal_patterns = [
            "Human:", "Assistant:", "You are an AI", "I am Claude",
        ]
        text_lower = text.lower()
        for pattern in abnormal_patterns:
            if pattern.lower() in text_lower:
                pattern_pos = text_lower.rfind(pattern.lower())
                if pattern_pos > len(text) * 0.5:
                    return True, f"abnormal_pattern"

        # 6. 检测重复内容
        if len(text) > 300:
            last_200 = text[-200:]
            prev_200 = text[-400:-200] if len(text) > 400 else ""
            if prev_200 and last_200 == prev_200:
                return True, "exact_repetition"

        # # 7. 检测过长生成
        # if len(text) > 2500:
        #     return True, "max_length_safety"

        return False, ""
