"""答案提取器模块"""
import re
from typing import Optional


class AnswerExtractor:
    """改进的答案提取器"""
    
    @staticmethod
    def extract_answer(text: str, strict: bool = False) -> Optional[str]:
        """改进的答案提取 - 使用更鲁棒的策略"""
        
        # 第一层：最高优先级格式
        high_confidence_patterns = [
            r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',  # GSM8K标准
            r'\\boxed\{([^}]+)\}',  # 完整的boxed
        ]
        
        for pattern in high_confidence_patterns:
            matches = list(re.finditer(pattern, text, re.I))
            if matches:
                answer = matches[-1].group(1).replace(',', '').strip()
                if answer:
                    # 处理 LaTeX 分数
                    frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', answer)
                    if frac_match:
                        return frac_match.group(1)
                    return answer
        
        # 改进：处理不完整的 boxed
        if '\\boxed{' in text and not strict:
            incomplete_answer = AnswerExtractor._extract_incomplete_boxed(text)
            if incomplete_answer:
                return incomplete_answer
        
        # 严格模式下只信任高优先级格式
        if strict:
            return None
        
        # 第二层：带有"final answer"的明确声明
        final_answer_patterns = [
            r'final answer is[:\s]+\$?(-?\d+(?:,\d+)*(?:\.\d+)?)',
            r'answer is[:\s]+\$?(-?\d+(?:,\d+)*(?:\.\d+)?)',
            r'Therefore,?\s+the answer is[:\s]+\$?(-?\d+(?:,\d+)*(?:\.\d+)?)',
        ]
        
        for pattern in final_answer_patterns:
            matches = list(re.finditer(pattern, text, re.I))
            if matches:
                answer = matches[-1].group(1).replace(',', '').strip()
                if answer:
                    return answer
        
        # 第三层：从等式中提取
        return AnswerExtractor._extract_from_equations(text)
    
    @staticmethod
    def _extract_incomplete_boxed(text: str) -> Optional[str]:
        """提取不完整的boxed答案"""
        last_boxed_pos = text.rfind('\\boxed{')
        content_after = text[last_boxed_pos + 7:]
        
        # 尝试提取到闭括号
        brace_count = 1
        answer_content = ""
        
        for i, char in enumerate(content_after):
            if char == '{':
                brace_count += 1
                answer_content += char
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    if answer_content.strip():
                        # 处理 LaTeX 分数
                        frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', answer_content)
                        if frac_match:
                            return frac_match.group(1)
                        
                        # 提取纯数字
                        num_match = re.search(r'-?\d+(?:\.\d+)?', answer_content)
                        if num_match:
                            return num_match.group(0)
                    break
                answer_content += char
            else:
                answer_content += char
        
        # 如果没找到闭括号，智能提取
        if brace_count > 0 and answer_content:
            for end_marker in ['\n', '\\text', 'Therefore', 'Thus']:
                if end_marker in answer_content:
                    answer_content = answer_content[:answer_content.index(end_marker)]
                    break
            
            answer_content = answer_content.strip()
            
            # 处理 LaTeX 分数
            frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', answer_content)
            if frac_match:
                return frac_match.group(1)
            
            # 提取纯数字
            if len(answer_content) < 50:
                num_match = re.search(r'-?\d+(?:\.\d+)?', answer_content)
                if num_match:
                    return num_match.group(0)
        
        return None
    
    @staticmethod
    def _extract_from_equations(text: str) -> Optional[str]:
        """从等式中智能提取答案"""
        lines = text.split('\n')
        keywords = ['total', 'answer', 'result', 'value', 'earnings', 'profit', 'money']
        
        # 倒序遍历，优先找最后的等式
        for line in reversed(lines):
            line_lower = line.lower()
            if any(k in line_lower for k in keywords) and '=' in line:
                rhs = line.split('=')[-1].strip()
                # 确保不包含运算符（不是中间计算）
                if not (re.search(r'[+*/]', rhs) or re.search(r'\s-\s', rhs)):
                    num_match = re.search(r'^\$?(-?\d+(?:,\d+)*(?:\.\d+)?)', rhs)
                    if num_match:
                        return num_match.group(1).replace(',', '')
        
        return None
