"""数据结构定义"""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class GenerationState:
    """生成状态管理"""
    tokens_used: int = 0
    full_response: str = ""
    full_sequence_ids: Optional[torch.Tensor] = None
    early_stopped: bool = False
    halt_reason: Optional[str] = None
    predicted_answer: Optional[str] = None


@dataclass
class CheckpointResult:
    """检查点结果数据类"""
    should_halt: bool
    answer: Optional[str]
    entropy: float
    halt_reason: Optional[str] = None
    confidence: Optional[float] = None
    
    
    def __repr__(self) -> str:
        """返回结果的字符串表示"""
        return (f"CheckpointResult(should_halt={self.should_halt}, "
                f"halt_reason={self.halt_reason}, answer={self.answer}, "
                f"entropy={self.entropy:.4f}, confidence={self.confidence})")
