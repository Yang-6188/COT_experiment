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
    """检查点结果"""
    should_halt: bool = False
    halt_reason: Optional[str] = None
    answer: Optional[str] = None
    entropy: float = 0.0
    confidence: float = 0.0
