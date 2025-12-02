"""智能早停决策器"""
from typing import Dict, Any
from collections import Counter

from ..data_structures import CheckpointResult
from .consistency_detector import AnswerConsistencyDetector
from .entropy_detector import EntropyHaltDetector


class SmartHaltDecisionMaker:
    """智能早停决策器 - 结合推理阶段"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化决策器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.use_consistency = config.get('early_stopping', {}).get('use_answer_consistency', True)
        self.use_entropy = config.get('early_stopping', {}).get('use_entropy_halt', True)
        
        # 初始化检测器
        self.consistency_detector = AnswerConsistencyDetector(
            k=config.get('early_stopping', {}).get('consistency_k', 3)
        )
        self.entropy_detector = EntropyHaltDetector(
            threshold=config.get('early_stopping', {}).get('entropy_threshold', 0.6),
            consecutive_steps=config.get('early_stopping', {}).get('entropy_consecutive_steps', 2)
        )
        
        self.last_check_token_count = 0
        self.last_stage = None
        self.stage_check_counts = Counter()
    
    def should_check_now(
        self, 
        full_text: str, 
        tokens_used: int, 
        stage: str,
        cooldown: int = 40
    ) -> bool:
        """
        智能检查判断 - 考虑推理阶段
        
        Args:
            full_text: 当前生成的完整文本
            tokens_used: 已使用的token数
            stage: 当前推理阶段
            cooldown: 冷却token数
            
        Returns:
            是否应该进行检查
        """
        # 如果早停功能完全禁用，不需要检查
        if not self.use_consistency and not self.use_entropy:
            return False
        
        # 如果在答案信号阶段,立即检查
        if stage == 'answer_signal':
            return True
        
        # 如果在结论阶段,且距离上次检查超过20个token
        if stage == 'conclusion' and (tokens_used - self.last_check_token_count) >= 20:
            return True
        
        # 如果在计算阶段,使用正常冷却
        if stage == 'calculation':
            if (tokens_used - self.last_check_token_count) >= cooldown:
                # 限制同一阶段的检查次数
                if self.stage_check_counts.get(stage, 0) < 3:
                    return True
        
        # 中间步骤阶段,更长的冷却时间
        if stage == 'intermediate':
            return (tokens_used - self.last_check_token_count) >= cooldown * 2
        
        return False
    
    def update_check_state(self, tokens_used: int, stage: str):
        """
        更新检查状态
        
        Args:
            tokens_used: 已使用的token数
            stage: 当前推理阶段
        """
        self.last_check_token_count = tokens_used
        if stage != self.last_stage:
            self.stage_check_counts[stage] = 0
            self.last_stage = stage
        self.stage_check_counts[stage] += 1
    
    def make_decision(
        self, 
        probe_result: CheckpointResult, 
        stage: str
    ) -> CheckpointResult:
        """
        智能决策 - 考虑推理阶段和配置
        
        Args:
            probe_result: 探针结果
            stage: 当前推理阶段
            
        Returns:
            决策结果
        """
        # 如果早停功能完全禁用,直接返回不停止
        if not self.use_consistency and not self.use_entropy:
            return CheckpointResult(
                should_halt=False,
                halt_reason=None,
                answer=probe_result.answer,
                entropy=probe_result.entropy,
                confidence=probe_result.confidence
            )
        
        if not probe_result.answer:
            return probe_result
        
        # 答案信号阶段的决策更激进
        if stage == 'answer_signal':
            if self.use_entropy and probe_result.entropy < 0.5:
                return CheckpointResult(
                    should_halt=True,
                    halt_reason=f"answer_signal_high_confidence",
                    answer=probe_result.answer,
                    entropy=probe_result.entropy,
                    confidence=probe_result.confidence
                )
        
        # 结论阶段 - 中等激进
        if stage == 'conclusion':
            # 一致性检测
            if self.use_consistency:
                is_consistent = self.consistency_detector.add_answer(probe_result.answer)
                if is_consistent and (not self.use_entropy or probe_result.entropy < 0.8):
                    return CheckpointResult(
                        should_halt=True,
                        halt_reason=f"conclusion_consistency",
                        answer=probe_result.answer,
                        entropy=probe_result.entropy,
                        confidence=probe_result.confidence
                    )
            
            # 极低熵也可以在结论阶段停止
            if self.use_entropy and probe_result.entropy < 0.3:
                return CheckpointResult(
                    should_halt=True,
                    halt_reason=f"conclusion_high_confidence",
                    answer=probe_result.answer,
                    entropy=probe_result.entropy,
                    confidence=probe_result.confidence
                )
        
        # 计算阶段 - 保守策略
        if stage == 'calculation':
            if self.use_entropy and probe_result.entropy < 0.15:
                if self.use_consistency:
                    is_consistent = self.consistency_detector.add_answer(probe_result.answer)
                    if is_consistent:
                        return CheckpointResult(
                            should_halt=True,
                            halt_reason=f"calculation_high_confidence_consistent",
                            answer=probe_result.answer,
                            entropy=probe_result.entropy,
                            confidence=probe_result.confidence
                        )
                else:
                    return CheckpointResult(
                        should_halt=True,
                        halt_reason=f"calculation_high_confidence",
                        answer=probe_result.answer,
                        entropy=probe_result.entropy,
                        confidence=probe_result.confidence
                    )
        
        # 记录答案但不停止
        if self.use_consistency:
            self.consistency_detector.add_answer(probe_result.answer)
        
        return CheckpointResult(
            should_halt=False,
            halt_reason=None,
            answer=probe_result.answer,
            entropy=probe_result.entropy,
            confidence=probe_result.confidence
        )
    
    def reset(self):
        """重置决策器状态"""
        self.consistency_detector.reset()
        self.entropy_detector.reset()
        self.last_check_token_count = 0
        self.last_stage = None
        self.stage_check_counts.clear()
