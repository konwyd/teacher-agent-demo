"""Deterministic, testable teaching-strategy capability boundary."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field


class StrategyId(str, Enum):
    """Stable strategy identifiers returned by generic math reasoning."""

    SYMBOLIC_TRANSFORMATION = "symbolic_transformation"
    QUANTITATIVE_MODELING = "quantitative_modeling"
    GEOMETRIC_REASONING = "geometric_reasoning"
    DATA_ANALYSIS = "data_analysis"
    UNKNOWN = "unknown"


class StrategyCandidate(BaseModel):
    """One reasoner-selected strategy with advisory confidence."""

    strategy_id: StrategyId = StrategyId.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class StrategySelection(BaseModel):
    """A primary strategy and optional supporting strategies."""

    primary: StrategyCandidate = Field(default_factory=StrategyCandidate)
    supporting: list[StrategyCandidate] = Field(default_factory=list)


class StrategyContext(BaseModel):
    """Verified and advisory context available to a capability."""

    problem_context: str
    student_message: str
    reasoning_next_step: str = ""
    verifier_name: str = ""
    verified_evidence: list[str] = Field(default_factory=list)
    unresolved_obstacles: list[str] = Field(default_factory=list)


class StrategyGuidance(BaseModel):
    """Deterministic guidance that does not choose a teaching action."""

    source_strategy: StrategyId
    representation: str = ""
    operation_types: list[str] = Field(default_factory=list)
    diagnostic_target: str = ""
    misconception_checks: list[str] = Field(default_factory=list)
    verifier_name: str = ""
    normalizers: list[str] = Field(default_factory=list)
    primary_next_step: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class StrategyBundle(BaseModel):
    """Isolated outputs from one primary and registered supporting plugins."""

    primary: StrategyGuidance = Field(
        default_factory=lambda: StrategyGuidance(source_strategy=StrategyId.UNKNOWN)
    )
    supporting: list[StrategyGuidance] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


class StrategyCapability(Protocol):
    """Interface for deterministic strategy modules."""

    strategy_id: StrategyId

    def advise(self, context: StrategyContext) -> StrategyGuidance:
        """Return deterministic guidance without choosing an action."""


def normalize_strategy_selection(selection: StrategySelection) -> StrategySelection:
    """Apply the confidence, uniqueness, and support-count contract."""

    primary = selection.primary
    if primary.confidence < 0.60:
        primary = StrategyCandidate(
            strategy_id=StrategyId.UNKNOWN,
            confidence=primary.confidence,
            rationale="primary_below_confidence_threshold",
        )

    supporting: list[StrategyCandidate] = []
    seen = {primary.strategy_id}
    for candidate in selection.supporting:
        if candidate.confidence < 0.60 or candidate.strategy_id in seen:
            continue
        seen.add(candidate.strategy_id)
        supporting.append(candidate)
        if len(supporting) == 2:
            break
    return StrategySelection(primary=primary, supporting=supporting)


class UnknownCapability:
    """Conservative fallback when no selected capability can advise."""

    strategy_id = StrategyId.UNKNOWN

    def advise(self, context: StrategyContext) -> StrategyGuidance:
        return StrategyGuidance(
            source_strategy=StrategyId.UNKNOWN,
            primary_next_step="",
            confidence=0.0,
        )


class SymbolicTransformationCapability:
    """Deterministic guidance for symbolic equivalence transformations."""

    strategy_id = StrategyId.SYMBOLIC_TRANSFORMATION

    def advise(self, context: StrategyContext) -> StrategyGuidance:
        return StrategyGuidance(
            source_strategy=StrategyId.SYMBOLIC_TRANSFORMATION,
            representation="使用题目已有的等式或代数关系表示当前状态。",
            operation_types=[
                "比较关系",
                "选择等价变换",
                "检查变换前后是否保持关系",
            ],
            diagnostic_target="学生能说明下一步变换的对象和依据。",
            misconception_checks=[
                "是否只改变了等式一侧",
                "是否把候选结论当成已验证事实",
            ],
            verifier_name=context.verifier_name,
            normalizers=["保留原题符号", "一次只提出一个主要下一步"],
            primary_next_step=context.reasoning_next_step,
            confidence=0.8,
        )


class StrategyRegistry:
    """Invoke registered deterministic capabilities with failure isolation."""

    def __init__(self, capabilities: Iterable[StrategyCapability]) -> None:
        self._capabilities = {
            capability.strategy_id: capability for capability in capabilities
        }

    @classmethod
    def default(cls) -> StrategyRegistry:
        """Return the initial registry with only unknown and symbolic support."""

        return cls([UnknownCapability(), SymbolicTransformationCapability()])

    def advise(
        self,
        selection: StrategySelection,
        context: StrategyContext,
    ) -> StrategyBundle:
        """Invoke the normalized selection without making any LLM call."""

        normalized = normalize_strategy_selection(selection)
        failures: list[str] = []
        primary_capability = self._capabilities.get(normalized.primary.strategy_id)
        if primary_capability is None:
            primary = self._unknown_capability().advise(context)
        else:
            try:
                primary = primary_capability.advise(context)
            except Exception as exc:
                failures.append(
                    f"{normalized.primary.strategy_id.value}:{type(exc).__name__}"
                )
                primary = self._unknown_capability().advise(context)

        supporting: list[StrategyGuidance] = []
        for candidate in normalized.supporting:
            capability = self._capabilities.get(candidate.strategy_id)
            if capability is None:
                continue
            try:
                supporting.append(capability.advise(context))
            except Exception as exc:
                failures.append(f"{candidate.strategy_id.value}:{type(exc).__name__}")

        return StrategyBundle(
            primary=primary,
            supporting=supporting,
            failure_reasons=failures,
        )

    def _unknown_capability(self) -> StrategyCapability:
        return self._capabilities.get(StrategyId.UNKNOWN, UnknownCapability())
