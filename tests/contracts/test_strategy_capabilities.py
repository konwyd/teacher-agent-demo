"""Permanent contracts for deterministic teaching strategy capabilities."""

import pytest

from app.agent.strategy_capabilities import (
    StrategyCandidate,
    StrategyContext,
    StrategyGuidance,
    StrategyId,
    StrategyRegistry,
    StrategySelection,
    normalize_strategy_selection,
)


pytestmark = pytest.mark.safety_contract


def test_low_confidence_primary_becomes_unknown() -> None:
    normalized = normalize_strategy_selection(
        StrategySelection(
            primary=StrategyCandidate(
                strategy_id=StrategyId.SYMBOLIC_TRANSFORMATION,
                confidence=0.59,
            )
        )
    )

    assert normalized.primary.strategy_id is StrategyId.UNKNOWN


def test_supporting_is_unique_and_capped_at_two() -> None:
    selection = StrategySelection(
        primary=StrategyCandidate(
            strategy_id=StrategyId.SYMBOLIC_TRANSFORMATION,
            confidence=0.9,
        ),
        supporting=[
            StrategyCandidate(
                strategy_id=StrategyId.QUANTITATIVE_MODELING,
                confidence=0.9,
            ),
            StrategyCandidate(
                strategy_id=StrategyId.QUANTITATIVE_MODELING,
                confidence=0.8,
            ),
            StrategyCandidate(
                strategy_id=StrategyId.GEOMETRIC_REASONING,
                confidence=0.7,
            ),
            StrategyCandidate(
                strategy_id=StrategyId.DATA_ANALYSIS,
                confidence=0.7,
            ),
        ],
    )

    normalized = normalize_strategy_selection(selection)

    assert [item.strategy_id for item in normalized.supporting] == [
        StrategyId.QUANTITATIVE_MODELING,
        StrategyId.GEOMETRIC_REASONING,
    ]


def test_registry_returns_one_primary_and_ignores_missing_supporting_plugins() -> None:
    bundle = StrategyRegistry.default().advise(
        StrategySelection(
            primary=StrategyCandidate(
                strategy_id=StrategyId.SYMBOLIC_TRANSFORMATION,
                confidence=0.9,
            )
        ),
        StrategyContext(
            problem_context="2x+3=7",
            student_message="我不知道第一步",
        ),
    )

    assert bundle.primary.source_strategy is StrategyId.SYMBOLIC_TRANSFORMATION
    assert bundle.supporting == []


def test_symbolic_capability_returns_deterministic_unsolved_guidance() -> None:
    bundle = StrategyRegistry.default().advise(
        StrategySelection(
            primary=StrategyCandidate(
                strategy_id=StrategyId.SYMBOLIC_TRANSFORMATION,
                confidence=0.6,
            )
        ),
        StrategyContext(
            problem_context="2x+3=7",
            student_message="我不知道第一步",
            reasoning_next_step="比较等式两边并选择等价变换。",
            verifier_name="fraction_checker",
        ),
    )

    assert bundle.primary.source_strategy is StrategyId.SYMBOLIC_TRANSFORMATION
    assert bundle.primary.primary_next_step == "比较等式两边并选择等价变换。"
    assert bundle.primary.verifier_name == "fraction_checker"
    assert bundle.primary.confidence == 0.8


def test_default_registry_falls_back_for_unregistered_primary() -> None:
    bundle = StrategyRegistry.default().advise(
        StrategySelection(
            primary=StrategyCandidate(
                strategy_id=StrategyId.QUANTITATIVE_MODELING,
                confidence=0.9,
            )
        ),
        StrategyContext(problem_context="一共有30人。", student_message="怎么建模？"),
    )

    assert bundle.primary.source_strategy is StrategyId.UNKNOWN
    assert bundle.primary.primary_next_step == ""
    assert bundle.primary.confidence == 0.0


class _RecordingCapability:
    def __init__(self, strategy_id: StrategyId, calls: list[StrategyId]) -> None:
        self.strategy_id = strategy_id
        self._calls = calls

    def advise(self, context: StrategyContext) -> StrategyGuidance:
        self._calls.append(self.strategy_id)
        return StrategyGuidance(
            source_strategy=self.strategy_id,
            primary_next_step=context.reasoning_next_step,
            confidence=0.7,
        )


class _FailingCapability:
    def __init__(self, strategy_id: StrategyId) -> None:
        self.strategy_id = strategy_id

    def advise(self, context: StrategyContext) -> StrategyGuidance:
        raise RuntimeError("simulated capability failure")


def test_registry_invokes_primary_once_and_only_registered_supporting() -> None:
    calls: list[StrategyId] = []
    registry = StrategyRegistry(
        [
            _RecordingCapability(StrategyId.UNKNOWN, calls),
            _RecordingCapability(StrategyId.SYMBOLIC_TRANSFORMATION, calls),
            _RecordingCapability(StrategyId.DATA_ANALYSIS, calls),
        ]
    )

    bundle = registry.advise(
        StrategySelection(
            primary=StrategyCandidate(
                strategy_id=StrategyId.SYMBOLIC_TRANSFORMATION,
                confidence=0.9,
            ),
            supporting=[
                StrategyCandidate(
                    strategy_id=StrategyId.QUANTITATIVE_MODELING,
                    confidence=0.8,
                ),
                StrategyCandidate(
                    strategy_id=StrategyId.DATA_ANALYSIS,
                    confidence=0.8,
                ),
            ],
        ),
        StrategyContext(problem_context="表格", student_message="下一步呢？"),
    )

    assert calls == [
        StrategyId.SYMBOLIC_TRANSFORMATION,
        StrategyId.DATA_ANALYSIS,
    ]
    assert [item.source_strategy for item in bundle.supporting] == [
        StrategyId.DATA_ANALYSIS
    ]


def test_supporting_failure_is_isolated_and_recorded() -> None:
    calls: list[StrategyId] = []
    registry = StrategyRegistry(
        [
            _RecordingCapability(StrategyId.UNKNOWN, calls),
            _RecordingCapability(StrategyId.SYMBOLIC_TRANSFORMATION, calls),
            _FailingCapability(StrategyId.DATA_ANALYSIS),
        ]
    )

    bundle = registry.advise(
        StrategySelection(
            primary=StrategyCandidate(
                strategy_id=StrategyId.SYMBOLIC_TRANSFORMATION,
                confidence=0.9,
            ),
            supporting=[
                StrategyCandidate(
                    strategy_id=StrategyId.DATA_ANALYSIS,
                    confidence=0.8,
                )
            ],
        ),
        StrategyContext(problem_context="表格", student_message="下一步呢？"),
    )

    assert bundle.primary.source_strategy is StrategyId.SYMBOLIC_TRANSFORMATION
    assert bundle.supporting == []
    assert bundle.failure_reasons == ["data_analysis:RuntimeError"]


@pytest.mark.parametrize(
    "primary_capability",
    [
        None,
        _FailingCapability(StrategyId.DATA_ANALYSIS),
    ],
)
def test_missing_or_failed_primary_falls_back_to_unknown(primary_capability) -> None:
    calls: list[StrategyId] = []
    capabilities = [_RecordingCapability(StrategyId.UNKNOWN, calls)]
    if primary_capability is not None:
        capabilities.append(primary_capability)
    registry = StrategyRegistry(capabilities)

    bundle = registry.advise(
        StrategySelection(
            primary=StrategyCandidate(
                strategy_id=StrategyId.DATA_ANALYSIS,
                confidence=0.8,
            )
        ),
        StrategyContext(problem_context="表格", student_message="下一步呢？"),
    )

    assert bundle.primary.source_strategy is StrategyId.UNKNOWN
    assert calls == [StrategyId.UNKNOWN]
    expected_failures = (
        [] if primary_capability is None else ["data_analysis:RuntimeError"]
    )
    assert bundle.failure_reasons == expected_failures
