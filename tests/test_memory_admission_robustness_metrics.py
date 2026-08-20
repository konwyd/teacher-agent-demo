from __future__ import annotations

from app.evaluation.memory_admission.robustness.metrics import (
    compute_robustness_metrics,
)
from app.evaluation.memory_admission.robustness.models import (
    CandidateSpec,
    FailureAttribution,
    RobustnessCase,
    RobustnessRecord,
)
from app.memory.contracts import (
    AdmissionAction,
    MemoryRelation,
    MemorySourceKind,
    MemoryType,
    TemporalScope,
)


def _case(case_id: str, expected: AdmissionAction, *labels: str, group: str = ""):
    return RobustnessCase(
        case_id=case_id,
        labels=labels or ("paraphrase",),
        rationale="literal metric fixture",
        candidate=CandidateSpec(
            memory_type=MemoryType.PREFERENCE,
            source_kind=MemorySourceKind.USER_STATED,
            temporal_scope=TemporalScope.PERSISTENT,
            payload={"teaching_preferences": ["图示讲解"]},
        ),
        expected_action=expected,
        expected_relation=MemoryRelation.NEW,
        metamorphic_group=group,
    )


def _record(
    case: RobustnessCase,
    predicted: AdmissionAction,
    *,
    relation: MemoryRelation | None = MemoryRelation.NEW,
    failure: FailureAttribution | None = None,
) -> RobustnessRecord:
    return RobustnessRecord(
        case_id=case.case_id,
        labels=case.labels,
        expected_action=case.expected_action,
        predicted_action=predicted,
        expected_relation=case.expected_relation,
        predicted_relation=relation,
        semantic_key="student:preference:teaching_style",
        evidence_maturity=0.0,
        support_count=1,
        distinct_turn_count=1,
        distinct_conversation_count=1,
        failure_attribution=failure,
    )


def test_classification_and_admit_metrics_use_literal_confusion_matrix() -> None:
    cases = [
        _case("m1", AdmissionAction.ADMIT),
        _case("m2", AdmissionAction.ADMIT),
        _case("m3", AdmissionAction.PENDING),
        _case("m4", AdmissionAction.REJECT),
    ]
    records = [
        _record(cases[0], AdmissionAction.ADMIT),
        _record(cases[1], AdmissionAction.PENDING, failure=FailureAttribution.POLICY_FAILURE),
        _record(cases[2], AdmissionAction.ADMIT, failure=FailureAttribution.POLICY_FAILURE),
        _record(cases[3], AdmissionAction.REJECT),
    ]

    metrics = compute_robustness_metrics(cases, records)

    assert metrics["robustness_accuracy"].value == 0.5
    assert metrics["admit_precision"].value == 0.5
    assert metrics["admit_recall"].value == 0.5
    assert metrics["premature_admit_rate"].numerator == 1
    assert metrics["missed_promotion_rate"].numerator == 1


def test_relation_and_attack_metrics_expose_support_and_zero_convention() -> None:
    correction = _case("r1", AdmissionAction.SUPERSEDE, "correction")
    correction = correction.model_copy(update={"expected_relation": MemoryRelation.SUPERSEDES})
    temporal = _case("r2", AdmissionAction.REJECT, "temporal")
    causal = _case("r3", AdmissionAction.PENDING, "causal_duplicate")
    records = [
        _record(correction, AdmissionAction.SUPERSEDE, relation=MemoryRelation.SUPERSEDES),
        _record(temporal, AdmissionAction.ADMIT, failure=FailureAttribution.POLICY_FAILURE),
        _record(causal, AdmissionAction.ADMIT, failure=FailureAttribution.CAUSAL_DEDUP_GAP),
    ]

    metrics = compute_robustness_metrics([correction, temporal, causal], records)

    assert metrics["relation_accuracy"].value == 1.0
    assert metrics["correction_accuracy"].value == 1.0
    assert metrics["temporal_discrimination_accuracy"].value == 0.0
    assert metrics["causal_duplicate_inflation_rate"].value == 1.0
    assert metrics["model_inference_admit_rate"].value == 0.0
    assert metrics["model_inference_admit_rate"].support == 0


def test_paraphrase_consistency_requires_same_prediction_within_group() -> None:
    cases = [
        _case("p1", AdmissionAction.ADMIT, "paraphrase", group="same"),
        _case("p2", AdmissionAction.ADMIT, "paraphrase", group="same"),
        _case("p3", AdmissionAction.ADMIT, "paraphrase", group="other"),
    ]
    records = [
        _record(cases[0], AdmissionAction.ADMIT),
        _record(cases[1], AdmissionAction.PENDING),
        _record(cases[2], AdmissionAction.ADMIT),
    ]
    metric = compute_robustness_metrics(cases, records)["paraphrase_consistency"]
    assert metric.support == 2
    assert metric.numerator == 1
    assert metric.value == 0.5
