"""Metrics for Phase 2A.5 adversarial robustness reports."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from app.evaluation.memory_admission.robustness.models import (
    FailureAttribution,
    RobustnessCase,
    RobustnessRecord,
)
from app.memory.contracts import AdmissionAction, MemorySourceKind, TemporalScope


class RobustnessMetric(BaseModel):
    value: float
    support: int
    numerator: int


def _ratio(numerator: int, support: int) -> RobustnessMetric:
    return RobustnessMetric(
        value=numerator / support if support else 0.0,
        support=support,
        numerator=numerator,
    )


def _accuracy(cases, records, label: str) -> RobustnessMetric:
    selected = [
        (case, record)
        for case, record in zip(cases, records)
        if label in case.labels
    ]
    return _ratio(
        sum(case.expected_action is record.predicted_action for case, record in selected),
        len(selected),
    )


def compute_robustness_metrics(
    cases: list[RobustnessCase], records: list[RobustnessRecord]
) -> dict[str, RobustnessMetric]:
    if len(cases) != len(records):
        raise ValueError("cases and records must be aligned")
    correct = sum(
        case.expected_action is record.predicted_action
        for case, record in zip(cases, records)
    )
    actions = tuple(AdmissionAction)
    f1_total = 0.0
    for action in actions:
        tp = sum(c.expected_action is action and r.predicted_action is action for c, r in zip(cases, records))
        fp = sum(c.expected_action is not action and r.predicted_action is action for c, r in zip(cases, records))
        fn = sum(c.expected_action is action and r.predicted_action is not action for c, r in zip(cases, records))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_total += 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    admit_tp = sum(c.expected_action is AdmissionAction.ADMIT and r.predicted_action is AdmissionAction.ADMIT for c, r in zip(cases, records))
    predicted_admit = sum(r.predicted_action is AdmissionAction.ADMIT for r in records)
    expected_admit = sum(c.expected_action is AdmissionAction.ADMIT for c in cases)
    relations = [(c, r) for c, r in zip(cases, records)]
    premature = [(c, r) for c, r in relations if c.expected_action is not AdmissionAction.ADMIT]
    promotions = [(c, r) for c, r in relations if c.expected_action is AdmissionAction.ADMIT]
    duplicate = [(c, r) for c, r in relations if "duplicate" in c.labels]
    causal = [(c, r) for c, r in relations if "causal_duplicate" in c.labels]
    inference = [(c, r) for c, r in relations if c.candidate.source_kind is MemorySourceKind.MODEL_INFERENCE]
    temporary = [(c, r) for c, r in relations if c.candidate.temporal_scope in {TemporalScope.TURN_ONLY, TemporalScope.SESSION}]
    stale = [(c, r) for c, r in relations if "stale_belief" in c.labels]

    groups: dict[str, list[RobustnessRecord]] = defaultdict(list)
    for case, record in relations:
        if "paraphrase" in case.labels and case.metamorphic_group:
            groups[case.metamorphic_group].append(record)
    group_results = [
        len({row.predicted_action for row in rows}) == 1
        for rows in groups.values()
    ]

    return {
        "robustness_accuracy": _ratio(correct, len(cases)),
        "robustness_macro_f1": RobustnessMetric(value=f1_total / len(actions), support=len(cases), numerator=correct),
        "admit_precision": _ratio(admit_tp, predicted_admit),
        "admit_recall": _ratio(admit_tp, expected_admit),
        "paraphrase_consistency": _ratio(sum(group_results), len(group_results)),
        "temporal_discrimination_accuracy": _accuracy(cases, records, "temporal"),
        "conditional_preference_accuracy": _accuracy(cases, records, "conditional_preference"),
        "correction_accuracy": _accuracy(cases, records, "correction"),
        "relation_accuracy": _ratio(sum(c.expected_relation is r.predicted_relation for c, r in relations), len(relations)),
        "premature_admit_rate": _ratio(sum(r.predicted_action is AdmissionAction.ADMIT for c, r in premature), len(premature)),
        "missed_promotion_rate": _ratio(sum(r.predicted_action is not AdmissionAction.ADMIT for c, r in promotions), len(promotions)),
        "stale_belief_rate": _ratio(sum(r.predicted_action is AdmissionAction.ADMIT and c.expected_action is not AdmissionAction.ADMIT for c, r in stale), len(stale)),
        "duplicate_evidence_inflation_rate": _ratio(sum(r.predicted_action is AdmissionAction.ADMIT for c, r in duplicate), len(duplicate)),
        "causal_duplicate_inflation_rate": _ratio(sum(r.failure_attribution is FailureAttribution.CAUSAL_DEDUP_GAP for c, r in causal), len(causal)),
        "model_inference_admit_rate": _ratio(sum(r.predicted_action is AdmissionAction.ADMIT for c, r in inference), len(inference)),
        "temporary_state_admit_rate": _ratio(sum(r.predicted_action is AdmissionAction.ADMIT for c, r in temporary), len(temporary)),
        "incorrect_supersede_rate": _ratio(sum(r.predicted_action is AdmissionAction.SUPERSEDE and c.expected_action is not AdmissionAction.SUPERSEDE for c, r in relations), len(relations)),
    }
