from __future__ import annotations

from pathlib import Path

from app.evaluation.memory_admission.robustness.layer_a import LayerAEvaluator
from app.evaluation.memory_admission.robustness.models import (
    FailureAttribution,
    RobustnessCase,
)


REQUIRED_ATTACK_LABELS = {
    "paraphrase",
    "deceptive_neighbor",
    "negation",
    "correction",
    "conditional_preference",
    "temporal",
    "misconception_trajectory",
    "causal_duplicate",
    "cross_session",
    "stale_belief",
    "noisy",
    "source_reliability",
    "capability_independent",
    "metamorphic",
    "duplicate",
    "contradiction",
    "refinement",
}


def _data_path() -> Path:
    return Path("app/evaluation/memory_admission/robustness/data")


def test_layer_a_attack_set_is_new_fixed_and_broad() -> None:
    cases = LayerAEvaluator.load_cases(_data_path())
    old = {
        *(f"dev-{index:02d}" for index in range(1, 31)),
        *(f"hold-{index:02d}" for index in range(1, 19)),
    }
    assert len(cases) >= 60
    assert len({case.case_id for case in cases}) == len(cases)
    assert not ({case.case_id for case in cases} & old)
    assert REQUIRED_ATTACK_LABELS <= {
        label for case in cases for label in case.labels
    }
    assert all(case.expected_action is not None for case in cases)
    assert all(case.rationale and len(case.rationale) <= 240 for case in cases)


def test_layer_a_schema_rejects_missing_ground_truth() -> None:
    payload = {
        "case_id": "attack-invalid",
        "labels": ["temporal"],
        "rationale": "missing decision",
        "candidate": {
            "memory_type": "preference",
            "source_kind": "user_stated",
            "temporal_scope": "persistent",
            "payload": {"teaching_preferences": ["分步骤讲解"]},
        },
    }
    try:
        RobustnessCase.model_validate(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("ground-truth decision must be mandatory")


def test_layer_a_runs_without_live_services_and_preserves_failures(tmp_path: Path) -> None:
    report = LayerAEvaluator(_data_path()).run(tmp_path)
    assert len(report.records) >= 60
    assert report.llm_calls == 0
    assert report.student_memory_writes == 0
    assert report.failures == [
        row for row in report.records if row.failure_attribution is not None
    ]
    assert all(
        row.failure_attribution in set(FailureAttribution) or row.failure_attribution is None
        for row in report.records
    )
    causal = [row for row in report.records if "causal_duplicate" in row.labels]
    assert causal
    assert all(
        row.failure_attribution in {None, FailureAttribution.CAUSAL_DEDUP_GAP}
        for row in causal
    )


def test_layer_a_replay_is_deterministic_except_latency(tmp_path: Path) -> None:
    first = LayerAEvaluator(_data_path()).run(tmp_path / "first")
    second = LayerAEvaluator(_data_path()).run(tmp_path / "second")
    compact = lambda report: [
        (
            row.case_id,
            row.predicted_action,
            row.predicted_relation,
            row.failure_attribution,
            row.evidence_maturity,
        )
        for row in report.records
    ]
    assert compact(first) == compact(second)
