from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.memory_admission.analytics import (
    reduce_shadow_analytics,
    export_shadow_review,
)
from app.evaluation.memory_admission.evaluator import (
    MemoryAdmissionEvaluator,
    classification_metrics,
)
from app.evaluation.memory_admission.models import REQUIRED_CATEGORIES
from app.memory.contracts import AdmissionAction


def test_dataset_is_fixed_disjoint_bounded_and_complete() -> None:
    evaluator = MemoryAdmissionEvaluator()
    dev = evaluator.load("development")
    holdout = evaluator.load("holdout")
    assert len(dev) >= 30
    assert len(holdout) >= 18
    assert {case.case_id for case in dev}.isdisjoint(
        {case.case_id for case in holdout}
    )
    all_cases = dev + holdout
    assert len({case.case_id for case in all_cases}) == len(all_cases) >= 48
    assert sum(bool(case.sequence_id) for case in all_cases) >= 12
    assert REQUIRED_CATEGORIES <= {case.category for case in all_cases}
    assert all(len(json.dumps(case.model_dump(mode="json"))) < 12_000 for case in all_cases)
    assert all("conversation" not in case.rationale.lower() for case in all_cases)


def test_literal_classification_metrics_and_zero_denominators() -> None:
    labels = [AdmissionAction.ADMIT, AdmissionAction.ADMIT, AdmissionAction.REJECT]
    predictions = [AdmissionAction.ADMIT, AdmissionAction.PENDING, AdmissionAction.REJECT]
    metrics = classification_metrics(labels, predictions)
    assert metrics["accuracy"]["value"] == 2 / 3
    assert metrics["admit_precision"]["value"] == 1.0
    assert metrics["per_action"]["supersede"]["precision"] == 0.0
    assert metrics["per_action"]["supersede"]["support"] == 0


def test_evaluator_reports_required_safety_sequence_and_baseline_metrics() -> None:
    report = MemoryAdmissionEvaluator().run("development")
    for name in (
        "inference_admit_rate",
        "temporary_admit_rate",
        "curated_to_student_admit_rate",
        "single_observation_admit_rate",
        "contradiction_supersede_rate",
        "duplicate_admit_rate",
    ):
        assert name in report.safety
        assert report.safety[name].support >= 0
    assert report.sequential["promotion_recall"].support >= 0
    assert report.baseline["baseline_enqueue_rate"].support == len(report.cases)
    assert len(report.divergence_cases) <= 12


def test_analytics_and_review_export_are_bounded(tmp_path: Path) -> None:
    records = MemoryAdmissionEvaluator().run("development").records[:4]
    analytics = reduce_shadow_analytics(records)
    assert analytics["candidate_count"] == 4
    assert "by_memory_type" in analytics
    target = tmp_path / "review.jsonl"
    written = export_shadow_review(records, target, max_records=2)
    assert written == 2
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(rows) == 2
    assert all("full_conversation" not in row for row in rows)
    assert all(set(row) <= {
        "case_id", "candidate", "provenance", "features", "relation",
        "evidence", "decision", "reasons", "baseline_action", "divergence",
        "controller_error",
    } for row in rows)
