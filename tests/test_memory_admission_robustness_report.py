from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.memory_admission.robustness.report import (
    RobustnessEvaluationReport,
    render_markdown,
    run_robustness_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "app/evaluation/memory_admission/robustness/policy_v1_manifest.json"


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> RobustnessEvaluationReport:
    return run_robustness_evaluation(
        ROOT,
        MANIFEST,
        tmp_path_factory.mktemp("robustness-report"),
        live_smoke_status="DEFERRED",
        live_smoke_reason="local 32B inference excluded from primary benchmark",
    )


def test_report_contains_every_required_metric_and_status(report) -> None:
    assert report.policy_v1_frozen is True
    assert report.layer_a_cases >= 60
    assert report.layer_b_cases >= 30
    assert report.long_trajectories == [20, 50, 100]
    required = {
        "robustness_accuracy", "robustness_macro_f1", "admit_precision",
        "admit_recall", "paraphrase_consistency",
        "temporal_discrimination_accuracy", "conditional_preference_accuracy",
        "correction_accuracy", "relation_accuracy", "premature_admit_rate",
        "missed_promotion_rate", "stale_belief_rate",
        "duplicate_evidence_inflation_rate", "causal_duplicate_inflation_rate",
        "model_inference_admit_rate", "temporary_state_admit_rate",
        "incorrect_supersede_rate",
    }
    assert required <= set(report.metrics)
    assert report.live_smoke.status == "DEFERRED"
    assert report.policy_v1_modified is False
    assert report.authoritative_admission_implemented is False
    assert report.memcon_implemented is False
    assert report.online_reward_implemented is False
    assert report.ready_for_phase_2b == "REVIEW_REQUIRED"


def test_failure_taxonomy_separates_causal_gap_from_policy(report) -> None:
    assert "CAUSAL_DEDUP_GAP" in report.failure_taxonomy
    causal_cases = report.failure_taxonomy["CAUSAL_DEDUP_GAP"]
    assert len(causal_cases) == 5
    policy_cases = set(report.failure_taxonomy.get("POLICY_FAILURE", []))
    assert not (set(causal_cases) & policy_cases)
    assert report.phase_2b_blockers


def test_long_trajectory_report_has_required_diagnostics(report) -> None:
    assert len(report.long_trajectory_results) == 3
    for item in report.long_trajectory_results:
        assert item.max_pending_ledger_size >= 0
        assert item.pending_survival_distribution
        assert item.promotion_latency_independent_events
        assert item.contradiction_recovery_behavior


def test_markdown_renderer_includes_manifest_results_and_stop_boundary(report) -> None:
    markdown = render_markdown(report)
    for text in (
        "PolicyV1 frozen manifest",
        "Layer A",
        "Layer B",
        "Causal duplicate",
        "Failure attribution",
        "Existing regression results",
        "Phase 2B blockers",
        "POLICY_V1_MODIFIED = NO",
        "PHASE_2A_5_COMPLETE = YES",
        "READY_FOR_PHASE_2B = REVIEW_REQUIRED",
    ):
        assert text in markdown
