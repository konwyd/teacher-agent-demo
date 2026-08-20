from __future__ import annotations

from pathlib import Path

from app.evaluation.memory_admission.robustness.models import FailureAttribution
from app.evaluation.memory_admission.robustness.trajectories import (
    run_causal_duplicate_attacks,
    run_long_trajectories,
)


DATA = Path("app/evaluation/memory_admission/robustness/data")


def test_five_causal_duplicate_variants_report_contract_gap(tmp_path: Path) -> None:
    report = run_causal_duplicate_attacks(DATA, tmp_path)
    assert len(report.records) == 5
    assert report.gap_code == "ROBUSTNESS_GAP_CAUSAL_DEDUP"
    assert report.inflated_count == 5
    assert all(
        row.failure_attribution is FailureAttribution.CAUSAL_DEDUP_GAP
        for row in report.records
    )
    assert all(row.support_count > row.distinct_turn_count for row in report.records)


def test_long_trajectories_use_real_ledger_at_20_50_100_turns(tmp_path: Path) -> None:
    report = run_long_trajectories(tmp_path, lengths=(20, 50, 100))
    assert [item.turn_count for item in report.trajectories] == [20, 50, 100]
    assert sum(item.turn_count for item in report.trajectories) == 170
    assert report.llm_calls == 0
    for item in report.trajectories:
        assert len(item.decisions) == item.turn_count
        assert item.final_ledger_size > 0
        assert item.max_pending_ledger_size >= 0
        assert item.pending_survival_distribution
        assert item.promotion_latency_independent_events
        assert item.contradiction_recovery_behavior


def test_long_trajectory_replay_is_deterministic(tmp_path: Path) -> None:
    first = run_long_trajectories(tmp_path / "first", lengths=(20,))
    second = run_long_trajectories(tmp_path / "second", lengths=(20,))
    left = first.trajectories[0].model_dump(exclude={"sqlite_path"})
    right = second.trajectories[0].model_dump(exclude={"sqlite_path"})
    assert left == right


def test_independent_verified_evidence_maturity_does_not_decrease(tmp_path: Path) -> None:
    item = run_long_trajectories(tmp_path, lengths=(20,)).trajectories[0]
    maturity = item.independent_evidence_maturity
    assert len(maturity) >= 3
    assert maturity == sorted(maturity)
