from __future__ import annotations

from pathlib import Path

from app.evaluation.memory_admission.robustness.report import (
    render_markdown,
    run_robustness_evaluation,
)


def test_final_report_includes_named_trajectory_a_to_e_observations(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = run_robustness_evaluation(
        root,
        root / "app/evaluation/memory_admission/robustness/policy_v1_manifest.json",
        tmp_path,
        live_smoke_status="DEFERRED",
        live_smoke_reason="bounded cost",
    )
    assert set(report.named_probe_results) == {"A", "B", "C", "D", "E"}
    assert report.named_probe_results["E"].gap_code == "ROBUSTNESS_GAP_CAUSAL_DEDUP"
    markdown = render_markdown(report)
    assert "Named misconception trajectories A–E" in markdown
    assert "outbox-retry" in markdown
