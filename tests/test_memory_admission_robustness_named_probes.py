from __future__ import annotations

from pathlib import Path

from app.evaluation.memory_admission.robustness.trajectories import (
    run_named_misconception_probes,
)


def test_named_probes_cover_required_trajectories_a_through_e(tmp_path: Path) -> None:
    probes = run_named_misconception_probes(tmp_path)
    assert set(probes) == {"A", "B", "C", "D", "E"}
    assert probes["A"].decisions[0] == "pending"
    assert "admit" in probes["A"].decisions[1:]
    assert "admit" not in probes["B"].decisions
    assert probes["C"].relations[1:] == ["contradicts", "contradicts"]
    assert len(set(probes["D"].semantic_keys)) == 3
    assert probes["E"].distinct_causal_turns == 1
    assert probes["E"].support_count == 3
    assert probes["E"].gap_code == "ROBUSTNESS_GAP_CAUSAL_DEDUP"


def test_named_probe_e_models_separate_regeneration_appends(tmp_path: Path) -> None:
    probe = run_named_misconception_probes(tmp_path)["E"]
    assert probe.append_count == 3
    assert probe.event_references == ["tool-call", "retry", "outbox-retry"]
    assert probe.decisions == ["pending", "admit", "admit"]
