"""Hybrid evaluation CLI pipeline tests."""

from __future__ import annotations

import json

from app.evaluation.hybrid_runner import TeacherTurnResult
from app.evaluation.mode_runner import EvalMode
from app.evaluation.student_simulator import load_student_scenarios
from scripts.run_hybrid_eval import run_hybrid_pipeline, select_scenarios


class _OneResponseTeacher:
    def respond(self, student_message: str) -> TeacherTurnResult:
        del student_message
        return TeacherTurnResult(response="继续想想。")


def _factory(mode, scenario, run_id, use_real_teacher_llm):
    del mode, scenario, run_id, use_real_teacher_llm
    return _OneResponseTeacher()


def test_select_scenarios_filters_concept_and_mastery() -> None:
    selected = select_scenarios(
        load_student_scenarios(),
        concept="move_terms",
        mastery="low",
    )

    assert len(selected) == 1
    assert selected[0].scenario_id == "move_terms_low"


def test_pipeline_writes_trace_then_score_and_complete_manifest(tmp_path) -> None:
    scenario = select_scenarios(
        load_student_scenarios(),
        concept="move_terms",
        mastery="high",
    )

    result = run_hybrid_pipeline(
        scenarios=scenario,
        modes=[EvalMode.AGENT_NO_GRAPH],
        output_dir=tmp_path,
        run_id="pipeline_one",
        max_turns=1,
        teacher_session_factory=_factory,
    )

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert result["trace_count"] == result["score_count"] == 1
    assert manifest["complete"] is True
    assert manifest["expected_count"] == 1
    assert manifest["trace_count"] == manifest["score_count"] == 1
    assert (tmp_path / "hybrid_traces.jsonl").is_file()
    assert (tmp_path / "hybrid_scores.jsonl").is_file()
    assert (tmp_path / "hybrid_summary.json").is_file()


def test_full_offline_stub_pipeline_persists_exactly_54_records(tmp_path) -> None:
    result = run_hybrid_pipeline(
        scenarios=load_student_scenarios(),
        modes=list(EvalMode),
        output_dir=tmp_path,
        run_id="pipeline_54",
        max_turns=1,
        teacher_session_factory=_factory,
    )

    trace_lines = (tmp_path / "hybrid_traces.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    score_lines = (tmp_path / "hybrid_scores.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert result["expected_count"] == 54
    assert len(trace_lines) == len(score_lines) == 54

