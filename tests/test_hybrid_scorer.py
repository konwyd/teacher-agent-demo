"""Disk-first deterministic hybrid scoring tests."""

from __future__ import annotations

import json

from app.evaluation.hybrid_runner import (
    TeacherTurnResult,
    run_hybrid_conversation,
    write_hybrid_traces,
)
from app.evaluation.hybrid_scorer import (
    HybridConversationScore,
    aggregate_hybrid_scores,
    score_hybrid_trace,
    score_trace_file,
)
from app.evaluation.mode_runner import EvalMode
from app.evaluation.student_simulator import (
    TerminationReason,
    UnderstandingStatus,
    load_student_scenarios,
)


def _scenario():
    return next(
        scenario
        for scenario in load_student_scenarios()
        if scenario.concept_id == "move_terms"
        and scenario.mastery_level.value == "high"
    )


class _Teacher:
    def __init__(self) -> None:
        self.responses = iter(
            [
                "你可以自己写出移项后的等式。",
                "这一步正确，请继续。",
                "答案正确，请代回原方程验算一下？",
                "验算正确，代回后两边相等，这题完成了。",
            ]
        )

    def respond(self, student_message: str) -> TeacherTurnResult:
        del student_message
        return TeacherTurnResult(response=next(self.responses))


def _solved_trace():
    return run_hybrid_conversation(
        _scenario(),
        EvalMode.AGENT_WITH_GRAPH,
        run_id="score_run",
        teacher_session=_Teacher(),
    )


def test_fixed_solved_trace_has_exact_trajectory_metrics() -> None:
    trace = _solved_trace()

    score = score_hybrid_trace(trace)

    assert score.task_success is True
    assert score.solution_progress == 1.0
    assert score.misconception_resolved is None
    assert score.understanding_gain == 2
    assert score.confidence_change == 0.2
    assert score.teacher_turns == 4
    assert score.hint_exposure == 1
    assert score.termination_reason is TerminationReason.SOLVED_AND_VERIFIED
    assert score.teacher_quality_pass_rate == 1.0
    assert score.teacher_quality_checks


def test_score_trace_file_reads_persisted_trace_not_mutated_memory(
    tmp_path,
) -> None:
    trace = _solved_trace()
    trace_path = tmp_path / "hybrid_traces.jsonl"
    score_path = tmp_path / "hybrid_scores.jsonl"
    summary_path = tmp_path / "hybrid_summary.json"
    write_hybrid_traces([trace], trace_path)

    trace.final_student_state.understanding_status = UnderstandingStatus.CONFUSED
    scores, summary = score_trace_file(
        trace_path,
        score_path,
        summary_path,
    )

    assert scores[0].understanding_gain == 2
    persisted = HybridConversationScore.model_validate_json(
        score_path.read_text(encoding="utf-8").strip()
    )
    assert persisted == scores[0]
    assert summary["total_conversations"] == 1
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_aggregate_groups_by_mode_concept_mastery_and_combination() -> None:
    score = score_hybrid_trace(_solved_trace())
    second = score.model_copy(
        update={
            "scenario_id": "move_terms_low",
            "mastery_level": "low",
            "mode": EvalMode.AGENT_NO_GRAPH,
            "task_success": False,
            "termination_reason": TerminationReason.MAX_TURNS,
        }
    )

    summary = aggregate_hybrid_scores([score, second])

    assert summary["total_conversations"] == 2
    assert set(summary["groups"]) == {
        "by_mode",
        "by_mode_concept",
        "by_mode_mastery",
        "by_mode_concept_mastery",
    }
    assert summary["groups"]["by_mode"]["agent_with_graph"][
        "task_success_rate"
    ] == 1.0
    assert summary["groups"]["by_mode"]["agent_no_graph"][
        "termination_distribution"
    ]["max_turns"]["count"] == 1
