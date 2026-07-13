"""Dynamic hybrid evaluation runner tests."""

from __future__ import annotations

import json

from app.evaluation.hybrid_runner import (
    HybridConversationTrace,
    TeacherTurnResult,
    run_hybrid_conversation,
    run_hybrid_matrix,
    write_hybrid_traces,
)
from app.evaluation.mode_runner import EvalMode
from app.evaluation.student_simulator import (
    TerminationReason,
    load_student_scenarios,
)


def _scenario(concept: str = "move_terms", mastery: str = "high"):
    return next(
        scenario
        for scenario in load_student_scenarios()
        if scenario.concept_id == concept
        and scenario.mastery_level.value == mastery
    )


class _ScriptedTeacher:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.messages: list[str] = []

    def respond(self, student_message: str) -> TeacherTurnResult:
        self.messages.append(student_message)
        index = min(len(self.messages) - 1, len(self.responses) - 1)
        return TeacherTurnResult(
            response=self.responses[index],
            metadata={"script_index": index},
        )


def test_conversation_stops_after_solution_is_verified() -> None:
    teacher = _ScriptedTeacher(
        [
            "你可以自己写出移项后的等式。",
            "这一步正确，请继续。",
            "答案正确，请代回原方程验算一下？",
            "验算正确，代回后两边相等，这题完成了。",
        ]
    )

    trace = run_hybrid_conversation(
        _scenario(),
        EvalMode.AGENT_WITH_GRAPH,
        run_id="run_verified",
        teacher_session=teacher,
    )

    assert trace.termination_reason is TerminationReason.SOLVED_AND_VERIFIED
    assert trace.teacher_turn_count == 4
    assert len(teacher.messages) == 4
    assert trace.transcript[0]["role"] == "student"
    assert trace.transcript[-1]["role"] == "teacher"
    assert trace.turns[-1].generated_student_message is None


def test_runner_enforces_eight_teacher_turn_hard_limit() -> None:
    teacher = _ScriptedTeacher(["继续想想。"])

    trace = run_hybrid_conversation(
        _scenario(concept="remove_parentheses"),
        EvalMode.DEEPSEEK_ONLY,
        run_id="run_max",
        teacher_session=teacher,
        max_turns=8,
    )

    assert trace.termination_reason is TerminationReason.MAX_TURNS
    assert trace.teacher_turn_count == 8
    assert len(teacher.messages) == 8


def test_each_mode_starts_from_equal_independent_hidden_state() -> None:
    scenario = _scenario()

    def factory(mode, scenario, run_id, use_real_teacher_llm):
        del mode, scenario, run_id, use_real_teacher_llm
        return _ScriptedTeacher(["继续想想。"])

    traces = run_hybrid_matrix(
        [scenario],
        list(EvalMode),
        run_id="run_isolation",
        max_turns=1,
        teacher_session_factory=factory,
    )

    assert len(traces) == 3
    initial_dumps = [trace.initial_student_state.model_dump() for trace in traces]
    assert initial_dumps[0] == initial_dumps[1] == initial_dumps[2]
    assert scenario.initial_state.current_solution_step == 0
    assert all(trace.turns[0].teacher_metadata for trace in traces)


def test_full_scenario_mode_product_contains_exactly_54_traces() -> None:
    def factory(mode, scenario, run_id, use_real_teacher_llm):
        del mode, scenario, run_id, use_real_teacher_llm
        return _ScriptedTeacher(["继续想想。"])

    traces = run_hybrid_matrix(
        load_student_scenarios(),
        list(EvalMode),
        run_id="run_54",
        max_turns=1,
        teacher_session_factory=factory,
    )

    assert len(traces) == 54
    assert len({(trace.scenario_id, trace.mode) for trace in traces}) == 54


def test_trace_jsonl_contains_full_state_changes_and_transcript(tmp_path) -> None:
    trace = run_hybrid_conversation(
        _scenario(),
        EvalMode.AGENT_NO_GRAPH,
        run_id="run_write",
        teacher_session=_ScriptedTeacher(["继续想想。"]),
        max_turns=1,
    )
    output = tmp_path / "hybrid_traces.jsonl"

    write_hybrid_traces([trace], output)

    row = json.loads(output.read_text(encoding="utf-8").strip())
    restored = HybridConversationTrace.model_validate(row)
    assert restored.transcript == trace.transcript
    assert restored.turns[0].student_state_before == trace.initial_student_state
    assert restored.turns[0].student_state_after == trace.final_student_state
    assert restored.turns[0].teacher_observation.model_dump()

