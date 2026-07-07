"""diagnose_node 结构化输出 + teaching_policy_node 策略 + 日志写入测试（全部离线）。"""

import json
import os
from pathlib import Path

from app.agent.state import (
    DiagnosisResult,
    StudentGap,
    StudentIntent,
    TeacherState,
    TeachingAction,
    TurnLog,
)
from app.agent.workflow import _choose_action, diagnose_node, run_turn
from app.memory.student_state import ConceptMastery


def _record(mastery: float = 0.5, attempts: int = 0) -> ConceptMastery:
    return ConceptMastery(student_id="s", concept_id="c", mastery_score=mastery, attempts=attempts)


# ---------------- diagnose_node ----------------


def test_diagnose_outputs_structured_result() -> None:
    state = TeacherState(student_id="s_diag", message="我不会解 2x+3=7")
    out = diagnose_node(state)
    diag = out["diagnosis"]
    assert isinstance(diag, DiagnosisResult)
    assert diag.intent is StudentIntent.PROBLEM_SOLVING
    assert diag.detected_concept == "一元一次方程"
    assert 0.0 <= diag.confidence <= 1.0
    assert diag.reason != ""
    assert out["detected_concept_id"] == "c_linear"


def test_diagnose_concept_question_intent() -> None:
    state = TeacherState(student_id="s_diag", message="什么是移项？")
    out = diagnose_node(state)
    assert out["diagnosis"].intent is StudentIntent.CONCEPT_QUESTION
    assert out["diagnosis"].student_gap is StudentGap.CONCEPT_MISSING
    assert out["detected_concept_id"] == "c_move"


def test_diagnose_calculation_error_gap() -> None:
    state = TeacherState(student_id="s_diag", message="去括号的时候总是错")
    out = diagnose_node(state)
    assert out["diagnosis"].student_gap is StudentGap.CALCULATION_ERROR
    assert out["detected_concept_id"] == "c_bracket"


# ---------------- teaching_policy_node ----------------


def test_policy_answer_checking() -> None:
    diag = DiagnosisResult(intent=StudentIntent.ANSWER_CHECKING, confidence=0.9)
    assert _choose_action(diag, _record()) is TeachingAction.GRADE_AND_FEEDBACK


def test_policy_review() -> None:
    diag = DiagnosisResult(intent=StudentIntent.REVIEW, confidence=0.9)
    assert _choose_action(diag, _record()) is TeachingAction.SCHEDULE_REVIEW


def test_policy_prerequisite_missing_explains_concept() -> None:
    diag = DiagnosisResult(
        intent=StudentIntent.PROBLEM_SOLVING,
        student_gap=StudentGap.PREREQUISITE_MISSING,
        confidence=0.9,
    )
    assert _choose_action(diag, _record()) is TeachingAction.EXPLAIN_CONCEPT


def test_policy_low_confidence_asks_diagnostic() -> None:
    diag = DiagnosisResult(intent=StudentIntent.UNKNOWN, confidence=0.3)
    assert _choose_action(diag, _record()) is TeachingAction.ASK_DIAGNOSTIC_QUESTION


def test_policy_hint_ladder() -> None:
    diag = DiagnosisResult(intent=StudentIntent.PROBLEM_SOLVING, confidence=0.8)
    assert _choose_action(diag, _record(attempts=0)) is TeachingAction.GIVE_HINT_LEVEL_1
    assert _choose_action(diag, _record(attempts=1)) is TeachingAction.GIVE_HINT_LEVEL_2
    # 多次求助也不默认丢例题（worked_example 只有三种显式触发）
    assert _choose_action(diag, _record(attempts=3)) is TeachingAction.GIVE_HINT_LEVEL_2


def test_policy_high_mastery_gives_practice() -> None:
    diag = DiagnosisResult(intent=StudentIntent.PROBLEM_SOLVING, confidence=0.8)
    assert _choose_action(diag, _record(mastery=0.8)) is TeachingAction.GIVE_PRACTICE_QUESTION


# ---------------- log_node ----------------


def test_turn_writes_jsonl_log() -> None:
    state = run_turn("s_log", "我不会解 2x+3=7")
    assert state.log_id

    log_file = Path(os.environ["TEACHER_LOG_DIR"]) / "turn_logs.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    entry = TurnLog.model_validate(json.loads(lines[-1]))
    assert entry.log_id == state.log_id
    assert entry.student_id == "s_log"
    assert entry.user_message == "我不会解 2x+3=7"
    assert entry.diagnosis.detected_concept == "一元一次方程"
    assert entry.teaching_action != ""
    assert entry.teacher_response != ""
    assert "c_linear" in entry.student_state_after


def test_multiple_turns_append_log_lines() -> None:
    run_turn("s_log2", "什么是移项？")
    run_turn("s_log2", "我不会解 2x+3=7")
    log_file = Path(os.environ["TEACHER_LOG_DIR"]) / "turn_logs.jsonl"
    assert len(log_file.read_text(encoding="utf-8").strip().splitlines()) == 2
