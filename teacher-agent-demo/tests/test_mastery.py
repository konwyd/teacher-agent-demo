"""Student Memory / Mastery Tracking 测试（全部离线；存储走 conftest 指定的 tmp JSON）。"""

import os
from pathlib import Path

from app.agent.state import DiagnosisResult, StudentIntent, TeachingAction
from app.agent.workflow import _choose_action, run_turn
from app.memory.student_state import ConceptMastery, JsonStudentStateStore


def _store(tmp_path: Path) -> JsonStudentStateStore:
    return JsonStudentStateStore(tmp_path / "profiles.json")


# ---------------- 存储层 ----------------


def test_new_student_default_mastery(tmp_path: Path) -> None:
    rec = _store(tmp_path).get_concept_mastery("s_new", "c_linear")
    assert rec.mastery_score == 0.5
    assert rec.attempts == 0
    assert rec.correct_steps == 0 and rec.wrong_steps == 0
    assert rec.common_misconceptions == []


def test_correct_step_increases_mastery(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rec = store.update_mastery_after_turn("s1", "c_linear", "correct_step")
    assert rec.mastery_score == 0.6
    assert rec.correct_steps == 1 and rec.attempts == 1
    rec = store.update_mastery_after_turn("s1", "c_linear", "correct_final_answer")
    assert rec.mastery_score == 0.8


def test_wrong_step_decreases_mastery(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rec = store.update_mastery_after_turn("s1", "c_linear", "incorrect_step", error_detail="2x = 4")
    assert rec.mastery_score == 0.4
    assert rec.wrong_steps == 1
    assert rec.recent_errors == ["2x = 4"]
    assert rec.next_review_at is not None  # 低于复习阈值，安排复习


def test_mastery_clamped_between_0_and_1(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(10):
        rec = store.update_mastery_after_turn("s1", "c1", "correct_final_answer")
    assert rec.mastery_score == 1.0
    for _ in range(20):
        rec = store.update_mastery_after_turn("s1", "c1", "incorrect_step")
    assert rec.mastery_score == 0.0


def test_persistence_across_store_instances(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    JsonStudentStateStore(path).update_mastery_after_turn("s1", "c_linear", "correct_step")
    reloaded = JsonStudentStateStore(path)  # 重新从文件加载
    assert reloaded.get_concept_mastery("s1", "c_linear").mastery_score == 0.6


def test_repeated_misconception_promoted_to_common(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rec = store.record_misconception("s1", "c_move", "移项时忘记变号。")
    assert rec.common_misconceptions == []  # 出现一次还不算"常见"
    rec = store.record_misconception("s1", "c_move", "移项时忘记变号。")
    assert rec.common_misconceptions == ["移项时忘记变号。"]


def test_recommend_review_concepts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.update_mastery_after_turn("s1", "c_move", "incorrect_step")      # 0.4
    store.update_mastery_after_turn("s1", "c_linear", "correct_final_answer")  # 0.7
    due = store.recommend_review_concepts("s1")
    ids = [r.concept_id for r in due]
    assert "c_move" in ids and "c_linear" not in ids


# ---------------- 策略随掌握度变化 ----------------


def _diag() -> DiagnosisResult:
    return DiagnosisResult(intent=StudentIntent.PROBLEM_SOLVING, confidence=0.85)


def _rec(m: float) -> ConceptMastery:
    return ConceptMastery(student_id="s", concept_id="c", mastery_score=m)


def test_policy_adapts_to_mastery() -> None:
    assert _choose_action(_diag(), _rec(0.2)) is TeachingAction.ASK_DIAGNOSTIC_QUESTION
    assert _choose_action(_diag(), _rec(0.5)) is TeachingAction.GIVE_HINT_LEVEL_1
    assert _choose_action(_diag(), _rec(0.75)) is TeachingAction.GIVE_PRACTICE_QUESTION


def test_practice_request_gated_by_mastery() -> None:
    diag = DiagnosisResult(intent=StudentIntent.UNKNOWN, confidence=0.3)
    assert _choose_action(diag, _rec(0.75), message="再来一题") is TeachingAction.GIVE_PRACTICE_QUESTION
    assert _choose_action(diag, _rec(0.3), message="再来一题") is TeachingAction.GIVE_HINT_LEVEL_2


# ---------------- 端到端：多轮后状态保存 ----------------


def test_multi_turn_mastery_saved_to_file() -> None:
    sid = "m_e2e"
    s1 = run_turn(sid, "老师 3x+6=10 怎么解？")   # help: 0.5 -> 0.45
    assert s1.mastery_update is not None
    assert s1.mastery_update.mastery_before == 0.5
    assert s1.mastery_delta == -0.05
    s2 = run_turn(sid, "3x=4")                     # correct step: 0.45 -> 0.55
    assert s2.mastery_update.mastery_before == 0.45
    assert s2.mastery_delta == 0.1
    s3 = run_turn(sid, "x=4/3")                    # final: 0.55 -> 0.75
    assert s3.mastery_update.mastery_after == 0.75
    assert Path(os.environ["TEACHER_PROFILE_PATH"]).exists()


def test_repeated_wrong_step_surfaces_known_misconception() -> None:
    sid = "m_miscon"
    run_turn(sid, "老师 3x+6=10 怎么解？")
    s1 = run_turn(sid, "3x=16")  # 移项忘记变号（第一次，尚未进入 common）
    assert s1.known_misconceptions == []
    s2 = run_turn(sid, "3x=16")  # 第二次同样错误 -> 识别为同一 misconception
    assert any("变号" in m for m in s2.known_misconceptions)
    assert s2.recommended_next_action != ""
