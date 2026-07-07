"""Targeted fixes 回归测试：例题污染 / 过早给答案 / 多问题 / 概念识别 / 激进例题。"""

import pytest

from app.agent.state import DiagnosisResult, StudentIntent, TeacherState, TeachingAction
from app.agent.workflow import (
    _choose_action,
    _keep_single_question,
    postprocess_teacher_response,
    run_turn,
)
from app.memory.student_state import ConceptMastery


def _norm(s: str) -> str:
    return s.replace(" ", "")


def _record(mastery: float = 0.5, attempts: int = 0) -> ConceptMastery:
    return ConceptMastery(student_id="s", concept_id="c", mastery_score=mastery, attempts=attempts)


# ---------------- 例题污染 / 过早给答案 ----------------


def test_first_ask_3x_never_mentions_2x_example() -> None:
    state = run_turn("tf_3x", "老师 3x+6=10 怎么解？")
    assert "2x+3=7" not in _norm(state.response)
    assert "3x=9" not in _norm(state.response)


def test_first_ask_2x_never_reveals_answer() -> None:
    state = run_turn("tf_2x", "2x+3=7 怎么解")
    assert state.action is TeachingAction.GIVE_HINT_LEVEL_1
    assert "x=2" not in _norm(state.response)


# ---------------- 概念识别（方程结构优先） ----------------


def test_bracket_equation_detected_as_bracket_concept() -> None:
    state = run_turn("tf_bracket", "3(x-2)=9 怎么做")
    assert state.detected_concept_name == "去括号"


def test_frac_equation_detected_as_frac_concept() -> None:
    state = run_turn("tf_frac", "x/2 - 1 = x/3 怎么解")
    assert state.detected_concept_name == "分数系数方程"


def test_x_on_both_sides_detected_as_move_concept() -> None:
    state = run_turn("tf_move", "5x-2=3x+6 怎么解")
    assert state.detected_concept_name == "移项"


# ---------------- 每轮最多一个问题 ----------------


def test_responses_have_at_most_one_question() -> None:
    sid = "tf_oneq"
    for msg in ("老师 3x+6=10 怎么解？", "为什么+6移过去变成-6", "3x=4", "听不懂"):
        state = run_turn(sid, msg)
        n = state.response.count("？") + state.response.count("?")
        assert n <= 1, (msg, state.response)


def test_keep_single_question_collapses_multi_questions() -> None:
    text = "先想想减几？你懂了吗？那再除以3好不好？"
    out = _keep_single_question(text)
    assert out.count("？") == 1
    assert "再除以3" in out  # 保留最后一个问题


# ---------------- postprocess 降级 ----------------


def _llm_state(response_irrelevant: str = "") -> TeacherState:
    return TeacherState(
        student_id="tf_pp",
        message="怎么办",
        original_problem="3x + 6 = 10",
        target_equation="3x + 6 = 10",
        detected_concept_name="一元一次方程",
        diagnosis=DiagnosisResult(intent=StudentIntent.PROBLEM_SOLVING),
        action=TeachingAction.GIVE_HINT_LEVEL_1,
    )


def test_postprocess_degrades_alien_equation() -> None:
    out = postprocess_teacher_response("我们看个新例子：5x = 20，很简单吧？", _llm_state())
    assert "5x=20" not in _norm(out)
    assert "3x+6=10" in _norm(out)  # 降级为围绕原题的模板


def test_postprocess_degrades_premature_final_answer() -> None:
    out = postprocess_teacher_response("其实答案就是 x = 4/3，记住即可。懂了吗？", _llm_state())
    assert "x=4/3" not in _norm(out)


# ---------------- worked_example 只有三种触发 ----------------


def test_how_to_solve_gets_hint1_not_worked_example() -> None:
    diag = DiagnosisResult(intent=StudentIntent.PROBLEM_SOLVING, confidence=0.85)
    assert _choose_action(diag, _record(), message="我不会做这道题") is TeachingAction.GIVE_HINT_LEVEL_1


def test_example_request_triggers_worked_example() -> None:
    diag = DiagnosisResult(intent=StudentIntent.PROBLEM_SOLVING, confidence=0.85)
    assert _choose_action(diag, _record(), message="能举个例子吗") is TeachingAction.SHOW_WORKED_EXAMPLE


def test_totally_lost_triggers_worked_example() -> None:
    diag = DiagnosisResult(intent=StudentIntent.CONFUSED, confidence=0.85)
    assert _choose_action(diag, _record(), message="我完全不懂") is TeachingAction.SHOW_WORKED_EXAMPLE
    assert _choose_action(diag, _record(), message="有点不明白") is TeachingAction.GIVE_HINT_LEVEL_2


def test_second_consecutive_wrong_step_gets_worked_example() -> None:
    sid = "tf_2wrong"
    run_turn(sid, "老师 3x+6=10 怎么解？")
    s1 = run_turn(sid, "2x=4")   # 第一次答错 -> 批改反馈
    assert s1.action is TeachingAction.GRADE_AND_FEEDBACK
    s2 = run_turn(sid, "5x=1")   # 连续第二次答错 -> 类比例题
    assert s2.action is TeachingAction.SHOW_WORKED_EXAMPLE
    assert "类比" in s2.response
    assert "3x + 6 = 10" in s2.response  # 回到原题


# ---------------- 真实 LLM 路径加固（gap 锚定 / 收尾问题） ----------------


def test_anchor_gap_downgrades_speculative_llm_gap() -> None:
    from app.agent.state import StudentGap
    from app.agent.workflow import _anchor_gap
    from app.retrieval.graph_retriever import get_retriever

    concept = get_retriever().get_concept("c_linear")
    diag = DiagnosisResult(
        intent=StudentIntent.PROBLEM_SOLVING,
        student_gap=StudentGap.PROCEDURE_ERROR,  # LLM 无证据的猜测
        confidence=0.9,
    )
    _anchor_gap(diag, "anchor_s1", concept, "我不会解 2x+3=7")
    assert diag.student_gap is StudentGap.UNKNOWN  # 回退，policy 走 hint1


def test_anchor_gap_keeps_evidenced_gap() -> None:
    from app.agent.state import StudentGap
    from app.agent.workflow import _anchor_gap
    from app.retrieval.graph_retriever import get_retriever

    concept = get_retriever().get_concept("c_bracket")
    diag = DiagnosisResult(
        intent=StudentIntent.UNKNOWN,
        student_gap=StudentGap.CALCULATION_ERROR,
        confidence=0.9,
    )
    _anchor_gap(diag, "anchor_s2", concept, "去括号的时候总是错")  # 有关键词证据
    assert diag.student_gap is StudentGap.CALCULATION_ERROR


def test_postprocess_appends_closing_question_when_missing() -> None:
    state = _llm_state()
    state.action = TeachingAction.GIVE_PRACTICE_QUESTION
    out = postprocess_teacher_response("很好，你已经掌握了。这道题留给你练习。", state)
    assert out.count("？") + out.count("?") == 1
    assert out.rstrip().endswith("？")


def test_postprocess_keeps_existing_single_question() -> None:
    out = postprocess_teacher_response("先想想第一步。你觉得两边同时减几？", _llm_state())
    assert out == "先想想第一步。你觉得两边同时减几？"


def test_postprocess_low_mastery_diagnostic_falls_back_to_basic_language() -> None:
    """低掌握度学生的诊断问题若缺基础语言，降级为含“两边同时”的模板。"""
    from app.memory.student_state import get_store

    sid = "tf_lowmastery_pp"
    store = get_store()
    for _ in range(3):  # 0.5 -> 0.2
        store.update_mastery_after_turn(sid, "c_linear", "incorrect_step")
    state = _llm_state()
    state.student_id = sid
    state.detected_concept_id = "c_linear"
    state.action = TeachingAction.ASK_DIAGNOSTIC_QUESTION
    out = postprocess_teacher_response("你具体是哪里不明白？", state)
    assert "两边同时" in out
    assert "3x+6=10" in _norm(out)  # 模板围绕原题
