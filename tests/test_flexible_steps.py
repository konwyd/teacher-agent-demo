"""灵活回答测试：跳步、多步计划、直接最终答案（全部离线）。"""

from fractions import Fraction

from app.agent import math_checker as mc
from app.agent.legacy_action_compat import LegacyTeachingAction as TeachingAction
from app.agent.math_checker import StepValidity
from app.agent.state import StudentIntent
from app.agent.workflow import run_turn

PROBLEM = "老师3x+4=7怎么解？"


# ---------------- math_checker ----------------


def test_classify_student_math_response() -> None:
    C = mc.classify_student_math_response
    assert C("为什么要减4") == "why_question"
    assert C("先都减四，然后同时除以3") == "multi_step_plan"
    assert C("3x=3，然后x=1") == "multi_step_plan"
    assert C("先减4") == "single_step"
    assert C("3x=3") == "intermediate_equation"
    assert C("x=1") == "final_answer"
    assert C("1") == "final_answer"
    assert C("最后等于1") == "final_answer"
    assert C("今天天气真好") == "unknown"


def test_multi_step_plan_correct() -> None:
    for msg in ("先都减四，然后同时除以3", "两边减4然后除以3", "先把4移过去，再除以3",
                "减4得到3x=3，然后除以3得到x=1", "3x=3，然后x=1"):
        plan = mc.check_multi_step_plan("3x + 4 = 7", msg)
        assert plan.valid and plan.reaches_final, (msg, plan)
        assert plan.final_value == Fraction(1)


def test_multi_step_plan_wrong_equation_rejected() -> None:
    plan = mc.check_multi_step_plan("3x + 4 = 7", "3x=5，然后x=5/3")
    assert not plan.valid
    assert "不同解" in plan.explanation


def test_single_operations_accepted() -> None:
    assert mc.check_single_operation("3x + 4 = 7", "先减4").validity is StepValidity.CORRECT_STEP
    assert mc.check_single_operation("3x + 4 = 7", "两边同时减4").student_equation == "3x = 3"
    r = mc.check_single_operation("3x = 3", "两边除以3")
    assert r.validity is StepValidity.CORRECT_FINAL_ANSWER and r.student_equation == "x = 1"


def test_check_final_answer_variants() -> None:
    for msg in ("x=1", "1", "最后等于1"):
        assert mc.check_final_answer("3x + 4 = 7", msg).validity is StepValidity.CORRECT_FINAL_ANSWER
    assert mc.check_final_answer("3x + 4 = 7", "2").validity is StepValidity.INCORRECT_STEP


# ---------------- workflow：三种回答方式 ----------------


def test_multi_step_plan_accepted_end_to_end() -> None:
    sid = "flex_multi"
    run_turn(sid, PROBLEM)
    state = run_turn(sid, "先都减四，然后同时除以3")
    assert state.response_kind == "multi_step_plan"
    assert state.student_step_validity == "correct_final_answer"
    assert state.diagnosis.intent is StudentIntent.PROVIDES_FINAL_ANSWER
    assert "思路" in state.response and "正确" in state.response
    assert "3x = 3" in state.response and "x = 1" in state.response
    assert "代回" in state.response
    # 不重复追问已回答的步骤，且只有一个问题
    assert "减去几" not in state.response and "减几" not in state.response
    assert state.response.count("？") + state.response.count("?") == 1


def test_equation_chain_plan_accepted() -> None:
    sid = "flex_chain"
    run_turn(sid, PROBLEM)
    state = run_turn(sid, "3x=3，然后x=1")
    assert state.student_step_validity == "correct_final_answer"
    assert "正确" in state.response


def test_single_op_then_single_op() -> None:
    sid = "flex_ops"
    run_turn(sid, PROBLEM)
    s1 = run_turn(sid, "先减4")
    assert s1.student_step_validity == "correct_step"
    assert s1.student_equation == "3x = 3"
    assert "除以" in s1.response  # 下一步引导指向除以
    s2 = run_turn(sid, "两边除以3")  # current 已推进到 3x=3
    assert s2.student_step_validity == "correct_final_answer"
    assert "x = 1" in s2.response


def test_direct_final_answer_accepted() -> None:
    sid = "flex_final"
    run_turn(sid, PROBLEM)
    state = run_turn(sid, "x=1")
    assert state.student_step_validity == "correct_final_answer"
    assert "正确" in state.response and "代回" in state.response


def test_low_mastery_student_multi_step_plan_still_accepted() -> None:
    """memory 只调教学深浅，不得覆盖 checker 的正确判定。"""
    from app.memory.student_state import get_store

    sid = "flex_lowmastery"
    store = get_store()
    for _ in range(4):  # 掌握度打到 0.1
        store.update_mastery_after_turn(sid, "c_linear", "incorrect_step")
    run_turn(sid, PROBLEM)
    state = run_turn(sid, "先都减四，然后同时除以3")
    assert state.student_step_validity == "correct_final_answer"
    assert state.action is TeachingAction.GRADE_AND_FEEDBACK  # 不是诊断/提示
    assert "思路" in state.response and "正确" in state.response
