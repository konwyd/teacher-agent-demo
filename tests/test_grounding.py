"""Grounding 测试：老师必须始终围绕学生的 original_problem（原题 3x+6=10 全套案例）。

全部离线（conftest 已移除 API key），跨轮状态依赖 conversation store 单例，
因此每个测试用不同的 student_id。
"""

import json
import os
from pathlib import Path

from app.agent import math_checker as mc
from app.agent.state import StudentIntent, TeachingAction, TurnLog
from app.agent.workflow import run_turn

PROBLEM = "老师 3x+6=10 怎么解？"


def _no_new_equation(response: str, target: str) -> bool:
    return not mc.uses_new_example(response, target)


def test_first_turn_extracts_original_problem() -> None:
    state = run_turn("g_extract", PROBLEM)
    assert state.original_problem == "3x + 6 = 10"
    assert state.target_equation == "3x + 6 = 10"
    assert state.lesson_goal == "solve_for_x"


def test_first_turn_does_not_teach_retrieved_example() -> None:
    """第一轮不得把检索例题（如 2x+3=7）当成目标题。"""
    state = run_turn("g_noswap", PROBLEM)
    assert "3x + 6 = 10" in state.response
    assert _no_new_equation(state.response, state.target_equation), state.response


def test_why_question_explains_subtract_six() -> None:
    """学生问“为什么+6移过去变成-6”，必须解释“等式两边同时减6”，不换新例题。"""
    run_turn("g_why", PROBLEM)
    state = run_turn("g_why", "为什么+6移过去变成-6")
    assert state.diagnosis is not None
    assert state.diagnosis.intent is StudentIntent.ASKS_WHY
    assert state.original_problem == "3x + 6 = 10"  # 未被覆盖
    assert "两边同时减" in state.response and "6" in state.response
    assert _no_new_equation(state.response, state.target_equation), state.response


def test_correct_step_3x_eq_4() -> None:
    """学生回答 3x=4：必须判定为正确步骤，先肯定，再回到原题的下一步（除以3）。"""
    run_turn("g_step", PROBLEM)
    state = run_turn("g_step", "3x=4")
    assert state.student_step_validity == "correct_step"
    assert state.action is TeachingAction.GRADE_AND_FEEDBACK
    assert "正确" in state.response
    assert "两边同时减去 6" in state.response
    assert "3x + 6 = 10" in state.response  # 回到原题
    assert "除以" in state.response  # 引导下一步
    assert state.response.count("？") <= 1  # 每轮最多一个问题
    assert _no_new_equation(state.response, state.target_equation), state.response


def test_correct_final_answer_x_eq_4_3() -> None:
    """学生回答 x=4/3：必须判定为正确最终答案。"""
    run_turn("g_final", PROBLEM)
    run_turn("g_final", "3x=4")
    state = run_turn("g_final", "x=4/3")
    assert state.student_step_validity == "correct_final_answer"
    assert "正确" in state.response
    assert _no_new_equation(state.response, state.target_equation), state.response


def test_wrong_step_2x_eq_4_is_rejected() -> None:
    """原题是 3x+6=10 时，2x=4 必须被指出不是当前原题的正确中间步骤。"""
    run_turn("g_wrong", PROBLEM)
    state = run_turn("g_wrong", "2x=4")
    assert state.student_step_validity == "incorrect_step"
    assert "不是当前原题" in state.response
    assert "3x + 6 = 10" in state.response  # 回到原题


def test_original_problem_not_overwritten_by_followups() -> None:
    """后续追问/步骤回答不得覆盖 original_problem。"""
    run_turn("g_keep", PROBLEM)
    for msg in ("为什么+6移过去变成-6", "3x=4", "听不懂"):
        state = run_turn("g_keep", msg)
        assert state.original_problem == "3x + 6 = 10", msg


def test_solved_steps_accumulate_and_current_step_advances() -> None:
    run_turn("g_adv", PROBLEM)
    run_turn("g_adv", "3x=4")
    state = run_turn("g_adv", "x=4/3")
    assert "3x = 4" in state.solved_steps
    assert state.current_step == "3x = 4"  # 本轮开始时的进度


def test_no_new_example_unless_worked_example() -> None:
    """teacher response 不得无故出现新题，除非 teaching_action 是 show_worked_example。"""
    sid = "g_nonew"
    run_turn(sid, PROBLEM)
    for msg in ("为什么+6移过去变成-6", "3x=4", "听不懂"):
        state = run_turn(sid, msg)
        if state.action is not TeachingAction.SHOW_WORKED_EXAMPLE:
            assert _no_new_equation(state.response, state.target_equation), (msg, state.response)


def test_requests_solution_gets_worked_example_and_returns_to_original() -> None:
    run_turn("g_sol", PROBLEM)
    state = run_turn("g_sol", "太难了，直接告诉我答案吧")
    assert state.diagnosis is not None
    assert state.diagnosis.intent is StudentIntent.REQUESTS_SOLUTION
    assert state.action is TeachingAction.SHOW_WORKED_EXAMPLE
    assert "3x + 6 = 10" in state.response  # 例题之后必须回到原题


def test_log_contains_grounding_fields() -> None:
    run_turn("g_log", PROBLEM)
    state = run_turn("g_log", "3x=4")
    log_file = Path(os.environ["TEACHER_LOG_DIR"]) / "turn_logs.jsonl"
    entry = TurnLog.model_validate(json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1]))
    assert entry.original_problem == "3x + 6 = 10"
    assert entry.target_equation == "3x + 6 = 10"
    assert entry.student_step_validity == "correct_step"
    assert entry.whether_response_used_new_example is False
    assert entry.whether_response_returned_to_original_problem is True
