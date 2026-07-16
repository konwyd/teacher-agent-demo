"""math_checker 单元测试（纯 stdlib 逻辑，无 LLM）。"""

from fractions import Fraction

from app.agent import math_checker as mc
from app.agent.math_checker import StepValidity


def test_extract_equation_from_natural_language() -> None:
    assert mc.extract_equation("老师 3x+6=10 怎么解？") == "3x + 6 = 10"
    assert mc.extract_equation("都除以2变成x=2") == "x = 2"
    assert mc.extract_equation("为什么+6移过去变成-6") is None
    assert mc.extract_equation("x＝4/3") == "x = 4/3"


def test_solve_and_equivalence() -> None:
    assert mc.solve_str("3x+6=10") == Fraction(4, 3)
    assert mc.solve_str("5x-2=3x+6") == 4
    assert mc.is_equivalent("3x+6=10", "3x=4")
    assert not mc.is_equivalent("3x+6=10", "2x=4")


def test_correct_intermediate_step() -> None:
    r = mc.check_step("3x + 6 = 10", "3x=4")
    assert r.validity is StepValidity.CORRECT_STEP
    assert "两边同时减去 6" in r.explanation


def test_correct_final_answer() -> None:
    r = mc.check_step("3x + 6 = 10", "x=4/3")
    assert r.validity is StepValidity.CORRECT_FINAL_ANSWER


def test_bare_number_as_answer() -> None:
    assert mc.check_step("3x + 6 = 10", "4/3").validity is StepValidity.CORRECT_FINAL_ANSWER
    assert mc.check_step("3x + 6 = 10", "2").validity is StepValidity.INCORRECT_STEP


def test_incorrect_step_for_wrong_problem() -> None:
    """2x=4 对原题 3x+6=10 不是正确变形（它是 2x+3=7 的步骤）。"""
    r = mc.check_step("3x + 6 = 10", "2x=4")
    assert r.validity is StepValidity.INCORRECT_STEP
    assert "不是当前原题" in r.explanation


def test_step_for_2x_plus_3_problem() -> None:
    """用户反馈场景：原题 2x+3=7 时，2x=4 和 x=2 都必须被正确识别。"""
    assert mc.check_step("2x + 3 = 7", "2x=4").validity is StepValidity.CORRECT_STEP
    assert mc.check_step("2x + 3 = 7", "都除以2变成x=2").validity is StepValidity.CORRECT_FINAL_ANSWER


def test_why_question_is_not_a_step() -> None:
    assert mc.check_step("3x + 6 = 10", "为什么要减6").validity is StepValidity.NOT_A_STEP


def test_suggest_next_step_trajectory() -> None:
    """3x+6=10 的正确教学轨迹：先减6得 3x=4，再除以3得 x=4/3。"""
    assert mc.suggest_next_step("3x + 6 = 10") == ("两边同时减去 6", "3x = 4")
    assert mc.suggest_next_step("3x = 4") == ("两边同时除以 3", "x = 4/3")
    assert mc.suggest_next_step("x = 4/3") is None


def test_uses_new_example_detection() -> None:
    assert mc.uses_new_example("看这道例题 2x+3=7", "3x + 6 = 10") is True
    assert mc.uses_new_example("对，3x=4，下一步除以3", "3x + 6 = 10") is False
    assert mc.mentions_equation("回到原题 3x + 6 = 10", "3x + 6 = 10") is True
