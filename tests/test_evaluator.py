"""Evaluation harness 测试（全部离线，conftest 已移除 API key）。"""

import pytest

from app.agent.state import TeacherState, TeachingAction
from app.evaluation.evaluator import (
    GoldenCase,
    GoldenTurn,
    evaluate_response,
    load_golden_cases,
    load_single_turn_cases,
    run_golden_conversation,
    run_single_case,
)


def _case(**overrides) -> GoldenCase:
    base = dict(
        case_id="test_case",
        original_problem="3x + 6 = 10",
        turns=[GoldenTurn(role="student", content="老师 3x+6=10 怎么解？")],
        expected_concept="一元一次方程",
        expected_final_answer="x=4/3",
        forbidden_patterns=["2x+3=7", "我们来看一道新题"],
        required_behaviors=["must_keep_original_problem"],
    )
    base.update(overrides)
    return GoldenCase.model_validate(base)


def _state(**overrides) -> TeacherState:
    base = dict(
        student_id="s_eval",
        message="3x=4",
        original_problem="3x + 6 = 10",
        target_equation="3x + 6 = 10",
        action=TeachingAction.GIVE_HINT_LEVEL_1,
        response="我们继续看原题 3x + 6 = 10。下一步做什么？",
    )
    base.update(overrides)
    return TeacherState.model_validate(base)


def test_eval_cases_load() -> None:
    golden = load_golden_cases()
    single = load_single_turn_cases()
    assert len(golden) >= 10
    assert len(single) >= 5
    ids = [c.case_id for c in golden]
    assert len(ids) == len(set(ids)), "case_id 不得重复"
    for c in golden:
        assert c.original_problem
        assert c.turns and all(t.role == "student" for t in c.turns)
        assert c.required_behaviors


def test_evaluator_detects_forbidden_patterns() -> None:
    case = _case()
    state = _state(response="我们来看一道新题：2x+3=7，先减3。好吗？")
    checks = {c.name: c for c in evaluate_response(case, "3x=4", state)}
    assert checks["no_forbidden_patterns"].passed is False
    assert "2x+3=7" in checks["no_forbidden_patterns"].detail


def test_evaluator_checks_original_problem() -> None:
    case = _case()
    state = _state(original_problem="2x + 3 = 7")  # 原题被篡改
    checks = {c.name: c for c in evaluate_response(case, "你好", state)}
    assert checks["keeps_original_problem"].passed is False


def test_evaluator_detects_unrelated_new_example() -> None:
    case = _case(forbidden_patterns=[])
    state = _state(response="看这个例子：5x = 20，所以 x = 4。懂了吗？")
    checks = {c.name: c for c in evaluate_response(case, "还是不懂", state)}
    assert checks["no_unrelated_new_example"].passed is False


def test_evaluator_detects_premature_final_answer() -> None:
    case = _case(forbidden_patterns=[])
    state = _state(response="其实答案就是 x = 4/3，记住就行。明白吗？")
    checks = {c.name: c for c in evaluate_response(case, "第一步做什么", state)}
    assert checks["no_premature_final_answer"].passed is False


def test_evaluator_detects_multiple_questions() -> None:
    case = _case()
    state = _state(response="先想想减几？再想想除几？最后答案是多少？")
    checks = {c.name: c for c in evaluate_response(case, "嗯", state)}
    assert checks["asks_at_most_one_question"].passed is False


def test_golden_conversation_runner_mock_mode() -> None:
    cases = {c.case_id: c for c in load_golden_cases()}
    result = run_golden_conversation(cases["linear_eq_001"])
    assert result.passed, result.failures


def test_all_golden_cases_pass_offline() -> None:
    """Golden 回归门：mock 模式下所有多轮案例必须全部通过。"""
    for case in load_golden_cases():
        result = run_golden_conversation(case)
        assert result.passed, f"{case.case_id} 失败：{result.failures}"


def test_all_single_turn_cases_pass_offline() -> None:
    for case in load_single_turn_cases():
        result = run_single_case(case)
        assert result.passed, f"{case.case_id} 失败：{result.failures}"
