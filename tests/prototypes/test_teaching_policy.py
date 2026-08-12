from __future__ import annotations

from prototypes.llm_led_teacher.orchestration_models import (
    RequestClassification,
    TeachingContract,
    TeachingResponseMode,
)
from prototypes.llm_led_teacher.teaching_policy import (
    has_explicit_full_solution_request,
    has_no_answer_request,
    normalize_teaching_contract,
)


def _math_classification() -> RequestClassification:
    return RequestClassification(
        scope="k9",
        subject="数学",
        stage="初中",
        is_new_problem=False,
        answer_mode="guided",
        reason="math follow-up",
    )


def _full_solution_contract() -> TeachingContract:
    return TeachingContract(
        response_mode="full_solution",
        requested_target="完整解答当前题",
        allowed_content=["完整推导", "最终答案"],
        forbidden_content=[],
        may_apply_to_active_problem=True,
        may_reveal_final_answer=True,
        reason="model proposed full solution",
    )


def test_request_marker_predicates_use_original_student_message() -> None:
    assert has_explicit_full_solution_request("请给出这道题的完整解答") is True
    assert has_explicit_full_solution_request("请给出答案") is True
    assert has_explicit_full_solution_request("这道题怎么解？") is False
    assert has_no_answer_request("讲讲思路，但不要告诉我答案") is True
    assert has_no_answer_request("讲完整思路，但不要答案") is True
    assert has_explicit_full_solution_request("请完整解答，但不要答案") is False


def test_no_answer_phrase_overrides_model_full_solution_contract() -> None:
    normalized = normalize_teaching_contract(
        _full_solution_contract(),
        "讲讲思路，但不要告诉我答案",
        _math_classification(),
    )

    assert normalized.response_mode is TeachingResponseMode.GUIDED_SOLUTION
    assert normalized.may_reveal_final_answer is False
    assert "最终答案" not in normalized.allowed_content
    assert "最终答案" in normalized.forbidden_content


def test_no_answer_request_wins_when_message_also_says_complete_solution() -> None:
    normalized = normalize_teaching_contract(
        _full_solution_contract(),
        "请完整解答，但不要答案",
        _math_classification(),
    )

    assert normalized.response_mode is TeachingResponseMode.GUIDED_SOLUTION
    assert normalized.may_reveal_final_answer is False


def test_full_solution_requires_explicit_complete_solution_marker() -> None:
    normalized = normalize_teaching_contract(
        _full_solution_contract(),
        "这道题怎么解？",
        _math_classification(),
    )

    assert normalized.response_mode is TeachingResponseMode.ONE_STEP
    assert normalized.may_reveal_final_answer is False
    assert not any("完整" in item or "答案" in item for item in normalized.allowed_content)
    assert "完整解答当前题" in normalized.forbidden_content


def test_normalization_keeps_required_prohibitions_with_full_forbidden_list() -> None:
    proposed = _full_solution_contract().model_copy(
        update={"forbidden_content": ["限制1", "限制2", "限制3", "限制4", "限制5"]}
    )

    normalized = normalize_teaching_contract(
        proposed,
        "这道题怎么解？",
        _math_classification(),
    )

    assert len(normalized.forbidden_content) == 5
    assert "完整解答当前题" in normalized.forbidden_content
    assert "最终答案" in normalized.forbidden_content


def test_explicit_complete_solution_request_keeps_full_solution_permission() -> None:
    proposed = _full_solution_contract()

    normalized = normalize_teaching_contract(
        proposed,
        "请给出这道题的完整解答",
        _math_classification(),
    )

    assert normalized == proposed


def test_off_topic_forces_brief_redirect_without_problem_permissions() -> None:
    classification = RequestClassification(
        scope="off_topic",
        subject=None,
        stage="未知",
        is_new_problem=True,
        answer_mode="brief_redirect",
        reason="not a K9 learning request",
    )

    normalized = normalize_teaching_contract(
        _full_solution_contract(),
        "帮我比较两款企业邮箱",
        classification,
    )

    assert normalized.response_mode is TeachingResponseMode.BRIEF_REDIRECT
    assert normalized.may_apply_to_active_problem is False
    assert normalized.may_reveal_final_answer is False
