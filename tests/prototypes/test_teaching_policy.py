from __future__ import annotations

import pytest

from prototypes.llm_led_teacher import teaching_policy
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
    assert has_explicit_full_solution_request("请完整解答，但只讲公式") is False


@pytest.mark.parametrize(
    "message",
    [
        "不要直接告诉我答案",
        "别直接告诉我答案",
        "不用告诉我答案",
        "先不公布最终答案",
    ],
)
def test_negated_disclosure_window_cannot_grant_full_solution(message: str) -> None:
    assert has_no_answer_request(message) is True
    assert has_explicit_full_solution_request(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "不要完整步骤，只给我一个提示",
        "不需要完整解答，只讲思路",
    ],
)
def test_negated_full_solution_request_never_reveals_answer(message: str) -> None:
    normalized = normalize_teaching_contract(
        _full_solution_contract(),
        message,
        _math_classification(),
    )

    assert has_explicit_full_solution_request(message) is False
    assert normalized.response_mode is not TeachingResponseMode.FULL_SOLUTION
    assert normalized.may_reveal_final_answer is False


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


def test_complete_concept_explanation_is_not_a_full_problem_solution() -> None:
    normalized = normalize_teaching_contract(
        _full_solution_contract().model_copy(
            update={
                "allowed_content": ["求根公式", "代入当前题", "最终答案"],
            }
        ),
        "请完整讲解求根公式，但不要代入当前题",
        _math_classification(),
    )

    assert normalized.response_mode is TeachingResponseMode.CONCEPT_EXPLANATION
    assert normalized.may_apply_to_active_problem is False
    assert normalized.may_reveal_final_answer is False
    assert "代入当前题" not in normalized.allowed_content
    assert "最终答案" not in normalized.allowed_content


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


def test_guided_contract_cannot_hide_answer_permission_in_allowed_content() -> None:
    proposed = TeachingContract(
        response_mode="guided_solution",
        requested_target="给学生一个提示",
        allowed_content=["分步提示", "完整解答", "最终答案"],
        forbidden_content=[],
        may_apply_to_active_problem=True,
        may_reveal_final_answer=False,
        reason="model proposed guided mode with contradictory content",
    )

    normalized = normalize_teaching_contract(
        proposed,
        "给我一点提示",
        _math_classification(),
    )

    assert normalized.response_mode is TeachingResponseMode.GUIDED_SOLUTION
    assert normalized.allowed_content == ["分步提示"]
    assert "完整解答当前题" in normalized.forbidden_content
    assert "最终答案" in normalized.forbidden_content


def test_no_apply_intent_cleans_application_permission_and_allowed_content() -> None:
    proposed = TeachingContract(
        response_mode="guided_solution",
        requested_target="解释求根公式",
        allowed_content=["通用公式", "套用本题"],
        forbidden_content=[],
        may_apply_to_active_problem=True,
        may_reveal_final_answer=False,
        reason="model proposed applying the formula",
    )

    normalized = normalize_teaching_contract(
        proposed,
        "只讲公式，不要套用到本题",
        _math_classification(),
    )

    assert normalized.response_mode is TeachingResponseMode.CONCEPT_EXPLANATION
    assert normalized.may_apply_to_active_problem is False
    assert normalized.allowed_content == ["通用公式"]
    assert "代入或套用当前题" in normalized.forbidden_content


def test_explicit_complete_solution_request_keeps_full_solution_permission() -> None:
    proposed = _full_solution_contract()

    normalized = normalize_teaching_contract(
        proposed,
        "请给出这道题的完整解答",
        _math_classification(),
    )

    assert normalized == proposed


def test_explicit_full_solution_with_why_cue_keeps_full_permissions() -> None:
    proposed = _full_solution_contract()

    normalized = normalize_teaching_contract(
        proposed,
        "请完整解答这道题，并解释为什么这样做",
        _math_classification(),
    )

    assert normalized.response_mode is TeachingResponseMode.FULL_SOLUTION
    assert normalized.may_apply_to_active_problem is True
    assert normalized.may_reveal_final_answer is True


def test_exact_explicit_full_solution_with_explanation_remains_authorized() -> None:
    normalized = normalize_teaching_contract(
        _full_solution_contract(),
        "请完整解答并解释为什么",
        _math_classification(),
    )

    assert normalized.response_mode is TeachingResponseMode.FULL_SOLUTION
    assert normalized.may_apply_to_active_problem is True
    assert normalized.may_reveal_final_answer is True


def test_negation_window_does_not_cross_into_later_explicit_request() -> None:
    assert (
        has_explicit_full_solution_request("不需要解释为什么，请完整解答")
        is True
    )


@pytest.mark.parametrize(
    "message",
    [
        "我不想要完整解答，只给一个提示",
        "我不想在还没理清每个符号和条件之前就看到你把全部计算过程展开成完整解答",
        "完整解答就不要了，给我一个提示",
        "完整解答就不要了给提示",
        "完整解答就不要了只给一个提示",
        "请完整解答，但是我改主意了，完整解答就不要了",
    ],
)
def test_last_full_solution_mention_is_negated_clause_aware(message: str) -> None:
    assert has_explicit_full_solution_request(message) is False

    normalized = normalize_teaching_contract(
        _full_solution_contract(),
        message,
        _math_classification(),
    )
    assert normalized.response_mode is not TeachingResponseMode.FULL_SOLUTION
    assert normalized.may_reveal_final_answer is False


@pytest.mark.parametrize(
    "message",
    [
        "不要解释为什么，请完整解答",
        "先不要完整解答。不过我改主意了，请完整解答",
    ],
)
def test_later_positive_full_solution_mention_wins(message: str) -> None:
    assert has_explicit_full_solution_request(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "完整解答不要解释为什么",
        "完整解答不用分很多段",
        "完整解答不要紧，我能看懂",
    ],
)
def test_post_marker_response_preferences_do_not_negate_full_solution(
    message: str,
) -> None:
    assert has_explicit_full_solution_request(message) is True

    normalized = normalize_teaching_contract(
        _full_solution_contract(),
        message,
        _math_classification(),
    )
    assert normalized.response_mode is TeachingResponseMode.FULL_SOLUTION
    assert normalized.may_apply_to_active_problem is True
    assert normalized.may_reveal_final_answer is True


def test_later_positive_mention_wins_within_the_same_clause() -> None:
    assert (
        has_explicit_full_solution_request(
            "不要完整解答我改主意了请完整解答"
        )
        is True
    )


@pytest.mark.parametrize(
    "message",
    [
        "先不要答案，但是请完整解答",
        "不要代入当前题，但是请完整解答",
    ],
)
def test_global_no_answer_and_no_apply_still_override_later_request(
    message: str,
) -> None:
    assert has_explicit_full_solution_request(message) is False


def test_explicit_full_solution_with_no_answer_still_cannot_reveal() -> None:
    normalized = normalize_teaching_contract(
        _full_solution_contract(),
        "请完整解答但不要告诉答案",
        _math_classification(),
    )

    assert normalized.response_mode is not TeachingResponseMode.FULL_SOLUTION
    assert normalized.may_reveal_final_answer is False


def test_explicit_full_solution_with_no_apply_cannot_apply_or_reveal() -> None:
    normalized = normalize_teaching_contract(
        _full_solution_contract(),
        "请完整解答但不要代入当前题",
        _math_classification(),
    )

    assert normalized.response_mode is not TeachingResponseMode.FULL_SOLUTION
    assert normalized.may_apply_to_active_problem is False
    assert normalized.may_reveal_final_answer is False


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


def test_shared_safe_contract_builds_concept_only_fallback() -> None:
    contract = teaching_policy.build_safe_teaching_contract(
        "求根公式是什么？",
        _math_classification(),
    )

    assert contract.response_mode is TeachingResponseMode.CONCEPT_EXPLANATION
    assert contract.may_apply_to_active_problem is False
    assert contract.may_reveal_final_answer is False


def test_shared_fallback_keeps_explicit_full_request_despite_why_cue() -> None:
    contract = teaching_policy.build_safe_teaching_contract(
        "请完整解答这道题，并解释为什么这样做",
        _math_classification(),
    )

    assert contract.response_mode is TeachingResponseMode.FULL_SOLUTION
    assert contract.may_apply_to_active_problem is True
    assert contract.may_reveal_final_answer is True
