"""Trusted teaching-contract policy derived from the original student message."""

from __future__ import annotations

from prototypes.llm_led_teacher.orchestration_models import (
    RequestClassification,
    RequestScope,
    TeachingContract,
    TeachingResponseMode,
)


_NO_ANSWER_MARKERS = (
    "不要答案",
    "别给答案",
    "不要给出答案",
    "别给出答案",
    "不要告诉我答案",
    "别告诉我答案",
    "不要直接给答案",
    "先别给答案",
)

_FULL_SOLUTION_REQUEST_MARKERS = (
    "给出答案",
    "告诉我答案",
    "直接告诉我答案",
    "直接给答案",
    "只要答案",
    "完整答案",
    "完整步骤",
    "全部步骤",
    "完整解答",
    "从头到尾",
    "完整讲解",
)

_GENERIC_SOLVE_REQUEST_MARKERS = (
    "怎么解",
    "如何解",
    "怎样解",
    "求解",
    "解方程",
    "帮我解",
)


def has_explicit_full_solution_request(message: str) -> bool:
    """Return whether the student's own message clearly asks for a full answer."""

    return not has_no_answer_request(message) and any(
        marker in message for marker in _FULL_SOLUTION_REQUEST_MARKERS
    )


def has_no_answer_request(message: str) -> bool:
    """Return whether the student's own message explicitly forbids answer disclosure."""

    return any(marker in message for marker in _NO_ANSWER_MARKERS)


def _is_generic_solve_request(message: str) -> bool:
    return any(marker in message for marker in _GENERIC_SOLVE_REQUEST_MARKERS)


def _without_answer_permissions(
    contract: TeachingContract,
    response_mode: TeachingResponseMode,
) -> TeachingContract:
    allowed_content = [
        item
        for item in contract.allowed_content
        if not any(
            marker in item for marker in ("答案", "完整", "全部步骤", "从头到尾")
        )
    ]
    required_prohibitions = ["完整解答当前题", "最终答案"]
    existing_prohibitions = [
        item
        for item in contract.forbidden_content
        if item not in required_prohibitions
    ]
    forbidden_content = existing_prohibitions[:3] + required_prohibitions
    return contract.model_copy(
        update={
            "response_mode": response_mode,
            "allowed_content": allowed_content,
            "forbidden_content": forbidden_content,
            "may_reveal_final_answer": False,
        }
    )


def normalize_teaching_contract(
    contract: TeachingContract,
    student_message: str,
    classification: RequestClassification,
) -> TeachingContract:
    """Normalize model-proposed permissions using only trusted request evidence."""

    if classification.scope is RequestScope.OFF_TOPIC:
        return TeachingContract(
            response_mode=TeachingResponseMode.BRIEF_REDIRECT,
            requested_target="简短引导学生回到九年义务教育学习问题",
            allowed_content=["简短范围说明", "回到 K9 学习的邀请"],
            forbidden_content=["展开题外内容", "套用当前题", "披露最终答案"],
            may_apply_to_active_problem=False,
            may_reveal_final_answer=False,
            reason="off-topic requests must use a brief redirect",
        )

    no_answer = has_no_answer_request(student_message)
    explicit_full_solution = has_explicit_full_solution_request(student_message)
    if contract.response_mode is TeachingResponseMode.FULL_SOLUTION and (
        no_answer or not explicit_full_solution
    ):
        safe_mode = (
            TeachingResponseMode.ONE_STEP
            if _is_generic_solve_request(student_message) and not no_answer
            else TeachingResponseMode.GUIDED_SOLUTION
        )
        return _without_answer_permissions(contract, safe_mode)

    if no_answer:
        return _without_answer_permissions(contract, contract.response_mode)

    return contract
