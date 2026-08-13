"""Trusted teaching-contract policy derived from the original student message."""

from __future__ import annotations

import re

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
)

_GENERIC_SOLVE_REQUEST_MARKERS = (
    "怎么解",
    "如何解",
    "怎样解",
    "求解",
    "解方程",
    "帮我解",
)

_CONCEPT_ONLY_MARKERS = (
    "只讲公式",
    "只讲概念",
    "只解释公式",
    "只解释概念",
    "只说明公式",
    "只说明概念",
)

_NEGATED_ANSWER_PATTERN = re.compile(
    r"(?:不要|别|不用|无需|不必|先不|暂时不).{0,12}(?:答案|最终结果)"
)
_NEGATED_APPLICATION_PATTERN = re.compile(
    r"(?:不要|别|不用|无需|不必|先不|暂时不).{0,12}"
    r"(?:"
    r"(?:代入|套用|应用|用于|计算)(?:到|在|于)?(?:当前题|本题)?"
    r"|解答(?:当前题|本题)"
    r")"
)
_CONCEPT_EXPLANATION_PATTERN = re.compile(
    r"(?:解释|讲解|讲讲|说明).{0,10}(?:公式|概念|定义|原理)"
)

_ANSWER_CONTENT_MARKERS = (
    "答案",
    "完整",
    "全部步骤",
    "从头到尾",
    "当前题的根",
)
_APPLICATION_CONTENT_MARKERS = (
    "代入",
    "套用",
    "应用当前题",
    "应用本题",
    "计算当前题",
    "解答当前题",
    "解当前题",
)
_INTENT_CLAUSE_BOUNDARY = re.compile(r"(?:[\n，。；！？,;!?]+|但是|不过|然而|但)")
_FULL_SOLUTION_NEGATION_MARKERS = (
    "不要",
    "别",
    "不用",
    "无需",
    "不必",
    "不需要",
    "不想",
    "不愿",
    "先不",
    "暂时不",
)
_FULL_SOLUTION_MENTION_PATTERN = re.compile(
    "|".join(
        re.escape(marker)
        for marker in sorted(
            _FULL_SOLUTION_REQUEST_MARKERS,
            key=len,
            reverse=True,
        )
    )
)
_FULL_SOLUTION_POST_NEGATION_PATTERN = re.compile(
    r"(?:(?:就)?(?:不要|不需要|不想要|不愿要|不必|免了)(?:了)?(?:吧)?"
    r"|(?:就)?算了(?:吧)?)$"
)


def _intent_clauses(message: str) -> tuple[str, ...]:
    return tuple(
        clause.strip()
        for clause in _INTENT_CLAUSE_BOUNDARY.split(message)
        if clause.strip()
    )


def _last_full_solution_intent(message: str) -> bool | None:
    last_intent: bool | None = None
    for clause in _intent_clauses(message):
        mentions = list(_FULL_SOLUTION_MENTION_PATTERN.finditer(clause))
        if not mentions:
            continue
        for index, mention in enumerate(mentions):
            scope_start = mentions[index - 1].end() if index else 0
            scope_end = (
                mentions[index + 1].start()
                if index + 1 < len(mentions)
                else len(clause)
            )
            prefix = clause[scope_start : mention.start()]
            suffix = clause[mention.end() : scope_end].strip()
            negated = any(
                marker in prefix
                for marker in _FULL_SOLUTION_NEGATION_MARKERS
            ) or bool(_FULL_SOLUTION_POST_NEGATION_PATTERN.search(suffix))
            last_intent = not negated
    return last_intent


def has_explicit_full_solution_request(message: str) -> bool:
    """Return whether the student's own message clearly asks for a full answer."""

    return (
        not has_no_answer_request(message)
        and not _has_no_apply_request(message)
        and _last_full_solution_intent(message) is True
    )


def has_no_answer_request(message: str) -> bool:
    """Return whether the student's own message explicitly forbids answer disclosure."""

    return any(
        bool(_NEGATED_ANSWER_PATTERN.search(clause))
        or any(marker in clause for marker in _NO_ANSWER_MARKERS)
        for clause in _intent_clauses(message)
    )


def _is_generic_solve_request(message: str) -> bool:
    return any(marker in message for marker in _GENERIC_SOLVE_REQUEST_MARKERS)


def _has_no_apply_request(message: str) -> bool:
    return any(
        bool(_NEGATED_APPLICATION_PATTERN.search(clause))
        or any(marker in clause for marker in _CONCEPT_ONLY_MARKERS)
        for clause in _intent_clauses(message)
    )


def is_concept_request(message: str) -> bool:
    return (
        any(marker in message for marker in ("是什么", "什么意思", "概念", "定义", "原理", "为什么"))
        or any(marker in message for marker in _CONCEPT_ONLY_MARKERS)
        or bool(_CONCEPT_EXPLANATION_PATTERN.search(message))
    )


def _has_boundary(items: list[str], markers: tuple[str, ...]) -> bool:
    return any(marker in item for item in items for marker in markers)


def _append_required_prohibitions(
    existing: list[str],
    required: list[str],
) -> list[str]:
    preserved = [item for item in existing if item not in required]
    return preserved[: 5 - len(required)] + required


def _normalize_permissions(
    contract: TeachingContract,
    *,
    response_mode: TeachingResponseMode,
    may_apply: bool,
    may_reveal: bool,
) -> TeachingContract:
    allowed_content = list(contract.allowed_content)
    required_prohibitions: list[str] = []
    if not may_reveal:
        allowed_content = [
            item
            for item in allowed_content
            if not any(marker in item for marker in _ANSWER_CONTENT_MARKERS)
        ]
        if not _has_boundary(
            contract.forbidden_content,
            ("答案", "完整解答", "完整求解", "当前题的根"),
        ):
            required_prohibitions.extend(["完整解答当前题", "最终答案"])
    if not may_apply:
        allowed_content = [
            item
            for item in allowed_content
            if not any(marker in item for marker in _APPLICATION_CONTENT_MARKERS)
        ]
        if not _has_boundary(
            contract.forbidden_content,
            ("代入", "套用", "应用当前题", "应用本题", "当前题的根"),
        ):
            required_prohibitions.append("代入或套用当前题")
    forbidden_content = _append_required_prohibitions(
        contract.forbidden_content,
        required_prohibitions,
    )
    return contract.model_copy(
        update={
            "response_mode": response_mode,
            "allowed_content": allowed_content,
            "forbidden_content": forbidden_content,
            "may_apply_to_active_problem": may_apply,
            "may_reveal_final_answer": may_reveal,
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
    no_apply = _has_no_apply_request(student_message)
    concept_request = is_concept_request(student_message)

    response_mode = contract.response_mode
    if no_apply:
        response_mode = TeachingResponseMode.CONCEPT_EXPLANATION
    elif (
        response_mode is TeachingResponseMode.FULL_SOLUTION
        and explicit_full_solution
        and not no_answer
    ):
        response_mode = TeachingResponseMode.FULL_SOLUTION
    elif concept_request:
        response_mode = TeachingResponseMode.CONCEPT_EXPLANATION
    elif response_mode is TeachingResponseMode.FULL_SOLUTION and (
        no_answer or not explicit_full_solution
    ):
        response_mode = (
            TeachingResponseMode.ONE_STEP
            if _is_generic_solve_request(student_message) and not no_answer
            else TeachingResponseMode.GUIDED_SOLUTION
        )

    may_reveal = bool(
        contract.may_reveal_final_answer
        and response_mode is TeachingResponseMode.FULL_SOLUTION
        and explicit_full_solution
        and not no_answer
    )
    may_apply = bool(
        contract.may_apply_to_active_problem
        and response_mode
        not in {
            TeachingResponseMode.CONCEPT_EXPLANATION,
            TeachingResponseMode.BRIEF_REDIRECT,
        }
        and not no_apply
    )
    return _normalize_permissions(
        contract,
        response_mode=response_mode,
        may_apply=may_apply,
        may_reveal=may_reveal,
    )


def build_safe_teaching_contract(
    student_message: str,
    classification: RequestClassification,
) -> TeachingContract:
    """Build a conservative fallback contract from trusted request evidence."""

    target = student_message.strip()[:300] or "当前学习问题"
    if classification.scope is RequestScope.OFF_TOPIC:
        response_mode = TeachingResponseMode.BRIEF_REDIRECT
        may_apply = False
        allowed_content = ["简短范围说明", "回到 K9 学习的邀请"]
    elif _has_no_apply_request(student_message):
        response_mode = TeachingResponseMode.CONCEPT_EXPLANATION
        may_apply = False
        allowed_content = ["概念定义", "通用解释"]
    elif has_explicit_full_solution_request(student_message):
        response_mode = TeachingResponseMode.FULL_SOLUTION
        may_apply = True
        allowed_content = ["完整解答", "最终答案", "步骤原因说明"]
    elif is_concept_request(student_message):
        response_mode = TeachingResponseMode.CONCEPT_EXPLANATION
        may_apply = False
        allowed_content = ["概念定义", "通用解释"]
    elif _is_generic_solve_request(student_message):
        response_mode = TeachingResponseMode.ONE_STEP
        may_apply = True
        allowed_content = ["必要概念", "一个有意义的步骤"]
    else:
        response_mode = TeachingResponseMode.GUIDED_SOLUTION
        may_apply = False
        allowed_content = ["必要概念", "一般性提示"]
    proposed = TeachingContract(
        response_mode=response_mode,
        requested_target=target,
        allowed_content=allowed_content,
        forbidden_content=[],
        may_apply_to_active_problem=may_apply,
        may_reveal_final_answer=(
            response_mode is TeachingResponseMode.FULL_SOLUTION
        ),
        reason="safe fallback derived from the original student message",
    )
    return normalize_teaching_contract(proposed, student_message, classification)
