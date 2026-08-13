"""Subject-agnostic semantic auditing for teaching-contract boundaries."""

from __future__ import annotations

import json
import unicodedata

from app.llm.deepseek_client import DeepSeekClient
from prototypes.llm_led_teacher.orchestration_models import (
    BoundaryAudit,
    BoundaryViolation,
    TeachingContract,
)


_AUDIT_SYSTEM_PROMPT = """你是跨学科教学回答的语义边界审计器。
你只判断回答草稿是否遵守给定教学契约规定的范围与披露边界，包括目标、当前题应用权限、最终答案权限和步骤限制。
不要要求、推断或输出模型的隐藏推理，也不要把本审计说成对所有学科通用事实正确性的验证。
请基于公开草稿给出可复核的简短证据；若越界，请给出可执行的保守修复指令。
只有草稿未违反教学契约时 passed 才能为 true。"""

_CONSERVATIVE_REPAIR = "边界检查不可用，按教学契约保守重写回答。"

_FINAL_MARKERS = (
    "最终答案",
    "披露最终答案",
    "完整解答",
    "完整求解",
)
_APPLICATION_MARKERS = (
    "代入",
    "套用",
    "应用当前题",
    "应用本题",
    "计算当前题",
    "解答当前题",
    "解当前题",
)
_MULTI_STEP_MARKERS = (
    "完整解答",
    "完整求解",
    "全部步骤",
    "多个步骤",
    "多步",
    "从头到尾",
)
_NO_ANSWER_MARKERS = ("不要答案", "别给答案", "不披露答案")
_OFF_TARGET_MARKERS = ("题外", "偏题", "展开题外内容")


class SemanticBoundaryAuditor:
    """Ask DeepSeek to judge disclosure scope without claiming fact checking."""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def audit(
        self,
        student_message: str,
        problem_context: str,
        contract: TeachingContract,
        draft: str,
    ) -> BoundaryAudit:
        payload = {
            "student_message": student_message,
            "problem_context": problem_context,
            "contract": contract.model_dump(mode="json"),
            "draft": draft,
        }
        try:
            result = self.client.chat_json(
                system=_AUDIT_SYSTEM_PROMPT,
                user=json.dumps(payload, ensure_ascii=False),
                schema=BoundaryAudit,
                temperature=0.0,
                max_tokens=320,
            )
            if result.passed and (result.violations or result.audit_error):
                raise ValueError("passed audit cannot contain violations or audit_error")
            return result
        except Exception as exc:  # noqa: BLE001 - every audit outage fails closed
            return safe_boundary_fallback(contract, exc)


def safe_boundary_fallback(
    contract: TeachingContract,
    error: Exception,
) -> BoundaryAudit:
    """Block an unaudited draft without inventing evidence or violations."""

    del contract  # The conservative result is intentionally contract-independent.
    return BoundaryAudit(
        passed=False,
        evidence=[],
        repair_instructions=[_CONSERVATIVE_REPAIR],
        audit_error=type(error).__name__,
    )


def hard_boundary_recheck(
    problem_context: str,
    contract: TeachingContract,
    response: str,
) -> list[BoundaryViolation]:
    """Find only literal high-confidence leaks after a semantic repair.

    This is not a solver or a universal factual verifier. ``problem_context``
    is an untyped string and is never treated as trusted answer provenance.
    """

    del problem_context
    normalized_response = _normalize_public_text(response)
    violations: list[BoundaryViolation] = []

    for marker in contract.forbidden_content:
        normalized_marker = _normalize_public_text(marker)
        if normalized_marker and normalized_marker in normalized_response:
            _append_unique(
                violations,
                _violation_for_forbidden_marker(marker, contract),
            )

    return violations


def _normalize_public_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _violation_for_forbidden_marker(
    marker: str,
    contract: TeachingContract,
) -> BoundaryViolation:
    normalized_marker = _normalize_public_text(marker)
    if not contract.may_reveal_final_answer and any(
        _normalize_public_text(token) in normalized_marker for token in _FINAL_MARKERS
    ):
        return BoundaryViolation.FINAL_ANSWER_LEAK
    if any(
        _normalize_public_text(token) in normalized_marker
        for token in _MULTI_STEP_MARKERS
    ):
        return BoundaryViolation.TOO_MANY_STEPS
    if not contract.may_apply_to_active_problem and any(
        _normalize_public_text(token) in normalized_marker
        for token in _APPLICATION_MARKERS
    ):
        return BoundaryViolation.APPLIED_WITHOUT_PERMISSION
    if any(
        _normalize_public_text(token) in normalized_marker
        for token in _NO_ANSWER_MARKERS
    ):
        return BoundaryViolation.IGNORED_NO_ANSWER
    if any(
        _normalize_public_text(token) in normalized_marker
        for token in _OFF_TARGET_MARKERS
    ):
        return BoundaryViolation.OFF_TARGET
    return BoundaryViolation.OFF_TARGET


def _append_unique(
    violations: list[BoundaryViolation],
    violation: BoundaryViolation,
) -> None:
    if violation not in violations:
        violations.append(violation)
