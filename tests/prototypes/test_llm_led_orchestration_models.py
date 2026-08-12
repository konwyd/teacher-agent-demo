from __future__ import annotations

import pytest
from pydantic import ValidationError

from prototypes.llm_led_teacher.orchestration_models import (
    AgentDecision,
    AgentOrchestrationResult,
    AnswerMode,
    BoundaryAudit,
    RequestClassification,
    TeachingContract,
    TeachingResponseMode,
)


def test_k9_classification_requires_supported_subject() -> None:
    with pytest.raises(ValidationError):
        RequestClassification(
            scope="k9",
            subject="计算机",
            stage="初中",
            is_new_problem=True,
            answer_mode="guided",
            reason="invalid subject",
        )


def test_off_topic_classification_forces_brief_redirect() -> None:
    item = RequestClassification(
        scope="off_topic",
        subject=None,
        stage="未知",
        is_new_problem=True,
        answer_mode="brief_redirect",
        reason="not K9",
    )

    assert item.subject is None
    assert item.answer_mode is AnswerMode.BRIEF_REDIRECT


@pytest.mark.parametrize(
    "subject", ["数学", "语文", "英语", "物理", "化学", "生物", "历史", "地理", "道法"]
)
def test_all_supported_k9_subjects_are_valid(subject: str) -> None:
    item = RequestClassification(
        scope="k9",
        subject=subject,
        stage="初中",
        is_new_problem=True,
        answer_mode="guided",
        reason="supported subject",
    )

    assert item.subject is not None
    assert item.subject.value == subject


@pytest.mark.parametrize(
    ("subject", "answer_mode"),
    [("数学", "brief_redirect"), (None, "guided")],
)
def test_off_topic_rejects_subject_or_non_redirect_mode(
    subject: str | None,
    answer_mode: str,
) -> None:
    with pytest.raises(ValidationError):
        RequestClassification(
            scope="off_topic",
            subject=subject,
            stage="未知",
            is_new_problem=True,
            answer_mode=answer_mode,
            reason="invalid off-topic shape",
        )


def test_concept_contract_cannot_reveal_or_apply_to_active_problem() -> None:
    with pytest.raises(ValidationError):
        TeachingContract(
            response_mode="concept_explanation",
            requested_target="解释求根公式",
            allowed_content=["给出通用公式"],
            forbidden_content=[],
            may_apply_to_active_problem=True,
            may_reveal_final_answer=False,
            reason="concept only",
        )


def test_only_full_solution_may_reveal_final_answer() -> None:
    with pytest.raises(ValidationError):
        TeachingContract(
            response_mode="guided_solution",
            requested_target="引导解题",
            may_reveal_final_answer=True,
            reason="invalid disclosure",
        )


def test_finish_decision_requires_teaching_contract() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(kind="finish", reason="done")


def test_full_solution_contract_may_reveal_final_answer() -> None:
    contract = TeachingContract(
        response_mode="full_solution",
        requested_target="完整解答当前题",
        may_apply_to_active_problem=True,
        may_reveal_final_answer=True,
        reason="student requested a complete solution",
    )

    assert contract.response_mode is TeachingResponseMode.FULL_SOLUTION
    assert contract.may_reveal_final_answer is True


def test_orchestration_result_without_contract_remains_legacy_compatible() -> None:
    result = AgentOrchestrationResult(
        classification=RequestClassification(
            scope="k9",
            subject="数学",
            stage="初中",
            is_new_problem=False,
            answer_mode="guided",
            reason="legacy result",
        )
    )

    assert result.teaching_contract is None


def test_boundary_audit_serializes_violation_enum_values() -> None:
    audit = BoundaryAudit(
        passed=False,
        violations=["final_answer_leak", "too_many_steps"],
        evidence=["draft exposed the final answer"],
        repair_instructions=["remove the final answer"],
    )

    assert audit.model_dump(mode="json")["violations"] == [
        "final_answer_leak",
        "too_many_steps",
    ]
