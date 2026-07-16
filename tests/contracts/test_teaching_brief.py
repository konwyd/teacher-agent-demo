import pytest

from app.agent.teaching_actions import (
    DisclosureLevel,
    SafetyConstraints,
    TeachingAction,
    TeachingActionPlan,
)
from app.agent.teaching_brief import TeachingBriefComposer


pytestmark = pytest.mark.safety_contract


def test_brief_contains_one_next_step_and_one_closing_task() -> None:
    brief = TeachingBriefComposer().compose(
        problem_context="2x+3=7",
        current_student_request="我不知道下一步",
        verified_facts=[],
        student_evidence=[],
        action_plan=TeachingActionPlan(
            action=TeachingAction.SCAFFOLD,
            scaffold_level=1,
        ),
        constraints=SafetyConstraints(
            allowed_actions=[TeachingAction.SCAFFOLD],
            disclosure_ceiling=DisclosureLevel.NEXT_STEP,
        ),
        next_step_candidates=["先观察常数项", "直接写出最终答案"],
        candidate_conclusion="x=2",
    )

    assert brief.primary_next_step == "先观察常数项"
    assert brief.closing_task
    assert "x=2" in brief.forbidden_disclosures


def test_brief_rejects_plan_outside_constraints() -> None:
    with pytest.raises(ValueError, match="allowed_actions"):
        TeachingBriefComposer().compose(
            problem_context="一个问题",
            current_student_request="请解释",
            verified_facts=[],
            student_evidence=[],
            action_plan=TeachingActionPlan(action=TeachingAction.MODEL),
            constraints=SafetyConstraints(
                allowed_actions=[TeachingAction.EXPLAIN],
            ),
            next_step_candidates=[],
            candidate_conclusion="",
        )
