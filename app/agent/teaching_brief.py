"""Policy-approved input boundary for teacher response generation."""

from __future__ import annotations

from pydantic import BaseModel

from app.agent.teaching_actions import (
    DisclosureLevel,
    SafetyConstraints,
    TeachingAction,
    TeachingActionPlan,
)


_CLOSING_TASKS = {
    TeachingAction.ELICIT: "你能先把完整题目和已经想到的一步发给我吗？",
    TeachingAction.FOCUS: "你愿意先指出自己最不确定的那一处吗？",
    TeachingAction.SCAFFOLD: "你能按这个提示先写出下一小步吗？",
    TeachingAction.EXPLAIN: "你能用自己的话复述这一步的依据吗？",
    TeachingAction.CORRECT: "你愿意先重新检查这处变化吗？",
    TeachingAction.CHECK: "你能说明这个结论为什么符合原题条件吗？",
    TeachingAction.MODEL: "你能把示范中的方法对应回原题第一步吗？",
    TeachingAction.PRACTICE: "你愿意先独立完成这道练习的第一步吗？",
    TeachingAction.CONSOLIDATE: "你能说出下次遇到同类关系时先看什么吗？",
}


class TeachingBrief(BaseModel):
    problem_context: str
    current_student_request: str
    verified_facts: list[str]
    student_evidence: list[str]
    action_plan: TeachingActionPlan
    disclosure_ceiling: DisclosureLevel
    primary_next_step: str
    allowed_content: list[str]
    forbidden_disclosures: list[str]
    closing_task: str


class TeachingBriefComposer:
    """Compress deterministic policy output into the response-safe boundary."""

    def compose(
        self,
        *,
        problem_context: str,
        current_student_request: str,
        verified_facts: list[str],
        student_evidence: list[str],
        action_plan: TeachingActionPlan,
        constraints: SafetyConstraints,
        next_step_candidates: list[str],
        candidate_conclusion: str,
    ) -> TeachingBrief:
        if action_plan.action not in constraints.allowed_actions:
            raise ValueError("action_plan.action must be in constraints.allowed_actions")

        primary_next_step = next(
            (candidate.strip() for candidate in next_step_candidates if candidate.strip()),
            "",
        )
        allowed_content = [
            item.strip()
            for item in [*verified_facts, *student_evidence, primary_next_step]
            if item.strip()
        ]
        forbidden_disclosures: list[str] = []
        conclusion = candidate_conclusion.strip()
        if conclusion:
            if constraints.disclosure_ceiling is DisclosureLevel.FINAL:
                allowed_content.append(conclusion)
            else:
                forbidden_disclosures.append(conclusion)

        return TeachingBrief(
            problem_context=problem_context,
            current_student_request=current_student_request,
            verified_facts=list(verified_facts),
            student_evidence=list(student_evidence),
            action_plan=action_plan.model_copy(deep=True),
            disclosure_ceiling=constraints.disclosure_ceiling,
            primary_next_step=primary_next_step,
            allowed_content=allowed_content,
            forbidden_disclosures=forbidden_disclosures,
            closing_task=_CLOSING_TASKS[action_plan.action],
        )
