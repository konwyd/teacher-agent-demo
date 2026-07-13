"""Deterministic, topic-agnostic teaching-action policy."""

from __future__ import annotations

from app.agent.state import (
    TeachingActionKind,
    TeachingActionPlan,
    TeachingIntent,
    TeachingState,
    VerificationStatus,
)


class TeachingPolicy:
    """Choose the next teaching action without solving or classifying a topic."""

    def choose(self, state: TeachingState) -> TeachingActionPlan:
        verification = state.verification.status
        if verification is VerificationStatus.VERIFIED_CORRECT:
            return TeachingActionPlan(
                action=TeachingActionKind.CHECK_UNDERSTANDING,
                rationale="已有明确验证结论，先检查学生是否理解依据。",
            )
        if verification is VerificationStatus.VERIFIED_INCORRECT:
            return TeachingActionPlan(
                action=TeachingActionKind.EXPLAIN_STEP,
                rationale="已有明确验证结论，先解释可修正的步骤。",
            )
        if not state.problem_context:
            return TeachingActionPlan(
                action=TeachingActionKind.ASSESS,
                rationale="缺少可教学的问题上下文，先收集题目和已有尝试。",
            )

        intent = state.math_analysis.intent
        if intent is TeachingIntent.REQUESTS_EXAMPLE:
            return TeachingActionPlan(
                action=TeachingActionKind.SHOW_WORKED_EXAMPLE,
                allow_worked_example=True,
                rationale="学生明确请求示范。",
            )
        if intent is TeachingIntent.REQUESTS_PRACTICE:
            return TeachingActionPlan(
                action=TeachingActionKind.OFFER_PRACTICE,
                rationale="学生明确请求练习。",
            )
        if intent is TeachingIntent.ASKS_WHY:
            return TeachingActionPlan(
                action=TeachingActionKind.EXPLAIN_STEP,
                rationale="学生询问推理依据。",
            )
        if state.math_analysis.needs_clarification or intent is TeachingIntent.UNKNOWN:
            return TeachingActionPlan(
                action=TeachingActionKind.DIAGNOSE,
                rationale="当前意图或问题信息不足，先诊断。",
            )
        return TeachingActionPlan(
            action=TeachingActionKind.GIVE_HINT,
            hint_level=1,
            rationale="给出不泄露答案的下一步提示。",
        )
