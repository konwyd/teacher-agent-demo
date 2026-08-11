"""Deterministic, topic-agnostic teaching-action policy."""

from __future__ import annotations

from app.agent.state import (
    TeachingIntent,
    TeachingState,
    VerificationStatus,
)
from app.agent.teaching_actions import (
    PracticeDifficulty,
    PracticeSimilarity,
    TeachingAction,
    TeachingActionPlan,
)


class TeachingPolicy:
    """Choose the next teaching action without solving or classifying a topic."""

    def choose(self, state: TeachingState) -> TeachingActionPlan:
        state.answer_disclosure_allowed = False
        verification = state.verification.status
        if verification is VerificationStatus.VERIFIED_CORRECT:
            return TeachingActionPlan(
                action=TeachingAction.CHECK,
                rationale="已有明确验证结论，先检查学生是否理解依据。",
            )
        if verification is VerificationStatus.VERIFIED_INCORRECT:
            return TeachingActionPlan(
                action=TeachingAction.CORRECT,
                rationale="已有明确验证结论，先解释可修正的步骤。",
            )
        if not state.problem_context:
            return TeachingActionPlan(
                action=TeachingAction.ELICIT,
                rationale="缺少可教学的问题上下文，先收集题目和已有尝试。",
            )

        intent = state.math_analysis.intent
        if (
            state.math_analysis.needs_clarification
            or state.math_reasoning.needs_clarification
        ):
            return TeachingActionPlan(
                action=TeachingAction.FOCUS,
                rationale="题意或推理不可靠，先澄清。",
            )
        if (
            intent is TeachingIntent.REQUESTS_ANSWER
            and state.math_reasoning.confidence >= 0.8
            and state.math_reasoning.candidate_conclusion
        ):
            state.answer_disclosure_allowed = True
            return TeachingActionPlan(
                action=TeachingAction.EXPLAIN,
                rationale="学生明确请求答案且推理置信度足够。",
            )
        if intent is TeachingIntent.REQUESTS_EXAMPLE:
            return TeachingActionPlan(
                action=TeachingAction.MODEL,
                rationale="学生明确请求示范。",
            )
        if intent is TeachingIntent.REQUESTS_PRACTICE:
            return TeachingActionPlan(
                action=TeachingAction.PRACTICE,
                practice_difficulty=PracticeDifficulty.SIMILAR,
                practice_similarity=PracticeSimilarity.SAME_STRATEGY,
                rationale="学生明确请求练习。",
            )
        if intent is TeachingIntent.ASKS_WHY:
            return TeachingActionPlan(
                action=TeachingAction.EXPLAIN,
                rationale="学生询问推理依据。",
            )
        if state.math_reasoning.next_step_hint or state.math_reasoning.reasoning_steps:
            return TeachingActionPlan(
                action=TeachingAction.SCAFFOLD,
                scaffold_level=1,
                rationale="已有可教学的下一步推理，但未获答案披露授权。",
            )
        if intent is TeachingIntent.UNKNOWN:
            return TeachingActionPlan(
                action=TeachingAction.FOCUS,
                rationale="当前意图或问题信息不足，先诊断。",
            )
        return TeachingActionPlan(
            action=TeachingAction.SCAFFOLD,
            scaffold_level=1,
            rationale="给出不泄露答案的下一步提示。",
        )
