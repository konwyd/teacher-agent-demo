"""Deterministic safety constraints for topic-agnostic teaching decisions."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.state import TeachingIntent, VerificationStatus
from app.agent.teaching_actions import (
    DisclosureLevel,
    SafetyConstraints,
    TeachingAction,
)


_TOTAL_CONFUSION_MARKERS = ("我完全不懂", "我一点都不会", "我完全不会做")
_HARMLESS_END_PUNCTUATION = "。.!！?？…"


class SafetyInput(BaseModel):
    """Evidence and intent signals that may restrict a teaching turn."""

    problem_context: str = ""
    intent: TeachingIntent = TeachingIntent.UNKNOWN
    verification_status: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    candidate_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate_available: bool = False
    consecutive_verified_errors: int = Field(default=0, ge=0)
    has_verified_error: bool = False
    has_confirmed_progress: bool = False
    expresses_total_confusion: bool = False

    @classmethod
    def from_turn(cls, *, problem_context: str, message: str) -> SafetyInput:
        normalized_message = message.strip().rstrip(_HARMLESS_END_PUNCTUATION)
        return cls(
            problem_context=problem_context,
            expresses_total_confusion=normalized_message in _TOTAL_CONFUSION_MARKERS,
        )


class SafetyGate:
    """Pure restriction rules applied before a teaching action is selected."""

    def evaluate(self, context: SafetyInput) -> SafetyConstraints:
        if not context.problem_context.strip():
            return SafetyConstraints(
                allowed_actions=[TeachingAction.ELICIT],
                disclosure_ceiling=DisclosureLevel.NONE,
                reason_codes=["missing_problem_context"],
            )

        example_allowed = (
            context.intent is TeachingIntent.REQUESTS_EXAMPLE
            or context.consecutive_verified_errors >= 2
            or context.expresses_total_confusion
        )

        if context.verification_status is VerificationStatus.VERIFIED_INCORRECT:
            allowed = [
                TeachingAction.FOCUS,
                TeachingAction.SCAFFOLD,
                TeachingAction.CORRECT,
                TeachingAction.EXPLAIN,
            ]
        elif context.verification_status is VerificationStatus.VERIFIED_CORRECT:
            allowed = [
                TeachingAction.CHECK,
                TeachingAction.EXPLAIN,
                TeachingAction.CONSOLIDATE,
                TeachingAction.PRACTICE,
            ]
        elif context.intent is TeachingIntent.REQUESTS_PRACTICE:
            allowed = [TeachingAction.PRACTICE, TeachingAction.CHECK]
        else:
            allowed = [
                TeachingAction.FOCUS,
                TeachingAction.SCAFFOLD,
                TeachingAction.EXPLAIN,
                TeachingAction.CHECK,
            ]

        if example_allowed:
            allowed.append(TeachingAction.MODEL)

        disclosure = DisclosureLevel.NEXT_STEP
        answer_authorized = (
            context.intent is TeachingIntent.REQUESTS_ANSWER
            and context.candidate_available
            and context.candidate_confidence >= 0.8
        )
        if answer_authorized:
            disclosure = DisclosureLevel.FINAL

        new_problem_allowed = (
            context.intent is TeachingIntent.REQUESTS_PRACTICE
            or context.verification_status is VerificationStatus.VERIFIED_CORRECT
        )

        reason_codes = [f"verifier_{context.verification_status.value}"]
        if context.intent is TeachingIntent.REQUESTS_EXAMPLE:
            reason_codes.append("example_requested")
        if context.consecutive_verified_errors >= 2:
            reason_codes.append("example_after_two_verified_errors")
        if context.expresses_total_confusion:
            reason_codes.append("example_after_total_confusion")
        reason_codes.append(
            "new_problem_allowed" if new_problem_allowed else "new_problem_blocked"
        )
        reason_codes.append(
            "answer_authorized" if answer_authorized else "final_disclosure_blocked"
        )

        return SafetyConstraints(
            allowed_actions=allowed,
            disclosure_ceiling=disclosure,
            worked_example_allowed=example_allowed,
            new_problem_allowed=new_problem_allowed,
            student_response_required=True,
            reason_codes=reason_codes,
        )
