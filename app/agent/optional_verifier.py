"""Optional mathematical-verification contracts for the teaching harness.

The neutral harness can operate without a verifier.  Domain implementations
live behind this module so that a topic-specific checker never becomes a
required dependency of policy or response generation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from app.agent import math_checker as mc
from app.agent.math_checker import StepValidity
from app.agent.state import VerificationResult, VerificationStatus


class VerificationRequest(BaseModel):
    """Context a verifier may use for one student mathematical attempt."""

    problem_context: str = ""
    student_attempt: str = ""
    previous_verified_step: str = ""


@runtime_checkable
class OptionalVerifier(Protocol):
    """A non-blocking source of mathematical evidence."""

    name: str

    def supports(self, request: VerificationRequest) -> bool:
        """Return whether this implementation can evaluate the request."""

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """Return evidence, never forcing a verdict outside its domain."""


class NoopVerifier:
    """Default verifier for ordinary teaching without a domain checker."""

    name = "none"

    def supports(self, request: VerificationRequest) -> bool:  # noqa: ARG002
        return False

    def verify(self, request: VerificationRequest) -> VerificationResult:  # noqa: ARG002
        return VerificationResult(
            status=VerificationStatus.NOT_APPLICABLE,
            summary="No verifier configured for this teaching turn.",
            verifier_name=self.name,
        )


class LinearEquationVerifier:
    """Adapter exposing the existing Fraction-based equation checker.

    This class deliberately delegates to ``math_checker``.  It is not a new
    generic solver and is only an optional implementation of the neutral
    verifier contract.
    """

    name = "linear_equation_fraction_checker"

    def supports(self, request: VerificationRequest) -> bool:
        return mc.extract_equation(request.problem_context) is not None

    def verify(self, request: VerificationRequest) -> VerificationResult:
        target_equation = mc.extract_equation(request.problem_context)
        if target_equation is None:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                summary="The problem is outside the linear-equation verifier domain.",
                verifier_name=self.name,
            )

        result = mc.check_step(
            target_equation,
            request.student_attempt,
            prev_step=request.previous_verified_step or None,
        )
        status = {
            StepValidity.CORRECT_STEP: VerificationStatus.VERIFIED_CORRECT,
            StepValidity.CORRECT_FINAL_ANSWER: VerificationStatus.VERIFIED_CORRECT,
            StepValidity.INCORRECT_STEP: VerificationStatus.VERIFIED_INCORRECT,
            StepValidity.NOT_A_STEP: VerificationStatus.UNABLE_TO_VERIFY,
            StepValidity.UNPARSEABLE: VerificationStatus.UNABLE_TO_VERIFY,
        }[result.validity]
        evidence = [result.explanation] if result.explanation else []
        if result.student_equation:
            evidence.append(result.student_equation)
        return VerificationResult(
            status=status,
            summary=result.validity.value,
            evidence=evidence,
            verifier_name=self.name,
        )
