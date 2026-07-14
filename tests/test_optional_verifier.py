"""Contract tests for optional, topic-agnostic verifier boundaries."""

from app.agent.optional_verifier import (
    LinearEquationVerifier,
    NoopVerifier,
    VerificationRequest,
)
from app.agent.state import VerificationStatus
from app.agent.teaching_harness import run_generic_teaching_turn
from app.retrieval.context_provider import EmptyGraphContextProvider


def test_noop_verifier_is_explicitly_not_applicable() -> None:
    result = NoopVerifier().verify(
        VerificationRequest(
            problem_context="男生和女生人数的比是3:2，总数30人",
            student_attempt="我先算总份数",
        )
    )

    assert result.status is VerificationStatus.NOT_APPLICABLE
    assert result.verifier_name == "none"


def test_linear_equation_verifier_maps_existing_checker_result() -> None:
    result = LinearEquationVerifier().verify(
        VerificationRequest(
            problem_context="2x+3=7",
            student_attempt="2x=4",
            previous_verified_step="2x+3=7",
        )
    )

    assert result.status is VerificationStatus.VERIFIED_CORRECT
    assert result.verifier_name == "linear_equation_fraction_checker"
    assert result.evidence


def test_linear_equation_verifier_declines_another_math_domain() -> None:
    result = LinearEquationVerifier().verify(
        VerificationRequest(
            problem_context="一个三角形的内角和为什么是180度",
            student_attempt="因为可以拼成平角",
        )
    )

    assert result.status is VerificationStatus.NOT_APPLICABLE


def test_generic_harness_runs_when_all_optional_services_are_empty() -> None:
    state = run_generic_teaching_turn(
        "generic-optional-services",
        "三角形的内角和为什么是180度？",
        verifier=NoopVerifier(),
        graph_context_provider=EmptyGraphContextProvider(),
    )

    assert state.verification.status is VerificationStatus.NOT_APPLICABLE
    assert state.graph_context.available is False
    assert state.response
