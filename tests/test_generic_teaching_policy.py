"""Contract tests for topic-agnostic policy and response generation."""

from app.agent.math_analysis import analyze_turn
from app.agent.response_generation import generate_response
from app.agent.state import (
    TeachingActionKind,
    TeachingIntent,
    TeachingState,
    VerificationResult,
    VerificationStatus,
)
from app.agent.teaching_policy import TeachingPolicy


def test_unverified_problem_uses_a_hint_without_disclosing_an_answer() -> None:
    state = TeachingState(
        student_id="ratio-student",
        message="男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
        problem_context="男生和女生人数的比是3:2，总数30人。",
        math_analysis=analyze_turn(
            "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
            "男生和女生人数的比是3:2，总数30人。",
        ),
    )
    state.action_plan = TeachingPolicy().choose(state)
    response = generate_response(state)

    assert state.action_plan.action is TeachingActionKind.GIVE_HINT
    assert state.answer_disclosure_allowed is False
    assert "18" not in response
    assert "12" not in response
    assert response.count("？") == 1
    assert response.endswith("？")


def test_verified_result_precedes_generic_mastery_signals() -> None:
    state = TeachingState(
        student_id="verified-student",
        message="我这样算对吗？",
        problem_context="某个数学问题",
        verification=VerificationResult(status=VerificationStatus.VERIFIED_CORRECT),
    )

    action = TeachingPolicy().choose(state)

    assert action.action is TeachingActionKind.CHECK_UNDERSTANDING


def test_example_request_is_an_explicit_action() -> None:
    state = TeachingState(
        student_id="example-student",
        message="能给我举个例子吗？",
        problem_context="三角形内角和",
        math_analysis={"intent": TeachingIntent.REQUESTS_EXAMPLE},
    )

    action = TeachingPolicy().choose(state)

    assert action.action is TeachingActionKind.SHOW_WORKED_EXAMPLE
    assert action.allow_worked_example is True
