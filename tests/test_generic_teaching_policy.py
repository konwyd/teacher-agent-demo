"""Contract tests for topic-agnostic policy and response generation."""

from app.agent.math_analysis import analyze_turn
from app.agent.response_generation import generate_response
from app.agent.state import (
    GenericMathReasoning,
    MathAnalysis,
    TeachingIntent,
    TeachingState,
    VerificationResult,
    VerificationStatus,
)
from app.agent.teaching_actions import (
    DisclosureLevel,
    SafetyConstraints,
    TeachingAction,
)
from app.agent.teaching_brief import TeachingBriefComposer
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
    brief = TeachingBriefComposer().compose(
        problem_context=state.problem_context,
        current_student_request=state.message,
        verified_facts=[],
        student_evidence=[],
        action_plan=state.action_plan,
        constraints=SafetyConstraints(
            allowed_actions=[state.action_plan.action],
            disclosure_ceiling=DisclosureLevel.NEXT_STEP,
        ),
        next_step_candidates=[state.math_reasoning.next_step_hint],
        candidate_conclusion=state.math_reasoning.candidate_conclusion,
    )
    response = generate_response(brief)

    assert state.action_plan.action is TeachingAction.SCAFFOLD
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

    assert action.action is TeachingAction.CHECK


def test_example_request_is_an_explicit_action() -> None:
    state = TeachingState(
        student_id="example-student",
        message="能给我举个例子吗？",
        problem_context="三角形内角和",
        math_analysis={"intent": TeachingIntent.REQUESTS_EXAMPLE},
    )

    action = TeachingPolicy().choose(state)

    assert action.action is TeachingAction.MODEL


def test_explicit_answer_request_allows_high_confidence_candidate_only() -> None:
    state = TeachingState(
        student_id="ratio-student",
        message="直接告诉我完整答案。",
        problem_context="男生和女生人数的比是3:2，总数30人。",
        math_analysis=MathAnalysis(intent=TeachingIntent.REQUESTS_ANSWER),
        math_reasoning=GenericMathReasoning(
            candidate_conclusion="男生18人，女生12人。",
            confidence=0.8,
        ),
    )

    action = TeachingPolicy().choose(state)

    assert action.action is TeachingAction.EXPLAIN
    assert state.answer_disclosure_allowed is True


def test_hint_request_never_allows_candidate_conclusion() -> None:
    state = TeachingState(
        student_id="ratio-student",
        message="给我一个提示。",
        problem_context="男生和女生人数的比是3:2，总数30人。",
        math_reasoning=GenericMathReasoning(
            next_step_hint="先求总份数。",
            candidate_conclusion="男生18人，女生12人。",
            confidence=0.99,
        ),
    )

    action = TeachingPolicy().choose(state)

    assert action.action is TeachingAction.SCAFFOLD
    assert state.answer_disclosure_allowed is False


def test_non_answer_action_revokes_stale_answer_disclosure() -> None:
    state = TeachingState(
        student_id="ratio-student",
        message="再给我一个提示。",
        problem_context="男生和女生人数的比是3:2，总数30人。",
        answer_disclosure_allowed=True,
        math_reasoning=GenericMathReasoning(
            next_step_hint="先求总份数。",
            candidate_conclusion="男生18人，女生12人。",
            confidence=0.99,
        ),
    )

    action = TeachingPolicy().choose(state)

    assert action.action is TeachingAction.SCAFFOLD
    assert state.answer_disclosure_allowed is False
