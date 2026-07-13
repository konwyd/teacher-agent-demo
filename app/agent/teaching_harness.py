"""Default topic-agnostic teaching-harness orchestration."""

from __future__ import annotations

from app.agent.math_analysis import analyze_turn
from app.agent.response_generation import generate_response
from app.agent.state import TeachingState
from app.agent.teaching_policy import TeachingPolicy


def run_generic_teaching_turn(
    student_id: str,
    message: str,
    conversation_id: str | None = None,
) -> TeachingState:
    """Run the neutral teaching pipeline without a solver, verifier, or graph.

    The incoming message remains the problem context until a future context
    provider supplies a more precise stable problem representation.  This is a
    conservative default: it permits diagnosis and hints but never fabricates a
    mathematical verdict or final answer.
    """

    problem_context = message.strip()
    analysis = analyze_turn(message, problem_context)
    state = TeachingState(
        student_id=student_id,
        message=message,
        conversation_id=conversation_id,
        problem_context=problem_context,
        math_analysis=analysis,
        trace=["generic_analysis"],
    )
    state.action_plan = TeachingPolicy().choose(state)
    state.current_phase = state.action_plan.action.value
    state.response = generate_response(state)
    state.trace.extend(["generic_teaching_policy", "generic_response_generation"])
    return state
