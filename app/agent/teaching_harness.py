"""Default topic-agnostic teaching-harness orchestration."""

from __future__ import annotations

from app.agent.generic_math_reasoning import GenericMathReasoner
from app.agent.math_analysis import analyze_turn
from app.agent.optional_verifier import NoopVerifier, OptionalVerifier, VerificationRequest
from app.agent.response_generation import generate_response
from app.agent.state import TeachingState
from app.agent.teaching_policy import TeachingPolicy
from app.retrieval.context_provider import (
    EmptyGraphContextProvider,
    GraphContextProvider,
    GraphContextRequest,
)


def run_generic_teaching_turn(
    student_id: str,
    message: str,
    conversation_id: str | None = None,
    *,
    verifier: OptionalVerifier | None = None,
    graph_context_provider: GraphContextProvider | None = None,
    reasoner: GenericMathReasoner | None = None,
    response_client=None,
    analysis_allow_llm: bool = True,
) -> TeachingState:
    """Run the neutral teaching pipeline without a solver, verifier, or graph.

    The incoming message remains the problem context until a future context
    provider supplies a more precise stable problem representation.  This is a
    conservative default: it permits diagnosis and hints but never fabricates a
    mathematical verdict or final answer.
    """

    problem_context = message.strip()
    analysis = analyze_turn(
        message,
        problem_context,
        allow_llm=analysis_allow_llm,
    )
    active_reasoner = reasoner or GenericMathReasoner()
    reasoning = active_reasoner.analyze(message, problem_context, analysis)
    active_verifier = verifier or NoopVerifier()
    active_graph_provider = graph_context_provider or EmptyGraphContextProvider()
    verification = active_verifier.verify(
        VerificationRequest(problem_context=problem_context)
    )
    graph_context = active_graph_provider.get_context(
        GraphContextRequest(problem_context=problem_context)
    )
    state = TeachingState(
        student_id=student_id,
        message=message,
        conversation_id=conversation_id,
        problem_context=problem_context,
        math_analysis=analysis,
        math_reasoning=reasoning,
        verification=verification,
        graph_context=graph_context,
        trace=[
            "generic_analysis",
            "generic_math_reasoning",
            "optional_verifier",
            "optional_graph_context",
        ],
    )
    state.action_plan = TeachingPolicy().choose(state)
    state.answer_disclosure_allowed = state.action_plan.answer_disclosure_allowed
    state.current_phase = state.action_plan.action.value
    state.response = generate_response(state, client=response_client)
    state.trace.extend(["generic_teaching_policy", "generic_response_generation"])
    return state
