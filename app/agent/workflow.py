"""Topic-agnostic teaching-workflow facade.

The default path orchestrates neutral analysis, policy, and response generation.
The retained linear-equation implementation is called only through its explicit
compatibility boundary, preserving the legacy ``run_turn`` return contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent import linear_equation_adapter as _legacy
from app.agent import math_checker as mc
from app.agent.optional_verifier import LinearEquationVerifier
from app.agent.runtime_router import RuntimePath, resolve_runtime_path
from app.agent.state import (
    TeacherState,
    TeachingState,
    project_teacher_state,
    project_teaching_state_to_teacher_state,
)
from app.agent.teaching_actions import TeachingAction
from app.agent.teaching_brief import TeachingBrief
from app.agent.teaching_harness import run_generic_teaching_turn
from app.memory.episode_state import get_episode_store
from app.retrieval.graph_retriever import get_retriever as _legacy_graph_retriever


@dataclass(frozen=True)
class TeachingTurnResult:
    teacher_state: TeacherState
    teaching_state: TeachingState


def linear_grounding_guard(response: str, brief: TeachingBrief) -> bool:
    introduced_example = mc.uses_new_example(response, brief.problem_context)
    return not introduced_example or brief.action_plan.action in {
        TeachingAction.MODEL,
        TeachingAction.PRACTICE,
    }


def execute_turn(
    student_id: str,
    message: str,
    conversation_id: str | None = None,
    **generic_dependencies,
) -> TeachingTurnResult:
    """Execute one route once while retaining legacy and unified views."""
    episode_store = get_episode_store()
    episode = episode_store.find(student_id, conversation_id)
    route = resolve_runtime_path(
        student_id,
        message,
        conversation_id,
        episode=episode,
    )
    active_episode = episode or episode_store.get(student_id, conversation_id)
    active_episode.orchestration_path = route.value
    episode_store.save(active_episode)

    if route is RuntimePath.LEGACY_EQUATION:
        legacy_state = _legacy.run_legacy_equation_turn(
            student_id,
            message,
            conversation_id,
        )
        return TeachingTurnResult(
            teacher_state=legacy_state,
            teaching_state=project_teacher_state(legacy_state),
        )
    if route is RuntimePath.UNIFIED_LINEAR_SLICE:
        generic_dependencies.update(
            verifier=LinearEquationVerifier(),
            activate_planner=True,
            grounding_guard=linear_grounding_guard,
        )
    teaching_state = run_generic_teaching_turn(
        student_id,
        message,
        conversation_id,
        **generic_dependencies,
    )
    return TeachingTurnResult(
        teacher_state=project_teaching_state_to_teacher_state(teaching_state),
        teaching_state=teaching_state,
    )


def run_teaching_turn(
    student_id: str,
    message: str,
    conversation_id: str | None = None,
    *,
    verifier=None,
    graph_context_provider=None,
    reasoner=None,
    planner=None,
    response_client=None,
    analysis_allow_llm: bool = True,
) -> TeachingState:
    """Run the default neutral teaching harness for one tutoring turn."""

    return execute_turn(
        student_id,
        message,
        conversation_id,
        verifier=verifier,
        graph_context_provider=graph_context_provider,
        reasoner=reasoner,
        planner=planner,
        response_client=response_client,
        analysis_allow_llm=analysis_allow_llm,
    ).teaching_state


def run_turn(
    student_id: str,
    message: str,
    conversation_id: str | None = None,
) -> TeacherState:
    """Compatibility API returning ``TeacherState`` for existing callers."""

    return execute_turn(student_id, message, conversation_id).teacher_state


def build_workflow():
    """Return the retained workflow for callers that inspect legacy nodes."""

    return _legacy.build_legacy_equation_workflow()


# Legacy test/evaluation imports remain at this compatibility edge until their
# callers move to the neutral contract. They are not used by the default path.
BASIC_LANGUAGE_MARKERS = _legacy.BASIC_LANGUAGE_MARKERS
is_step_message = _legacy.is_step_message
_choose_action = _legacy._choose_action
_keep_single_question = _legacy._keep_single_question
postprocess_teacher_response = _legacy.postprocess_teacher_response
_anchor_gap = _legacy._anchor_gap
# This read-only export remains for a pre-existing integration test and legacy
# inspection callers. Runtime teaching goes through the context provider above.
get_retriever = _legacy_graph_retriever
context_resolver_node = _legacy.context_resolver_node
retrieve_node = _legacy.retrieve_node
teaching_policy_node = _legacy.teaching_policy_node
memory_update_node = _legacy.memory_update_node
log_node = _legacy.log_node
_deepseek_response = _legacy._deepseek_response
_llm_diagnosis = _legacy._llm_diagnosis


def teach_node(state: TeacherState) -> dict:
    """Run legacy teaching with monkeypatch-compatible response injection."""

    _legacy._deepseek_response = _deepseek_response
    return _legacy.teach_node(state)


def diagnose_node(state: TeacherState) -> dict:
    """Run legacy diagnosis with monkeypatch-compatible analysis injection."""

    _legacy._llm_diagnosis = _llm_diagnosis
    return _legacy.diagnose_node(state)
