"""Default topic-agnostic teaching-harness orchestration."""

from __future__ import annotations

from app.agent.generic_math_reasoning import GenericMathReasoner
from app.agent.math_analysis import analyze_turn
from app.agent.optional_verifier import NoopVerifier, OptionalVerifier, VerificationRequest
from app.agent.response_generation import generate_response
from app.agent.state import TeachingState
from app.agent.teaching_policy import TeachingPolicy
from app.memory.episode_state import get_episode_store
from app.memory.schemas import ContextResolution, now_iso
from app.retrieval.context_provider import (
    EmptyGraphContextProvider,
    GraphContextProvider,
    GraphContextRequest,
)


def _last_question(response: str) -> str | None:
    text = response.strip()
    question_end = max(text.rfind("？"), text.rfind("?"))
    if question_end < 0:
        return None
    question_start = max(
        text.rfind("。", 0, question_end),
        text.rfind("！", 0, question_end),
        text.rfind("!", 0, question_end),
        text.rfind("\n", 0, question_end),
    )
    return text[question_start + 1 : question_end + 1].strip() or None


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
    """Run the neutral teaching pipeline with stable episode problem context."""

    episode_store = get_episode_store()
    episode = episode_store.get(student_id, conversation_id)
    had_problem_context = bool(episode.problem_context)
    if not had_problem_context:
        episode.problem_context = message.strip()
        episode_store.save(episode)
    problem_context = episode.problem_context or message.strip()
    context_resolution = ContextResolution(
        is_reply_to_previous_teacher=had_problem_context,
        dialogue_act="follow_up" if had_problem_context else "new_problem",
        semantic_meaning=(
            "复用会话原题理解当前消息"
            if had_problem_context
            else "初始化通用数学问题上下文"
        ),
        confidence=1.0,
        should_start_new_problem=not had_problem_context,
        reason=(
            "复用已有通用 problem_context 处理当前追问。"
            if had_problem_context
            else "通用 episode 尚无 problem_context，使用当前消息初始化。"
        ),
    )
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
        conversation_id=episode.conversation_id,
        episode_state=episode,
        context_resolution=context_resolution,
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
    state.current_phase = state.action_plan.action.value
    state.response = generate_response(state, client=response_client)
    state.trace.extend(["generic_teaching_policy", "generic_response_generation"])
    episode.current_phase = state.current_phase
    episode.last_teacher_question = _last_question(state.response)
    episode.expected_student_response = episode.last_teacher_question
    episode.recent_turns = (episode.recent_turns + [{
        "timestamp": now_iso(),
        "student_message": message,
        "teacher_response": state.response,
        "dialogue_act": context_resolution.dialogue_act,
        "teaching_action": state.action_plan.action.value,
    }])[-8:]
    episode_store.save(episode)
    state.episode_state = episode
    return state
