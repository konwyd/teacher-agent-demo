"""Default topic-agnostic teaching-harness orchestration."""

from __future__ import annotations

from app.agent.generic_math_reasoning import GenericMathReasoner
from app.agent.math_analysis import analyze_turn
from app.agent.optional_verifier import NoopVerifier, OptionalVerifier, VerificationRequest
from app.agent.orchestration_trace import ModuleDecision, OrchestrationTrace
from app.agent.response_generation import generate_response
from app.agent.safety_gate import SafetyGate, SafetyInput
from app.agent.state import (
    GenericMathReasoning,
    GraphContext,
    TeachingState,
    VerificationStatus,
)
from app.agent.strategy_capabilities import (
    StrategyBundle,
    StrategyContext,
    StrategyRegistry,
)
from app.agent.student_model import StudentModel
from app.agent.teaching_planner import PlannerInput, PlannerResult, TeachingPlanner
from app.agent.teaching_policy import TeachingPolicy
from app.llm.deepseek_client import is_configured
from app.memory.episode_state import get_episode_store
from app.memory.schemas import ContextResolution, EpisodeState, now_iso
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
    planner: TeachingPlanner | None = None,
    response_client=None,
    analysis_allow_llm: bool = True,
    activate_planner: bool = False,
    episode_store=None,
) -> TeachingState:
    """Run the neutral teaching pipeline with stable episode problem context."""

    active_episode_store = (
        episode_store if episode_store is not None else get_episode_store()
    )
    fallback_reasons: list[str] = []
    try:
        loaded_episode = active_episode_store.get(student_id, conversation_id)
        episode = loaded_episode.model_copy(deep=True)
    except Exception:
        fallback_reasons.append("persistence_load_failed")
        episode = EpisodeState(
            conversation_id=conversation_id or student_id,
            student_id=student_id,
        )
    had_problem_context = bool(episode.problem_context)
    if not had_problem_context:
        episode.problem_context = message.strip()
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
    decisions = [
        ModuleDecision(
            module="context_resolution",
            called=True,
            reason="context_resolved",
            llm_calls=int(analysis_allow_llm and is_configured()),
        )
    ]

    student_model = StudentModel()
    evidence_snapshot = student_model.snapshot(episode, scope_id=problem_context)
    active_verifier = verifier or NoopVerifier()
    student_attempt = message if had_problem_context else ""
    verification_request = VerificationRequest(
        problem_context=problem_context,
        student_attempt=student_attempt,
        previous_verified_step=evidence_snapshot.latest_verified_step,
    )
    try:
        verifier_supported = active_verifier.supports(verification_request)
        if verifier_supported:
            verification = active_verifier.verify(verification_request)
            decisions.append(
                ModuleDecision(
                    module="verifier",
                    called=True,
                    reason="verification_completed",
                )
            )
        else:
            verification = NoopVerifier().verify(verification_request)
            decisions.append(
                ModuleDecision(
                    module="verifier",
                    called=False,
                    reason="verifier_unsupported",
                )
            )
    except Exception:
        verification = NoopVerifier().verify(verification_request)
        fallback_reasons.append("verifier_failed")
        decisions.append(
            ModuleDecision(
                module="verifier",
                called=True,
                reason="verifier_failed",
            )
        )

    analysis = analyze_turn(
        message,
        problem_context,
        allow_llm=analysis_allow_llm,
    )
    active_reasoner = reasoner or GenericMathReasoner()
    verifier_evidence_sufficient = (
        verification.status
        in {
            VerificationStatus.VERIFIED_CORRECT,
            VerificationStatus.VERIFIED_INCORRECT,
        }
        and bool(verification.evidence)
    )
    reasoner_llm_calls = 0
    if verifier_evidence_sufficient:
        reasoning = GenericMathReasoning()
        decisions.append(
            ModuleDecision(
                module="reasoner",
                called=False,
                reason="verifier_evidence_sufficient",
            )
        )
    else:
        reasoner_llm_calls = int(
            isinstance(active_reasoner, GenericMathReasoner)
            and (
                getattr(active_reasoner, "_client", None) is not None
                or is_configured()
            )
            and bool(problem_context.strip())
            and not analysis.needs_clarification
        )
        try:
            reasoning = GenericMathReasoning.model_validate(
                active_reasoner.analyze(message, problem_context, analysis)
            )
            decisions.append(
                ModuleDecision(
                    module="reasoner",
                    called=True,
                    reason="reasoning_completed",
                    llm_calls=reasoner_llm_calls,
                )
            )
        except Exception:
            reasoning = GenericMathReasoning(
                needs_clarification=True,
                uncertainty_note="通用数学推理不可用，已降级为澄清教学。",
            )
            fallback_reasons.append("reasoner_failed")
            decisions.append(
                ModuleDecision(
                    module="reasoner",
                    called=True,
                    reason="reasoner_failed",
                    llm_calls=reasoner_llm_calls,
                )
            )

    strategy_context = StrategyContext(
        problem_context=problem_context,
        student_message=message,
        reasoning_next_step=reasoning.next_step_hint,
        verifier_name=verification.verifier_name,
        verified_evidence=list(verification.evidence),
        unresolved_obstacles=[
            record.text for record in evidence_snapshot.unresolved_obstacles
        ],
    )
    try:
        strategy_bundle = StrategyRegistry.default().advise(
            reasoning.strategy_selection,
            strategy_context,
        )
        decisions.append(
            ModuleDecision(
                module="strategy_registry",
                called=True,
                reason=(
                    "strategy_guidance_degraded"
                    if strategy_bundle.failure_reasons
                    else "strategy_guidance_assembled"
                ),
            )
        )
    except Exception:
        strategy_bundle = StrategyBundle(failure_reasons=["registry_failure"])
        fallback_reasons.append("strategy_registry_failed")
        decisions.append(
            ModuleDecision(
                module="strategy_registry",
                called=True,
                reason="strategy_registry_failed",
            )
        )

    active_graph_provider = graph_context_provider or EmptyGraphContextProvider()
    try:
        graph_context = active_graph_provider.get_context(
            GraphContextRequest(problem_context=problem_context)
        )
        decisions.append(
            ModuleDecision(
                module="graph_provider",
                called=True,
                reason=(
                    "graph_context_loaded"
                    if graph_context.available
                    else "graph_context_empty"
                ),
            )
        )
    except Exception:
        graph_context = GraphContext()
        fallback_reasons.append("graph_context_failed")
        decisions.append(
            ModuleDecision(
                module="graph_provider",
                called=True,
                reason="graph_context_failed",
            )
        )

    has_verified_error = (
        verification.status is VerificationStatus.VERIFIED_INCORRECT
        or bool(evidence_snapshot.verified_errors)
    )
    has_confirmed_progress = (
        verification.status is VerificationStatus.VERIFIED_CORRECT
        or bool(evidence_snapshot.confirmed_correct)
    )
    safety_input = SafetyInput.from_turn(
        problem_context=problem_context,
        message=message,
    ).model_copy(
        update={
            "intent": analysis.intent,
            "verification_status": verification.status,
            "candidate_confidence": reasoning.confidence,
            "candidate_available": bool(reasoning.candidate_conclusion),
            "consecutive_verified_errors": (
                evidence_snapshot.consecutive_verified_errors
                + int(
                    verification.status
                    is VerificationStatus.VERIFIED_INCORRECT
                )
            ),
            "has_verified_error": has_verified_error,
            "has_confirmed_progress": has_confirmed_progress,
        }
    )
    safety_constraints = SafetyGate().evaluate(safety_input)
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
        safety_constraints=safety_constraints,
        student_evidence_snapshot=evidence_snapshot,
        strategy_bundle=strategy_bundle,
        trace=[
            "generic_analysis",
            "generic_math_reasoning",
            "optional_verifier",
            "optional_graph_context",
        ],
    )
    state.action_plan = TeachingPolicy().choose(state)
    active_plan = state.action_plan

    planner_input = PlannerInput(
        problem_context=problem_context,
        current_student_request=message,
        constraints=safety_constraints,
        verified_facts=list(verification.evidence),
        student_evidence=evidence_snapshot.model_dump(mode="json"),
        strategy_bundle=strategy_bundle.model_dump(mode="json"),
        has_verified_error=has_verified_error,
        has_confirmed_progress=has_confirmed_progress,
    )
    active_planner = planner or TeachingPlanner()
    response_llm_calls = int(
        bool(reasoning.reasoning_steps or reasoning.next_step_hint)
        and (response_client is not None or is_configured())
    )
    calls_before_planner = sum(decision.llm_calls for decision in decisions)
    if calls_before_planner + response_llm_calls >= 3:
        validated = active_planner.validate_or_fallback(active_plan, planner_input)
        planner_result = PlannerResult(
            plan=validated.plan,
            used_llm=False,
            reason="llm_call_budget_exhausted",
        )
        planner_called = False
    else:
        try:
            planner_result = active_planner.plan(planner_input)
            planner_called = True
        except Exception:
            fallback_reasons.append("planner_failed")
            validated = TeachingPlanner().validate_or_fallback(
                active_plan,
                planner_input,
            )
            planner_result = PlannerResult(
                plan=validated.plan,
                used_llm=False,
                reason="planner_failed",
            )
            planner_called = True
    decisions.append(
        ModuleDecision(
            module="planner",
            called=planner_called,
            reason=planner_result.reason,
            llm_calls=int(planner_result.used_llm),
        )
    )

    # Task 6 is shadow-only. ``activate_planner`` is accepted for forward
    # compatibility but cannot change the active compatibility policy yet.
    _ = activate_planner
    state.current_phase = state.action_plan.action.value
    state.response = generate_response(state, client=response_client)
    decisions.extend(
        [
            ModuleDecision(
                module="response_generation",
                called=True,
                reason="response_generated",
                llm_calls=response_llm_calls,
            ),
            ModuleDecision(
                module="guard",
                called=True,
                reason="response_guard_applied",
            ),
        ]
    )
    state.trace.extend(["generic_teaching_policy", "generic_response_generation"])

    orchestration_trace = OrchestrationTrace(
        decisions=decisions,
        shadow_mode=True,
        active_plan=active_plan,
        shadow_plan=planner_result.plan,
        fallback_reasons=fallback_reasons,
    )
    orchestration_trace.llm_call_count = sum(
        decision.llm_calls for decision in orchestration_trace.decisions
    )
    assert orchestration_trace.llm_call_count <= 3
    state.orchestration_trace = orchestration_trace

    student_model.record_turn(
        episode,
        verification=verification,
        action_plan=active_plan,
        analysis=analysis,
        source_turn=len(episode.recent_turns) + 1,
        scope_id=problem_context,
    )
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
    persistence_decision = ModuleDecision(
        module="persistence",
        called=True,
        reason="episode_saved",
    )
    orchestration_trace.decisions.append(persistence_decision)
    episode.last_orchestration_trace = orchestration_trace.model_dump(mode="json")
    try:
        active_episode_store.save(episode)
    except Exception:
        persistence_decision.reason = "persistence_failed"
        orchestration_trace.fallback_reasons.append("persistence_failed")
        episode.last_orchestration_trace = orchestration_trace.model_dump(mode="json")
    state.episode_state = episode
    return state
