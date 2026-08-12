"""Bounded, model-led teaching loop with deterministic guardrails."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Protocol

from app.agent.math_analysis import analyze_turn
from app.agent.response_generation import generate_response
from app.agent.safety_gate import SafetyGate, SafetyInput
from app.agent.state import VerificationResult
from app.agent.student_model import StudentModel
from app.agent.teaching_actions import TeachingAction, TeachingActionPlan
from app.agent.teaching_brief import TeachingBriefComposer
from app.agent.teaching_planner import PlannerInput, TeachingPlanner
from app.llm.deepseek_client import DeepSeekClient
from app.memory.schemas import EpisodeState
from app.retrieval.context_provider import GraphContextProvider
from prototypes.tool_loop_teacher.models import (
    DecisionKind,
    LoopDecision,
    LoopResult,
    LoopTraceEvent,
    ToolResult,
)
from prototypes.tool_loop_teacher.skills import SkillCatalog
from prototypes.tool_loop_teacher.tools import (
    ToolRegistry,
    ToolRuntimeContext,
    build_default_registry,
)


class ActionProvider(Protocol):
    def decide(self, payload: dict[str, Any]) -> LoopDecision:
        """Choose one tool call or finish with a proposed teaching plan."""


class ResponseClient(Protocol):
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str: ...


class DeepSeekActionProvider:
    """Use the existing JSON-mode client as a portable tool-call adapter."""

    _SYSTEM_PROMPT = """你是受约束的初中数学 Teacher Agent 编排器。

每轮只能做一个决定：调用一个工具，或者结束并提出 TeachingActionPlan。
工具结果会在下一轮作为 observations 返回。只在确有必要时调用工具，不要机械地调用全部工具。

硬约束：
1. deterministic_verification 的优先级高于你的数学判断，不能推翻它。
2. proposed_plan.action 必须来自 safety_constraints.allowed_actions。
3. RAG 和学生记忆只能辅助教学决策，不能推翻 verifier。
4. Skill 是教学方法说明；需要时先 load_skill，再依据其内容做决定。
5. 不要生成学生可见回复，只负责选工具或输出结构化教学计划。
6. observations 已足够时应 finish，避免重复调用同一工具。
"""

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        *,
        max_tokens: int = 1200,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self._client = client or DeepSeekClient()
        self._max_tokens = max_tokens

    @property
    def client(self) -> DeepSeekClient:
        return self._client

    def decide(self, payload: dict[str, Any]) -> LoopDecision:
        return self._client.chat_json(
            system=self._SYSTEM_PROMPT,
            user=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            schema=LoopDecision,
            temperature=0.0,
            # The local DeepSeek-V4 GGUF may spend several hundred tokens in
            # reasoning_content before emitting the JSON action. A small
            # completion budget therefore looks like an empty `{}` decision.
            max_tokens=self._max_tokens,
        )


class AgentLoop:
    """Run an isolated, bounded tool loop and close through existing guards."""

    _MANDATORY_VERIFIER_TOOL = "verify_math_attempt"

    def __init__(
        self,
        action_provider: ActionProvider,
        *,
        response_client: ResponseClient | None = None,
        registry: ToolRegistry | None = None,
        skill_catalog: SkillCatalog | None = None,
        max_iterations: int = 6,
        max_calls_per_tool: int = 2,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if max_calls_per_tool < 1:
            raise ValueError("max_calls_per_tool must be positive")
        self._action_provider = action_provider
        self._response_client = response_client
        self._registry = registry or build_default_registry()
        self._skills = skill_catalog or SkillCatalog()
        self._max_iterations = max_iterations
        self._max_calls_per_tool = max_calls_per_tool

    def run_turn(
        self,
        *,
        student_id: str,
        problem_context: str,
        student_message: str,
        conversation_id: str = "prototype",
        episode: EpisodeState | None = None,
        graph_provider: GraphContextProvider | None = None,
        reasoner=None,
    ) -> LoopResult:
        active_episode = episode or EpisodeState(
            conversation_id=conversation_id,
            student_id=student_id,
            original_problem=problem_context,
            problem_context=problem_context,
        )
        analysis = analyze_turn(
            student_message,
            problem_context,
            allow_llm=False,
        )
        snapshot = StudentModel().snapshot(
            active_episode,
            scope_id=problem_context,
        )
        runtime_context = ToolRuntimeContext(
            problem_context=problem_context,
            student_message=student_message,
            analysis=analysis,
            episode=active_episode,
            previous_verified_step=snapshot.latest_verified_step,
            graph_provider=graph_provider,
            reasoner=reasoner,
            skill_catalog=self._skills,
        )

        observations: list[ToolResult] = []
        trace: list[LoopTraceEvent] = []
        call_counts: Counter[str] = Counter()

        # Mathematical checking is a runtime invariant rather than an optional
        # model choice.  Other capabilities remain genuinely on-demand.
        verifier_observation = self._registry.execute(
            self._MANDATORY_VERIFIER_TOOL,
            {},
            runtime_context,
        )
        observations.append(verifier_observation)
        call_counts[self._MANDATORY_VERIFIER_TOOL] += 1
        trace.append(
            LoopTraceEvent(
                iteration=0,
                event="mandatory_tool",
                detail=verifier_observation.model_dump(mode="json"),
            )
        )
        verification = self._verification_from(verifier_observation)

        safety_input = SafetyInput.from_turn(
            problem_context=problem_context,
            message=student_message,
        ).model_copy(
            update={
                "intent": analysis.intent,
                "verification_status": verification.status,
                "candidate_confidence": 0.0,
                "candidate_available": False,
                "consecutive_verified_errors": snapshot.consecutive_verified_errors,
                "has_verified_error": bool(snapshot.verified_errors),
                "has_confirmed_progress": bool(snapshot.confirmed_correct),
            }
        )
        constraints = SafetyGate().evaluate(safety_input)

        proposed_plan: TeachingActionPlan | None = None
        termination_reason = "iteration_budget_exhausted"
        fallback_used = False

        for iteration in range(1, self._max_iterations + 1):
            payload = self._decision_payload(
                student_id=student_id,
                problem_context=problem_context,
                student_message=student_message,
                analysis=analysis.model_dump(mode="json"),
                verification=verification.model_dump(mode="json"),
                constraints=constraints.model_dump(mode="json"),
                observations=observations,
                call_counts=call_counts,
            )
            try:
                decision = LoopDecision.model_validate(
                    self._action_provider.decide(payload)
                )
            except Exception as exc:  # noqa: BLE001 - bounded fallback is intentional
                trace.append(
                    LoopTraceEvent(
                        iteration=iteration,
                        event="provider_error",
                        detail={"error": f"{type(exc).__name__}: {exc}"},
                    )
                )
                termination_reason = "provider_error"
                fallback_used = True
                break

            trace.append(
                LoopTraceEvent(
                    iteration=iteration,
                    event="decision",
                    detail=decision.model_dump(mode="json"),
                )
            )
            if decision.kind is DecisionKind.FINISH:
                proposed_plan = decision.proposed_plan
                termination_reason = "model_finished"
                break

            tool_name = decision.tool_name
            if call_counts[tool_name] >= self._max_calls_per_tool:
                result = ToolResult(
                    tool_name=tool_name,
                    ok=False,
                    error="per-tool call budget exhausted",
                )
            else:
                result = self._registry.execute(
                    tool_name,
                    decision.arguments,
                    runtime_context,
                )
                call_counts[tool_name] += 1
            observations.append(result)
            trace.append(
                LoopTraceEvent(
                    iteration=iteration,
                    event="tool_result",
                    detail=result.model_dump(mode="json"),
                )
            )
        else:
            fallback_used = True

        if proposed_plan is None:
            proposed_plan = TeachingActionPlan(action=TeachingAction.FOCUS)

        planner_input = PlannerInput(
            problem_context=problem_context,
            current_student_request=student_message,
            constraints=constraints,
            verified_facts=list(verification.evidence),
            student_evidence={
                "records": [
                    record.model_dump(mode="json") for record in snapshot.records
                ]
            },
            has_verified_error=(
                verification.status.value == "verified_incorrect"
                or bool(snapshot.verified_errors)
            ),
            has_confirmed_progress=(
                verification.status.value == "verified_correct"
                or bool(snapshot.confirmed_correct)
            ),
        )
        planner_result = TeachingPlanner().validate_or_fallback(
            proposed_plan,
            planner_input,
            used_llm=True,
        )
        if planner_result.reason == "planner_output_rejected":
            fallback_used = True
            trace.append(
                LoopTraceEvent(
                    iteration=len(trace),
                    event="plan_rejected",
                    detail={
                        "proposed": proposed_plan.model_dump(mode="json"),
                        "accepted": planner_result.plan.model_dump(mode="json"),
                    },
                )
            )

        next_steps = self._next_step_candidates(observations)
        brief = TeachingBriefComposer().compose(
            problem_context=problem_context,
            current_student_request=student_message,
            verified_facts=list(verification.evidence),
            student_evidence=[record.text for record in snapshot.records],
            action_plan=planner_result.plan,
            constraints=constraints,
            next_step_candidates=next_steps,
            candidate_conclusion="",
        )
        response = generate_response(brief, client=self._response_client)

        return LoopResult(
            response=response,
            action_plan=planner_result.plan,
            teaching_brief=brief,
            analysis=analysis,
            verification=verification,
            safety_constraints=constraints,
            observations=observations,
            trace=trace,
            loaded_skills=self._loaded_skills(observations),
            termination_reason=termination_reason,
            fallback_used=fallback_used,
        )

    def _decision_payload(
        self,
        *,
        student_id: str,
        problem_context: str,
        student_message: str,
        analysis: dict[str, Any],
        verification: dict[str, Any],
        constraints: dict[str, Any],
        observations: list[ToolResult],
        call_counts: Counter[str],
    ) -> dict[str, Any]:
        return {
            "student_id": student_id,
            "problem_context": problem_context,
            "current_student_message": student_message,
            "deterministic_analysis": analysis,
            "deterministic_verification": verification,
            "safety_constraints": constraints,
            "available_tools": self._registry.descriptors(),
            "available_skills": [summary.__dict__ for summary in self._skills.list()],
            "tool_call_counts": dict(call_counts),
            "observations": [
                observation.model_dump(mode="json") for observation in observations
            ],
        }

    @staticmethod
    def _verification_from(observation: ToolResult) -> VerificationResult:
        if not observation.ok:
            return VerificationResult(
                summary=observation.error,
                verifier_name="none",
            )
        return VerificationResult.model_validate(
            observation.payload.get("verification", {})
        )

    @staticmethod
    def _next_step_candidates(observations: list[ToolResult]) -> list[str]:
        candidates: list[str] = []
        for observation in observations:
            if not observation.ok or observation.tool_name != "reason_about_problem":
                continue
            reasoning = observation.payload.get("reasoning", {})
            next_step = str(reasoning.get("next_step_hint", "")).strip()
            if next_step:
                candidates.append(next_step)
        return candidates

    @staticmethod
    def _loaded_skills(observations: list[ToolResult]) -> list[str]:
        return [
            str(observation.payload["skill_name"])
            for observation in observations
            if observation.ok
            and observation.tool_name == "load_skill"
            and observation.payload.get("skill_name")
        ]
