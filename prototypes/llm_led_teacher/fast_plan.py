"""Deterministic execution plan for requests with a high-confidence route."""

from __future__ import annotations

from typing import Callable

from prototypes.llm_led_teacher.fast_routing import FastRoute
from prototypes.llm_led_teacher.orchestration_models import (
    AgentOrchestrationResult,
    AgentTraceEvent,
    SchoolStage,
    TeachingContract,
    ToolResult,
)
from prototypes.llm_led_teacher.orchestration_tools import (
    OrchestrationRuntime,
    OrchestrationToolRegistry,
)
from prototypes.llm_led_teacher.rag_hook import RagContext
from prototypes.llm_led_teacher.teaching_policy import (
    build_safe_teaching_contract,
    is_concept_request,
)


TraceCallback = Callable[[str | AgentTraceEvent], None] | None


def run_fast_plan(
    fast_route: FastRoute,
    runtime: OrchestrationRuntime,
    registry: OrchestrationToolRegistry,
    student_message: str,
    trace: list[AgentTraceEvent],
    callback: TraceCallback,
) -> AgentOrchestrationResult:
    """Load only the needed skill/tool and finish without an action-model call."""

    classification = fast_route.classification
    contract = build_safe_teaching_contract(student_message, classification)
    skill_name = _skill_for_contract(contract)
    needs_local_search = bool(
        classification.subject is not None
        and classification.stage is not SchoolStage.UNKNOWN
        and (classification.is_new_problem or is_concept_request(student_message))
    )
    planned_tools = (["load_skill"] if skill_name else []) + (
        ["search_local_knowledge"] if needs_local_search else []
    )
    _append_trace(
        trace,
        AgentTraceEvent(
            step=1,
            event_type="fast_route",
            summary=f"高置信度快速路由：{fast_route.reason}",
            details={
                "skill": skill_name,
                "planned_tools": planned_tools,
            },
        ),
        callback,
    )

    observations: list[ToolResult] = []
    loaded_skills: list[str] = []
    loaded_skill_guidance: dict[str, str] = {}
    rag_context = RagContext(source="none")
    step = 2

    if skill_name:
        result = registry.execute("load_skill", {"name": skill_name}, runtime)
        observations.append(result)
        if result.ok:
            loaded_skills.append(skill_name)
            loaded_skill_guidance[skill_name] = str(
                result.data.get("content", "")
            )
        _trace_tool_result(trace, step, result, callback)
        step += 1

    if needs_local_search:
        assert classification.subject is not None
        result = registry.execute(
            "search_local_knowledge",
            {
                "query": student_message[:800],
                "subject": classification.subject.value,
                "stage": classification.stage.value,
                "top_k": 5,
                "sources": ["textbook", "exercise"],
            },
            runtime,
        )
        observations.append(result)
        if result.ok:
            rag_context = _rag_from_tool(result)
        _trace_tool_result(trace, step, result, callback)
        step += 1

    _append_trace(
        trace,
        AgentTraceEvent(
            step=step,
            event_type="teaching_contract",
            summary=(
                f"教学契约已确定：{contract.response_mode.value}；"
                f"应用当前题={'是' if contract.may_apply_to_active_problem else '否'}；"
                f"披露最终答案={'是' if contract.may_reveal_final_answer else '否'}"
            ),
            details=contract.model_dump(mode="json"),
        ),
        callback,
    )
    return AgentOrchestrationResult(
        classification=classification,
        teaching_contract=contract,
        rag_context=rag_context,
        observations=observations,
        trace=trace,
        loaded_skills=loaded_skills,
        loaded_skill_guidance=loaded_skill_guidance,
        termination_reason="deterministic_fast_path",
        routing_guidance={
            "answer_mode": classification.answer_mode.value,
            "router": "deterministic_fast_path",
        },
    )


def _skill_for_contract(contract: TeachingContract) -> str:
    mode = contract.response_mode.value
    if mode == "concept_explanation":
        return "concept_explanation"
    if mode in {"one_step", "guided_solution"}:
        return "socratic_guidance"
    if mode == "full_solution":
        return "complete_solution"
    return ""


def _trace_tool_result(
    trace: list[AgentTraceEvent],
    step: int,
    result: ToolResult,
    callback: TraceCallback,
) -> None:
    hit_count = len(result.data.get("hits", [])) if result.ok else 0
    _append_trace(
        trace,
        AgentTraceEvent(
            step=step,
            event_type="tool_result",
            summary=(
                f"{result.name} 完成，命中 {hit_count} 条"
                if result.ok
                else f"{result.name} 未执行：{result.error}"
            ),
            tool_name=result.name,
            ok=result.ok,
            elapsed_seconds=result.elapsed_seconds,
            details={"hit_count": hit_count, "error": result.error},
        ),
        callback,
    )


def _append_trace(
    trace: list[AgentTraceEvent],
    event: AgentTraceEvent,
    callback: TraceCallback,
) -> None:
    trace.append(event)
    if callback is not None:
        try:
            callback(event)
        except Exception:  # noqa: BLE001 - UI reporting must not block teaching
            pass


def _rag_from_tool(result: ToolResult) -> RagContext:
    data = result.data
    return RagContext(
        text=str(data.get("text", "")),
        refs=list(data.get("refs", [])),
        is_exact_match=bool(data.get("is_exact_match", False)),
        source=str(data.get("source", "none")),
        hits=list(data.get("hits", [])),
        online_triggered=bool(data.get("online_triggered", False)),
        online_status=str(data.get("online_status", "not_triggered")),
        online_reason=str(data.get("online_reason", "")),
        local_top_score=float(data.get("local_top_score", 0.0)),
        diagnostics=dict(data.get("diagnostics", {})),
    )
