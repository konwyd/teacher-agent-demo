"""Deterministic execution plan for requests with a high-confidence route."""

from __future__ import annotations

from typing import Callable

from prototypes.llm_led_teacher import qa_cache
from prototypes.llm_led_teacher.fast_routing import FastRoute, needs_local_evidence
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
from prototypes.llm_led_teacher.retrieval_policy import RetrievalDecision, RetrievalMode
from prototypes.llm_led_teacher.skill_policy import select_skill_for_contract
from prototypes.llm_led_teacher.teaching_policy import build_safe_teaching_contract


TraceCallback = Callable[[str | AgentTraceEvent], None] | None


def run_fast_plan(
    fast_route: FastRoute,
    runtime: OrchestrationRuntime,
    registry: OrchestrationToolRegistry,
    student_message: str,
    trace: list[AgentTraceEvent],
    callback: TraceCallback,
    *,
    router_type: str = "deterministic_fast_path",
    force_local_search: bool = False,
) -> AgentOrchestrationResult:
    """Load only the needed skill/tool and finish without an action-model call."""

    classification = fast_route.classification
    contract = build_safe_teaching_contract(student_message, classification)
    skill_name = select_skill_for_contract(contract)
    needs_local_search = bool(
        classification.subject is not None
        and classification.stage is not SchoolStage.UNKNOWN
        and (force_local_search or needs_local_evidence(student_message))
    )
    needs_quality_qa = bool(
        qa_cache.lookup_enabled() and classification.use_quality_qa
    )
    if needs_local_search:
        retrieval_decision = RetrievalDecision(
            mode=RetrievalMode.LOCAL,
            reason=(
                "动态工具请求先完成本地课程资料检索"
                if force_local_search
                else "学生明确要求教材、课标或题库证据"
            ),
            confidence=1.0,
        )
    elif needs_quality_qa:
        retrieval_decision = RetrievalDecision(
            mode=RetrievalMode.LOCAL,
            reason="当前经验、常见错误或反馈型问题可参考历史优质问答",
            confidence=1.0,
        )
    else:
        retrieval_decision = RetrievalDecision(
            mode=RetrievalMode.NONE,
            reason="当前请求可直接作答，且未明确要求外部证据",
            confidence=1.0,
        )
    planned_tools = (
        (["load_skill"] if skill_name else [])
        + (["search_local_knowledge"] if needs_local_search else [])
        + (["search_quality_qa"] if needs_quality_qa else [])
    )
    _append_trace(
        trace,
        AgentTraceEvent(
            step=1,
            event_type="deterministic_plan",
            summary=f"确定性执行计划：{fast_route.reason}",
            details={
                "router": router_type,
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
        _trace_tool_start(trace, step, "load_skill", callback)
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
        _trace_tool_start(trace, step, "search_local_knowledge", callback)
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
            rag_context = _merge_rag(rag_context, _rag_from_tool(result))
        _trace_tool_result(trace, step, result, callback)
        step += 1

    if needs_quality_qa:
        assert classification.subject is not None
        _trace_tool_start(trace, step, "search_quality_qa", callback)
        result = registry.execute(
            "search_quality_qa",
            {
                "query": student_message[:800],
                "subject": classification.subject.value,
                "stage": classification.stage.value,
                "top_k": 3,
            },
            runtime,
        )
        observations.append(result)
        if result.ok:
            rag_context = _merge_rag(rag_context, _rag_from_tool(result))
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
        retrieval_decision=retrieval_decision,
        rag_context=rag_context,
        observations=observations,
        trace=trace,
        loaded_skills=loaded_skills,
        loaded_skill_guidance=loaded_skill_guidance,
        termination_reason=router_type,
        routing_guidance={
            "answer_mode": classification.answer_mode.value,
            "router": router_type,
        },
    )


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


def _trace_tool_start(
    trace: list[AgentTraceEvent],
    step: int,
    tool_name: str,
    callback: TraceCallback,
) -> None:
    """Publish a safe start marker before a potentially slow tool blocks."""

    _append_trace(
        trace,
        AgentTraceEvent(
            step=step,
            event_type="decision",
            summary=f"准备执行 {tool_name}",
            tool_name=tool_name,
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


def _merge_rag(current: RagContext, incoming: RagContext) -> RagContext:
    if not current.text and not current.hits:
        return incoming
    if not incoming.text and not incoming.hits:
        return RagContext(
            text=current.text,
            refs=current.refs,
            is_exact_match=current.is_exact_match,
            source=current.source,
            hits=current.hits,
            online_triggered=current.online_triggered,
            online_status=current.online_status,
            online_reason=current.online_reason,
            local_top_score=current.local_top_score,
            diagnostics={**current.diagnostics, **incoming.diagnostics},
        )
    return RagContext(
        text="\n\n".join(part for part in (current.text, incoming.text) if part),
        refs=list(dict.fromkeys(current.refs + incoming.refs)),
        is_exact_match=current.is_exact_match or incoming.is_exact_match,
        source="hybrid",
        hits=current.hits + incoming.hits,
        online_triggered=current.online_triggered or incoming.online_triggered,
        online_status=(
            incoming.online_status
            if incoming.online_triggered
            else current.online_status
        ),
        online_reason=incoming.online_reason or current.online_reason,
        local_top_score=max(current.local_top_score, incoming.local_top_score),
        diagnostics={**current.diagnostics, **incoming.diagnostics},
    )
