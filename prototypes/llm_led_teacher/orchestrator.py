"""Mandatory request classification and bounded multisubject agent loop."""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol

from pydantic import ValidationError

from app.llm.deepseek_client import DeepSeekClient, DeepSeekError
from prototypes.llm_led_teacher.fast_plan import run_fast_plan
from prototypes.llm_led_teacher.fast_routing import choose_fast_route
from prototypes.llm_led_teacher.orchestration_models import (
    AgentActionKind,
    AgentDecision,
    AgentOrchestrationResult,
    AgentTraceEvent,
    AnswerMode,
    RequestClassification,
    RequestScope,
    SchoolStage,
    Subject,
    TeachingContract,
)
from prototypes.llm_led_teacher.orchestration_tools import (
    OrchestrationRuntime,
    OrchestrationToolRegistry,
)
from prototypes.llm_led_teacher.rag_hook import RagContext
from prototypes.llm_led_teacher.skill_policy import (
    DEFAULT_SKILL_CATALOG,
    available_skill_descriptors,
    select_skill_for_contract,
)
from prototypes.llm_led_teacher.teaching_policy import (
    build_safe_teaching_contract,
    is_concept_request,
    normalize_teaching_contract,
)


class Classifier(Protocol):
    def classify(
        self,
        message: str,
        problem_context: str,
        active_subject: str,
        active_stage: str,
    ) -> RequestClassification: ...


class ActionProvider(Protocol):
    def decide(self, payload: dict[str, Any]) -> AgentDecision: ...


class RequestClassifier:
    _SYSTEM_PROMPT = """你是九年义务教育请求分类器，不回答问题。
只能使用九个学科：数学、语文、英语、物理、化学、生物、历史、地理、道法。
scope 只能是 k9 或 off_topic。普通产品、公司、娱乐、生活闲聊属于 off_topic；
off_topic 必须 subject=null、stage=未知、answer_mode=brief_redirect。
当前消息若是 active_problem 的追问，is_new_problem=false；新的自包含题目为 true。
小学或初中无法确定时 stage=未知。学生明确要完整答案时 complete，否则 guided。"""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def classify(
        self,
        message: str,
        problem_context: str,
        active_subject: str,
        active_stage: str,
    ) -> RequestClassification:
        payload = {
            "active_problem": problem_context[-1600:],
            "current_message": message[-1200:],
            "active_subject": active_subject,
            "active_stage": active_stage,
        }
        for attempt in range(2):
            try:
                result = self.client.chat_json(
                    system=self._SYSTEM_PROMPT,
                    user=json.dumps(
                        {**payload, "format_repair_attempt": attempt},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    schema=RequestClassification,
                    temperature=0.0,
                    max_tokens=320,
                )
                return RequestClassification.model_validate(result)
            except DeepSeekError:
                break
            except (ValidationError, ValueError, TypeError):
                continue
        return RequestClassification(
            scope=RequestScope.OFF_TOPIC,
            subject=None,
            stage=SchoolStage.UNKNOWN,
            is_new_problem=True,
            answer_mode=AnswerMode.BRIEF_REDIRECT,
            reason="分类格式连续失败，安全降级为简短引导",
        )


class DeepSeekActionProvider:
    _SYSTEM_PROMPT = """你是有边界的九学科教师 Agent 编排器，不直接回答学生。
每步只输出一个动作：call_tool 或 finish。可用工具：
- load_skill(name)
load_skill 的 name 必须逐字复制 payload.available_skills 中的规范名称，不得创建新名称。
检索工具的 subject 和 stage 必须逐字复制 payload.classification 中的值。
- inspect_student_context()
- search_local_knowledge(query, subject, stage, top_k, sources)，其中 sources 只能是 ["textbook","exercise"] 的非空子集
- search_web(query, subject, stage, evidence_gap)

规则：K9 问题优先查本地教材/题库；只有本地工具成功后且存在具体证据缺口才可联网。
有相关本地片段即可 finish，不要求凑满五条，也不要用固定相似度阈值替代内容判断。
同一参数不要重复调用；联网最多一次。reason 只写可公开的简短动作理由，不写思维链。
finish 必须提交 contract。response_mode 只能是以下五种：
- concept_explanation：完整解释学生请求的概念，但不得自动套用当前题；
- one_step：当前多步题只推进一个有意义的步骤；
- guided_solution：引导解题，但不披露最终答案；
- full_solution：完整求解当前题；
- brief_redirect：题外请求的简短范围引导。
“完整解释所问概念”不等于“完整求解当前题”。只有学生原消息明确要求完整解答或给出答案时，
contract 才可使用 full_solution 且 may_reveal_final_answer=true；普通“怎么解”不构成授权，
学生说“不要答案”时绝不可披露。finish 时 plan 用 guided、complete 或 brief_redirect 表示回答节奏。"""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def decide(self, payload: dict[str, Any]) -> AgentDecision:
        return self.client.chat_json(
            system=self._SYSTEM_PROMPT,
            user=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            schema=AgentDecision,
            temperature=0.0,
            max_tokens=480,
        )


class MultisubjectAgentLoop:
    _ACTION_FORMAT_REPAIR = "上次 JSON 不符合 schema，只输出合法 action。"

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        *,
        classifier: Classifier | None = None,
        action_provider: ActionProvider | None = None,
        registry: OrchestrationToolRegistry | None = None,
        skill_catalog: dict[str, str] | None = None,
        max_iterations: int = 4,
        enable_fast_routing: bool | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        active_client = client
        uses_default_components = classifier is None and action_provider is None
        if classifier is None or action_provider is None:
            active_client = active_client or DeepSeekClient()
        self.classifier = classifier or RequestClassifier(active_client)
        self.action_provider = action_provider or DeepSeekActionProvider(active_client)
        self.registry = registry or OrchestrationToolRegistry()
        configured_skills = (
            DEFAULT_SKILL_CATALOG if skill_catalog is None else skill_catalog
        )
        self.skill_catalog = {
            name: description
            for name, description in configured_skills.items()
            if name != "error_diagnosis"
        }
        self.max_iterations = max_iterations
        self.enable_fast_routing = (
            uses_default_components
            if enable_fast_routing is None
            else enable_fast_routing
        )

    def run(
        self,
        problem_context: str,
        student_message: str,
        active_subject: str,
        active_stage: str,
        progress_callback: Callable[[str | AgentTraceEvent], None] | None = None,
    ) -> AgentOrchestrationResult:
        self._report_phase(progress_callback, "classification")
        fast_route = (
            choose_fast_route(
                student_message,
                problem_context,
                active_subject,
                active_stage,
            )
            if self.enable_fast_routing
            else None
        )
        classification = (
            fast_route.classification
            if fast_route is not None
            else self.classifier.classify(
                student_message,
                problem_context,
                active_subject,
                active_stage,
            )
        )
        classification = self._inherit_continuity(
            classification,
            active_subject,
            active_stage,
        )
        trace: list[AgentTraceEvent] = []
        subject_label = classification.subject.value if classification.subject else "非 K9"
        self._trace(
            trace,
            AgentTraceEvent(
                step=0,
                event_type="classification",
                summary=f"识别为{classification.stage.value}{subject_label}，{'新题' if classification.is_new_problem else '继续当前题'}",
                details={
                    **classification.model_dump(mode="json"),
                    "router": (
                        "deterministic_fast_path"
                        if fast_route is not None
                        else "model_classifier"
                    ),
                },
            ),
            progress_callback,
        )
        if classification.scope is RequestScope.OFF_TOPIC:
            teaching_contract = build_safe_teaching_contract(
                student_message,
                classification,
            )
            self._trace_contract(trace, 0, teaching_contract, progress_callback)
            return AgentOrchestrationResult(
                classification=classification,
                teaching_contract=teaching_contract,
                rag_context=RagContext(source="off_topic"),
                trace=trace,
                termination_reason="off_topic_redirect",
                routing_guidance={"answer_mode": AnswerMode.BRIEF_REDIRECT.value},
                direct_response=(
                    "这个请求不在当前中小学课程辅导范围内。"
                    "你可以换一个数学、语文、英语、物理等课程学习问题。"
                ),
            )

        self._report_phase(progress_callback, "planning")
        runtime = OrchestrationRuntime(
            classification=classification,
            problem_context=problem_context,
            student_message=student_message,
            skill_catalog=dict(self.skill_catalog),
        )
        observations = []
        loaded_skills: list[str] = []
        rag_context = RagContext(source="none")
        termination_reason = "iteration_budget_exhausted"
        routing_guidance = {"answer_mode": classification.answer_mode.value}
        teaching_contract: TeachingContract | None = None
        loaded_skill_guidance: dict[str, str] = {}

        if fast_route is not None:
            return run_fast_plan(
                fast_route,
                runtime,
                self.registry,
                student_message,
                trace,
                progress_callback,
            )

        for step in range(1, self.max_iterations + 1):
            payload = {
                "classification": classification.model_dump(mode="json"),
                "problem_context": problem_context[-1600:],
                "student_message": student_message[-1200:],
                "available_skills": available_skill_descriptors(runtime.skill_catalog),
                "observations": [item.model_dump(mode="json") for item in observations],
                "budgets": {
                    "remaining_steps": self.max_iterations - step + 1,
                    "local_calls": len(runtime.local_search_signatures),
                    "web_calls": runtime.web_search_count,
                },
            }
            decision: AgentDecision | None = None
            action_error: Exception | None = None
            for action_attempt in range(2):
                action_payload = dict(payload)
                if action_attempt:
                    action_payload["action_format_repair"] = self._ACTION_FORMAT_REPAIR
                try:
                    decision = AgentDecision.model_validate(
                        self.action_provider.decide(action_payload)
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - one structured repair
                    action_error = exc
                    if isinstance(exc, DeepSeekError):
                        break
                    if action_attempt == 0:
                        self._trace(
                            trace,
                            AgentTraceEvent(
                                step=step,
                                event_type="action_retry",
                                summary=self._ACTION_FORMAT_REPAIR,
                                ok=False,
                                details={"error_type": type(exc).__name__},
                            ),
                            progress_callback,
                        )

            if decision is None:
                assert action_error is not None
                termination_reason = "provider_error"
                self._trace(
                    trace,
                    AgentTraceEvent(
                        step=step,
                        event_type="fallback",
                        summary="编排决策失败，使用已有证据安全收口",
                        ok=False,
                        details={"error_type": type(action_error).__name__},
                    ),
                    progress_callback,
                )
                if self._needs_fallback_local_search(
                    classification,
                    runtime,
                    student_message,
                ):
                    fallback_arguments = {
                        "query": student_message[:800],
                        "subject": classification.subject.value,
                        "stage": classification.stage.value,
                        "top_k": 5,
                        "sources": ["textbook", "exercise"],
                    }
                    result = self.registry.execute(
                        "search_local_knowledge",
                        fallback_arguments,
                        runtime,
                    )
                    observations.append(result)
                    if result.ok:
                        rag_context = self._merge_rag(rag_context, result.data)
                    hit_count = len(result.data.get("hits", [])) if result.ok else 0
                    self._trace(
                        trace,
                        AgentTraceEvent(
                            step=step,
                            event_type="fallback_tool",
                            summary=(
                                f"编排降级后执行本地检索，命中 {hit_count} 条"
                                if result.ok
                                else f"编排降级本地检索未执行：{result.error}"
                            ),
                            tool_name="search_local_knowledge",
                            ok=result.ok,
                            elapsed_seconds=result.elapsed_seconds,
                            details={"hit_count": hit_count, "error": result.error},
                        ),
                        progress_callback,
                    )
                break
            self._trace(
                trace,
                AgentTraceEvent(
                    step=step,
                    event_type="decision",
                    summary=decision.reason,
                    tool_name=decision.tool_name,
                ),
                progress_callback,
            )
            if decision.kind is AgentActionKind.FINISH:
                termination_reason = "model_finished"
                teaching_contract = normalize_teaching_contract(
                    decision.contract,
                    student_message,
                    classification,
                )
                if decision.plan in {mode.value for mode in AnswerMode}:
                    routing_guidance["answer_mode"] = decision.plan
                break

            tool_arguments = dict(decision.arguments)
            if decision.tool_name in {"search_local_knowledge", "search_web"}:
                if classification.subject is not None:
                    tool_arguments["subject"] = classification.subject.value
                tool_arguments["stage"] = classification.stage.value
            result = self.registry.execute(
                decision.tool_name,
                tool_arguments,
                runtime,
            )
            observations.append(result)
            fallback_details: dict[str, Any] = {}
            stop_after_tool = False
            if (
                result.name == "load_skill"
                and not result.ok
                and result.error == "unknown skill"
            ):
                requested_skill = str(decision.arguments.get("name", ""))
                fallback_contract = build_safe_teaching_contract(
                    student_message,
                    classification,
                )
                selected_skill = select_skill_for_contract(fallback_contract)
                if not selected_skill or selected_skill not in runtime.skill_catalog:
                    termination_reason = "skill_catalog_unavailable"
                    stop_after_tool = True
                    fallback_details = {
                        "fallback_reason": "skill_catalog_unavailable",
                        "requested_skill": requested_skill,
                        "selected_skill": selected_skill,
                        "original_error": result.error,
                    }
                else:
                    recovered = self.registry.execute(
                        "load_skill",
                        {"name": selected_skill},
                        runtime,
                    )
                    observations.append(recovered)
                    fallback_details = {
                        "fallback_reason": "unknown_skill_fallback",
                        "requested_skill": requested_skill,
                        "selected_skill": selected_skill,
                        "original_error": result.error,
                    }
                    result = recovered
            if (
                result.name == "load_skill"
                and result.ok
                and result.data.get("idempotent") is True
            ):
                termination_reason = "duplicate_skill_short_circuit"
                teaching_contract = build_safe_teaching_contract(
                    student_message,
                    classification,
                )
                stop_after_tool = True
            if result.ok and result.name == "load_skill":
                skill_name = str(result.data.get("name", ""))
                if skill_name and skill_name not in loaded_skills:
                    loaded_skills.append(skill_name)
                    loaded_skill_guidance[skill_name] = str(
                        result.data.get("content", "")
                    )
            if result.ok and result.name in {"search_local_knowledge", "search_web"}:
                rag_context = self._merge_rag(rag_context, result.data)
            hit_count = len(result.data.get("hits", [])) if result.ok else 0
            summary = (
                f"{result.name} 完成，命中 {hit_count} 条"
                if result.ok
                else f"{result.name} 未执行：{result.error}"
            )
            self._trace(
                trace,
                AgentTraceEvent(
                    step=step,
                    event_type="tool_result",
                    summary=summary,
                    tool_name=result.name,
                    ok=result.ok,
                    elapsed_seconds=result.elapsed_seconds,
                    details={
                        "hit_count": hit_count,
                        "error": result.error,
                        **fallback_details,
                    },
                ),
                progress_callback,
            )
            if stop_after_tool:
                break

        if teaching_contract is None:
            teaching_contract = build_safe_teaching_contract(
                student_message,
                classification,
            )
        self._trace_contract(
            trace,
            min(self.max_iterations, len(trace)),
            teaching_contract,
            progress_callback,
        )
        return AgentOrchestrationResult(
            classification=classification,
            teaching_contract=teaching_contract,
            rag_context=rag_context,
            observations=observations,
            trace=trace,
            loaded_skills=loaded_skills,
            loaded_skill_guidance=loaded_skill_guidance,
            termination_reason=termination_reason,
            routing_guidance=routing_guidance,
        )

    @staticmethod
    def _needs_fallback_local_search(
        classification: RequestClassification,
        runtime: OrchestrationRuntime,
        student_message: str,
    ) -> bool:
        return bool(
            classification.scope is RequestScope.K9
            and classification.subject is not None
            and classification.stage is not SchoolStage.UNKNOWN
            and not runtime.local_search_succeeded
            and not runtime.local_search_signatures
            and (
                classification.is_new_problem
                or not runtime.problem_context
                or is_concept_request(student_message)
            )
        )

    @classmethod
    def _trace_contract(
        cls,
        trace: list[AgentTraceEvent],
        step: int,
        contract: TeachingContract,
        callback: Callable[[str | AgentTraceEvent], None] | None,
    ) -> None:
        cls._trace(
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

    @staticmethod
    def _inherit_continuity(
        classification: RequestClassification,
        active_subject: str,
        active_stage: str,
    ) -> RequestClassification:
        if classification.is_new_problem or classification.scope is RequestScope.OFF_TOPIC:
            return classification
        updates: dict[str, Any] = {}
        try:
            updates["subject"] = Subject(active_subject)
        except ValueError:
            pass
        try:
            updates["stage"] = SchoolStage(active_stage)
        except ValueError:
            pass
        return classification.model_copy(update=updates)

    @staticmethod
    def _merge_rag(current: RagContext, data: dict[str, Any]) -> RagContext:
        if not current.text:
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
        new_text = str(data.get("text", ""))
        return RagContext(
            text="\n\n".join(part for part in (current.text, new_text) if part),
            refs=list(dict.fromkeys(current.refs + list(data.get("refs", [])))),
            source="hybrid" if new_text else current.source,
            hits=current.hits + list(data.get("hits", [])),
            online_triggered=current.online_triggered or bool(data.get("online_triggered", False)),
            online_status=str(data.get("online_status", current.online_status)),
            online_reason=str(data.get("online_reason", current.online_reason)),
            local_top_score=max(current.local_top_score, float(data.get("local_top_score", 0.0))),
            diagnostics={**current.diagnostics, **dict(data.get("diagnostics", {}))},
        )

    @staticmethod
    def _trace(
        trace: list[AgentTraceEvent],
        event: AgentTraceEvent,
        callback: Callable[[str | AgentTraceEvent], None] | None,
    ) -> None:
        trace.append(event)
        if callback is not None:
            try:
                callback(event)
            except Exception:  # noqa: BLE001 - UI progress must not block teaching
                pass

    @staticmethod
    def _report_phase(
        callback: Callable[[str | AgentTraceEvent], None] | None,
        phase: str,
    ) -> None:
        if callback is not None:
            try:
                callback(phase)
            except Exception:  # noqa: BLE001 - UI progress must not block teaching
                pass
