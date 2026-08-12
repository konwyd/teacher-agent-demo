"""Mandatory request classification and bounded multisubject agent loop."""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol

from app.llm.deepseek_client import DeepSeekClient
from prototypes.llm_led_teacher.orchestration_models import (
    AgentOrchestrationResult,
    AgentTraceEvent,
    AnswerMode,
    RequestClassification,
    RequestScope,
    SchoolStage,
    Subject,
)
from prototypes.llm_led_teacher.orchestration_tools import (
    OrchestrationRuntime,
    OrchestrationToolRegistry,
)
from prototypes.llm_led_teacher.rag_hook import RagContext
from prototypes.llm_led_teacher.retrieval_policy import (
    RetrievalDecision,
    RetrievalMode,
    apply_retrieval_guards,
)


class Classifier(Protocol):
    def classify(
        self,
        message: str,
        problem_context: str,
        active_subject: str,
        active_stage: str,
    ) -> RequestClassification: ...


class RetrievalProvider(Protocol):
    def decide(self, payload: dict[str, Any]) -> RetrievalDecision: ...


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
                    max_tokens=700,
                )
                return RequestClassification.model_validate(result)
            except Exception:  # noqa: BLE001 - one repair then safe classification
                continue
        return RequestClassification(
            scope=RequestScope.OFF_TOPIC,
            subject=None,
            stage=SchoolStage.UNKNOWN,
            is_new_problem=True,
            answer_mode=AnswerMode.BRIEF_REDIRECT,
            reason="分类格式连续失败，安全降级为简短引导",
        )


class RetrievalDecisionProvider:
    _SYSTEM_PROMPT = """你是九年义务教育教师的检索路由器，只判断回答当前请求是否需要外部证据。
不要回答学生，不要生成教学内容，不要编排自由步骤或工具调用计划。
mode 只能是 none、local、local_then_web：无需资料选 none；需要教材或题库证据选 local；
只有需要核对今天、最新、刚发布等时效信息时才选 local_then_web，并设置 needs_current_information=true。
reason 只写可公开的简短检索理由，不写私有思维链；confidence 表示路由置信度。"""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def decide(self, payload: dict[str, Any]) -> RetrievalDecision:
        return RetrievalDecision.model_validate(
            self.client.chat_json(
                system=self._SYSTEM_PROMPT,
                user=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                schema=RetrievalDecision,
                temperature=0.0,
                max_tokens=500,
            )
        )


class MultisubjectAgentLoop:
    def __init__(
        self,
        client: DeepSeekClient | None = None,
        *,
        classifier: Classifier | None = None,
        retrieval_provider: RetrievalProvider | None = None,
        registry: OrchestrationToolRegistry | None = None,
    ) -> None:
        active_client = client
        if classifier is None or retrieval_provider is None:
            active_client = active_client or DeepSeekClient()
        self.classifier = classifier or RequestClassifier(active_client)
        self.retrieval_provider = retrieval_provider or RetrievalDecisionProvider(
            active_client
        )
        self.registry = registry or OrchestrationToolRegistry()

    def run(
        self,
        problem_context: str,
        student_message: str,
        active_subject: str,
        active_stage: str,
        progress_callback: Callable[[AgentTraceEvent], None] | None = None,
    ) -> AgentOrchestrationResult:
        classification = self.classifier.classify(
            student_message,
            problem_context,
            active_subject,
            active_stage,
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
                details=classification.model_dump(mode="json"),
            ),
            progress_callback,
        )
        if classification.scope is RequestScope.OFF_TOPIC:
            return AgentOrchestrationResult(
                classification=classification,
                retrieval_decision=RetrievalDecision(
                    mode=RetrievalMode.NONE,
                    reason="当前请求无需课程资料",
                    confidence=1.0,
                ),
                rag_context=RagContext(source="off_topic"),
                trace=trace,
                termination_reason="off_topic_redirect",
                routing_guidance={"answer_mode": AnswerMode.BRIEF_REDIRECT.value},
            )

        runtime = OrchestrationRuntime(
            classification=classification,
            problem_context=problem_context,
            student_message=student_message,
            skill_catalog={
                "socratic_guidance": "一次推进一个关键步骤，用一个必要问题检查理解。",
                "error_diagnosis": "指出具体错误证据，再给最小可执行修正。",
                "concept_explanation": "先讲核心因果，再用贴近学生的例子解释。",
            },
        )
        observations = []
        rag_context = RagContext(source="none")
        routing_guidance = {"answer_mode": classification.answer_mode.value}

        payload = {
            "classification": classification.model_dump(mode="json"),
            "problem_context": problem_context[-1600:],
            "student_message": student_message[-1200:],
        }
        try:
            decision = apply_retrieval_guards(
                RetrievalDecision.model_validate(
                    self.retrieval_provider.decide(payload)
                ),
                message=student_message,
                classification=classification,
            )
            termination_reason = "retrieval_routed"
        except Exception as exc:  # noqa: BLE001 - bounded safe fallback
            decision = RetrievalDecision(
                mode=RetrievalMode.NONE,
                reason="检索路由失败，使用已有知识继续",
                confidence=0.0,
            )
            termination_reason = "provider_error"
            self._trace(
                trace,
                AgentTraceEvent(
                    step=1,
                    event_type="fallback",
                    summary=decision.reason,
                    ok=False,
                    details={"error_type": type(exc).__name__},
                ),
                progress_callback,
            )
        else:
            self._trace(
                trace,
                AgentTraceEvent(
                    step=1,
                    event_type="decision",
                    summary=decision.reason,
                    details=decision.model_dump(mode="json"),
                ),
                progress_callback,
            )

        local_arguments = {
            "query": student_message,
            "subject": classification.subject.value,
            "stage": classification.stage.value,
            "top_k": 5,
            "sources": ["textbook", "exercise"],
        }
        web_arguments = {
            "query": student_message,
            "subject": classification.subject.value,
            "stage": classification.stage.value,
            "evidence_gap": decision.reason,
        }

        def record(result: Any, step: int) -> None:
            nonlocal rag_context
            observations.append(result)
            if result.ok:
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
                    details={"hit_count": hit_count, "error": result.error},
                ),
                progress_callback,
            )

        if decision.mode in {RetrievalMode.LOCAL, RetrievalMode.LOCAL_THEN_WEB}:
            local = self.registry.execute(
                "search_local_knowledge", local_arguments, runtime
            )
            record(local, 2)
        if decision.mode is RetrievalMode.LOCAL_THEN_WEB:
            web = self.registry.execute("search_web", web_arguments, runtime)
            record(web, 3)

        return AgentOrchestrationResult(
            classification=classification,
            retrieval_decision=decision,
            rag_context=rag_context,
            observations=observations,
            trace=trace,
            termination_reason=termination_reason,
            routing_guidance=routing_guidance,
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
        callback: Callable[[AgentTraceEvent], None] | None,
    ) -> None:
        trace.append(event)
        if callback is not None:
            try:
                callback(event)
            except Exception:  # noqa: BLE001 - UI progress must not block teaching
                pass
