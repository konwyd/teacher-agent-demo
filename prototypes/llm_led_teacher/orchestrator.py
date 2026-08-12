"""Mandatory request classification and bounded multisubject agent loop."""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol

from app.llm.deepseek_client import DeepSeekClient
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
)
from prototypes.llm_led_teacher.orchestration_tools import (
    OrchestrationRuntime,
    OrchestrationToolRegistry,
)
from prototypes.llm_led_teacher.rag_hook import RagContext


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


class DeepSeekActionProvider:
    _SYSTEM_PROMPT = """你是有边界的九学科教师 Agent 编排器，不直接回答学生。
每步只输出一个动作：call_tool 或 finish。可用工具：
- load_skill(name)
- inspect_student_context()
- search_local_knowledge(query, subject, stage, top_k, sources)，其中 sources 只能是 ["textbook","exercise"] 的非空子集
- search_web(query, subject, stage, evidence_gap)

规则：K9 问题优先查本地教材/题库；只有本地工具成功后且存在具体证据缺口才可联网。
有相关本地片段即可 finish，不要求凑满五条，也不要用固定相似度阈值替代内容判断。
同一参数不要重复调用；联网最多一次。reason 只写可公开的简短动作理由，不写思维链。
finish 时 plan 用 guided、complete 或 brief_redirect 表示回答节奏。"""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def decide(self, payload: dict[str, Any]) -> AgentDecision:
        return self.client.chat_json(
            system=self._SYSTEM_PROMPT,
            user=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            schema=AgentDecision,
            temperature=0.0,
            max_tokens=900,
        )


class MultisubjectAgentLoop:
    def __init__(
        self,
        client: DeepSeekClient | None = None,
        *,
        classifier: Classifier | None = None,
        action_provider: ActionProvider | None = None,
        registry: OrchestrationToolRegistry | None = None,
        max_iterations: int = 6,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        active_client = client
        if classifier is None or action_provider is None:
            active_client = active_client or DeepSeekClient()
        self.classifier = classifier or RequestClassifier(active_client)
        self.action_provider = action_provider or DeepSeekActionProvider(active_client)
        self.registry = registry or OrchestrationToolRegistry()
        self.max_iterations = max_iterations

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
        loaded_skills: list[str] = []
        rag_context = RagContext(source="none")
        termination_reason = "iteration_budget_exhausted"
        routing_guidance = {"answer_mode": classification.answer_mode.value}

        for step in range(1, self.max_iterations + 1):
            payload = {
                "classification": classification.model_dump(mode="json"),
                "problem_context": problem_context[-1600:],
                "student_message": student_message[-1200:],
                "observations": [item.model_dump(mode="json") for item in observations],
                "budgets": {
                    "remaining_steps": self.max_iterations - step + 1,
                    "local_calls": len(runtime.local_search_signatures),
                    "web_calls": runtime.web_search_count,
                },
            }
            try:
                decision = AgentDecision.model_validate(self.action_provider.decide(payload))
            except Exception as exc:  # noqa: BLE001 - bounded safe fallback
                termination_reason = "provider_error"
                self._trace(
                    trace,
                    AgentTraceEvent(
                        step=step,
                        event_type="fallback",
                        summary="编排决策失败，使用已有证据安全收口",
                        ok=False,
                        details={"error_type": type(exc).__name__},
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
                if decision.plan in {mode.value for mode in AnswerMode}:
                    routing_guidance["answer_mode"] = decision.plan
                break

            result = self.registry.execute(decision.tool_name, decision.arguments, runtime)
            observations.append(result)
            if result.ok and result.name == "load_skill":
                skill_name = str(result.data.get("name", ""))
                if skill_name and skill_name not in loaded_skills:
                    loaded_skills.append(skill_name)
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
                    details={"hit_count": hit_count, "error": result.error},
                ),
                progress_callback,
            )

        return AgentOrchestrationResult(
            classification=classification,
            rag_context=rag_context,
            observations=observations,
            trace=trace,
            loaded_skills=loaded_skills,
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
