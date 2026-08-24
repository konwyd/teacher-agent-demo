"""Whitelisted, student-visible summaries of real teacher-agent work."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Literal

from pydantic import BaseModel, Field, model_validator

from app.agent.state import VerificationResult, VerificationStatus
from prototypes.llm_led_teacher.orchestration_models import (
    AgentTraceEvent,
    TeachingContract,
    TeachingResponseMode,
)


class PublicThinkingKind(str, Enum):
    CLASSIFICATION = "classification"
    STRATEGY = "strategy"
    SKILL = "skill"
    RETRIEVAL = "retrieval"
    VERIFICATION = "verification"
    GENERATION = "generation"
    AUDIT = "audit"
    REPAIR = "repair"
    COMPLETE = "complete"
    WARNING = "warning"


class PublicThinkingStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    WARNING = "warning"


class ReasoningTraceStageId(str, Enum):
    REQUEST_ANALYSIS = "request_analysis"
    LEARNING_CONTEXT = "learning_context"
    KNOWLEDGE_DECISION = "knowledge_decision"
    TOOL_EXECUTION = "tool_execution"
    STEP_VERIFICATION = "step_verification"
    TEACHING_STRATEGY = "teaching_strategy"
    ANSWER_GENERATION = "answer_generation"
    QUALITY_VALIDATION = "quality_validation"
    ANSWER_REPAIR = "answer_repair"
    FINAL_RECHECK = "final_recheck"


class ReasoningTraceSource(str, Enum):
    REQUEST_CLASSIFICATION = "RequestClassification"
    MEMORY_RESOLUTION = "StudentMemoryResolution"
    RETRIEVAL_DECISION = "RetrievalDecision"
    RAG_CONTEXT = "RagContext"
    AGENT_TRACE = "AgentTraceEvent"
    TEACHING_CONTRACT = "TeachingContract"
    VERIFICATION_RESULT = "VerificationResult"
    BOUNDARY_AUDIT = "BoundaryAudit"
    RUNTIME_STAGE = "RuntimeStage"


class ReasoningTraceEvidence(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    value: str = Field(min_length=1, max_length=160)
    source: ReasoningTraceSource
    source_field: str = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_source_field(self) -> "ReasoningTraceEvidence":
        if not self.source_field.startswith(f"{self.source.value}."):
            raise ValueError("source_field must belong to source")
        return self


class ReasoningTraceStage(BaseModel):
    sequence: int = Field(ge=1)
    stage_id: ReasoningTraceStageId
    stage: str = Field(min_length=1, max_length=24)
    icon: str = Field(min_length=1, max_length=8)
    summary: str = Field(min_length=1, max_length=240)
    content: list[ReasoningTraceEvidence] = Field(default_factory=list, max_length=6)
    status: PublicThinkingStatus
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


class PublicThinkingEvent(BaseModel):
    sequence: int = Field(ge=1)
    kind: PublicThinkingKind
    status: PublicThinkingStatus
    summary: str = Field(min_length=1, max_length=240)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


class PublicThinkingSection(BaseModel):
    kind: PublicThinkingKind
    title: Literal["题意", "写法", "资料", "成稿"]
    summary: str = Field(min_length=1, max_length=240)
    details: list[str] = Field(default_factory=list, max_length=4)
    status: PublicThinkingStatus
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


class PublicThinkingUpdate(BaseModel):
    """One safe event plus the section snapshot available at that moment."""

    event: PublicThinkingEvent
    sections: list[PublicThinkingSection] = Field(default_factory=list)
    reasoning_trace: list[ReasoningTraceStage] = Field(default_factory=list)


ThinkingEventCallback = Callable[[PublicThinkingEvent], None]
ThinkingUpdateCallback = Callable[[PublicThinkingUpdate], None]


_STAGE_COPY = {
    "classification": (
        PublicThinkingKind.CLASSIFICATION,
        "正在识别题目和学生意图……",
    ),
    "observing": (
        PublicThinkingKind.CLASSIFICATION,
        "正在识别题目和学生意图……",
    ),
    "planning": (
        PublicThinkingKind.STRATEGY,
        "正在选择合适的教学路径……",
    ),
    "retrieving": (
        PublicThinkingKind.RETRIEVAL,
        "正在查找可用的教材与题目参考……",
    ),
    "generating": (
        PublicThinkingKind.GENERATION,
        "正在根据题目、学生情况和参考资料组织回答……",
    ),
    "auditing": (
        PublicThinkingKind.AUDIT,
        "正在核验解题内容和教学节奏……",
    ),
    "repairing": (
        PublicThinkingKind.REPAIR,
        "发现需要调整的内容，正在进行安全修正……",
    ),
}

_ALLOWED_SUBJECTS = {
    "数学",
    "语文",
    "英语",
    "物理",
    "化学",
    "生物",
    "历史",
    "地理",
    "道法",
}
_ALLOWED_STAGES = {"小学", "初中", "未知"}
_PUBLIC_INTENTS = {
    "把普通问句改成作文开头",
    "解释知识点",
    "检查解题步骤",
    "引导完成当前题目",
    "给出完整解答",
}
_PUBLIC_STRATEGIES = {
    "从画面、情绪、悬念三个方向改写",
    "先确认题意，再分步讲解",
    "先验证学生步骤，再给下一步提示",
    "先解释概念，再给一个练习",
    "直接回应并简短说明",
}
_PUBLIC_COMPOSITIONS = {
    "先解释方法，再给三个示例和迁移提示",
    "先给结论，再解释依据",
    "先回应当前步骤，再给下一步提示",
    "先解释概念，再用例子巩固",
}
_ROUTE_MODE_COPY = {
    "none": "未使用 RAG",
    "local": "使用本地 RAG",
    "local_then_web": "先查本地 RAG，必要时联网补充",
}
_PUBLIC_STRATEGY_EVENT_COPY = {
    "从画面、情绪、悬念三个方向改写": (
        "将分别从画面、情绪和悬念三个角度组织示例，帮助比较表达效果。"
    ),
    "先确认题意，再分步讲解": "将先明确题目要求和已知条件，再按关键步骤展开讲解。",
    "先验证学生步骤，再给下一步提示": (
        "将先验证学生当前步骤，再只推进一个关键步骤，避免一次给完答案。"
    ),
    "先解释概念，再给一个练习": "将先用直白语言解释概念，再用一道练习帮助巩固。",
    "直接回应并简短说明": "将直接回应当前问题，并补充必要依据。",
}
_PUBLIC_COMPOSITION_EVENT_COPY = {
    "先解释方法，再给三个示例和迁移提示": (
        "回答会先说明方法，再给出三个示例，并补充可迁移的写法提示。"
    ),
    "先给结论，再解释依据": "回答会先给出明确结论，再解释关键依据。",
    "先回应当前步骤，再给下一步提示": "回答会先回应当前步骤，再给出下一步提示。",
    "先解释概念，再用例子巩固": "回答会先解释核心概念，再用例子帮助巩固。",
}
_ROUTE_EVENT_COPY = {
    "none": "本题可直接依据题目与确定性工具处理，因此不调用 RAG。",
    "local": "将检索本地教材与题库，用课程内资料核对概念和解题依据。",
    "local_then_web": (
        "将先查本地教材与题库；若证据不足，再补充联网资料并标明来源。"
    ),
}

_SECTION_ORDER = {"题意": 0, "写法": 1, "资料": 2, "成稿": 3}
_SLOT_ORDER = {
    "题意": ("classification",),
    "写法": ("strategy", "skill", "context", "verification"),
    "资料": ("route", "local_tool", "quality_tool", "web_tool", "retrieval"),
    "成稿": ("composition", "audit", "completion"),
}
_TOOL_START_COPY = {
    "load_skill": (PublicThinkingKind.SKILL, "正在准备合适的教学方法……"),
    "inspect_student_context": (
        PublicThinkingKind.STRATEGY,
        "正在查看与当前题目有关的学习情况……",
    ),
    "search_local_knowledge": (
        PublicThinkingKind.RETRIEVAL,
        "正在检索本地教材与题库……",
    ),
    "search_quality_qa": (
        PublicThinkingKind.RETRIEVAL,
        "正在检索优质问答参考……",
    ),
    "search_web": (
        PublicThinkingKind.RETRIEVAL,
        "本地资料不足，正在补充联网参考……",
    ),
}

_TRACE_STAGE_COPY = {
    ReasoningTraceStageId.REQUEST_ANALYSIS: ("问题分析", "🔍"),
    ReasoningTraceStageId.LEARNING_CONTEXT: ("学习上下文", "🧠"),
    ReasoningTraceStageId.KNOWLEDGE_DECISION: ("知识判断", "📚"),
    ReasoningTraceStageId.TOOL_EXECUTION: ("工具执行", "🛠️"),
    ReasoningTraceStageId.STEP_VERIFICATION: ("步骤验证", "🧮"),
    ReasoningTraceStageId.TEACHING_STRATEGY: ("教学策略", "📝"),
    ReasoningTraceStageId.ANSWER_GENERATION: ("回答生成", "✍️"),
    ReasoningTraceStageId.QUALITY_VALIDATION: ("质量验证", "✅"),
    ReasoningTraceStageId.ANSWER_REPAIR: ("回答修正", "🩹"),
    ReasoningTraceStageId.FINAL_RECHECK: ("最终复核", "🛡️"),
}
_RUNTIME_STAGE_TO_TRACE = {
    "classification": ReasoningTraceStageId.REQUEST_ANALYSIS,
    "observing": ReasoningTraceStageId.REQUEST_ANALYSIS,
    "planning": ReasoningTraceStageId.TEACHING_STRATEGY,
    "retrieving": ReasoningTraceStageId.KNOWLEDGE_DECISION,
    "generating": ReasoningTraceStageId.ANSWER_GENERATION,
    "auditing": ReasoningTraceStageId.QUALITY_VALIDATION,
    "repairing": ReasoningTraceStageId.ANSWER_REPAIR,
}
_TRACE_STAGE_IN_PROGRESS_COPY = {
    ReasoningTraceStageId.REQUEST_ANALYSIS: "正在识别任务类型和教学目标。",
    ReasoningTraceStageId.KNOWLEDGE_DECISION: "正在判断本轮需要哪些知识与资料。",
    ReasoningTraceStageId.TEACHING_STRATEGY: "正在确定教学方式和回答边界。",
    ReasoningTraceStageId.ANSWER_GENERATION: "正在按教学决策组织回答。",
    ReasoningTraceStageId.QUALITY_VALIDATION: "正在检查回答边界与可验证内容。",
    ReasoningTraceStageId.ANSWER_REPAIR: "检测到需要修正的内容，正在调整回答。",
}
_TEACHING_MODE_COPY = {
    TeachingResponseMode.CONCEPT_EXPLANATION: "概念解释",
    TeachingResponseMode.ONE_STEP: "推进一个关键步骤",
    TeachingResponseMode.GUIDED_SOLUTION: "分步引导解题",
    TeachingResponseMode.FULL_SOLUTION: "完整解答",
    TeachingResponseMode.BRIEF_REDIRECT: "简短回应并引导回学习",
}
_TEACHING_STRUCTURE_COPY = {
    TeachingResponseMode.CONCEPT_EXPLANATION: "核心概念 + 关键关系 + 简单例子",
    TeachingResponseMode.ONE_STEP: "当前步骤反馈 + 一个下一步提示",
    TeachingResponseMode.GUIDED_SOLUTION: "题意确认 + 分步提示",
    TeachingResponseMode.FULL_SOLUTION: "结论 + 完整步骤 + 关键依据",
    TeachingResponseMode.BRIEF_REDIRECT: "简短回应 + 学习方向建议",
}
_TOOL_PUBLIC_NAME = {
    "load_skill": "教学 Skill",
    "inspect_student_context": "学习上下文",
    "search_local_knowledge": "本地资料检索",
    "search_quality_qa": "优质问答检索",
    "search_web": "联网资料检索",
}


class PublicThinkingRecorder:
    """Build public events from a deliberately narrow set of scalar inputs."""

    def __init__(
        self,
        callback: ThinkingEventCallback | None = None,
        update_callback: ThinkingUpdateCallback | None = None,
    ) -> None:
        self.events: list[PublicThinkingEvent] = []
        self.sections: list[PublicThinkingSection] = []
        self.reasoning_trace: list[ReasoningTraceStage] = []
        self._section_slots: dict[str, dict[str, list[str]]] = {}
        self._tool_trace_evidence: dict[str, ReasoningTraceEvidence] = {}
        self._retrieval_trace_evidence: dict[str, ReasoningTraceEvidence] = {}
        self._retrieval_mode = ""
        self._callback = callback
        self._update_callback = update_callback
        self._audit_has_warnings = False

    @staticmethod
    def _evidence(
        *,
        label: str,
        value: str,
        source: ReasoningTraceSource,
        source_field: str,
    ) -> ReasoningTraceEvidence:
        return ReasoningTraceEvidence(
            label=label,
            value=value,
            source=source,
            source_field=source_field,
        )

    def _upsert_trace_stage(
        self,
        *,
        stage_id: ReasoningTraceStageId,
        summary: str,
        content: list[ReasoningTraceEvidence],
        status: PublicThinkingStatus,
        elapsed_seconds: float = 0.0,
    ) -> None:
        stage, icon = _TRACE_STAGE_COPY[stage_id]
        for index, current in enumerate(self.reasoning_trace):
            if current.stage_id is stage_id:
                self.reasoning_trace[index] = ReasoningTraceStage(
                    sequence=current.sequence,
                    stage_id=stage_id,
                    stage=stage,
                    icon=icon,
                    summary=summary,
                    content=content[:6],
                    status=status,
                    elapsed_seconds=max(float(elapsed_seconds), 0.0),
                )
                return
        self.reasoning_trace.append(
            ReasoningTraceStage(
                sequence=len(self.reasoning_trace) + 1,
                stage_id=stage_id,
                stage=stage,
                icon=icon,
                summary=summary,
                content=content[:6],
                status=status,
                elapsed_seconds=max(float(elapsed_seconds), 0.0),
            )
        )

    def _upsert_section(
        self,
        *,
        slot: str,
        kind: PublicThinkingKind,
        title: Literal["题意", "写法", "资料", "成稿"],
        summary: str,
        details: list[str],
        status: PublicThinkingStatus,
        elapsed_seconds: float = 0.0,
    ) -> None:
        slots = self._section_slots.setdefault(title, {})
        slots[slot] = details
        ordered_details = [
            detail
            for slot_name in _SLOT_ORDER[title]
            for detail in slots.get(slot_name, [])
        ][:4]
        section = PublicThinkingSection(
            kind=kind,
            title=title,
            summary=summary,
            details=ordered_details,
            status=status,
            elapsed_seconds=max(float(elapsed_seconds), 0.0),
        )
        for index, current in enumerate(self.sections):
            if current.title == title:
                self.sections[index] = section
                break
        else:
            self.sections.append(section)
        self.sections.sort(key=lambda item: _SECTION_ORDER[item.title])

    def _append(
        self,
        kind: PublicThinkingKind,
        status: PublicThinkingStatus,
        summary: str,
        *,
        elapsed_seconds: float = 0.0,
    ) -> None:
        event = PublicThinkingEvent(
            sequence=len(self.events) + 1,
            kind=kind,
            status=status,
            summary=summary,
            elapsed_seconds=max(float(elapsed_seconds), 0.0),
        )
        self.events.append(event)
        if self._callback is not None:
            try:
                self._callback(event)
            except Exception:  # noqa: BLE001 - UI reporting is best effort
                pass
        if self._update_callback is not None:
            update = PublicThinkingUpdate(
                event=event,
                sections=[section.model_copy(deep=True) for section in self.sections],
                reasoning_trace=[
                    stage.model_copy(deep=True) for stage in self.reasoning_trace
                ],
            )
            try:
                self._update_callback(update)
            except Exception:  # noqa: BLE001 - UI reporting is best effort
                pass

    def stage(self, name: str) -> None:
        mapped = _STAGE_COPY.get(name)
        if mapped is None:
            return
        trace_stage_id = _RUNTIME_STAGE_TO_TRACE[name]
        self._upsert_trace_stage(
            stage_id=trace_stage_id,
            summary=_TRACE_STAGE_IN_PROGRESS_COPY[trace_stage_id],
            content=[
                self._evidence(
                    label="运行阶段",
                    value=name,
                    source=ReasoningTraceSource.RUNTIME_STAGE,
                    source_field="RuntimeStage.name",
                )
            ],
            status=PublicThinkingStatus.IN_PROGRESS,
        )
        kind, summary = mapped
        self._append(kind, PublicThinkingStatus.IN_PROGRESS, summary)

    def classification(
        self,
        *,
        subject: object,
        stage: object,
        is_new_problem: object,
        intent: object = "",
    ) -> None:
        if (
            isinstance(subject, str)
            and subject in _ALLOWED_SUBJECTS
            and isinstance(stage, str)
            and stage in _ALLOWED_STAGES
            and isinstance(is_new_problem, bool)
        ):
            relation = "新题" if is_new_problem else "当前题的追问"
            details = [f"{subject}{relation}"]
            stage_label = "" if stage == "未知" else stage
            target = f"{stage_label}{subject}{relation}"
            event_summary = f"已识别为{target}；题目与作答目标已经确认。"
            if isinstance(intent, str) and intent in _PUBLIC_INTENTS:
                details.append(intent)
                event_summary = f"已识别为{target}；本轮目标是{intent}。"
        else:
            details = ["题目类型已确认"]
            event_summary = "题目类型与作答目标已经确认。"
        summary = "已确认题目与作答目标。"
        trace_content: list[ReasoningTraceEvidence] = []
        if isinstance(subject, str) and subject in _ALLOWED_SUBJECTS:
            trace_content.append(
                self._evidence(
                    label="学科",
                    value=subject,
                    source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                    source_field="RequestClassification.subject",
                )
            )
        if isinstance(stage, str) and stage in _ALLOWED_STAGES:
            trace_content.append(
                self._evidence(
                    label="学段",
                    value=stage,
                    source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                    source_field="RequestClassification.stage",
                )
            )
        if isinstance(is_new_problem, bool):
            trace_content.append(
                self._evidence(
                    label="问题关系",
                    value="新问题" if is_new_problem else "当前问题的追问",
                    source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                    source_field="RequestClassification.is_new_problem",
                )
            )
        if isinstance(intent, str) and intent in _PUBLIC_INTENTS:
            trace_content.append(
                self._evidence(
                    label="回答目标",
                    value=intent,
                    source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                    source_field="RequestClassification.answer_mode",
                )
            )
        self._upsert_trace_stage(
            stage_id=ReasoningTraceStageId.REQUEST_ANALYSIS,
            summary=event_summary,
            content=trace_content,
            status=PublicThinkingStatus.COMPLETED,
        )
        self._upsert_section(
            slot="classification",
            kind=PublicThinkingKind.CLASSIFICATION,
            title="题意",
            summary=summary,
            details=details,
            status=PublicThinkingStatus.COMPLETED,
        )
        self._append(
            PublicThinkingKind.CLASSIFICATION,
            PublicThinkingStatus.COMPLETED,
            event_summary,
        )

    def strategy(self, label: object) -> None:
        detail = (
            label
            if isinstance(label, str) and label in _PUBLIC_STRATEGIES
            else "处理方式已确定"
        )
        summary = "已确定本轮处理方式。"
        event_summary = (
            _PUBLIC_STRATEGY_EVENT_COPY.get(label, summary)
            if isinstance(label, str) else summary
        )
        self._upsert_trace_stage(
            stage_id=ReasoningTraceStageId.TEACHING_STRATEGY,
            summary=event_summary,
            content=[
                self._evidence(
                    label="教学节奏",
                    value=detail,
                    source=ReasoningTraceSource.RUNTIME_STAGE,
                    source_field="RuntimeStage.public_strategy",
                )
            ],
            status=PublicThinkingStatus.COMPLETED,
        )
        self._upsert_section(
            slot="strategy",
            kind=PublicThinkingKind.STRATEGY,
            title="写法",
            summary=summary,
            details=[detail],
            status=PublicThinkingStatus.COMPLETED,
        )
        self._append(
            PublicThinkingKind.STRATEGY,
            PublicThinkingStatus.COMPLETED,
            event_summary,
        )

    @staticmethod
    def _public_reason(reason: object) -> str:
        if not isinstance(reason, str):
            return ""
        return " ".join(reason.split())[:120]

    def retrieval_decision(self, *, mode: object, reason: object = "") -> None:
        self._retrieval_mode = (
            mode if isinstance(mode, str) and mode in _ROUTE_MODE_COPY else ""
        )
        if isinstance(mode, str) and mode in _ROUTE_MODE_COPY:
            details = [_ROUTE_MODE_COPY[mode]]
            public_reason = self._public_reason(reason)
            if public_reason:
                details.append(public_reason)
        else:
            details = ["资料使用方式已确定"]
        if mode == "none":
            self._section_slots.setdefault("资料", {}).pop("retrieval", None)
        summary = "已确定本轮资料使用方式。"
        event_summary = (
            _ROUTE_EVENT_COPY.get(mode, summary)
            if isinstance(mode, str) else summary
        )
        trace_mode = (
            _ROUTE_MODE_COPY[mode]
            if isinstance(mode, str) and mode in _ROUTE_MODE_COPY
            else "资料策略已确定"
        )
        self._retrieval_trace_evidence["mode"] = self._evidence(
            label="资料策略",
            value=trace_mode,
            source=ReasoningTraceSource.RETRIEVAL_DECISION,
            source_field="RetrievalDecision.mode",
        )
        if self._retrieval_mode == "none":
            self._retrieval_trace_evidence.pop("result", None)
        self._upsert_trace_stage(
            stage_id=ReasoningTraceStageId.KNOWLEDGE_DECISION,
            summary=event_summary,
            content=list(self._retrieval_trace_evidence.values()),
            status=PublicThinkingStatus.COMPLETED,
        )
        self._upsert_section(
            slot="route",
            kind=PublicThinkingKind.RETRIEVAL,
            title="资料",
            summary=summary,
            details=details,
            status=PublicThinkingStatus.COMPLETED,
        )
        self._append(
            PublicThinkingKind.RETRIEVAL,
            PublicThinkingStatus.COMPLETED,
            event_summary,
        )

    def composition(self, label: object) -> None:
        detail = (
            label
            if isinstance(label, str) and label in _PUBLIC_COMPOSITIONS
            else "成稿结构已确定"
        )
        summary = "已确定回答的公开结构。"
        event_summary = (
            _PUBLIC_COMPOSITION_EVENT_COPY.get(label, summary)
            if isinstance(label, str) else summary
        )
        self._upsert_trace_stage(
            stage_id=ReasoningTraceStageId.ANSWER_GENERATION,
            summary=event_summary,
            content=[
                self._evidence(
                    label="公开结构",
                    value=detail,
                    source=ReasoningTraceSource.RUNTIME_STAGE,
                    source_field="RuntimeStage.public_composition",
                )
            ],
            status=PublicThinkingStatus.COMPLETED,
        )
        self._upsert_section(
            slot="composition",
            kind=PublicThinkingKind.GENERATION,
            title="成稿",
            summary=summary,
            details=[detail],
            status=PublicThinkingStatus.COMPLETED,
        )
        self._append(
            PublicThinkingKind.GENERATION,
            PublicThinkingStatus.COMPLETED,
            event_summary,
        )

    def teaching_plan(self, contract: TeachingContract) -> None:
        mode = _TEACHING_MODE_COPY[contract.response_mode]
        structure = _TEACHING_STRUCTURE_COPY[contract.response_mode]
        self._upsert_trace_stage(
            stage_id=ReasoningTraceStageId.TEACHING_STRATEGY,
            summary=f"已确定采用“{mode}”的教学方式。",
            content=[
                self._evidence(
                    label="教学方式",
                    value=mode,
                    source=ReasoningTraceSource.TEACHING_CONTRACT,
                    source_field="TeachingContract.response_mode",
                ),
                self._evidence(
                    label="回答结构",
                    value=structure,
                    source=ReasoningTraceSource.TEACHING_CONTRACT,
                    source_field="TeachingContract.response_mode",
                ),
                self._evidence(
                    label="应用到当前题",
                    value="允许" if contract.may_apply_to_active_problem else "禁止",
                    source=ReasoningTraceSource.TEACHING_CONTRACT,
                    source_field="TeachingContract.may_apply_to_active_problem",
                ),
                self._evidence(
                    label="披露最终答案",
                    value="允许" if contract.may_reveal_final_answer else "禁止",
                    source=ReasoningTraceSource.TEACHING_CONTRACT,
                    source_field="TeachingContract.may_reveal_final_answer",
                ),
            ],
            status=PublicThinkingStatus.COMPLETED,
        )

    def agent(self, event: AgentTraceEvent) -> None:
        if event.event_type == "classification":
            self._record_classification(event.details)
            return
        if event.event_type == "memory_resolution":
            self._record_memory_resolution(event)
            return
        if event.event_type == "decision":
            mapped = _TOOL_START_COPY.get(event.tool_name)
            if mapped is not None:
                kind, summary = mapped
                self._tool_trace_evidence[event.tool_name] = self._evidence(
                    label=_TOOL_PUBLIC_NAME[event.tool_name],
                    value="执行中",
                    source=ReasoningTraceSource.AGENT_TRACE,
                    source_field="AgentTraceEvent.tool_name",
                )
                self._upsert_trace_stage(
                    stage_id=ReasoningTraceStageId.TOOL_EXECUTION,
                    summary="正在执行本轮所需工具。",
                    content=list(self._tool_trace_evidence.values()),
                    status=PublicThinkingStatus.IN_PROGRESS,
                )
                self._append(kind, PublicThinkingStatus.IN_PROGRESS, summary)
            return
        if event.event_type == "tool_result":
            self._record_tool_result(event)
            return
        if event.event_type == "fallback":
            self._append(
                PublicThinkingKind.WARNING,
                PublicThinkingStatus.WARNING,
                "编排判断暂时不可用，将使用已有证据安全完成回答。",
            )

    def _record_memory_resolution(self, event: AgentTraceEvent) -> None:
        if event.ok is not True:
            self._upsert_trace_stage(
                stage_id=ReasoningTraceStageId.LEARNING_CONTEXT,
                summary="学习上下文暂时不可用，不影响本轮回答。",
                content=[
                    self._evidence(
                        label="记忆读取",
                        value="未完成，按 fail-open 继续",
                        source=ReasoningTraceSource.AGENT_TRACE,
                        source_field="AgentTraceEvent.ok",
                    )
                ],
                status=PublicThinkingStatus.WARNING,
            )
            self._append(
                PublicThinkingKind.WARNING,
                PublicThinkingStatus.WARNING,
                "学习情况暂时不可用，不影响本轮回答。",
            )
            return
        status = event.details.get("status")
        if status not in {"none", "unique", "ambiguous"}:
            return
        status_copy = {
            "none": "未找到相关学习主题",
            "unique": "找到唯一相关学习主题",
            "ambiguous": "找到多个可能的学习主题，需要澄清",
        }[status]
        evidence = [
            self._evidence(
                label="主题匹配",
                value=status_copy,
                source=ReasoningTraceSource.MEMORY_RESOLUTION,
                source_field="StudentMemoryResolution.status",
            )
        ]
        candidate_count = event.details.get("candidate_count")
        if (
            isinstance(candidate_count, int)
            and not isinstance(candidate_count, bool)
            and candidate_count >= 0
        ):
            evidence.append(
                self._evidence(
                    label="候选主题",
                    value=f"{candidate_count} 个",
                    source=ReasoningTraceSource.MEMORY_RESOLUTION,
                    source_field="StudentMemoryResolution.candidates",
                )
            )
        self._upsert_trace_stage(
            stage_id=ReasoningTraceStageId.LEARNING_CONTEXT,
            summary=status_copy + "。",
            content=evidence,
            status=(
                PublicThinkingStatus.WARNING
                if status == "ambiguous"
                else PublicThinkingStatus.COMPLETED
            ),
        )

    def _record_classification(self, details: dict[str, object]) -> None:
        scope = details.get("scope")
        subject = details.get("subject")
        stage = details.get("stage")
        is_new_problem = details.get("is_new_problem")
        if (
            scope == "k9"
            and isinstance(subject, str)
            and subject in _ALLOWED_SUBJECTS
            and isinstance(stage, str)
            and stage in _ALLOWED_STAGES
            and isinstance(is_new_problem, bool)
        ):
            relation = "新题" if is_new_problem else "当前题的追问"
            section_details = [f"{subject}{relation}"]
            intent = details.get("intent")
            if isinstance(intent, str) and intent in _PUBLIC_INTENTS:
                section_details.append(intent)
            trace_content = [
                self._evidence(
                    label="学科",
                    value=subject,
                    source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                    source_field="RequestClassification.subject",
                ),
                self._evidence(
                    label="学段",
                    value=stage,
                    source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                    source_field="RequestClassification.stage",
                ),
                self._evidence(
                    label="问题关系",
                    value="新问题" if is_new_problem else "当前问题的追问",
                    source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                    source_field="RequestClassification.is_new_problem",
                ),
            ]
            self._upsert_trace_stage(
                stage_id=ReasoningTraceStageId.REQUEST_ANALYSIS,
                summary=f"已确定为{stage}{subject}{relation}。",
                content=trace_content,
                status=PublicThinkingStatus.COMPLETED,
            )
            self._upsert_section(
                slot="classification",
                kind=PublicThinkingKind.CLASSIFICATION,
                title="题意",
                summary="已确认题目与作答目标。",
                details=section_details,
                status=PublicThinkingStatus.COMPLETED,
            )
            self._append(
                PublicThinkingKind.CLASSIFICATION,
                PublicThinkingStatus.COMPLETED,
                f"已确定为{stage}{subject}{relation}。",
            )
            return
        if scope == "off_topic":
            self._upsert_trace_stage(
                stage_id=ReasoningTraceStageId.REQUEST_ANALYSIS,
                summary="当前请求不属于 K9 学习范围。",
                content=[
                    self._evidence(
                        label="请求范围",
                        value="非 K9 学习请求",
                        source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                        source_field="RequestClassification.scope",
                    )
                ],
                status=PublicThinkingStatus.COMPLETED,
            )
            self._upsert_section(
                slot="classification",
                kind=PublicThinkingKind.CLASSIFICATION,
                title="题意",
                summary="已确认题目与作答目标。",
                details=["非 K9 学习请求"],
                status=PublicThinkingStatus.COMPLETED,
            )
            self._append(
                PublicThinkingKind.CLASSIFICATION,
                PublicThinkingStatus.COMPLETED,
                "当前问题不属于小学或初中学习范围，将简短回应并引导回学习内容。",
            )
            return
        self._upsert_trace_stage(
            stage_id=ReasoningTraceStageId.REQUEST_ANALYSIS,
            summary="已完成请求范围判断。",
            content=[
                self._evidence(
                    label="请求范围",
                    value="已确认",
                    source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                    source_field="RequestClassification.scope",
                )
            ],
            status=PublicThinkingStatus.COMPLETED,
        )
        self._upsert_section(
            slot="classification",
            kind=PublicThinkingKind.CLASSIFICATION,
            title="题意",
            summary="已确认题目与作答目标。",
            details=["题目类型已确认"],
            status=PublicThinkingStatus.COMPLETED,
        )
        self._append(
            PublicThinkingKind.CLASSIFICATION,
            PublicThinkingStatus.COMPLETED,
            "题目类型已经确认。",
        )

    @staticmethod
    def _hit_count(details: dict[str, object]) -> int:
        value = details.get("hit_count")
        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        return max(value, 0)

    def _record_tool_result(self, event: AgentTraceEvent) -> None:
        if event.tool_name not in _TOOL_START_COPY:
            return
        if event.ok is not True:
            failures = {
                "load_skill": "教学方法加载未完成，将使用基础教学策略继续。",
                "inspect_student_context": "学习情况暂时不可用，不影响本轮回答。",
                "search_local_knowledge": "本地参考暂时不可用，将按现有证据继续。",
                "search_quality_qa": "优质问答参考暂时不可用，将按现有证据继续。",
                "search_web": "联网参考暂时不可用，将使用已有证据继续。",
            }
            kind = _TOOL_START_COPY[event.tool_name][0]
            summary = failures[event.tool_name]
            self._tool_trace_evidence[event.tool_name] = self._evidence(
                label=_TOOL_PUBLIC_NAME[event.tool_name],
                value="未完成，使用已有证据继续",
                source=ReasoningTraceSource.AGENT_TRACE,
                source_field="AgentTraceEvent.ok",
            )
            self._upsert_trace_stage(
                stage_id=ReasoningTraceStageId.TOOL_EXECUTION,
                summary="部分工具未完成，Agent 将使用已有证据继续。",
                content=list(self._tool_trace_evidence.values()),
                status=PublicThinkingStatus.WARNING,
                elapsed_seconds=event.elapsed_seconds,
            )
            title = (
                "资料"
                if kind is PublicThinkingKind.RETRIEVAL
                else "写法"
            )
            self._upsert_section(
                slot=(
                    "local_tool"
                    if event.tool_name == "search_local_knowledge"
                    else "quality_tool"
                    if event.tool_name == "search_quality_qa"
                    else "web_tool"
                    if event.tool_name == "search_web"
                    else "skill"
                    if event.tool_name == "load_skill"
                    else "context"
                ),
                kind=kind,
                title=title,
                summary=summary,
                details=[
                    (
                        f"{summary} · {event.elapsed_seconds:.3f} 秒"
                        if event.elapsed_seconds > 0
                        else summary
                    )
                ],
                status=PublicThinkingStatus.WARNING,
                elapsed_seconds=event.elapsed_seconds,
            )
            self._append(
                kind,
                PublicThinkingStatus.WARNING,
                summary,
                elapsed_seconds=event.elapsed_seconds,
            )
            return

        hit_count = self._hit_count(event.details)
        successes = {
            "load_skill": "合适的教学方法已经准备完成。",
            "inspect_student_context": "已查看与当前题目有关的学习情况。",
            "search_local_knowledge": f"找到 {hit_count} 条可用的本地参考。",
            "search_quality_qa": f"找到 {hit_count} 条可用的优质问答参考。",
            "search_web": f"找到 {hit_count} 条可用的联网参考。",
        }
        kind = _TOOL_START_COPY[event.tool_name][0]
        summary = successes[event.tool_name]
        result_value = "已完成"
        if event.tool_name in {
            "search_local_knowledge",
            "search_quality_qa",
            "search_web",
        }:
            result_value = f"已完成，命中 {hit_count} 条"
        self._tool_trace_evidence[event.tool_name] = self._evidence(
            label=_TOOL_PUBLIC_NAME[event.tool_name],
            value=result_value,
            source=ReasoningTraceSource.AGENT_TRACE,
            source_field=(
                "AgentTraceEvent.details.hit_count"
                if event.tool_name
                in {"search_local_knowledge", "search_quality_qa", "search_web"}
                else "AgentTraceEvent.ok"
            ),
        )
        self._upsert_trace_stage(
            stage_id=ReasoningTraceStageId.TOOL_EXECUTION,
            summary="本轮所需工具已返回可用结果。",
            content=list(self._tool_trace_evidence.values()),
            status=PublicThinkingStatus.COMPLETED,
            elapsed_seconds=event.elapsed_seconds,
        )
        title = "资料" if kind is PublicThinkingKind.RETRIEVAL else "写法"
        self._upsert_section(
            slot=(
                "local_tool"
                if event.tool_name == "search_local_knowledge"
                else "quality_tool"
                if event.tool_name == "search_quality_qa"
                else "web_tool"
                if event.tool_name == "search_web"
                else "skill"
                if event.tool_name == "load_skill"
                else "context"
            ),
            kind=kind,
            title=title,
            summary=summary,
            details=[
                (
                    f"{summary} · {event.elapsed_seconds:.3f} 秒"
                    if event.elapsed_seconds > 0
                    else summary
                )
            ],
            status=PublicThinkingStatus.COMPLETED,
            elapsed_seconds=event.elapsed_seconds,
        )
        self._append(
            kind,
            PublicThinkingStatus.COMPLETED,
            summary,
            elapsed_seconds=event.elapsed_seconds,
        )

    def verification(self, result: VerificationResult) -> None:
        if result.status is VerificationStatus.VERIFIED_CORRECT:
            status = PublicThinkingStatus.COMPLETED
            summary = "学生当前步骤已经通过确定性验证。"
        elif result.status is VerificationStatus.VERIFIED_INCORRECT:
            status = PublicThinkingStatus.WARNING
            summary = "已发现当前步骤中的确定性错误，将据此进行纠正。"
        elif result.status is VerificationStatus.NOT_APPLICABLE:
            status = PublicThinkingStatus.COMPLETED
            summary = "当前内容不需要运行数学确定性验证。"
        else:
            status = PublicThinkingStatus.WARNING
            summary = "当前题型暂无确定性验证，回答将保留必要的不确定性说明。"
        if result.status in {
            VerificationStatus.VERIFIED_CORRECT,
            VerificationStatus.VERIFIED_INCORRECT,
        }:
            self._upsert_trace_stage(
                stage_id=ReasoningTraceStageId.STEP_VERIFICATION,
                summary=summary,
                content=[
                    self._evidence(
                        label="确定性校验",
                        value=(
                            "通过"
                            if result.status is VerificationStatus.VERIFIED_CORRECT
                            else "发现错误"
                        ),
                        source=ReasoningTraceSource.VERIFICATION_RESULT,
                        source_field="VerificationResult.status",
                    )
                ],
                status=status,
            )
        self._upsert_section(
            slot="verification",
            kind=PublicThinkingKind.VERIFICATION,
            title="写法",
            summary="已确定本轮处理方式。",
            details=[summary],
            status=status,
        )
        self._append(PublicThinkingKind.VERIFICATION, status, summary)

    def retrieval(
        self,
        *,
        source: str,
        hit_count: object,
        online_triggered: bool,
        online_status: str,
    ) -> None:
        safe_count = (
            max(hit_count, 0)
            if isinstance(hit_count, int) and not isinstance(hit_count, bool)
            else 0
        )
        if source.startswith("retrieval_error"):
            status = PublicThinkingStatus.WARNING
            summary = "参考资料检索暂时不可用，将使用已有知识继续。"
            details = ["资料检索暂时不可用"]
        elif safe_count == 0:
            status = PublicThinkingStatus.WARNING
            summary = "暂未找到可用参考，将基于题目和已有知识继续。"
            details = ["未找到可用参考"]
        elif online_triggered and online_status == "used":
            status = PublicThinkingStatus.COMPLETED
            summary = f"已找到 {safe_count} 条可用参考，其中包含联网补充资料。"
            details = [f"找到 {safe_count} 条可用参考", "包含联网补充资料"]
        else:
            status = PublicThinkingStatus.COMPLETED
            summary = f"已找到 {safe_count} 条可用参考。"
            details = [f"找到 {safe_count} 条可用参考"]
        if self._retrieval_mode != "none":
            self._retrieval_trace_evidence["result"] = self._evidence(
                label="资料结果",
                value=(
                    f"命中 {safe_count} 条"
                    if safe_count > 0
                    else "未获得可用命中"
                ),
                source=ReasoningTraceSource.RAG_CONTEXT,
                source_field="RagContext.hits",
            )
            self._upsert_trace_stage(
                stage_id=ReasoningTraceStageId.KNOWLEDGE_DECISION,
                summary=summary,
                content=list(self._retrieval_trace_evidence.values()),
                status=status,
            )
        self._upsert_section(
            slot="retrieval",
            kind=PublicThinkingKind.RETRIEVAL,
            title="资料",
            summary="已整理本轮可用资料。",
            details=details,
            status=status,
        )
        self._append(PublicThinkingKind.RETRIEVAL, status, summary)

    def audit(
        self,
        *,
        repair_performed: bool,
        remaining_warning_count: int,
        semantic_audit_performed: bool | None = None,
        deterministic_boundary_checked: bool = False,
        boundary_passed: bool | None = None,
        post_repair_audit_performed: bool = False,
        post_repair_boundary_checked: bool = False,
    ) -> None:
        self._audit_has_warnings = remaining_warning_count > 0
        if self._audit_has_warnings:
            status = PublicThinkingStatus.WARNING
            summary = "回答已经完成审计，但仍保留必要的风险提示。"
            detail = "成稿保留必要的风险提示"
        elif repair_performed:
            status = PublicThinkingStatus.COMPLETED
            summary = "修正后的回答已经通过教学内容与节奏审计。"
            detail = "修正后的成稿已通过审计"
        else:
            status = PublicThinkingStatus.COMPLETED
            summary = "最终回答已经通过教学内容与节奏审计。"
            detail = "成稿已通过内容与节奏审计"
        trace_evidence = [
            self._evidence(
                label="公开警告",
                value=str(max(remaining_warning_count, 0)),
                source=ReasoningTraceSource.RUNTIME_STAGE,
                source_field="RuntimeStage.remaining_warning_count",
            )
        ]
        if semantic_audit_performed is not None:
            trace_evidence.append(
                self._evidence(
                    label="语义边界审计",
                    value="已执行" if semantic_audit_performed else "未执行",
                    source=ReasoningTraceSource.RUNTIME_STAGE,
                    source_field="RuntimeStage.semantic_audit_performed",
                )
            )
        if semantic_audit_performed and boundary_passed is not None:
            trace_evidence.append(
                self._evidence(
                    label="边界审计结果",
                    value="通过" if boundary_passed else "需要修正",
                    source=ReasoningTraceSource.BOUNDARY_AUDIT,
                    source_field="BoundaryAudit.passed",
                )
            )
        if deterministic_boundary_checked:
            trace_evidence.append(
                self._evidence(
                    label="确定性边界复核",
                    value="已执行",
                    source=ReasoningTraceSource.RUNTIME_STAGE,
                    source_field="RuntimeStage.deterministic_boundary_checked",
                )
            )
        if semantic_audit_performed:
            trace_summary = (
                "语义边界审计已通过。"
                if boundary_passed is not False
                else "语义边界审计发现需要修正的内容。"
            )
        elif deterministic_boundary_checked:
            trace_summary = "已完成确定性边界复核。"
        else:
            trace_summary = "已完成公开输出检查。"
        if self._audit_has_warnings:
            trace_summary = "公开输出检查完成，并保留必要风险提示。"
        self._upsert_trace_stage(
            stage_id=ReasoningTraceStageId.QUALITY_VALIDATION,
            summary=trace_summary,
            content=trace_evidence,
            status=status,
        )
        if repair_performed:
            self._upsert_trace_stage(
                stage_id=ReasoningTraceStageId.ANSWER_REPAIR,
                summary="已根据验证或边界检查结果修正回答。",
                content=[
                    self._evidence(
                        label="回答修正",
                        value="已执行",
                        source=ReasoningTraceSource.RUNTIME_STAGE,
                        source_field="RuntimeStage.repair_performed",
                    )
                ],
                status=PublicThinkingStatus.COMPLETED,
            )
        if post_repair_audit_performed or post_repair_boundary_checked:
            final_source = (
                ReasoningTraceSource.BOUNDARY_AUDIT
                if post_repair_audit_performed
                else ReasoningTraceSource.RUNTIME_STAGE
            )
            final_source_field = (
                "BoundaryAudit.passed"
                if post_repair_audit_performed
                else "RuntimeStage.post_repair_boundary_checked"
            )
            self._upsert_trace_stage(
                stage_id=ReasoningTraceStageId.FINAL_RECHECK,
                summary="修正后的回答已完成最终边界复核。",
                content=[
                    self._evidence(
                        label="修正后复核",
                        value=(
                            "通过"
                            if boundary_passed is not False
                            else "仍需安全回退"
                        ),
                        source=final_source,
                        source_field=final_source_field,
                    )
                ],
                status=(
                    PublicThinkingStatus.WARNING
                    if boundary_passed is False
                    else PublicThinkingStatus.COMPLETED
                ),
            )
        self._upsert_section(
            slot="audit",
            kind=PublicThinkingKind.AUDIT,
            title="成稿",
            summary="已完成回答整理与公开审计。",
            details=[detail],
            status=status,
        )
        self._append(PublicThinkingKind.AUDIT, status, summary)

    def complete(self, elapsed_seconds: float) -> None:
        status = (
            PublicThinkingStatus.WARNING
            if self._audit_has_warnings
            else PublicThinkingStatus.COMPLETED
        )
        summary = (
            "思考过程已完成，回答中保留了必要的风险提示。"
            if self._audit_has_warnings
            else "思考过程已完成。"
        )
        completion_details = (
            []
            if self._section_slots.get("成稿")
            else ["回答已组织完成"]
        )
        for index, trace_stage in enumerate(self.reasoning_trace):
            if trace_stage.status is PublicThinkingStatus.IN_PROGRESS:
                self.reasoning_trace[index] = trace_stage.model_copy(
                    update={
                        "status": PublicThinkingStatus.COMPLETED,
                        "elapsed_seconds": max(float(elapsed_seconds), 0.0),
                    },
                    deep=True,
                )
        self._upsert_section(
            slot="completion",
            kind=PublicThinkingKind.COMPLETE,
            title="成稿",
            summary=summary,
            details=completion_details,
            status=status,
            elapsed_seconds=elapsed_seconds,
        )
        self._append(
            PublicThinkingKind.COMPLETE,
            status,
            summary,
            elapsed_seconds=elapsed_seconds,
        )


def format_public_thinking_event(event: PublicThinkingEvent) -> str:
    icon = {
        PublicThinkingStatus.IN_PROGRESS: "◦",
        PublicThinkingStatus.COMPLETED: "✓",
        PublicThinkingStatus.WARNING: "⚠️",
    }[event.status]
    elapsed = (
        f" · {event.elapsed_seconds:.3f} 秒"
        if event.elapsed_seconds > 0
        else ""
    )
    return f"{icon} {event.summary}{elapsed}"
