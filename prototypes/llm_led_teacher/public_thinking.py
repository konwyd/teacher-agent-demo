"""Whitelisted, student-visible summaries of real teacher-agent work."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Literal

from pydantic import BaseModel, Field

from app.agent.state import VerificationResult, VerificationStatus
from prototypes.llm_led_teacher.orchestration_models import AgentTraceEvent


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


ThinkingEventCallback = Callable[[PublicThinkingEvent], None]


_STAGE_COPY = {
    "observing": (
        PublicThinkingKind.CLASSIFICATION,
        "正在识别题目和学生意图……",
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
_SECTION_ORDER = {"题意": 0, "写法": 1, "资料": 2, "成稿": 3}
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
    "search_web": (
        PublicThinkingKind.RETRIEVAL,
        "本地资料不足，正在补充联网参考……",
    ),
}


class PublicThinkingRecorder:
    """Build public events from a deliberately narrow set of scalar inputs."""

    def __init__(self, callback: ThinkingEventCallback | None = None) -> None:
        self.events: list[PublicThinkingEvent] = []
        self.sections: list[PublicThinkingSection] = []
        self._callback = callback
        self._audit_has_warnings = False

    def _upsert_section(
        self,
        *,
        kind: PublicThinkingKind,
        title: Literal["题意", "写法", "资料", "成稿"],
        summary: str,
        details: list[str],
        status: PublicThinkingStatus,
        elapsed_seconds: float = 0.0,
    ) -> None:
        section = PublicThinkingSection(
            kind=kind,
            title=title,
            summary=summary,
            details=details,
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

    def stage(self, name: str) -> None:
        mapped = _STAGE_COPY.get(name)
        if mapped is None:
            return
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
            if isinstance(intent, str) and intent in _PUBLIC_INTENTS:
                details.append(intent)
        else:
            details = ["题目类型已确认"]
        summary = "已确认题目与作答目标。"
        self._upsert_section(
            kind=PublicThinkingKind.CLASSIFICATION,
            title="题意",
            summary=summary,
            details=details,
            status=PublicThinkingStatus.COMPLETED,
        )
        self._append(
            PublicThinkingKind.CLASSIFICATION,
            PublicThinkingStatus.COMPLETED,
            summary,
        )

    def strategy(self, label: object) -> None:
        detail = (
            label
            if isinstance(label, str) and label in _PUBLIC_STRATEGIES
            else "处理方式已确定"
        )
        summary = "已确定本轮处理方式。"
        self._upsert_section(
            kind=PublicThinkingKind.STRATEGY,
            title="写法",
            summary=summary,
            details=[detail],
            status=PublicThinkingStatus.COMPLETED,
        )
        self._append(
            PublicThinkingKind.STRATEGY,
            PublicThinkingStatus.COMPLETED,
            summary,
        )

    @staticmethod
    def _public_reason(reason: object) -> str:
        if not isinstance(reason, str):
            return ""
        return " ".join(reason.split())[:120]

    def retrieval_decision(self, *, mode: object, reason: object = "") -> None:
        if isinstance(mode, str) and mode in _ROUTE_MODE_COPY:
            details = [_ROUTE_MODE_COPY[mode]]
            public_reason = self._public_reason(reason)
            if public_reason:
                details.append(public_reason)
        else:
            details = ["资料使用方式已确定"]
        summary = "已确定本轮资料使用方式。"
        self._upsert_section(
            kind=PublicThinkingKind.RETRIEVAL,
            title="资料",
            summary=summary,
            details=details,
            status=PublicThinkingStatus.COMPLETED,
        )
        self._append(
            PublicThinkingKind.RETRIEVAL,
            PublicThinkingStatus.COMPLETED,
            summary,
        )

    def composition(self, label: object) -> None:
        detail = (
            label
            if isinstance(label, str) and label in _PUBLIC_COMPOSITIONS
            else "成稿结构已确定"
        )
        summary = "已确定回答的公开结构。"
        self._upsert_section(
            kind=PublicThinkingKind.GENERATION,
            title="成稿",
            summary=summary,
            details=[detail],
            status=PublicThinkingStatus.COMPLETED,
        )
        self._append(
            PublicThinkingKind.GENERATION,
            PublicThinkingStatus.COMPLETED,
            summary,
        )

    def agent(self, event: AgentTraceEvent) -> None:
        if event.event_type == "classification":
            self._record_classification(event.details)
            return
        if event.event_type == "decision":
            mapped = _TOOL_START_COPY.get(event.tool_name)
            if mapped is not None:
                kind, summary = mapped
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
            self._upsert_section(
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
            self._upsert_section(
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
        self._upsert_section(
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
                "search_web": "联网参考暂时不可用，将使用已有证据继续。",
            }
            kind = _TOOL_START_COPY[event.tool_name][0]
            summary = failures[event.tool_name]
            title = (
                "资料"
                if kind is PublicThinkingKind.RETRIEVAL
                else "写法"
            )
            self._upsert_section(
                kind=kind,
                title=title,
                summary=summary,
                details=[summary],
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
            "search_web": f"找到 {hit_count} 条可用的联网参考。",
        }
        kind = _TOOL_START_COPY[event.tool_name][0]
        summary = successes[event.tool_name]
        title = "资料" if kind is PublicThinkingKind.RETRIEVAL else "写法"
        self._upsert_section(
            kind=kind,
            title=title,
            summary=summary,
            details=[summary],
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
        self._upsert_section(
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
        self._upsert_section(
            kind=PublicThinkingKind.RETRIEVAL,
            title="资料",
            summary="已整理本轮可用资料。",
            details=details,
            status=status,
        )
        self._append(PublicThinkingKind.RETRIEVAL, status, summary)

    def audit(self, *, repair_performed: bool, remaining_warning_count: int) -> None:
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
        self._upsert_section(
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
        current_details = next(
            (
                section.details
                for section in self.sections
                if section.title == "成稿"
            ),
            ["回答已组织完成"],
        )
        self._upsert_section(
            kind=PublicThinkingKind.COMPLETE,
            title="成稿",
            summary=summary,
            details=current_details,
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
