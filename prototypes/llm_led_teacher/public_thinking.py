"""Whitelisted, student-visible summaries of real teacher-agent work."""

from __future__ import annotations

from enum import Enum
from typing import Callable

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
        self._callback = callback
        self._audit_has_warnings = False

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
            self._append(
                PublicThinkingKind.CLASSIFICATION,
                PublicThinkingStatus.COMPLETED,
                f"已确定为{stage}{subject}{relation}。",
            )
            return
        if scope == "off_topic":
            self._append(
                PublicThinkingKind.CLASSIFICATION,
                PublicThinkingStatus.COMPLETED,
                "当前问题不属于小学或初中学习范围，将简短回应并引导回学习内容。",
            )
            return
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
            self._append(
                _TOOL_START_COPY[event.tool_name][0],
                PublicThinkingStatus.WARNING,
                failures[event.tool_name],
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
        self._append(
            _TOOL_START_COPY[event.tool_name][0],
            PublicThinkingStatus.COMPLETED,
            successes[event.tool_name],
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
        self._append(PublicThinkingKind.VERIFICATION, status, summary)

    def retrieval(
        self,
        *,
        source: str,
        hit_count: int,
        online_triggered: bool,
        online_status: str,
    ) -> None:
        safe_count = max(int(hit_count), 0)
        if source.startswith("retrieval_error"):
            self._append(
                PublicThinkingKind.RETRIEVAL,
                PublicThinkingStatus.WARNING,
                "参考资料检索暂时不可用，将使用已有知识继续。",
            )
        elif safe_count == 0:
            self._append(
                PublicThinkingKind.RETRIEVAL,
                PublicThinkingStatus.WARNING,
                "暂未找到可用参考，将基于题目和已有知识继续。",
            )
        elif online_triggered and online_status == "used":
            self._append(
                PublicThinkingKind.RETRIEVAL,
                PublicThinkingStatus.COMPLETED,
                f"已找到 {safe_count} 条可用参考，其中包含联网补充资料。",
            )
        else:
            self._append(
                PublicThinkingKind.RETRIEVAL,
                PublicThinkingStatus.COMPLETED,
                f"已找到 {safe_count} 条可用参考。",
            )

    def audit(self, *, repair_performed: bool, remaining_warning_count: int) -> None:
        self._audit_has_warnings = remaining_warning_count > 0
        if self._audit_has_warnings:
            self._append(
                PublicThinkingKind.AUDIT,
                PublicThinkingStatus.WARNING,
                "回答已经完成审计，但仍保留必要的风险提示。",
            )
        elif repair_performed:
            self._append(
                PublicThinkingKind.AUDIT,
                PublicThinkingStatus.COMPLETED,
                "修正后的回答已经通过教学内容与节奏审计。",
            )
        else:
            self._append(
                PublicThinkingKind.AUDIT,
                PublicThinkingStatus.COMPLETED,
                "最终回答已经通过教学内容与节奏审计。",
            )

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
