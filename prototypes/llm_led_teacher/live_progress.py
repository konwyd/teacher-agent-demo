"""User-visible, auditable progress trace for one teaching turn."""

from __future__ import annotations

from string import punctuation
from time import perf_counter
from typing import Callable, Protocol

from prototypes.llm_led_teacher.orchestration_models import AgentTraceEvent


class StatusWriter(Protocol):
    def write(self, value: str) -> object: ...


class TraceTurn(Protocol):
    agent_trace: list[AgentTraceEvent]


_STAGES = {
    "classification": (
        "Agent · 请求分类",
        "识别学科、学段、新题/续题与学生要求的回答模式。",
    ),
    "planning": (
        "Agent · 规划",
        "根据分类与已有工具结果选择下一步；真实决策和 Tool 结果会逐条显示。",
    ),
    "tool": (
        "Agent · Tool",
        "执行 Agent 选择的公开工具调用。",
    ),
    "generating": (
        "Answer Guardrails · 生成回答",
        "按教学契约生成面向学生的回答。",
    ),
    "boundary_check": (
        "Answer Guardrails · 边界检查",
        "核对允许内容、应用权限、答案披露权限与可验证事实。",
    ),
    "repairing": (
        "Answer Guardrails · 修正",
        "按边界审计给出的修正指令生成安全版本。",
    ),
    "formatting": (
        "Answer Guardrails · Markdown 格式化",
        "展示已持久化的规范化 Markdown，并保持块级公式完整。",
    ),
}

_STAGE_ALIASES = {
    "observing": "classification",
    "retrieving": "planning",
    "auditing": "boundary_check",
}


def escape_markdown_text(value: object) -> str:
    """Escape untrusted text before interpolating it into Streamlit Markdown."""

    return "".join(
        f"\\{character}" if character in punctuation else character
        for character in str(value)
    )


def agent_trace_rows(turn: TraceTurn) -> list[dict[str, str]]:
    """Return the explicitly public trace fields for persistent UI rendering."""

    rows = []
    for event in getattr(turn, "agent_trace", []) or []:
        status = "成功" if event.ok is True else ("失败" if event.ok is False else "信息")
        rows.append(
            {
                "步骤": str(event.step),
                "类型": event.event_type,
                "Tool": event.tool_name or "—",
                "状态": status,
                "耗时": (
                    f"{event.elapsed_seconds:.3f} 秒"
                    if event.elapsed_seconds > 0
                    else "—"
                ),
                "摘要": event.summary,
            }
        )
    return rows


class LiveProgressTrace:
    """Append completed work and elapsed time to a Streamlit-like status box."""

    def __init__(
        self,
        status: StatusWriter,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._status = status
        self._clock = clock
        self._stage = "classification"
        self._started_at = clock()
        self._write_start(self._stage)

    def _write_start(self, stage: str) -> None:
        title, detail = _STAGES[stage]
        self._status.write(f"⏳ **{title}**\n\n{detail}")

    def update(self, stage: str) -> None:
        stage = _STAGE_ALIASES.get(stage, stage)
        if stage == self._stage:
            return
        now = self._clock()
        elapsed = now - self._started_at
        title, _ = _STAGES[self._stage]
        self._status.write(f"✅ {title}完成 · **{elapsed:.3f} 秒**")
        if stage == "complete":
            self._status.write("✅ **Agent Loop + Answer Guardrails · 完成**")
            return
        self._stage = stage
        self._started_at = now
        self._write_start(stage)

    def update_agent_event(self, event: AgentTraceEvent) -> None:
        """Render only the event's explicitly public summary."""
        icon = "✅" if event.ok is True else ("⚠️" if event.ok is False else "🧭")
        event_label = {
            "classification": "分类",
            "decision": "规划",
            "tool_result": "Tool",
            "teaching_contract": "教学契约",
            "fallback": "安全收口",
        }.get(event.event_type, event.event_type)
        event_label = escape_markdown_text(event_label)
        tool = (
            f" · Tool: {escape_markdown_text(event.tool_name)}"
            if event.tool_name
            else ""
        )
        summary = escape_markdown_text(event.summary)
        elapsed = (
            f" · {event.elapsed_seconds:.3f} 秒"
            if event.elapsed_seconds > 0
            else ""
        )
        self._status.write(
            f"{icon} **Agent {event.step} · {event_label}**{tool}："
            f"{summary}{elapsed}"
        )
