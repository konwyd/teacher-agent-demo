"""DeepSeek-style rendering for whitelisted public thinking events."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Protocol

from prototypes.llm_led_teacher.public_thinking import (
    PublicThinkingEvent,
    format_public_thinking_event,
)


class StatusWriter(Protocol):
    def write(self, value: str) -> object: ...

    def update(self, **kwargs: object) -> object: ...


class LiveThinkingPanel:
    """Append safe events and keep a Streamlit-like status title current."""

    def __init__(
        self,
        status: StatusWriter,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._status = status
        self._clock = clock
        self._started_at = clock()

    def append(self, event: PublicThinkingEvent) -> None:
        self._status.write(format_public_thinking_event(event))
        elapsed = max(round(self._clock() - self._started_at), 0)
        self._status.update(label=f"🧠 思考中…（已用时 {elapsed} 秒）")

    def complete(self, total_seconds: float) -> None:
        seconds = max(round(total_seconds), 0)
        self._status.update(
            label=f"🧠 已思考（用时 {seconds} 秒）",
            state="complete",
            expanded=True,
        )

    def fail(self) -> None:
        self._status.update(
            label="🧠 思考中断",
            state="error",
            expanded=True,
        )
