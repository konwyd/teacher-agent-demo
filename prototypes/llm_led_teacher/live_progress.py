"""DeepSeek-style rendering for whitelisted public thinking events."""

from __future__ import annotations

from typing import Callable, Protocol

from prototypes.llm_led_teacher.public_thinking import (
    PublicThinkingEvent,
    PublicThinkingUpdate,
    format_public_thinking_event,
)
from prototypes.llm_led_teacher.thinking_state import (
    begin_thinking,
    complete_thinking,
    fail_thinking,
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
        render_update: Callable[[PublicThinkingUpdate], None] | None = None,
        **_legacy_options: object,
    ) -> None:
        self._status = status
        self._render_update = render_update
        self._state = begin_thinking(None)

    def append(self, update: PublicThinkingEvent | PublicThinkingUpdate) -> None:
        if isinstance(update, PublicThinkingUpdate):
            if self._render_update is not None:
                self._render_update(update)
                return
            event = update.event
        else:
            event = update
        self._status.write(format_public_thinking_event(event))

    def complete(self, total_seconds: float) -> None:
        completed = complete_thinking(self._state)
        if completed == self._state:
            return
        self._state = completed
        seconds = max(round(total_seconds), 0)
        self._status.update(
            label=f"🧠 已思考（用时 {seconds} 秒）",
            state="complete",
            expanded=False,
        )

    def fail(self) -> None:
        failed = fail_thinking(self._state)
        if failed == self._state:
            return
        self._state = failed
        self._status.update(
            label="🧠 思考中断",
            state="error",
            expanded=True,
        )
