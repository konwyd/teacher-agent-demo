"""Pure display semantics for the LLM context-usage meter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextUsageDisplay:
    """Current request occupancy plus read-only historical capacity metadata."""

    used_tokens: int
    current_window_tokens: int
    historical_window_tokens: int | None
    usage_ratio: float
    historical_limit_changed: bool


def build_context_usage_display(
    metadata: Mapping[str, object],
    current_window_tokens: int,
) -> ContextUsageDisplay:
    """Build meter values without letting historical limits replace runtime config."""

    current_window = max(int(current_window_tokens), 0)
    used = max(int(metadata.get("context_used_tokens", 0) or 0), 0)
    historical_raw = metadata.get("context_window_tokens")
    historical = int(historical_raw) if historical_raw else None
    ratio = used / current_window if current_window else 0.0
    return ContextUsageDisplay(
        used_tokens=used,
        current_window_tokens=current_window,
        historical_window_tokens=historical,
        usage_ratio=ratio,
        historical_limit_changed=(
            historical is not None and historical != current_window
        ),
    )
