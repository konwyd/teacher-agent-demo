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


@dataclass(frozen=True)
class KvPersistenceDisplay:
    enabled: bool
    slot_id: int | None
    restore_status: str
    save_status: str
    filename: str
    directory: str
    error: str


def build_kv_persistence_display(
    metadata: Mapping[str, object],
) -> KvPersistenceDisplay:
    """Read KV lifecycle fields without evaluating metadata keys as variables."""

    slot_raw = metadata.get("kv_persistence_slot")
    return KvPersistenceDisplay(
        enabled=bool(metadata.get("kv_persistence_enabled", False)),
        slot_id=int(slot_raw) if slot_raw is not None else None,
        restore_status=str(
            metadata.get("kv_persistence_restore_status", "not_attempted")
        ),
        save_status=str(metadata.get("kv_persistence_save_status", "not_attempted")),
        filename=str(metadata.get("kv_persistence_filename", "") or ""),
        directory=str(metadata.get("kv_persistence_directory", "") or ""),
        error=str(metadata.get("kv_persistence_error", "") or ""),
    )


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
