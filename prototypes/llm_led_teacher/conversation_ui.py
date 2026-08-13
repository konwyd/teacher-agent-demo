"""Pure display semantics for the prototype conversation history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from app.memory.conversation_archive import ConversationRecord


_UNTITLED = "新对话"


@dataclass(frozen=True)
class ConversationHistoryItem:
    """A visible conversation entry ready for sidebar rendering."""

    conversation_id: str
    title: str
    group: str
    updated_at: str
    timestamp_label: str
    is_current: bool = False


def _normalize_title(value: object) -> str:
    return " ".join(str(value or "").split()).strip()[:80]


def episode_has_content(episode: Mapping[str, object]) -> bool:
    """Return whether a stored episode contains a completed or pending turn."""

    turns = episode.get("turns") or []
    pending = str(episode.get("pending_student_message") or "").strip()
    return bool(turns or pending)


def derive_conversation_title(record: ConversationRecord) -> str:
    """Prefer a meaningful stored title, then derive one from student text."""

    stored_title = _normalize_title(record.title)
    if stored_title and stored_title != _UNTITLED:
        return stored_title

    turns = record.episode.get("turns") or []
    if isinstance(turns, Sequence) and turns:
        first_turn = turns[0]
        if isinstance(first_turn, Mapping):
            first_message = _normalize_title(first_turn.get("student_message"))
            if first_message:
                return first_message

    pending = _normalize_title(record.episode.get("pending_student_message"))
    return pending or _UNTITLED


def _parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _history_group(updated_at: datetime, now: datetime) -> str:
    localized = updated_at.astimezone(now.tzinfo or timezone.utc)
    days_ago = max((now.date() - localized.date()).days, 0)
    if days_ago == 0:
        return "今天"
    if days_ago < 7:
        return "7 天内"
    if days_ago < 30:
        return "30 天内"
    return "更早"


def _timestamp_label(updated_at: datetime, now: datetime) -> str:
    """Format a compact, real timestamp for the visible directory entry."""

    timezone_info = now.tzinfo or timezone.utc
    localized = updated_at.astimezone(timezone_info)
    localized_now = now.astimezone(timezone_info)
    if localized.date() == localized_now.date():
        return localized.strftime("%H:%M")
    if localized.year == localized_now.year:
        return f"{localized.month}月{localized.day}日"
    return f"{localized.year}年{localized.month}月{localized.day}日"


def build_conversation_history(
    records: Sequence[ConversationRecord],
    *,
    now: datetime,
    current_conversation_id: str | None = None,
) -> list[ConversationHistoryItem]:
    """Filter, sort, title, and group stored conversations for the sidebar."""

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    visible = [record for record in records if episode_has_content(record.episode)]
    visible.sort(
        key=lambda record: (_parse_timestamp(record.updated_at), record.conversation_id),
        reverse=True,
    )
    items: list[ConversationHistoryItem] = []
    for record in visible:
        updated_at = _parse_timestamp(record.updated_at)
        items.append(
            ConversationHistoryItem(
                conversation_id=record.conversation_id,
                title=derive_conversation_title(record),
                group=_history_group(updated_at, now),
                updated_at=record.updated_at,
                timestamp_label=_timestamp_label(updated_at, now),
                is_current=record.conversation_id == current_conversation_id,
            )
        )
    return items
