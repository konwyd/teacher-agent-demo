from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.memory.conversation_archive import ConversationRecord
from prototypes.llm_led_teacher.conversation_ui import (
    build_conversation_history,
    derive_conversation_title,
    episode_has_content,
)


NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def _record(
    conversation_id: str,
    title: str,
    *,
    episode: dict | None = None,
    pending: str = "",
    days_ago: int = 0,
    hours_ago: int = 0,
) -> ConversationRecord:
    timestamp = NOW - timedelta(days=days_ago, hours=hours_ago)
    payload = episode or {"turns": [], "pending_student_message": pending}
    return ConversationRecord(
        conversation_id=conversation_id,
        user_id="usr_test",
        title=title,
        episode=payload,
        created_at=timestamp.isoformat(),
        updated_at=timestamp.isoformat(),
    )


def test_episode_content_requires_a_turn_or_pending_student_message() -> None:
    assert episode_has_content({"turns": [], "pending_student_message": ""}) is False
    assert episode_has_content({"turns": [], "pending_student_message": "  "}) is False
    assert episode_has_content({"turns": [{"student_message": "你好"}]}) is True
    assert episode_has_content({"turns": [], "pending_student_message": "待回复"}) is True


def test_history_omits_empty_records_and_uses_first_student_message_as_title() -> None:
    records = [
        _record("conv_empty", "新对话"),
        _record(
            "conv_math",
            "新对话",
            episode={
                "turns": [{"student_message": "  杨辉三角   是什么？  "}],
                "pending_student_message": "",
            },
            hours_ago=2,
        ),
    ]

    items = build_conversation_history(records, now=NOW)

    assert [(item.conversation_id, item.title, item.group) for item in items] == [
        ("conv_math", "杨辉三角 是什么？", "今天")
    ]


def test_pending_message_is_visible_and_history_groups_use_fixed_boundaries() -> None:
    records = [
        _record("today", "今天的题", pending="今天", hours_ago=2),
        _record("week", "一周内的题", pending="一周", days_ago=3),
        _record("month", "一月内的题", pending="一月", days_ago=12),
        _record("older", "更早的题", pending="更早", days_ago=40),
    ]

    items = build_conversation_history(records, now=NOW)

    assert [(item.conversation_id, item.group) for item in items] == [
        ("today", "今天"),
        ("week", "7 天内"),
        ("month", "30 天内"),
        ("older", "更早"),
    ]


def test_history_is_sorted_by_updated_time_even_if_input_is_not() -> None:
    records = [
        _record("older", "旧问题", pending="旧问题", days_ago=5),
        _record("newer", "新问题", pending="新问题", hours_ago=1),
    ]

    items = build_conversation_history(records, now=NOW)

    assert [item.conversation_id for item in items] == ["newer", "older"]


def test_meaningful_stored_title_wins_and_all_titles_are_normalized_and_bounded() -> None:
    meaningful = _record(
        "meaningful",
        "  已有   标题  ",
        episode={"turns": [{"student_message": "被忽略的问题"}]},
    )
    long_pending = _record(
        "pending",
        "新对话",
        pending="问" * 100,
    )

    assert derive_conversation_title(meaningful) == "已有 标题"
    assert derive_conversation_title(long_pending) == "问" * 80
