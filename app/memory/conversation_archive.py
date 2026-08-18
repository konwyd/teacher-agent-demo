"""Persistent per-user conversation archives for the Streamlit prototype."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_CONVERSATION_DB = "data/runtime/conversations.sqlite3"


class ConversationArchiveError(RuntimeError):
    """Base class for persistent conversation errors."""


class ConversationNotFoundError(ConversationArchiveError):
    """The requested conversation does not exist for the specified user."""


class InvalidConversationDataError(ConversationArchiveError):
    """Stored episode JSON is missing or malformed."""


@dataclass(frozen=True)
class ConversationRecord:
    conversation_id: str
    user_id: str
    title: str
    episode: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ConversationSummary:
    conversation_id: str
    title: str
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _normalize_title(title: str) -> str:
    normalized = " ".join(title.split()).strip()
    return (normalized or "新对话")[:80]


def _encode_episode(episode: dict[str, Any]) -> str:
    return json.dumps(episode, ensure_ascii=False, separators=(",", ":"))


def _decode_episode(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidConversationDataError("历史对话数据已损坏，无法读取。") from exc
    if not isinstance(payload, dict):
        raise InvalidConversationDataError("历史对话数据格式不正确。")
    return payload


class LocalConversationStore:
    """SQLite store that always scopes conversations by their owning user."""

    def __init__(self, path: str | Path = DEFAULT_CONVERSATION_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    episode_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                ON conversations (user_id, updated_at DESC)
                """
            )
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def create(
        self,
        user_id: str,
        episode: dict[str, Any],
        *,
        title: str = "新对话",
    ) -> ConversationRecord:
        conversation_id = f"conv_{uuid.uuid4().hex}"
        timestamp = _now()
        record = ConversationRecord(
            conversation_id=conversation_id,
            user_id=user_id,
            title=_normalize_title(title),
            episode=episode,
            created_at=timestamp,
            updated_at=timestamp,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id, user_id, title, episode_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.conversation_id,
                    record.user_id,
                    record.title,
                    _encode_episode(record.episode),
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def save(
        self,
        user_id: str,
        conversation_id: str,
        episode: dict[str, Any],
        *,
        title: str | None = None,
    ) -> ConversationRecord:
        updated_at = _now()
        with self._connect() as connection:
            if title is None:
                cursor = connection.execute(
                    """
                    UPDATE conversations
                    SET episode_json = ?, updated_at = ?
                    WHERE conversation_id = ? AND user_id = ?
                    """,
                    (_encode_episode(episode), updated_at, conversation_id, user_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE conversations
                    SET title = ?, episode_json = ?, updated_at = ?
                    WHERE conversation_id = ? AND user_id = ?
                    """,
                    (
                        _normalize_title(title),
                        _encode_episode(episode),
                        updated_at,
                        conversation_id,
                        user_id,
                    ),
                )
            if cursor.rowcount != 1:
                raise ConversationNotFoundError("找不到当前账号下的这段历史对话。")
        return self.load(user_id, conversation_id)

    def load(self, user_id: str, conversation_id: str) -> ConversationRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT conversation_id, user_id, title, episode_json, created_at, updated_at
                FROM conversations
                WHERE conversation_id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            ).fetchone()
        if row is None:
            raise ConversationNotFoundError("找不到当前账号下的这段历史对话。")
        return ConversationRecord(
            conversation_id=row["conversation_id"],
            user_id=row["user_id"],
            title=row["title"],
            episode=_decode_episode(row["episode_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_for_user(self, user_id: str) -> list[ConversationSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT conversation_id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC, conversation_id DESC
                """,
                (user_id,),
            ).fetchall()
        return [
            ConversationSummary(
                conversation_id=row["conversation_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def list_updated_between(
        self,
        updated_after: str,
        updated_through: str,
    ) -> list[ConversationRecord]:
        """System-only incremental scan for the curated Q&A batch pipeline.

        The caller receives ownership metadata for traceability, but must never
        send user_id to the extraction model or store it in Chroma.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT conversation_id, user_id, title, episode_json,
                       created_at, updated_at
                FROM conversations
                WHERE updated_at > ? AND updated_at <= ?
                ORDER BY updated_at ASC, conversation_id ASC
                """,
                (updated_after, updated_through),
            ).fetchall()
        return [
            ConversationRecord(
                conversation_id=row["conversation_id"],
                user_id=row["user_id"],
                title=row["title"],
                episode=_decode_episode(row["episode_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]


@lru_cache
def get_local_conversation_store() -> LocalConversationStore:
    return LocalConversationStore(
        os.getenv("TEACHER_CONVERSATION_DB", DEFAULT_CONVERSATION_DB)
    )
