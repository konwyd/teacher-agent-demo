"""Structured, per-user long-term memory for the 8502 Teacher Agent.

The module deliberately keeps recall and write-back deterministic.  Memory is a
bounded, untrusted teaching reference; it never overrides the current message,
the teaching contract, or deterministic verification.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from app.memory.conversation_archive import (
    DEFAULT_CONVERSATION_DB,
    LocalConversationStore,
)


DEFAULT_STUDENT_MEMORY_DB = DEFAULT_CONVERSATION_DB
_CONTINUATION_MARKERS = (
    "上次",
    "之前",
    "刚才",
    "继续",
    "接着",
    "这个",
    "那个",
    "这篇",
    "那篇",
)
_SUBJECT_KEYS = {
    "语文": "chinese",
    "数学": "mathematics",
    "英语": "english",
    "物理": "physics",
    "化学": "chemistry",
    "生物": "biology",
    "历史": "history",
    "地理": "geography",
    "道法": "politics",
}
_DOMAIN_CUES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "语文": (
        ("composition", ("作文", "写作", "开头", "结尾", "植物朋友")),
        ("reading", ("阅读", "文章", "段落", "中心思想", "赏析")),
    ),
    "数学": (
        ("fractions", ("分数", "分子", "分母", "通分", "约分")),
        ("word_problems", ("应用题", "一共", "还剩", "速度", "路程")),
        ("equations", ("方程", "移项", "等式", "x=", "x =")),
        ("geometry", ("几何", "三角形", "圆", "面积", "周长")),
    ),
    "英语": (
        ("vocabulary", ("单词", "词汇", "词义", "拼写", "背词")),
        ("grammar", ("语法", "时态", "句型", "从句", "主谓")),
        ("reading", ("阅读", "短文", "完形", "理解")),
    ),
}
_DOMAIN_LABELS = {
    "composition": "作文",
    "reading": "阅读",
    "fractions": "分数",
    "word_problems": "应用题",
    "equations": "方程",
    "geometry": "几何",
    "vocabulary": "词汇",
    "grammar": "语法",
    "general": "综合学习",
}
_PREFERENCE_CUES = {
    "一步一步": "分步骤讲解",
    "讲慢点": "慢节奏讲解",
    "慢一点": "慢节奏讲解",
    "多举例": "多使用例子",
    "举个例子": "多使用例子",
    "不要直接给答案": "先引导，不直接给最终答案",
    "别直接给答案": "先引导，不直接给最终答案",
    "简单一点": "使用更基础的语言",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _clean(value: object, limit: int) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def _json_load(raw: object, default: Any) -> Any:
    try:
        return json.loads(str(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def _normalize_topic(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()[:120]


def _contains_continuation(message: str) -> bool:
    return any(marker in message for marker in _CONTINUATION_MARKERS)


def detect_domain(subject: str, text: str) -> str:
    for domain, cues in _DOMAIN_CUES.get(subject, ()):
        if any(cue.lower() in text.lower() for cue in cues):
            return domain
    return "general"


def extract_topic(subject: str, domain: str, text: str) -> str:
    """Extract an explicit topic, falling back to a stable domain label."""

    for pattern in (r"《([^》]{1,60})》", r"[“\"]([^”\"]{1,60})[”\"]"):
        match = re.search(pattern, text)
        if match:
            return _clean(match.group(1), 80)
    if subject == "语文" and domain == "composition":
        match = re.search(r"(我的[\u4e00-\u9fff]{2,20})", text)
        if match:
            candidate = re.split(
                r"怎么|如何|继续|开头|结尾|这篇|那篇|作文|[，。？！]",
                match.group(1),
                maxsplit=1,
            )[0]
            if len(candidate) >= 4:
                return _clean(candidate, 80)
    return _DOMAIN_LABELS.get(domain, _DOMAIN_LABELS["general"])


class MemoryResolutionStatus(str, Enum):
    NONE = "none"
    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"


class GlobalStudentProfile(BaseModel):
    grade: str = Field(default="", max_length=20)
    preferred_name: str = Field(default="", max_length=40)
    teaching_preferences: list[str] = Field(default_factory=list, max_length=8)

    def has_content(self) -> bool:
        return bool(self.grade or self.preferred_name or self.teaching_preferences)


class StudentProfileSnapshot(BaseModel):
    student_profile_id: str
    global_profile: GlobalStudentProfile = Field(
        default_factory=GlobalStudentProfile
    )
    created_at: str = ""
    updated_at: str = ""


class TopicMemoryEvidence(BaseModel):
    conversation_id: str
    turn_id: int = Field(ge=1)
    kind: str = Field(max_length=60)
    summary: str = Field(max_length=240)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_now)


class TopicMemory(BaseModel):
    memory_id: str
    student_profile_id: str
    subject: str = Field(max_length=20)
    subject_key: str = Field(max_length=40)
    domain: str = Field(max_length=60)
    topic: str = Field(max_length=120)
    normalized_topic: str = Field(max_length=120)
    problem_context: str = Field(default="", max_length=800)
    summary: str = Field(default="", max_length=400)
    progress: str = Field(default="", max_length=400)
    difficulties: list[str] = Field(default_factory=list, max_length=8)
    next_step: str = Field(default="", max_length=300)
    evidence: list[TopicMemoryEvidence] = Field(default_factory=list, max_length=20)
    source_conversation_ids: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    status: str = Field(default="active", max_length=20)
    created_at: str = ""
    updated_at: str = ""
    last_accessed_at: str = ""
    version: int = Field(default=1, ge=1)

    def prompt_payload(self) -> dict[str, Any]:
        """Return bounded fields only; raw historic utterances stay out of prompts."""

        return {
            "subject": self.subject,
            "domain": self.domain,
            "topic": self.topic,
            "problem_context": self.problem_context,
            "summary": self.summary,
            "progress": self.progress,
            "difficulties": self.difficulties[-4:],
            "next_step": self.next_step,
            "confidence": self.confidence,
            "status": self.status,
            "source_conversation_ids": self.source_conversation_ids[-3:],
        }


class MemoryCandidate(BaseModel):
    memory: TopicMemory
    match_score: float = Field(ge=0.0, le=1.0)
    source: str = Field(default="profile", max_length=30)


class StudentMemoryResolution(BaseModel):
    student_profile_id: str
    global_profile: GlobalStudentProfile = Field(
        default_factory=GlobalStudentProfile
    )
    status: MemoryResolutionStatus = MemoryResolutionStatus.NONE
    subject: str = ""
    domain: str = ""
    query_topic: str = ""
    selected: TopicMemory | None = None
    candidates: list[MemoryCandidate] = Field(default_factory=list, max_length=5)
    clarification_question: str = ""
    should_restore_context: bool = False
    reason: str = ""

    def prompt_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "student_profile_id": self.student_profile_id,
            "global": self.global_profile.model_dump(mode="json"),
            "usage": (
                "soft_personalization_only; current message, teaching_contract and "
                "deterministic verification always win"
            ),
        }
        if self.selected is not None:
            payload["topic_memory"] = self.selected.prompt_payload()
        return payload


@dataclass(frozen=True)
class StudentMemoryIdentity:
    user_id: str
    conversation_id: str


class LearnerMemoryStore:
    """SQLite persistence for global profiles, topic memories and an outbox."""

    def __init__(self, path: str | Path) -> None:
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
                CREATE TABLE IF NOT EXISTS student_profiles (
                    student_profile_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL UNIQUE,
                    global_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS student_topic_memories (
                    memory_id TEXT PRIMARY KEY,
                    student_profile_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    subject_key TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    normalized_topic TEXT NOT NULL,
                    memory_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    UNIQUE(student_profile_id, subject, domain, normalized_topic),
                    FOREIGN KEY(student_profile_id) REFERENCES student_profiles(student_profile_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_topic_memory_lookup
                ON student_topic_memories (
                    student_profile_id, subject, status, updated_at DESC
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS student_memory_outbox (
                    job_id TEXT PRIMARY KEY,
                    student_profile_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    error TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(student_profile_id, conversation_id, turn_id)
                )
                """
            )
            connection.execute(
                """
                UPDATE student_memory_outbox
                SET status = 'pending', updated_at = ?
                WHERE status = 'processing'
                """,
                (_now(),),
            )
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def get_or_create_profile(self, user_id: str) -> StudentProfileSnapshot:
        clean_user_id = _clean(user_id, 160)
        if not clean_user_id:
            raise ValueError("user_id must not be empty")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT student_profile_id, global_json, created_at, updated_at
                FROM student_profiles WHERE user_id = ?
                """,
                (clean_user_id,),
            ).fetchone()
            if row is None:
                now = _now()
                profile_id = f"profile_{uuid.uuid4().hex}"
                try:
                    connection.execute(
                        """
                        INSERT INTO student_profiles (
                            student_profile_id, user_id, global_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (profile_id, clean_user_id, "{}", now, now),
                    )
                except sqlite3.IntegrityError:
                    row = connection.execute(
                        """
                        SELECT student_profile_id, global_json, created_at, updated_at
                        FROM student_profiles WHERE user_id = ?
                        """,
                        (clean_user_id,),
                    ).fetchone()
                else:
                    return StudentProfileSnapshot(
                        student_profile_id=profile_id,
                        created_at=now,
                        updated_at=now,
                    )
        assert row is not None
        return StudentProfileSnapshot(
            student_profile_id=row["student_profile_id"],
            global_profile=GlobalStudentProfile.model_validate(
                _json_load(row["global_json"], {})
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_global_profile(
        self,
        student_profile_id: str,
        *,
        grade: str = "",
        preferred_name: str = "",
        teaching_preferences: Sequence[str] = (),
    ) -> StudentProfileSnapshot:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, global_json, created_at
                FROM student_profiles WHERE student_profile_id = ?
                """,
                (student_profile_id,),
            ).fetchone()
            if row is None:
                raise KeyError("student profile not found")
            current = GlobalStudentProfile.model_validate(
                _json_load(row["global_json"], {})
            )
            preferences = list(current.teaching_preferences)
            for preference in teaching_preferences:
                clean_preference = _clean(preference, 80)
                if clean_preference and clean_preference not in preferences:
                    preferences.append(clean_preference)
            updated = current.model_copy(
                update={
                    "grade": _clean(grade, 20) or current.grade,
                    "preferred_name": _clean(preferred_name, 40)
                    or current.preferred_name,
                    "teaching_preferences": preferences[-8:],
                }
            )
            now = _now()
            connection.execute(
                """
                UPDATE student_profiles SET global_json = ?, updated_at = ?
                WHERE student_profile_id = ?
                """,
                (
                    json.dumps(updated.model_dump(mode="json"), ensure_ascii=False),
                    now,
                    student_profile_id,
                ),
            )
        return StudentProfileSnapshot(
            student_profile_id=student_profile_id,
            global_profile=updated,
            created_at=row["created_at"],
            updated_at=now,
        )

    def list_topic_memories(
        self,
        student_profile_id: str,
        subject: str,
        *,
        limit: int = 24,
    ) -> list[TopicMemory]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_json FROM student_topic_memories
                WHERE student_profile_id = ? AND subject = ? AND status != 'deleted'
                ORDER BY updated_at DESC LIMIT ?
                """,
                (student_profile_id, subject, max(1, min(limit, 100))),
            ).fetchall()
        memories: list[TopicMemory] = []
        for row in rows:
            try:
                memories.append(
                    TopicMemory.model_validate(_json_load(row["memory_json"], {}))
                )
            except ValueError:
                continue
        return memories

    def touch_memory(self, memory_id: str) -> None:
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT memory_json FROM student_topic_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return
            memory = TopicMemory.model_validate(_json_load(row["memory_json"], {}))
            memory.last_accessed_at = now
            connection.execute(
                """
                UPDATE student_topic_memories
                SET memory_json = ?, last_accessed_at = ? WHERE memory_id = ?
                """,
                (json.dumps(memory.model_dump(mode="json"), ensure_ascii=False), now, memory_id),
            )

    def upsert_topic_memory(self, incoming: TopicMemory) -> TopicMemory:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT memory_json FROM student_topic_memories
                WHERE student_profile_id = ? AND subject = ?
                  AND domain = ? AND normalized_topic = ?
                """,
                (
                    incoming.student_profile_id,
                    incoming.subject,
                    incoming.domain,
                    incoming.normalized_topic,
                ),
            ).fetchone()
            now = _now()
            if row is None:
                memory = incoming.model_copy(
                    update={
                        "created_at": incoming.created_at or now,
                        "updated_at": now,
                        "last_accessed_at": incoming.last_accessed_at or now,
                    }
                )
                connection.execute(
                    """
                    INSERT INTO student_topic_memories (
                        memory_id, student_profile_id, subject, subject_key, domain,
                        topic, normalized_topic, memory_json, status, confidence,
                        created_at, updated_at, last_accessed_at, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.memory_id,
                        memory.student_profile_id,
                        memory.subject,
                        memory.subject_key,
                        memory.domain,
                        memory.topic,
                        memory.normalized_topic,
                        json.dumps(memory.model_dump(mode="json"), ensure_ascii=False),
                        memory.status,
                        memory.confidence,
                        memory.created_at,
                        memory.updated_at,
                        memory.last_accessed_at,
                        memory.version,
                    ),
                )
                return memory
            current = TopicMemory.model_validate(_json_load(row["memory_json"], {}))
            evidence_by_key = {
                (item.conversation_id, item.turn_id, item.kind): item
                for item in current.evidence
            }
            for item in incoming.evidence:
                evidence_by_key[(item.conversation_id, item.turn_id, item.kind)] = item
            sources = list(
                dict.fromkeys(
                    current.source_conversation_ids + incoming.source_conversation_ids
                )
            )[-12:]
            difficulties = list(
                dict.fromkeys(current.difficulties + incoming.difficulties)
            )[-8:]
            memory = current.model_copy(
                update={
                    "problem_context": incoming.problem_context or current.problem_context,
                    "summary": incoming.summary or current.summary,
                    "progress": incoming.progress or current.progress,
                    "difficulties": difficulties,
                    "next_step": incoming.next_step or current.next_step,
                    "evidence": list(evidence_by_key.values())[-20:],
                    "source_conversation_ids": sources,
                    "confidence": max(current.confidence, incoming.confidence),
                    "status": incoming.status or current.status,
                    "updated_at": now,
                    "version": current.version + 1,
                }
            )
            connection.execute(
                """
                UPDATE student_topic_memories
                SET topic = ?, memory_json = ?, status = ?, confidence = ?,
                    updated_at = ?, version = ? WHERE memory_id = ?
                """,
                (
                    memory.topic,
                    json.dumps(memory.model_dump(mode="json"), ensure_ascii=False),
                    memory.status,
                    memory.confidence,
                    memory.updated_at,
                    memory.version,
                    memory.memory_id,
                ),
            )
            return memory

    def enqueue(self, profile_id: str, conversation_id: str, turn_id: int, payload: dict[str, Any]) -> str:
        now = _now()
        job_id = f"memory_job_{uuid.uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO student_memory_outbox (
                    job_id, student_profile_id, conversation_id, turn_id,
                    payload_json, status, attempts, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', 0, '', ?, ?)
                """,
                (
                    job_id,
                    profile_id,
                    conversation_id,
                    turn_id,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT job_id FROM student_memory_outbox
                WHERE student_profile_id = ? AND conversation_id = ? AND turn_id = ?
                """,
                (profile_id, conversation_id, turn_id),
            ).fetchone()
        assert row is not None
        return str(row["job_id"])

    def claim_next_job(self) -> tuple[str, dict[str, Any]] | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT job_id, payload_json FROM student_memory_outbox
                WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE student_memory_outbox
                SET status = 'processing', attempts = attempts + 1, updated_at = ?
                WHERE job_id = ? AND status = 'pending'
                """,
                (_now(), row["job_id"]),
            )
        return str(row["job_id"]), dict(_json_load(row["payload_json"], {}))

    def finish_job(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE student_memory_outbox
                SET status = 'completed', error = '', updated_at = ? WHERE job_id = ?
                """,
                (_now(), job_id),
            )

    def fail_job(self, job_id: str, error: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT attempts FROM student_memory_outbox WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            terminal = row is not None and int(row["attempts"]) >= 3
            connection.execute(
                """
                UPDATE student_memory_outbox
                SET status = ?, error = ?, updated_at = ? WHERE job_id = ?
                """,
                (
                    "failed" if terminal else "pending",
                    _clean(error, 500),
                    _now(),
                    job_id,
                ),
            )

    def job_status(self, job_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM student_memory_outbox WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return str(row["status"]) if row else "missing"


class StudentMemoryService:
    """Two-stage profile load, topic resolution and durable async write-back."""

    def __init__(
        self,
        store: LearnerMemoryStore,
        *,
        archive_store: LocalConversationStore | None = None,
    ) -> None:
        self.store = store
        self.archive_store = archive_store
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="teacher-memory",
        )
        self._drain_lock = threading.Lock()
        self._futures_lock = threading.Lock()
        self._futures: set[Future[None]] = set()

    def load_profile(self, user_id: str) -> StudentProfileSnapshot:
        return self.store.get_or_create_profile(user_id)

    def resolve(
        self,
        profile: StudentProfileSnapshot,
        *,
        user_id: str,
        conversation_id: str,
        subject: str,
        stage: str,
        message: str,
        problem_context: str,
        is_new_problem: bool,
    ) -> StudentMemoryResolution:
        del stage, is_new_problem  # retained in the public contract for future policy
        domain = detect_domain(subject, f"{problem_context} {message}")
        query_topic = extract_topic(subject, domain, message)
        continuation = _contains_continuation(message)
        memories = self.store.list_topic_memories(
            profile.student_profile_id,
            subject,
        )
        memories.extend(
            self._archive_candidates(
                profile.student_profile_id,
                user_id,
                conversation_id,
                subject,
            )
        )
        memories = self._deduplicate(memories)
        explicit_topic = self._is_explicit_topic(subject, domain, message, query_topic)
        if not continuation and not explicit_topic:
            return StudentMemoryResolution(
                student_profile_id=profile.student_profile_id,
                global_profile=profile.global_profile,
                subject=subject,
                domain=domain,
                query_topic=query_topic,
                reason="当前消息没有承接表达或明确历史主题，不注入主题记忆",
            )
        candidates = self._score_candidates(
            memories,
            domain=domain,
            query_topic=query_topic,
            continuation=continuation,
            explicit_topic=explicit_topic,
        )
        eligible = [item for item in candidates if item.match_score >= 0.68]
        if not eligible:
            return StudentMemoryResolution(
                student_profile_id=profile.student_profile_id,
                global_profile=profile.global_profile,
                subject=subject,
                domain=domain,
                query_topic=query_topic,
                candidates=candidates[:5],
                reason="没有达到置信阈值的历史主题",
            )
        top = eligible[0]
        runner_up = eligible[1] if len(eligible) > 1 else None
        unique = runner_up is None or (
            top.match_score >= 0.85
            and top.match_score - runner_up.match_score >= 0.15
        )
        if unique:
            if top.source == "profile":
                self.store.touch_memory(top.memory.memory_id)
            return StudentMemoryResolution(
                student_profile_id=profile.student_profile_id,
                global_profile=profile.global_profile,
                status=MemoryResolutionStatus.UNIQUE,
                subject=subject,
                domain=domain,
                query_topic=query_topic,
                selected=top.memory,
                candidates=eligible[:5],
                should_restore_context=bool(
                    continuation and top.memory.problem_context
                ),
                reason="找到唯一高置信度历史主题",
            )
        choices = [item.memory.topic for item in eligible[:3]]
        question = self._clarification_question(choices)
        return StudentMemoryResolution(
            student_profile_id=profile.student_profile_id,
            global_profile=profile.global_profile,
            status=MemoryResolutionStatus.AMBIGUOUS,
            subject=subject,
            domain=domain,
            query_topic=query_topic,
            candidates=eligible[:5],
            clarification_question=question,
            reason="存在多个相近历史主题，需要学生确认",
        )

    def resolve_pending_choice(
        self,
        profile: StudentProfileSnapshot,
        *,
        subject: str,
        message: str,
        candidates: Sequence[TopicMemory],
    ) -> StudentMemoryResolution:
        """Resolve the student's answer to a previous ambiguity question."""

        bounded = list(candidates)[:3]
        if not bounded:
            return StudentMemoryResolution(
                student_profile_id=profile.student_profile_id,
                global_profile=profile.global_profile,
                subject=subject,
                reason="没有待确认的记忆候选",
            )
        clean_message = _clean(message, 160)
        ordinal_map = {
            "第一个": 0,
            "第1个": 0,
            "一": 0,
            "1": 0,
            "第二个": 1,
            "第2个": 1,
            "二": 1,
            "2": 1,
            "第三个": 2,
            "第3个": 2,
            "三": 2,
            "3": 2,
        }
        selected: TopicMemory | None = None
        for marker, index in ordinal_map.items():
            if marker in clean_message and index < len(bounded):
                selected = bounded[index]
                break
        normalized_message = _normalize_topic(clean_message)
        if selected is None and normalized_message:
            matches: list[tuple[float, TopicMemory]] = []
            for memory in bounded:
                similarity = SequenceMatcher(
                    None,
                    normalized_message,
                    memory.normalized_topic,
                ).ratio()
                exact_or_contained = (
                    normalized_message == memory.normalized_topic
                    or (
                        len(normalized_message) >= 3
                        and normalized_message in memory.normalized_topic
                    )
                    or (
                        len(memory.normalized_topic) >= 3
                        and memory.normalized_topic in normalized_message
                    )
                )
                if exact_or_contained or similarity >= 0.72:
                    matches.append(
                        (1.0 if exact_or_contained else similarity, memory)
                    )
            matches.sort(key=lambda item: item[0], reverse=True)
            if matches and (
                len(matches) == 1
                or matches[0][0] - matches[1][0] >= 0.15
            ):
                selected = matches[0][1]
        if selected is None:
            abandons_choice = bool(
                len(clean_message) > 80
                or any(
                    cue in clean_message
                    for cue in ("都不是", "换个话题", "新问题", "重新问")
                )
            )
            return StudentMemoryResolution(
                student_profile_id=profile.student_profile_id,
                global_profile=profile.global_profile,
                status=(
                    MemoryResolutionStatus.NONE
                    if abandons_choice else MemoryResolutionStatus.AMBIGUOUS
                ),
                subject=subject,
                candidates=[
                    MemoryCandidate(
                        memory=memory,
                        match_score=0.0,
                        source=(
                            "archive"
                            if memory.memory_id.startswith("archive_")
                            else "profile"
                        ),
                    )
                    for memory in bounded
                ],
                clarification_question=self._clarification_question(
                    [memory.topic for memory in bounded]
                ),
                reason=(
                    "学生放弃本次历史主题选择"
                    if abandons_choice else "学生回复尚未唯一指向待确认主题"
                ),
            )
        if not selected.memory_id.startswith("archive_"):
            self.store.touch_memory(selected.memory_id)
        return StudentMemoryResolution(
            student_profile_id=profile.student_profile_id,
            global_profile=profile.global_profile,
            status=MemoryResolutionStatus.UNIQUE,
            subject=subject,
            domain=selected.domain,
            query_topic=selected.topic,
            selected=selected,
            candidates=[
                MemoryCandidate(
                    memory=selected,
                    match_score=1.0,
                    source=(
                        "archive"
                        if selected.memory_id.startswith("archive_")
                        else "profile"
                    ),
                )
            ],
            should_restore_context=bool(selected.problem_context),
            reason="学生已确认唯一历史主题",
        )

    def enqueue_turn(
        self,
        identity: StudentMemoryIdentity,
        profile: StudentProfileSnapshot,
        payload: dict[str, Any],
    ) -> str:
        classification = dict(payload.get("classification") or {})
        if classification.get("scope") != "k9":
            return ""
        if (
            payload.get("memory_resolution_status")
            == MemoryResolutionStatus.AMBIGUOUS.value
        ):
            return ""
        turn_id = int(payload.get("turn_id") or 0)
        if turn_id < 1:
            return ""
        job_id = self.store.enqueue(
            profile.student_profile_id,
            identity.conversation_id,
            turn_id,
            {
                **payload,
                "student_profile_id": profile.student_profile_id,
                "user_id": identity.user_id,
                "conversation_id": identity.conversation_id,
            },
        )
        future = self._executor.submit(self._drain_outbox)
        with self._futures_lock:
            self._futures.add(future)
        future.add_done_callback(self._forget_future)
        return job_id

    def wait_for_idle(self, timeout: float = 5.0) -> None:
        with self._futures_lock:
            futures = list(self._futures)
        if futures:
            wait(futures, timeout=timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _forget_future(self, future: Future[None]) -> None:
        with self._futures_lock:
            self._futures.discard(future)

    def _drain_outbox(self) -> None:
        if not self._drain_lock.acquire(blocking=False):
            return
        try:
            while True:
                claimed = self.store.claim_next_job()
                if claimed is None:
                    return
                job_id, payload = claimed
                try:
                    self._extract_and_write(payload)
                except Exception as exc:  # noqa: BLE001 - durable retry boundary
                    self.store.fail_job(job_id, f"{type(exc).__name__}: {exc}")
                else:
                    self.store.finish_job(job_id)
        finally:
            self._drain_lock.release()

    def _extract_and_write(self, payload: Mapping[str, Any]) -> None:
        profile_id = _clean(payload.get("student_profile_id"), 160)
        message = _clean(payload.get("student_message"), 1200)
        grade_match = re.search(
            r"(?:我是|我上)?([一二三四五六七八九1-9])年级",
            message,
        )
        name_match = re.search(
            r"(?:我叫|叫我)([A-Za-z0-9_\-\u4e00-\u9fff]{1,12})",
            message,
        )
        preferences = [
            preference
            for cue, preference in _PREFERENCE_CUES.items()
            if cue in message
        ]
        if grade_match or name_match or preferences:
            self.store.update_global_profile(
                profile_id,
                grade=(f"{grade_match.group(1)}年级" if grade_match else ""),
                preferred_name=(name_match.group(1) if name_match else ""),
                teaching_preferences=preferences,
            )
        classification = dict(payload.get("classification") or {})
        subject = _clean(classification.get("subject"), 20)
        if not subject:
            return
        problem_context = _clean(payload.get("problem_context"), 800)
        domain = detect_domain(subject, f"{problem_context} {message}")
        topic = extract_topic(subject, domain, f"{problem_context} {message}")
        normalized_topic = _normalize_topic(topic)
        if not normalized_topic:
            return
        verification = dict(payload.get("verification") or {})
        verification_status = str(verification.get("status") or "")
        if verification_status == "verified_correct":
            progress = "学生本轮步骤已通过确定性验证。"
            evidence_kind = "verified_correct"
            confidence = 0.95
            difficulties: list[str] = []
        elif verification_status == "verified_incorrect":
            progress = "学生本轮步骤存在已验证错误，仍需继续讲解。"
            evidence_kind = "verified_incorrect"
            confidence = 0.95
            difficulties = [_clean(verification.get("summary"), 180)]
        elif any(word in message for word in ("不会", "不懂", "不明白", "没懂")):
            progress = "学生明确表示当前主题仍有困难。"
            evidence_kind = "expressed_difficulty"
            confidence = 0.85
            difficulties = ["学生明确表达不理解或需要帮助"]
        else:
            progress = f"学生最近继续学习“{topic}”。"
            evidence_kind = "learning_turn"
            confidence = 0.75
            difficulties = []
        contract = dict(payload.get("teaching_contract") or {})
        requested_target = _clean(contract.get("requested_target"), 180)
        next_step = (
            f"继续围绕“{requested_target}”推进。"
            if requested_target
            else f"继续学习“{topic}”。"
        )
        conversation_id = _clean(payload.get("conversation_id"), 160)
        turn_id = int(payload.get("turn_id") or 1)
        incoming = TopicMemory(
            memory_id=f"memory_{uuid.uuid4().hex}",
            student_profile_id=profile_id,
            subject=subject,
            subject_key=_SUBJECT_KEYS.get(subject, subject),
            domain=domain,
            topic=topic,
            normalized_topic=normalized_topic,
            problem_context=problem_context,
            summary=f"学生正在学习“{topic}”。",
            progress=progress,
            difficulties=[item for item in difficulties if item],
            next_step=next_step,
            evidence=[
                TopicMemoryEvidence(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    kind=evidence_kind,
                    summary=progress,
                    confidence=confidence,
                )
            ],
            source_conversation_ids=[conversation_id],
            confidence=confidence,
            status="active",
        )
        self.store.upsert_topic_memory(incoming)

    def _archive_candidates(
        self,
        profile_id: str,
        user_id: str,
        exclude_conversation_id: str,
        subject: str,
    ) -> list[TopicMemory]:
        if self.archive_store is None:
            return []
        candidates: list[TopicMemory] = []
        for summary in self.archive_store.list_for_user(user_id)[:24]:
            if summary.conversation_id == exclude_conversation_id:
                continue
            try:
                record = self.archive_store.load(
                    user_id,
                    summary.conversation_id,
                )
            except Exception:  # noqa: BLE001 - one bad archive must not block recall
                continue
            episode = record.episode
            if _clean(episode.get("active_subject"), 20) != subject:
                continue
            problem = _clean(episode.get("problem_context"), 800)
            turns = episode.get("turns") or []
            if not problem or not isinstance(turns, Sequence) or not turns:
                continue
            domain = detect_domain(subject, problem)
            topic = extract_topic(subject, domain, problem)
            normalized = _normalize_topic(topic)
            if not normalized:
                continue
            candidates.append(
                TopicMemory(
                    memory_id=f"archive_{record.conversation_id}",
                    student_profile_id=profile_id,
                    subject=subject,
                    subject_key=_SUBJECT_KEYS.get(subject, subject),
                    domain=domain,
                    topic=topic,
                    normalized_topic=normalized,
                    problem_context=problem,
                    summary=f"该学生过去学习过“{topic}”。",
                    progress="存在可继续承接的历史学习会话。",
                    source_conversation_ids=[record.conversation_id],
                    confidence=0.72,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    last_accessed_at=record.updated_at,
                )
            )
        return candidates

    @staticmethod
    def _deduplicate(memories: Sequence[TopicMemory]) -> list[TopicMemory]:
        by_key: dict[tuple[str, str, str], TopicMemory] = {}
        for memory in memories:
            key = (memory.subject, memory.domain, memory.normalized_topic)
            current = by_key.get(key)
            if current is None or (
                current.memory_id.startswith("archive_")
                and not memory.memory_id.startswith("archive_")
            ):
                by_key[key] = memory
        return list(by_key.values())

    @staticmethod
    def _is_explicit_topic(
        subject: str,
        domain: str,
        message: str,
        query_topic: str,
    ) -> bool:
        if re.search(r"《[^》]+》|[“\"][^”\"]+[”\"]", message):
            return True
        if (
            subject == "语文"
            and domain == "composition"
            and query_topic.startswith("我的")
        ):
            return True
        label = _DOMAIN_LABELS.get(domain, "")
        return bool(label and label in message and domain != "general")

    @staticmethod
    def _score_candidates(
        memories: Sequence[TopicMemory],
        *,
        domain: str,
        query_topic: str,
        continuation: bool,
        explicit_topic: bool,
    ) -> list[MemoryCandidate]:
        normalized_query = _normalize_topic(query_topic)
        candidates: list[MemoryCandidate] = []
        for memory in memories:
            score = 0.0
            if memory.domain == domain:
                score += 0.45
            elif domain != "general":
                continue
            if continuation:
                score += 0.25
            similarity = (
                SequenceMatcher(
                    None,
                    normalized_query,
                    memory.normalized_topic,
                ).ratio()
                if normalized_query
                else 0.0
            )
            if explicit_topic:
                if normalized_query == memory.normalized_topic:
                    score += 0.5
                elif (
                    normalized_query in memory.normalized_topic
                    or memory.normalized_topic in normalized_query
                ):
                    score += 0.4
                else:
                    score += 0.3 * similarity
            score += min(0.1, memory.confidence * 0.1)
            candidates.append(
                MemoryCandidate(
                    memory=memory,
                    match_score=min(1.0, round(score, 4)),
                    source=(
                        "archive"
                        if memory.memory_id.startswith("archive_")
                        else "profile"
                    ),
                )
            )
        return sorted(
            candidates,
            key=lambda item: (
                item.match_score,
                item.memory.updated_at,
            ),
            reverse=True,
        )

    @staticmethod
    def _clarification_question(choices: Sequence[str]) -> str:
        unique = list(
            dict.fromkeys(
                _clean(choice, 60)
                for choice in choices
                if choice
            )
        )
        if len(unique) == 2:
            return f"你想继续“{unique[0]}”，还是“{unique[1]}”？"
        if len(unique) >= 3:
            return (
                f"你想继续“{unique[0]}”、“{unique[1]}”，"
                f"还是“{unique[2]}”？"
            )
        return "你想继续之前的哪个学习主题？"


_SERVICE_LOCK = threading.Lock()
_SERVICES: dict[tuple[str, str], StudentMemoryService] = {}


def get_student_memory_service(
    memory_path: str | Path | None = None,
    conversation_path: str | Path | None = None,
) -> StudentMemoryService:
    resolved_memory = str(
        Path(
            memory_path
            or os.getenv("TEACHER_STUDENT_MEMORY_DB")
            or os.getenv("TEACHER_CONVERSATION_DB")
            or DEFAULT_STUDENT_MEMORY_DB
        ).resolve()
    )
    resolved_conversation = str(
        Path(
            conversation_path
            or os.getenv("TEACHER_CONVERSATION_DB")
            or DEFAULT_CONVERSATION_DB
        ).resolve()
    )
    key = (resolved_memory, resolved_conversation)
    with _SERVICE_LOCK:
        service = _SERVICES.get(key)
        if service is None:
            service = StudentMemoryService(
                LearnerMemoryStore(resolved_memory),
                archive_store=LocalConversationStore(resolved_conversation),
            )
            _SERVICES[key] = service
        return service


def reset_student_memory_services() -> None:
    """Shut down cached workers; intended for isolated tests and process teardown."""

    with _SERVICE_LOCK:
        services = list(_SERVICES.values())
        _SERVICES.clear()
    for service in services:
        service.shutdown()
