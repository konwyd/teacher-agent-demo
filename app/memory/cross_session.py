"""Bounded, per-user recall of learning context across conversations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.memory.conversation_archive import LocalConversationStore


_CONTINUITY_MARKERS = (
    "上次",
    "之前",
    "刚才",
    "继续",
    "接着",
    "这个作文",
    "那个作文",
    "这篇作文",
    "那篇作文",
    "这个话题",
    "那个话题",
)
_CATEGORY_CUES = {
    "作文": ("作文", "写作", "开头", "结尾", "题目", "素材"),
    "数学": ("数学", "方程", "函数", "几何", "计算"),
    "英语": ("英语", "英文", "单词", "语法"),
    "物理": ("物理", "力", "电路", "压强", "浮力"),
    "化学": ("化学", "元素", "反应", "方程式"),
}


@dataclass(frozen=True)
class RecalledLearningContext:
    """Small continuity record selected from one of the user's conversations."""

    source_conversation_id: str
    problem_context: str
    active_subject: str
    active_stage: str
    recent_student_messages: tuple[str, ...]
    recent_teacher_response: str


def _clean(value: object, limit: int) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def _turn_value(turn: object, key: str) -> str:
    if isinstance(turn, Mapping):
        value: Any = turn.get(key, "")
        if key == "teacher_output" and isinstance(value, Mapping):
            value = value.get("response", "")
        return _clean(value, 320)
    return ""


def _message_category(message: str) -> str:
    for category, cues in _CATEGORY_CUES.items():
        if any(cue in message for cue in cues):
            return category
    return ""


def _candidate_matches_category(episode: Mapping[str, object], category: str) -> bool:
    if not category:
        return True
    problem = _clean(episode.get("problem_context"), 800)
    subject = _clean(episode.get("active_subject"), 20)
    if category == "作文":
        return "作文" in problem or "写作" in problem
    return category in problem or subject == category


def recall_learning_context(
    store: LocalConversationStore,
    user_id: str,
    message: str,
    *,
    exclude_conversation_id: str | None = None,
    scan_limit: int = 24,
) -> RecalledLearningContext | None:
    """Recall only explicit continuations; ordinary new questions stay memory-free."""

    clean_message = _clean(message, 600)
    if not clean_message or not any(
        marker in clean_message for marker in _CONTINUITY_MARKERS
    ):
        return None
    category = _message_category(clean_message)
    for summary in store.list_for_user(user_id)[:scan_limit]:
        if summary.conversation_id == exclude_conversation_id:
            continue
        record = store.load(user_id, summary.conversation_id)
        episode = record.episode
        problem_context = _clean(episode.get("problem_context"), 800)
        turns = episode.get("turns") or []
        if not problem_context or not isinstance(turns, Sequence) or not turns:
            continue
        if not _candidate_matches_category(episode, category):
            continue
        recent_turns = list(turns)[-2:]
        student_messages = tuple(
            value
            for value in (_turn_value(turn, "student_message") for turn in recent_turns)
            if value
        )
        last_teacher = _turn_value(recent_turns[-1], "teacher_output")
        return RecalledLearningContext(
            source_conversation_id=record.conversation_id,
            problem_context=problem_context,
            active_subject=_clean(episode.get("active_subject"), 20),
            active_stage=_clean(episode.get("active_stage"), 20),
            recent_student_messages=student_messages,
            recent_teacher_response=last_teacher,
        )
    return None
