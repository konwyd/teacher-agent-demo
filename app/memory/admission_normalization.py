"""Deterministic structured normalization for Memory v2 admission matching."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.memory.contracts import MemoryCandidate, MemoryType


class NormalizedCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    semantic_key: str = Field(default="", max_length=320)
    normalized_payload: dict[str, Any] = Field(default_factory=dict)
    explicit_correction: bool = False


def _normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return re.sub(r"\s+", " ", text).strip(" ,，。.!！?？;；:")


def _normalized_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalized_text(value)
    if isinstance(value, list):
        normalized = [_normalized_value(item) for item in value]
        return sorted({str(item): item for item in normalized if item not in ("", None)}.values(), key=str)
    if isinstance(value, dict):
        return {
            str(key): _normalized_value(item)
            for key, item in sorted(value.items())
            if item not in ("", None, [], {})
        }
    return value


class CandidateNormalizer:
    """Create stable identities only when current typed payloads are sufficient."""

    def normalize(self, candidate: MemoryCandidate) -> NormalizedCandidate:
        payload = {
            key: _normalized_value(value)
            for key, value in candidate.payload.items()
            if key not in {"explicit_correction", "correction_of"}
            and value not in ("", None, [], {})
        }
        explicit_correction = bool(
            candidate.payload.get("explicit_correction")
            or candidate.payload.get("correction_of")
        )
        semantic_key = self._semantic_key(candidate.memory_type, payload)
        return NormalizedCandidate(
            semantic_key=semantic_key,
            normalized_payload=payload,
            explicit_correction=explicit_correction,
        )

    @staticmethod
    def _semantic_key(memory_type: MemoryType, payload: dict[str, Any]) -> str:
        prefix = f"student:{memory_type.value}"
        if memory_type is MemoryType.PREFERENCE:
            return (
                f"{prefix}:teaching_style"
                if payload.get("teaching_preferences")
                else ""
            )
        if memory_type is MemoryType.PROFILE_FACT:
            fields = [key for key in ("grade", "preferred_name") if payload.get(key)]
            return f"{prefix}:{fields[0]}" if len(fields) == 1 else ""
        if memory_type is MemoryType.LEARNING_GOAL:
            subject = payload.get("subject")
            return f"{prefix}:{subject}" if subject and payload.get("goal") else ""
        if memory_type is MemoryType.DIFFICULTY:
            subject = payload.get("subject")
            topic = payload.get("topic") or payload.get("concept") or payload.get("statement")
            return f"{prefix}:{subject}:{topic}" if subject and topic else ""
        if memory_type is MemoryType.BEHAVIOR_EVIDENCE:
            subject = payload.get("subject")
            pattern = payload.get("evidence_pattern") or payload.get("summary")
            return f"{prefix}:{subject}:{pattern}" if subject and pattern else ""
        if memory_type is MemoryType.TOPIC_PROGRESS:
            subject = payload.get("subject")
            topic = payload.get("topic")
            return f"{prefix}:{subject}:{topic}" if subject and topic else ""
        if memory_type is MemoryType.TASK_STATE:
            subject = payload.get("subject")
            return f"{prefix}:{subject}" if subject and payload.get("requested_target") else ""
        return ""
