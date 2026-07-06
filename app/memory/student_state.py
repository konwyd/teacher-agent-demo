"""Student mastery-state interface.

Tracks per-student, per-concept mastery. Backed by SQLite/PostgreSQL later;
an in-memory implementation is provided for tests and early development.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import lru_cache

from pydantic import BaseModel, Field


class MasteryRecord(BaseModel):
    student_id: str
    concept_id: str
    mastery: float = Field(default=0.0, ge=0.0, le=1.0)
    attempts: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StudentStateStore(ABC):
    """Interface for reading/updating student mastery."""

    @abstractmethod
    def get_mastery(self, student_id: str, concept_id: str) -> MasteryRecord:
        """Return the record (default mastery=0.0 if unseen)."""

    @abstractmethod
    def update_mastery(
        self, student_id: str, concept_id: str, correct: bool, weight: float = 0.3
    ) -> MasteryRecord:
        """Update mastery after a graded attempt and return the new record."""

    @abstractmethod
    def get_weak_concepts(self, student_id: str, threshold: float = 0.6) -> list[MasteryRecord]:
        """Concepts below the mastery threshold (candidates for review)."""


class InMemoryStudentStateStore(StudentStateStore):
    """Simple EMA-based mastery tracking, in memory."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], MasteryRecord] = {}

    def get_mastery(self, student_id: str, concept_id: str) -> MasteryRecord:
        return self._records.get(
            (student_id, concept_id),
            MasteryRecord(student_id=student_id, concept_id=concept_id),
        )

    def update_mastery(
        self, student_id: str, concept_id: str, correct: bool, weight: float = 0.3
    ) -> MasteryRecord:
        rec = self.get_mastery(student_id, concept_id)
        target = 1.0 if correct else 0.0
        new_mastery = (1 - weight) * rec.mastery + weight * target
        updated = MasteryRecord(
            student_id=student_id,
            concept_id=concept_id,
            mastery=round(new_mastery, 4),
            attempts=rec.attempts + 1,
        )
        self._records[(student_id, concept_id)] = updated
        return updated

    def get_weak_concepts(self, student_id: str, threshold: float = 0.6) -> list[MasteryRecord]:
        return [
            r
            for (sid, _), r in self._records.items()
            if sid == student_id and r.mastery < threshold
        ]


@lru_cache
def get_store() -> InMemoryStudentStateStore:
    """进程级单例（demo 用；后续换 SQLite/PostgreSQL 实现）。"""
    return InMemoryStudentStateStore()
