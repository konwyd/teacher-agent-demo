"""Student mastery tracking：按 (student_id, concept_id) 跟踪掌握度，本地 JSON 持久化。

存储位置：data/runtime/student_profiles.json（可用 TEACHER_PROFILE_PATH 覆盖）。
清空学生记忆：删除该 JSON 文件，或调用 store.reset_student(student_id) / reset_all()。

掌握度更新规则：
- correct_step          +0.1
- correct_final_answer  +0.2
- incorrect_step        -0.1
- help_requested        -0.05（求助视为轻微未掌握信号）
- 始终裁剪到 [0.0, 1.0]；新学生初始 0.5（中性：先给一级提示观察）
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

DEFAULT_MASTERY = 0.5
REVIEW_THRESHOLD = 0.6
RECENT_ERRORS_KEPT = 5

MASTERY_DELTAS: dict[str, float] = {
    "correct_step": 0.1,
    "correct_final_answer": 0.2,
    "incorrect_step": -0.1,
    "help_requested": -0.05,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConceptMastery(BaseModel):
    """单个学生在单个知识点上的掌握状态。"""

    student_id: str
    concept_id: str
    mastery_score: float = Field(default=DEFAULT_MASTERY, ge=0.0, le=1.0)
    attempts: int = 0
    correct_steps: int = 0
    wrong_steps: int = 0
    last_seen_at: datetime = Field(default_factory=_now)
    common_misconceptions: list[str] = Field(default_factory=list)
    recent_errors: list[str] = Field(default_factory=list)
    next_review_at: Optional[datetime] = None
    # 内部计数：同一误因出现 >=2 次才进入 common_misconceptions
    misconception_counts: dict[str, int] = Field(default_factory=dict)


class StudentProfile(BaseModel):
    student_id: str
    concepts: dict[str, ConceptMastery] = Field(default_factory=dict)


class StudentStateStore(ABC):
    """学生掌握度存储接口。"""

    @abstractmethod
    def get_student_profile(self, student_id: str) -> StudentProfile: ...

    @abstractmethod
    def get_concept_mastery(self, student_id: str, concept_id: str) -> ConceptMastery: ...

    @abstractmethod
    def update_mastery_after_turn(
        self, student_id: str, concept_id: str, step_result: str, error_detail: str = ""
    ) -> ConceptMastery: ...

    @abstractmethod
    def record_misconception(
        self, student_id: str, concept_id: str, misconception: str
    ) -> ConceptMastery: ...

    @abstractmethod
    def recommend_review_concepts(self, student_id: str) -> list[ConceptMastery]: ...


class JsonStudentStateStore(StudentStateStore):
    """本地 JSON 文件存储（demo 规模足够；后续可平替 SQLite）。"""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._profiles: dict[str, StudentProfile] = {}
        self._load()

    # ---------- 持久化 ----------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for sid, pdata in raw.get("students", {}).items():
                self._profiles[sid] = StudentProfile.model_validate(pdata)
        except (json.JSONDecodeError, ValueError):
            self._profiles = {}  # 损坏文件：从空开始，不让运行时崩溃

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"students": {sid: p.model_dump(mode="json") for sid, p in self._profiles.items()}}
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 查询 ----------

    def get_student_profile(self, student_id: str) -> StudentProfile:
        if student_id not in self._profiles:
            self._profiles[student_id] = StudentProfile(student_id=student_id)
        return self._profiles[student_id]

    def get_concept_mastery(self, student_id: str, concept_id: str) -> ConceptMastery:
        profile = self.get_student_profile(student_id)
        if concept_id not in profile.concepts:
            return ConceptMastery(student_id=student_id, concept_id=concept_id)
        return profile.concepts[concept_id]

    # ---------- 更新 ----------

    def update_mastery_after_turn(
        self, student_id: str, concept_id: str, step_result: str, error_detail: str = ""
    ) -> ConceptMastery:
        rec = self.get_concept_mastery(student_id, concept_id).model_copy(deep=True)
        delta = MASTERY_DELTAS.get(step_result, 0.0)
        rec.mastery_score = min(1.0, max(0.0, round(rec.mastery_score + delta, 4)))
        rec.attempts += 1
        if step_result in ("correct_step", "correct_final_answer"):
            rec.correct_steps += 1
        elif step_result == "incorrect_step":
            rec.wrong_steps += 1
            if error_detail:
                rec.recent_errors = (rec.recent_errors + [error_detail])[-RECENT_ERRORS_KEPT:]
        rec.last_seen_at = _now()
        rec.next_review_at = (
            _now() + timedelta(days=1) if rec.mastery_score < REVIEW_THRESHOLD else None
        )
        self.get_student_profile(student_id).concepts[concept_id] = rec
        self._flush()
        return rec

    def record_misconception(
        self, student_id: str, concept_id: str, misconception: str
    ) -> ConceptMastery:
        rec = self.get_concept_mastery(student_id, concept_id).model_copy(deep=True)
        rec.misconception_counts[misconception] = rec.misconception_counts.get(misconception, 0) + 1
        if rec.misconception_counts[misconception] >= 2 and misconception not in rec.common_misconceptions:
            rec.common_misconceptions.append(misconception)
        self.get_student_profile(student_id).concepts[concept_id] = rec
        self._flush()
        return rec

    def recommend_review_concepts(self, student_id: str) -> list[ConceptMastery]:
        profile = self.get_student_profile(student_id)
        due = [
            r for r in profile.concepts.values()
            if r.mastery_score < REVIEW_THRESHOLD
            or (r.next_review_at is not None and r.next_review_at <= _now())
        ]
        return sorted(due, key=lambda r: r.mastery_score)

    # ---------- 清理 ----------

    def reset_student(self, student_id: str) -> None:
        self._profiles.pop(student_id, None)
        self._flush()

    def reset_all(self) -> None:
        self._profiles = {}
        self._flush()


DEFAULT_PROFILE_PATH = "data/runtime/student_profiles.json"

_stores: dict[str, JsonStudentStateStore] = {}


def get_store() -> JsonStudentStateStore:
    """按路径缓存的 store（TEACHER_PROFILE_PATH 可为测试/评测指定隔离文件）。"""
    path = os.getenv("TEACHER_PROFILE_PATH", DEFAULT_PROFILE_PATH)
    if path not in _stores:
        _stores[path] = JsonStudentStateStore(path)
    return _stores[path]
