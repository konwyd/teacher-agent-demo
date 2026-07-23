"""Layer 3 -> Layer 1 student profile aggregation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.memory.schemas import LearningEvent, StudentProfile, now_iso

DEFAULT_THREE_LAYER_PROFILE_PATH = "data/runtime/student_profiles_v2.json"


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, round(value, 4)))


class JsonStudentProfileStore:
    def __init__(self, path: Path | str = DEFAULT_THREE_LAYER_PROFILE_PATH) -> None:
        self._path = Path(path)
        self._profiles: dict[str, StudentProfile] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._profiles = {
                sid: StudentProfile.model_validate(value)
                for sid, value in raw.get("students", {}).items()
            }
        except (json.JSONDecodeError, ValueError):
            self._profiles = {}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"students": {sid: p.model_dump(mode="json") for sid, p in self._profiles.items()}}
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_profile(self, student_id: str) -> StudentProfile:
        if student_id not in self._profiles:
            self._profiles[student_id] = StudentProfile(student_id=student_id)
        return self._profiles[student_id]

    def save_profile(self, profile: StudentProfile) -> None:
        profile.last_updated_at = now_iso()
        self._profiles[profile.student_id] = profile
        self._flush()

    def reset_profile(self, student_id: str) -> StudentProfile:
        profile = StudentProfile(student_id=student_id)
        self._profiles[student_id] = profile
        self._flush()
        return profile


_stores: dict[str, JsonStudentProfileStore] = {}


def get_profile_store() -> JsonStudentProfileStore:
    path = os.getenv("TEACHER_THREE_LAYER_PROFILE_PATH", DEFAULT_THREE_LAYER_PROFILE_PATH)
    if path not in _stores:
        _stores[path] = JsonStudentProfileStore(path)
    return _stores[path]


def update_profile_from_event(event: LearningEvent) -> StudentProfile:
    store = get_profile_store()
    profile = store.get_profile(event.student_id).model_copy(deep=True)

    concept_delta = 0.0
    skill_delta = 0.0
    if event.event_type == "correct_step":
        concept_delta = 0.03
        skill_delta = 0.05
    elif event.event_type == "solved_problem":
        concept_delta = 0.1
        skill_delta = 0.05
    elif event.event_type == "verification_success":
        concept_delta = 0.05
        skill_delta = 0.03
    elif event.event_type == "wrong_step":
        concept_delta = -0.04
        skill_delta = -0.06
    elif event.event_type == "misconception":
        concept_delta = -0.03
        skill_delta = -0.08
        misconception = str(event.metadata.get("misconception") or event.evidence_text)
        profile.repeated_misconceptions[misconception] = (
            profile.repeated_misconceptions.get(misconception, 0) + 1
        )
        if (
            profile.repeated_misconceptions[misconception] >= 2
            and misconception not in profile.common_weaknesses
        ):
            profile.common_weaknesses.append(misconception)
    elif event.event_type == "expressed_confusion":
        concept_delta = -0.02

    if event.concept_id:
        before = profile.concept_mastery.get(event.concept_id, 0.5)
        profile.concept_mastery[event.concept_id] = _clamp(before + concept_delta)
    if event.skill_id:
        before = profile.skill_mastery.get(event.skill_id, 0.5)
        profile.skill_mastery[event.skill_id] = _clamp(before + skill_delta)

    store.save_profile(profile)
    return profile


def summarize_profile(student_id: str) -> dict:
    profile = get_profile_store().get_profile(student_id)
    return {
        "student_id": student_id,
        "concept_mastery": profile.concept_mastery,
        "skill_mastery": profile.skill_mastery,
        "common_weaknesses": profile.common_weaknesses,
        "repeated_misconceptions": profile.repeated_misconceptions,
        "preferred_teaching_style": profile.preferred_teaching_style,
        "last_updated_at": profile.last_updated_at,
    }
