"""Three-layer memory schemas for student profile, episode state, and events."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StudentProfile(BaseModel):
    """Layer 1: long-term student mastery summary."""

    student_id: str
    concept_mastery: dict[str, float] = Field(default_factory=dict)
    skill_mastery: dict[str, float] = Field(default_factory=dict)
    common_weaknesses: list[str] = Field(default_factory=list)
    repeated_misconceptions: dict[str, int] = Field(default_factory=dict)
    preferred_teaching_style: str | None = None
    last_updated_at: str | None = None


class StudentEvidenceKind(str, Enum):
    CONFIRMED_CORRECT = "confirmed_correct"
    VERIFIED_ERROR = "verified_error"
    UNRESOLVED_OBSTACLE = "unresolved_obstacle"
    TEACHING_ACTION = "teaching_action"
    HINT_OUTCOME = "hint_outcome"
    SOFT_CLAIM = "soft_claim"


class StudentEvidenceRecord(BaseModel):
    evidence_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = Field(default_factory=now_iso)
    source_turn: int = Field(ge=1)
    scope_id: str = ""
    kind: StudentEvidenceKind
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    verifier_name: str = ""
    teaching_action_code: str = ""
    scaffold_level: int = Field(default=0, ge=0, le=3)
    progressed_after_hint: bool | None = None


class StudentEvidenceSnapshot(BaseModel):
    records: list[StudentEvidenceRecord] = Field(default_factory=list)
    confirmed_correct: list[StudentEvidenceRecord] = Field(default_factory=list)
    verified_errors: list[StudentEvidenceRecord] = Field(default_factory=list)
    unresolved_obstacles: list[StudentEvidenceRecord] = Field(default_factory=list)
    recent_action_codes: list[str] = Field(default_factory=list)
    latest_scaffold_level: int = Field(default=0, ge=0, le=3)
    latest_verified_step: str = ""
    progress_after_latest_hint: bool | None = None
    consecutive_verified_errors: int = Field(default=0, ge=0)


class EpisodeState(BaseModel):
    """Layer 2: current teaching episode working memory."""

    conversation_id: str
    student_id: str
    original_problem: str | None = None
    problem_context: str | None = None
    target_equation: str | None = None
    current_phase: str = "problem_setup"
    current_solution_state: str | None = None
    solved_steps: list[str] = Field(default_factory=list)
    last_teacher_question: str | None = None
    expected_student_response: str | None = None
    pending_question_type: str | None = None
    pending_practice_problem: str | None = None
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    student_evidence: list[StudentEvidenceRecord] = Field(default_factory=list)
    last_orchestration_trace: dict[str, Any] = Field(default_factory=dict)
    is_problem_solved: bool = False


class LearningEvent(BaseModel):
    """Layer 3: structured evidence across sessions."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = Field(default_factory=now_iso)
    student_id: str
    conversation_id: str
    concept_id: str | None = None
    skill_id: str | None = None
    event_type: str
    evidence_text: str
    problem: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextResolution(BaseModel):
    """Context resolver output before concept diagnosis/RAG."""

    is_reply_to_previous_teacher: bool = False
    dialogue_act: str = "unknown"
    student_request_type: str | None = None
    reply_to_question_type: str | None = None
    semantic_meaning: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    should_start_new_problem: bool = False
    reason: str = ""
