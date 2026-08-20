"""Bounded benchmark and report contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.memory.contracts import (
    AdmissionAction,
    BaselineMemoryAction,
    MemoryRelation,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    TemporalScope,
)


REQUIRED_CATEGORIES = {
    "stable_preference", "profile_fact", "learning_goal", "task_state",
    "temporary_scope", "model_inference", "curated_separation",
    "agent_experience_separation", "single_verified_error",
    "single_correct_result", "repeated_verified_error", "repeated_progress",
    "duplicate", "contradiction", "explicit_correction", "refinement",
    "ambiguous_relation", "weak_source", "unknown_temporal", "difficulty",
    "baseline_divergence", "prewrite_snapshot", "ledger_dedup",
    "cross_conversation_evidence", "quality_qa_separation",
}


class BenchmarkCase(BaseModel):
    case_id: str
    category: str
    rationale: str = Field(max_length=240)
    memory_type: MemoryType
    source_kind: MemorySourceKind
    temporal_scope: TemporalScope
    payload: dict[str, Any]
    expected_action: AdmissionAction
    expected_relation: MemoryRelation | None = None
    target: MemoryTargetClass = MemoryTargetClass.STUDENT
    confidence: float = Field(default=0.9, ge=0, le=1)
    baseline_action: BaselineMemoryAction = BaselineMemoryAction.ENQUEUE
    existing_payload: dict[str, Any] | None = None
    existing_source: MemorySourceKind | None = None
    ambiguous_existing: bool = False
    prior_supports: int = Field(default=0, ge=0, le=6)
    prior_contradictions: int = Field(default=0, ge=0, le=6)
    sequence_id: str = ""
    sequence_step: int = Field(default=0, ge=0)


class MetricValue(BaseModel):
    value: float
    support: int
    numerator: int = 0


class EvaluationRecord(BaseModel):
    case_id: str
    category: str
    expected_action: AdmissionAction
    predicted_action: AdmissionAction
    expected_relation: MemoryRelation | None
    predicted_relation: MemoryRelation | None
    candidate: dict[str, Any]
    provenance: list[dict[str, Any]]
    features: dict[str, Any]
    evidence: dict[str, Any]
    reasons: list[str]
    baseline_action: BaselineMemoryAction
    divergence: bool
    latency_seconds: float
    controller_error: str = ""


class MemoryAdmissionEvaluationReport(BaseModel):
    split: str
    cases: list[str]
    records: list[EvaluationRecord]
    classification: dict[str, Any]
    safety: dict[str, MetricValue]
    sequential: dict[str, MetricValue]
    baseline: dict[str, MetricValue]
    divergence_cases: list[dict[str, Any]]
