"""Contracts for the sealed Phase 2A.5 robustness evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.memory.contracts import (
    AdmissionAction,
    MemoryRelation,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    TemporalScope,
)


class FailureAttribution(str, Enum):
    CANDIDATE_EXTRACTION_FAILURE = "CANDIDATE_EXTRACTION_FAILURE"
    NORMALIZATION_FAILURE = "NORMALIZATION_FAILURE"
    RELATION_MATCH_FAILURE = "RELATION_MATCH_FAILURE"
    FEATURE_FAILURE = "FEATURE_FAILURE"
    EVIDENCE_LEDGER_FAILURE = "EVIDENCE_LEDGER_FAILURE"
    POLICY_FAILURE = "POLICY_FAILURE"
    CAUSAL_DEDUP_GAP = "CAUSAL_DEDUP_GAP"
    LIVE_AGENT_LOOP_FAILURE = "LIVE_AGENT_LOOP_FAILURE"
    LIVE_MODEL_FAILURE = "LIVE_MODEL_FAILURE"


class CandidateSpec(BaseModel):
    memory_type: MemoryType
    source_kind: MemorySourceKind
    temporal_scope: TemporalScope
    payload: dict[str, Any]
    target: MemoryTargetClass = MemoryTargetClass.STUDENT
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    provenance_count: int = Field(default=1, ge=1, le=12)
    same_causal_turn: bool = False


class PriorEvidenceSpec(BaseModel):
    payload: dict[str, Any]
    source_kind: MemorySourceKind
    conversation_id: str
    turn_id: int = Field(ge=1)
    event_reference: str = ""
    tool_reference: str = ""
    verifier_reference: str = ""


class ExistingSpec(BaseModel):
    payload: dict[str, Any]
    source_kind: MemorySourceKind | None = None
    duplicate_count: int = Field(default=1, ge=1, le=3)


class RobustnessCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    labels: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=240)
    candidate: CandidateSpec
    expected_action: AdmissionAction
    expected_relation: MemoryRelation | None
    expected_semantic_key: str = ""
    existing: ExistingSpec | None = None
    prior_evidence: tuple[PriorEvidenceSpec, ...] = ()
    metamorphic_group: str = ""
    trajectory_id: str = ""


class RobustnessRecord(BaseModel):
    case_id: str
    labels: tuple[str, ...]
    expected_action: AdmissionAction
    predicted_action: AdmissionAction
    expected_relation: MemoryRelation | None
    predicted_relation: MemoryRelation | None
    semantic_key: str
    evidence_maturity: float
    support_count: int
    distinct_turn_count: int
    distinct_conversation_count: int
    failure_attribution: FailureAttribution | None = None
    reason_codes: tuple[str, ...] = ()
    latency_seconds: float = 0.0


class LayerAReport(BaseModel):
    records: list[RobustnessRecord]
    failures: list[RobustnessRecord]
    llm_calls: int = 0
    student_memory_writes: int = 0

