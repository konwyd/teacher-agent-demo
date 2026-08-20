"""Canonical Memory v2 contracts for the authoritative 8502 runtime.

This module defines data, lineage, and observation boundaries only.  It does
not contain an admission policy, an access policy, semantic matching, reward
logic, persistence, or migration behavior.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0.0"
MAX_CANDIDATE_PAYLOAD_BYTES = 4_096


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryTargetClass(str, Enum):
    """Semantic ownership class; operational archives and caches are excluded."""

    STUDENT = "student"
    AGENT_EXPERIENCE = "agent_experience"
    CURATED_KNOWLEDGE = "curated_knowledge"


class MemoryType(str, Enum):
    """Minimal taxonomy grounded in current 8502 data and source boundaries."""

    PROFILE_FACT = "profile_fact"
    PREFERENCE = "preference"
    TOPIC_PROGRESS = "topic_progress"
    DIFFICULTY = "difficulty"
    BEHAVIOR_EVIDENCE = "behavior_evidence"
    LEARNING_GOAL = "learning_goal"
    LEARNING_EPISODE = "learning_episode"
    TASK_STATE = "task_state"
    TEACHING_EXPERIENCE = "teaching_experience"
    KNOWLEDGE_REFERENCE = "knowledge_reference"


class MemorySourceKind(str, Enum):
    """Who asserted an item, or which process produced its evidence."""

    USER_STATED = "user_stated"
    OBSERVED_BEHAVIOR = "observed_behavior"
    VERIFIED_RESULT = "verified_result"
    MODEL_INFERENCE = "model_inference"
    EXTERNAL_TOOL = "external_tool"
    CURATED_KNOWLEDGE = "curated_knowledge"


class TemporalScope(str, Enum):
    TURN_ONLY = "turn_only"
    SESSION = "session"
    SHORT_LIVED = "short_lived"
    PERSISTENT = "persistent"
    UNKNOWN = "unknown"


class MemoryRelation(str, Enum):
    """Possible relation between a candidate and an existing semantic memory."""

    NEW = "new"
    DUPLICATE = "duplicate"
    SUPPORTS = "supports"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"


class MemoryLifecycleStatus(str, Enum):
    """Semantic lifecycle, independent from physical backend retention."""

    PENDING = "pending"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    DELETED = "deleted"


class AdmissionAction(str, Enum):
    REJECT = "reject"
    PENDING = "pending"
    ADMIT = "admit"
    SUPERSEDE = "supersede"


class BaselineMemoryAction(str, Enum):
    """Observed action of the existing system; not an admission decision."""

    ENQUEUE = "enqueue"
    SKIP = "skip"


class AdmissionReasonCode(str, Enum):
    """Structured public reasons, never private model reasoning."""

    BASELINE_ELIGIBLE = "baseline_eligible"
    BASELINE_SKIPPED_SCOPE = "baseline_skipped_scope"
    BASELINE_SKIPPED_AMBIGUOUS = "baseline_skipped_ambiguous"
    BASELINE_SKIPPED_INVALID_TURN = "baseline_skipped_invalid_turn"
    DUPLICATE = "duplicate"
    TEMPORARY = "temporary"
    MODEL_INFERENCE = "model_inference"
    CONTRADICTION = "contradiction"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    USER_CORRECTION = "user_correction"
    SOURCE_SEPARATION = "source_separation"
    STABLE_USER_PREFERENCE = "stable_user_preference"
    STABLE_PROFILE_FACT = "stable_profile_fact"
    STABLE_LEARNING_GOAL = "stable_learning_goal"
    MATURE_EVIDENCE = "mature_evidence"
    TEMPORARY_STATE = "temporary_state"
    UNSUPPORTED_INFERENCE = "unsupported_inference"
    EXPLICIT_SUPERSESSION = "explicit_supersession"
    AMBIGUOUS_RELATION = "ambiguous_relation"
    WEAK_TEMPORAL_VALIDITY = "weak_temporal_validity"
    SINGLE_OBSERVATION = "single_observation"
    QUALITY_QA_SEPARATION = "quality_qa_separation"
    CONTROLLER_ERROR = "controller_error"


class ProvenanceReference(BaseModel):
    """Bounded reference to source evidence; raw conversations are not copied."""

    schema_version: str = SCHEMA_VERSION
    conversation_id: str = Field(min_length=1, max_length=160)
    turn_id: int = Field(ge=1)
    source_kind: MemorySourceKind
    event_reference: str = Field(default="", max_length=160)
    tool_reference: str = Field(default="", max_length=160)
    verifier_reference: str = Field(default="", max_length=160)


class AdmissionFeatureValue(BaseModel):
    """Bounded explainable feature value; source is public, not model reasoning."""

    value: float = Field(ge=0.0, le=1.0)
    source: str = Field(min_length=1, max_length=120)


class AdmissionFeatures(BaseModel):
    future_utility: AdmissionFeatureValue
    factual_confidence: AdmissionFeatureValue
    novelty: AdmissionFeatureValue
    stability: AdmissionFeatureValue
    temporal_validity: AdmissionFeatureValue
    evidence_maturity: AdmissionFeatureValue
    contradiction_strength: AdmissionFeatureValue
    content_type_prior: AdmissionFeatureValue


class EvidenceSummary(BaseModel):
    support_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    distinct_turn_count: int = Field(default=0, ge=0)
    distinct_conversation_count: int = Field(default=0, ge=0)
    verified_support_count: int = Field(default=0, ge=0)
    source_diversity_count: int = Field(default=0, ge=0)
    maturity: float = Field(default=0.0, ge=0.0, le=1.0)


class ExistingMemorySnapshot(BaseModel):
    """Immutable committed-memory view used only for deterministic matching."""

    model_config = ConfigDict(frozen=True)

    memory_id: str = Field(min_length=1, max_length=160)
    target_memory_class: MemoryTargetClass
    memory_type: MemoryType
    semantic_key: str = Field(min_length=1, max_length=320)
    payload: dict[str, Any] = Field(default_factory=dict)
    source_kind: MemorySourceKind | None = None
    lifecycle_status: MemoryLifecycleStatus = MemoryLifecycleStatus.ACTIVE
    evidence_eligible: bool = False

    @model_validator(mode="after")
    def validate_match_only(self) -> "ExistingMemorySnapshot":
        if self.evidence_eligible:
            raise ValueError("committed memory snapshots are match-only, never evidence")
        return self


class EvidenceClusterSnapshot(BaseModel):
    """Immutable pre-write snapshot of independent shadow evidence."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str = Field(min_length=1, max_length=160)
    semantic_key: str = Field(min_length=1, max_length=320)
    normalized_payload: dict[str, Any] = Field(default_factory=dict)
    memory_type: MemoryType
    supporting_provenance: tuple[ProvenanceReference, ...] = ()
    contradicting_provenance: tuple[ProvenanceReference, ...] = ()
    support_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    distinct_turn_count: int = Field(default=0, ge=0)
    distinct_conversation_count: int = Field(default=0, ge=0)
    verified_support_count: int = Field(default=0, ge=0)
    source_kinds: tuple[MemorySourceKind, ...] = ()
    first_seen: datetime
    last_seen: datetime
    current_evidence_maturity: float = Field(default=0.0, ge=0.0, le=1.0)
    last_proposed_decision: AdmissionAction | None = None


class FrozenAdmissionContext(BaseModel):
    """Immutable context captured before the authoritative baseline enqueue."""

    model_config = ConfigDict(frozen=True)

    student_key: str = Field(min_length=1, max_length=160)
    conversation_id: str = Field(min_length=1, max_length=160)
    turn_id: int = Field(ge=1)
    known_subject: str = Field(default="", max_length=40)
    known_topic: str = Field(default="", max_length=160)
    existing_memories: tuple[ExistingMemorySnapshot, ...] = ()
    evidence_clusters: tuple[EvidenceClusterSnapshot, ...] = ()
    captured_at: datetime = Field(default_factory=utc_now)

class MemoryCandidate(BaseModel):
    """Typed semantic-memory candidate; creating one does not authorize a write."""

    schema_version: str = SCHEMA_VERSION
    candidate_id: str = Field(
        default_factory=lambda: f"candidate_{uuid.uuid4().hex}",
        min_length=1,
        max_length=160,
    )
    target_memory_class: MemoryTargetClass
    memory_type: MemoryType
    payload: dict[str, Any] = Field(default_factory=dict)
    source_kind: MemorySourceKind
    provenance: list[ProvenanceReference] = Field(min_length=1, max_length=12)
    confidence: float = Field(ge=0.0, le=1.0)
    temporal_scope: TemporalScope
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_source_boundary(self) -> "MemoryCandidate":
        provenance_kinds = {item.source_kind for item in self.provenance}
        if self.source_kind not in provenance_kinds:
            raise ValueError(
                "candidate source_kind must have matching provenance; model "
                "inference cannot masquerade as user_stated or verified_result"
            )
        if (
            self.target_memory_class is MemoryTargetClass.STUDENT
            and self.source_kind is MemorySourceKind.CURATED_KNOWLEDGE
        ):
            raise ValueError("curated knowledge cannot target student memory")
        encoded = json.dumps(
            self.payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_CANDIDATE_PAYLOAD_BYTES:
            raise ValueError("candidate payload exceeds bounded contract size")
        return self


class AdmissionDecision(BaseModel):
    """Future policy output.  Phase 1.5 creates no such decisions at runtime."""

    schema_version: str = SCHEMA_VERSION
    candidate_id: str = Field(min_length=1, max_length=160)
    decision: AdmissionAction
    matched_memory_id: str = Field(default="", max_length=160)
    relation: MemoryRelation | None = None
    reason_codes: list[AdmissionReasonCode] = Field(default_factory=list, max_length=12)
    policy_name: str = Field(min_length=1, max_length=120)
    policy_version: str = Field(min_length=1, max_length=80)
    resulting_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    lifecycle_status: MemoryLifecycleStatus | None = None
    created_at: datetime = Field(default_factory=utc_now)


class MemoryAdmissionShadowRecord(BaseModel):
    """Read-only comparison record for a baseline action and an unset proposal."""

    schema_version: str = SCHEMA_VERSION
    observer_version: str = "memory-v2-foundation-1"
    record_id: str = Field(
        default_factory=lambda: f"shadow_{uuid.uuid4().hex}",
        min_length=1,
        max_length=160,
    )
    candidate: MemoryCandidate
    baseline_action: BaselineMemoryAction
    baseline_skip_reason: str = Field(default="", max_length=160)
    proposed_decision: AdmissionDecision | None = None
    policy_name: str = Field(default="none", max_length=120)
    policy_version: str = Field(default="unimplemented", max_length=80)
    matched_existing_memory_id: str = Field(default="", max_length=160)
    relation: MemoryRelation | None = None
    features: AdmissionFeatures | None = None
    evidence_summary: EvidenceSummary | None = None
    divergence: bool | None = None
    decision_latency_seconds: float = Field(default=0.0, ge=0.0)
    controller_error: str = Field(default="", max_length=240)
    conversation_id: str = Field(min_length=1, max_length=160)
    turn_id: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=utc_now)


class MemoryOutcomeObservation(BaseModel):
    """Observable runtime signals only; deliberately contains no reward field."""

    schema_version: str = SCHEMA_VERSION
    observer_version: str = "memory-v2-foundation-1"
    observation_id: str = Field(
        default_factory=lambda: f"outcome_{uuid.uuid4().hex}",
        min_length=1,
        max_length=160,
    )
    conversation_id: str = Field(min_length=1, max_length=160)
    turn_id: int = Field(ge=1)
    verified_student_step_status: str = Field(default="", max_length=80)
    tool_success_count: int = Field(default=0, ge=0)
    tool_failure_count: int = Field(default=0, ge=0)
    retrieval_hit_count: int = Field(default=0, ge=0)
    top_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    teacher_turn_count: int = Field(default=1, ge=1)
    explicit_correction: bool = False
    timestamp: datetime = Field(default_factory=utc_now)
