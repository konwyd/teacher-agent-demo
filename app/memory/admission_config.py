"""Central deterministic configuration for Memory v2 Admission Policy v1."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.memory.contracts import (
    MemoryRelation,
    MemorySourceKind,
    MemoryType,
    TemporalScope,
)


class AdmissionPolicyConfig(BaseModel):
    """All Phase 2A policy priors and thresholds, versioned in one place."""

    policy_name: str = "teacher-memory-admission"
    policy_version: str = "v1"
    source_reliability_priors: dict[MemorySourceKind, float] = Field(
        default_factory=lambda: {
            MemorySourceKind.USER_STATED: 0.90,
            MemorySourceKind.OBSERVED_BEHAVIOR: 0.55,
            MemorySourceKind.VERIFIED_RESULT: 0.95,
            MemorySourceKind.MODEL_INFERENCE: 0.30,
            MemorySourceKind.EXTERNAL_TOOL: 0.75,
            MemorySourceKind.CURATED_KNOWLEDGE: 0.85,
        },
        description="Reliability priors by source; candidate confidence remains a separate input.",
    )
    temporal_validity_priors: dict[TemporalScope, float] = Field(
        default_factory=lambda: {
            TemporalScope.TURN_ONLY: 0.05,
            TemporalScope.SESSION: 0.15,
            TemporalScope.SHORT_LIVED: 0.50,
            TemporalScope.PERSISTENT: 0.95,
            TemporalScope.UNKNOWN: 0.30,
        },
        description="Suitability of each temporal scope for persistent semantic memory.",
    )
    stability_priors: dict[TemporalScope, float] = Field(
        default_factory=lambda: {
            TemporalScope.TURN_ONLY: 0.05,
            TemporalScope.SESSION: 0.15,
            TemporalScope.SHORT_LIVED: 0.45,
            TemporalScope.PERSISTENT: 0.90,
            TemporalScope.UNKNOWN: 0.25,
        },
        description="Expected cross-interaction stability by declared temporal scope.",
    )
    type_utility_priors: dict[MemoryType, float] = Field(
        default_factory=lambda: {
            MemoryType.PROFILE_FACT: 0.85,
            MemoryType.PREFERENCE: 0.90,
            MemoryType.TOPIC_PROGRESS: 0.60,
            MemoryType.DIFFICULTY: 0.75,
            MemoryType.BEHAVIOR_EVIDENCE: 0.65,
            MemoryType.LEARNING_GOAL: 0.85,
            MemoryType.LEARNING_EPISODE: 0.45,
            MemoryType.TASK_STATE: 0.15,
            MemoryType.TEACHING_EXPERIENCE: 0.75,
            MemoryType.KNOWLEDGE_REFERENCE: 0.40,
        },
        description="Future teaching utility priors by canonical memory type.",
    )
    content_type_priors: dict[MemoryType, float] = Field(
        default_factory=lambda: {
            MemoryType.PROFILE_FACT: 0.85,
            MemoryType.PREFERENCE: 0.90,
            MemoryType.TOPIC_PROGRESS: 0.55,
            MemoryType.DIFFICULTY: 0.70,
            MemoryType.BEHAVIOR_EVIDENCE: 0.65,
            MemoryType.LEARNING_GOAL: 0.80,
            MemoryType.LEARNING_EPISODE: 0.40,
            MemoryType.TASK_STATE: 0.10,
            MemoryType.TEACHING_EXPERIENCE: 0.70,
            MemoryType.KNOWLEDGE_REFERENCE: 0.35,
        },
        description="Type-specific prior for durable student-memory admission.",
    )
    single_verified_evidence_maturity: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Maturity of one verified observation; deliberately below admission.",
    )
    evidence_admit_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum independent evidence maturity for evidence-based admission.",
    )
    contradiction_pending_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Contradiction strength at which ambiguous claims remain pending.",
    )
    stable_admit_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum stability/temporal validity for direct durable items.",
    )
    refinement_novelty: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Novelty assigned to an unambiguous structured refinement.",
    )
    relation_novelty_priors: dict[MemoryRelation, float] = Field(
        default_factory=lambda: {
            MemoryRelation.NEW: 1.0,
            MemoryRelation.DUPLICATE: 0.0,
            MemoryRelation.SUPPORTS: 0.2,
            MemoryRelation.REFINES: 0.55,
            MemoryRelation.CONTRADICTS: 0.4,
            MemoryRelation.SUPERSEDES: 0.8,
        },
        description="Novelty priors for deterministic relation outcomes.",
    )
    unknown_relation_novelty: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Conservative novelty fallback when relation is unset.",
    )
    maturity_support_weight: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Per-independent-support maturity contribution, capped at three.",
    )
    maturity_verified_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Per-verified-support maturity contribution, capped at two.",
    )
    maturity_conversation_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Per-distinct-conversation contribution, capped at two.",
    )
    maturity_source_diversity_weight: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Per-supporting-source-kind contribution, capped at two.",
    )
    maturity_contradiction_penalty: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Per-independent-contradiction maturity penalty, capped at two.",
    )
