"""Conservative deterministic relation matching for shadow admission."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.memory.admission_normalization import NormalizedCandidate
from app.memory.contracts import (
    FrozenAdmissionContext,
    MemoryCandidate,
    MemoryRelation,
    MemorySourceKind,
)


class RelationMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    relation: MemoryRelation | None = None
    matched_memory_id: str = Field(default="", max_length=160)
    matched_cluster_id: str = Field(default="", max_length=160)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason_code: str = Field(default="", max_length=120)


def _is_refinement(existing: dict, candidate: dict) -> bool:
    if len(candidate) <= len(existing):
        return False
    return all(candidate.get(key) == value for key, value in existing.items())


class MemoryRelationMatcher:
    """Match against immutable pre-write memory and ledger snapshots only."""

    def match(
        self,
        candidate: MemoryCandidate,
        normalized: NormalizedCandidate,
        context: FrozenAdmissionContext,
    ) -> RelationMatch:
        if not normalized.semantic_key:
            return RelationMatch(reason_code="insufficient_structured_identity")

        existing = [
            item
            for item in context.existing_memories
            if item.semantic_key == normalized.semantic_key
        ]
        if len(existing) > 1:
            return RelationMatch(reason_code="ambiguous_existing_match")
        if len(existing) == 1:
            item = existing[0]
            current_payload = {
                key: value
                for key, value in item.payload.items()
                if key not in {"explicit_correction", "correction_of"}
            }
            if current_payload == normalized.normalized_payload:
                return RelationMatch(
                    relation=MemoryRelation.DUPLICATE,
                    matched_memory_id=item.memory_id,
                    confidence=1.0,
                    reason_code="exact_active_duplicate",
                )
            if _is_refinement(current_payload, normalized.normalized_payload):
                return RelationMatch(
                    relation=MemoryRelation.REFINES,
                    matched_memory_id=item.memory_id,
                    confidence=0.85,
                    reason_code="structured_refinement",
                )
            if (
                normalized.explicit_correction
                and candidate.source_kind is MemorySourceKind.USER_STATED
                and item.source_kind in {None, MemorySourceKind.USER_STATED}
            ):
                return RelationMatch(
                    relation=MemoryRelation.SUPERSEDES,
                    matched_memory_id=item.memory_id,
                    confidence=0.95,
                    reason_code="explicit_user_correction",
                )
            return RelationMatch(
                relation=MemoryRelation.CONTRADICTS,
                matched_memory_id=item.memory_id,
                confidence=0.75,
                reason_code="same_identity_different_value",
            )

        clusters = [
            item
            for item in context.evidence_clusters
            if item.semantic_key == normalized.semantic_key
        ]
        if len(clusters) == 1:
            cluster = clusters[0]
            if cluster.normalized_payload != normalized.normalized_payload:
                return RelationMatch(
                    relation=MemoryRelation.CONTRADICTS,
                    matched_cluster_id=cluster.cluster_id,
                    confidence=0.8,
                    reason_code="same_shadow_identity_different_value",
                )
            return RelationMatch(
                relation=MemoryRelation.SUPPORTS,
                matched_cluster_id=cluster.cluster_id,
                confidence=0.8,
                reason_code="independent_shadow_support",
            )
        if len(clusters) > 1:
            return RelationMatch(reason_code="ambiguous_evidence_match")
        return RelationMatch(
            relation=MemoryRelation.NEW,
            confidence=0.9,
            reason_code="new_structured_identity",
        )
