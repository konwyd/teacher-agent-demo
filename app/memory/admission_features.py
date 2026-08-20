"""Deterministic explainable features for Memory v2 Admission Policy v1."""

from __future__ import annotations

import hashlib

from app.memory.admission_config import AdmissionPolicyConfig
from app.memory.admission_matcher import RelationMatch
from app.memory.admission_normalization import NormalizedCandidate
from app.memory.contracts import (
    AdmissionFeatures,
    AdmissionFeatureValue,
    FrozenAdmissionContext,
    MemoryCandidate,
    MemoryRelation,
    MemorySourceKind,
    ProvenanceReference,
)


def _fingerprint(reference: ProvenanceReference) -> str:
    payload = reference.model_dump_json(exclude={"schema_version"})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


class AdmissionFeatureExtractor:
    """Extract features only from a candidate and frozen pre-write context."""

    def __init__(self, config: AdmissionPolicyConfig | None = None) -> None:
        self.config = config or AdmissionPolicyConfig()

    def extract(
        self,
        candidate: MemoryCandidate,
        normalized: NormalizedCandidate,
        match: RelationMatch,
        context: FrozenAdmissionContext,
    ) -> AdmissionFeatures:
        utility = self.config.type_utility_priors[candidate.memory_type]
        source_prior = self.config.source_reliability_priors[candidate.source_kind]
        factual_confidence = _bounded(source_prior * 0.7 + candidate.confidence * 0.3)
        stability = self.config.stability_priors[candidate.temporal_scope]
        temporal = self.config.temporal_validity_priors[candidate.temporal_scope]
        novelty = (
            self.config.relation_novelty_priors[match.relation]
            if match.relation is not None
            else self.config.unknown_relation_novelty
        )
        maturity, contradiction = self._evidence_values(
            candidate,
            normalized,
            match,
            context,
        )
        return AdmissionFeatures(
            future_utility=AdmissionFeatureValue(
                value=utility,
                source=f"type_prior:{candidate.memory_type.value}",
            ),
            factual_confidence=AdmissionFeatureValue(
                value=factual_confidence,
                source=f"source_prior:{candidate.source_kind.value}+candidate_signal",
            ),
            novelty=AdmissionFeatureValue(
                value=novelty,
                source=f"relation:{match.relation.value if match.relation else 'unset'}",
            ),
            stability=AdmissionFeatureValue(
                value=stability,
                source=f"temporal_stability:{candidate.temporal_scope.value}",
            ),
            temporal_validity=AdmissionFeatureValue(
                value=temporal,
                source=f"temporal_validity:{candidate.temporal_scope.value}",
            ),
            evidence_maturity=AdmissionFeatureValue(
                value=maturity,
                source="independent_provenance+frozen_shadow_ledger",
            ),
            contradiction_strength=AdmissionFeatureValue(
                value=contradiction,
                source="relation+independent_contradicting_provenance",
            ),
            content_type_prior=AdmissionFeatureValue(
                value=self.config.content_type_priors[candidate.memory_type],
                source=f"content_type:{candidate.memory_type.value}",
            ),
        )

    def _evidence_values(
        self,
        candidate: MemoryCandidate,
        normalized: NormalizedCandidate,
        match: RelationMatch,
        context: FrozenAdmissionContext,
    ) -> tuple[float, float]:
        # Committed existing_memories are intentionally never read here. They are
        # match-only and cannot become evidence or increase maturity.
        cluster = next(
            (
                item
                for item in context.evidence_clusters
                if normalized.semantic_key
                and item.semantic_key == normalized.semantic_key
            ),
            None,
        )
        supports: dict[str, ProvenanceReference] = {}
        contradictions: dict[str, ProvenanceReference] = {}
        if cluster is not None:
            supports.update(
                (_fingerprint(item), item)
                for item in cluster.supporting_provenance
            )
            contradictions.update(
                (_fingerprint(item), item)
                for item in cluster.contradicting_provenance
            )
        supports.update((_fingerprint(item), item) for item in candidate.provenance)

        support_refs = tuple(supports.values())
        contradiction_refs = tuple(contradictions.values())
        verified_count = sum(
            item.source_kind is MemorySourceKind.VERIFIED_RESULT
            for item in support_refs
        )
        if (
            len(support_refs) == 1
            and verified_count == 1
            and not contradiction_refs
        ):
            maturity = self.config.single_verified_evidence_maturity
        else:
            conversations = {item.conversation_id for item in support_refs}
            source_kinds = {item.source_kind for item in support_refs}
            maturity = (
                min(3, len(support_refs)) * self.config.maturity_support_weight
                + min(2, verified_count) * self.config.maturity_verified_weight
                + min(2, len(conversations))
                * self.config.maturity_conversation_weight
                + min(2, len(source_kinds))
                * self.config.maturity_source_diversity_weight
                - min(2, len(contradiction_refs))
                * self.config.maturity_contradiction_penalty
            )
        total = len(support_refs) + len(contradiction_refs)
        ledger_strength = len(contradiction_refs) / total if total else 0.0
        relation_strength = (
            match.confidence
            if match.relation in {MemoryRelation.CONTRADICTS, MemoryRelation.SUPERSEDES}
            else 0.0
        )
        return _bounded(maturity), _bounded(max(ledger_strength, relation_strength))
