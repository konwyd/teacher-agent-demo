"""Pure orchestration boundary for Memory v2 shadow admission evaluation."""

from __future__ import annotations

import hashlib
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from app.memory.admission_config import AdmissionPolicyConfig
from app.memory.admission_features import AdmissionFeatureExtractor
from app.memory.admission_matcher import MemoryRelationMatcher, RelationMatch
from app.memory.admission_normalization import CandidateNormalizer, NormalizedCandidate
from app.memory.admission_policy import AdmissionPolicyV1
from app.memory.contracts import (
    AdmissionDecision,
    AdmissionFeatures,
    EvidenceSummary,
    FrozenAdmissionContext,
    MemoryCandidate,
    MemorySourceKind,
    ProvenanceReference,
)


class AdmissionEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate: MemoryCandidate
    normalized: NormalizedCandidate
    match: RelationMatch
    features: AdmissionFeatures
    evidence_summary: EvidenceSummary
    decision: AdmissionDecision
    latency_seconds: float = Field(ge=0.0)


def _fingerprint(reference: ProvenanceReference) -> str:
    payload = reference.model_dump_json(exclude={"schema_version"})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MemoryAdmissionController:
    """Evaluate one candidate without repositories, stores, retrieval, or LLMs."""

    def __init__(
        self,
        config: AdmissionPolicyConfig | None = None,
        *,
        normalizer: CandidateNormalizer | None = None,
        matcher: MemoryRelationMatcher | None = None,
        feature_extractor: AdmissionFeatureExtractor | None = None,
        policy: AdmissionPolicyV1 | None = None,
    ) -> None:
        self.config = config or AdmissionPolicyConfig()
        self.normalizer = normalizer or CandidateNormalizer()
        self.matcher = matcher or MemoryRelationMatcher()
        self.feature_extractor = feature_extractor or AdmissionFeatureExtractor(
            self.config
        )
        self.policy = policy or AdmissionPolicyV1(self.config)

    def evaluate(
        self,
        candidate: MemoryCandidate,
        context: FrozenAdmissionContext,
    ) -> AdmissionEvaluation:
        started = perf_counter()
        normalized = self.normalizer.normalize(candidate)
        match = self.matcher.match(candidate, normalized, context)
        features = self.feature_extractor.extract(
            candidate,
            normalized,
            match,
            context,
        )
        evidence = self._evidence_summary(
            candidate,
            normalized,
            context,
            features.evidence_maturity.value,
        )
        decision = self.policy.decide(candidate, features, match, evidence)
        return AdmissionEvaluation(
            candidate=candidate,
            normalized=normalized,
            match=match,
            features=features,
            evidence_summary=evidence,
            decision=decision,
            latency_seconds=max(0.0, perf_counter() - started),
        )

    @staticmethod
    def _evidence_summary(
        candidate: MemoryCandidate,
        normalized: NormalizedCandidate,
        context: FrozenAdmissionContext,
        maturity: float,
    ) -> EvidenceSummary:
        supports: dict[str, ProvenanceReference] = {}
        contradictions: dict[str, ProvenanceReference] = {}
        cluster = next(
            (
                item
                for item in context.evidence_clusters
                if normalized.semantic_key
                and item.semantic_key == normalized.semantic_key
            ),
            None,
        )
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
        all_refs = tuple(supports.values()) + tuple(contradictions.values())
        return EvidenceSummary(
            support_count=len(supports),
            contradiction_count=len(contradictions),
            distinct_turn_count=len(
                {(item.conversation_id, item.turn_id) for item in all_refs}
            ),
            distinct_conversation_count=len(
                {item.conversation_id for item in all_refs}
            ),
            verified_support_count=sum(
                item.source_kind is MemorySourceKind.VERIFIED_RESULT
                for item in supports.values()
            ),
            source_diversity_count=len(
                {item.source_kind for item in supports.values()}
            ),
            maturity=maturity,
        )
