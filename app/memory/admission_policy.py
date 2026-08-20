"""Type-aware deterministic Admission Policy v1 for shadow evaluation only."""

from __future__ import annotations

from app.memory.admission_config import AdmissionPolicyConfig
from app.memory.admission_matcher import RelationMatch
from app.memory.contracts import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionFeatures,
    AdmissionReasonCode,
    EvidenceSummary,
    MemoryCandidate,
    MemoryLifecycleStatus,
    MemoryRelation,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    TemporalScope,
)


class AdmissionPolicyV1:
    """Hard gates followed by relation and memory-type-specific policy."""

    def __init__(self, config: AdmissionPolicyConfig | None = None) -> None:
        self.config = config or AdmissionPolicyConfig()

    def decide(
        self,
        candidate: MemoryCandidate,
        features: AdmissionFeatures,
        match: RelationMatch,
        evidence: EvidenceSummary,
    ) -> AdmissionDecision:
        action: AdmissionAction
        reasons: list[AdmissionReasonCode]

        if candidate.target_memory_class is not MemoryTargetClass.STUDENT:
            action = AdmissionAction.REJECT
            reasons = [
                AdmissionReasonCode.QUALITY_QA_SEPARATION
                if candidate.target_memory_class is MemoryTargetClass.AGENT_EXPERIENCE
                else AdmissionReasonCode.SOURCE_SEPARATION
            ]
        elif candidate.source_kind is MemorySourceKind.CURATED_KNOWLEDGE:
            action = AdmissionAction.REJECT
            reasons = [AdmissionReasonCode.SOURCE_SEPARATION]
        elif (
            candidate.memory_type is MemoryType.TASK_STATE
            or candidate.temporal_scope in {TemporalScope.TURN_ONLY, TemporalScope.SESSION}
        ):
            action = AdmissionAction.REJECT
            reasons = [AdmissionReasonCode.TEMPORARY_STATE]
        elif candidate.source_kind is MemorySourceKind.MODEL_INFERENCE:
            action = AdmissionAction.PENDING
            reasons = [AdmissionReasonCode.UNSUPPORTED_INFERENCE]
        elif match.relation is MemoryRelation.DUPLICATE:
            action = AdmissionAction.REJECT
            reasons = [AdmissionReasonCode.DUPLICATE]
        elif match.relation is MemoryRelation.SUPERSEDES:
            action = AdmissionAction.SUPERSEDE
            reasons = [AdmissionReasonCode.EXPLICIT_SUPERSESSION]
        elif match.relation is MemoryRelation.CONTRADICTS:
            action = AdmissionAction.PENDING
            reasons = [AdmissionReasonCode.CONTRADICTION]
        elif match.relation is None:
            action = AdmissionAction.PENDING
            reasons = [AdmissionReasonCode.AMBIGUOUS_RELATION]
        elif candidate.memory_type in {
            MemoryType.PREFERENCE,
            MemoryType.PROFILE_FACT,
            MemoryType.LEARNING_GOAL,
        }:
            action, reasons = self._stable_direct_policy(candidate, features)
        elif candidate.memory_type is MemoryType.DIFFICULTY:
            action, reasons = self._difficulty_policy(candidate, features)
        elif candidate.memory_type in {
            MemoryType.BEHAVIOR_EVIDENCE,
            MemoryType.TOPIC_PROGRESS,
        }:
            action, reasons = self._evidence_policy(features, evidence)
        else:
            action = AdmissionAction.PENDING
            reasons = [AdmissionReasonCode.INSUFFICIENT_EVIDENCE]

        return AdmissionDecision(
            candidate_id=candidate.candidate_id,
            decision=action,
            matched_memory_id=match.matched_memory_id,
            relation=match.relation,
            reason_codes=reasons,
            policy_name=self.config.policy_name,
            policy_version=self.config.policy_version,
            resulting_confidence=features.factual_confidence.value,
            lifecycle_status={
                AdmissionAction.PENDING: MemoryLifecycleStatus.PENDING,
                AdmissionAction.ADMIT: MemoryLifecycleStatus.ACTIVE,
                AdmissionAction.SUPERSEDE: MemoryLifecycleStatus.SUPERSEDED,
            }.get(action),
        )

    def _stable_direct_policy(
        self,
        candidate: MemoryCandidate,
        features: AdmissionFeatures,
    ) -> tuple[AdmissionAction, list[AdmissionReasonCode]]:
        if (
            candidate.source_kind is MemorySourceKind.USER_STATED
            and candidate.temporal_scope is TemporalScope.PERSISTENT
            and features.stability.value >= self.config.stable_admit_threshold
            and features.temporal_validity.value >= self.config.stable_admit_threshold
        ):
            reason = {
                MemoryType.PREFERENCE: AdmissionReasonCode.STABLE_USER_PREFERENCE,
                MemoryType.PROFILE_FACT: AdmissionReasonCode.STABLE_PROFILE_FACT,
                MemoryType.LEARNING_GOAL: AdmissionReasonCode.STABLE_LEARNING_GOAL,
            }[candidate.memory_type]
            return AdmissionAction.ADMIT, [reason]
        return AdmissionAction.PENDING, [AdmissionReasonCode.INSUFFICIENT_EVIDENCE]

    def _difficulty_policy(
        self,
        candidate: MemoryCandidate,
        features: AdmissionFeatures,
    ) -> tuple[AdmissionAction, list[AdmissionReasonCode]]:
        if (
            candidate.source_kind is MemorySourceKind.USER_STATED
            and candidate.temporal_scope is TemporalScope.PERSISTENT
            and features.stability.value >= self.config.stable_admit_threshold
        ):
            return AdmissionAction.ADMIT, [AdmissionReasonCode.MATURE_EVIDENCE]
        return AdmissionAction.PENDING, [AdmissionReasonCode.INSUFFICIENT_EVIDENCE]

    def _evidence_policy(
        self,
        features: AdmissionFeatures,
        evidence: EvidenceSummary,
    ) -> tuple[AdmissionAction, list[AdmissionReasonCode]]:
        if (
            features.evidence_maturity.value >= self.config.evidence_admit_threshold
            and features.contradiction_strength.value
            < self.config.contradiction_pending_threshold
        ):
            return AdmissionAction.ADMIT, [AdmissionReasonCode.MATURE_EVIDENCE]
        if evidence.support_count <= 1:
            return AdmissionAction.PENDING, [AdmissionReasonCode.SINGLE_OBSERVATION]
        return AdmissionAction.PENDING, [AdmissionReasonCode.INSUFFICIENT_EVIDENCE]
