from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.memory.admission_config import AdmissionPolicyConfig
from app.memory.admission_features import AdmissionFeatureExtractor
from app.memory.admission_matcher import RelationMatch
from app.memory.admission_normalization import CandidateNormalizer
from app.memory.contracts import (
    EvidenceClusterSnapshot,
    ExistingMemorySnapshot,
    FrozenAdmissionContext,
    MemoryCandidate,
    MemoryLifecycleStatus,
    MemoryRelation,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    ProvenanceReference,
    TemporalScope,
)


NOW = datetime(2026, 8, 20, tzinfo=timezone.utc)


def _provenance(
    source: MemorySourceKind,
    conversation_id: str,
    turn_id: int,
) -> ProvenanceReference:
    return ProvenanceReference(
        conversation_id=conversation_id,
        turn_id=turn_id,
        source_kind=source,
        event_reference=f"turn:{turn_id}",
        verifier_reference=(
            f"checker:{turn_id}"
            if source is MemorySourceKind.VERIFIED_RESULT
            else ""
        ),
    )


def _candidate(
    *,
    memory_type: MemoryType = MemoryType.PREFERENCE,
    source: MemorySourceKind = MemorySourceKind.USER_STATED,
    temporal_scope: TemporalScope = TemporalScope.PERSISTENT,
    confidence: float = 0.9,
    conversation_id: str = "conv-3",
    turn_id: int = 3,
    payload: dict | None = None,
) -> MemoryCandidate:
    defaults = {
        MemoryType.PREFERENCE: {"teaching_preferences": ["图示讲解"]},
        MemoryType.TASK_STATE: {
            "subject": "数学",
            "requested_target": "完成第12题",
        },
        MemoryType.BEHAVIOR_EVIDENCE: {
            "subject": "数学",
            "summary": "移项符号错误",
            "verification_status": "verified_incorrect",
        },
        MemoryType.DIFFICULTY: {
            "subject": "数学",
            "concept": "配方法",
        },
    }
    return MemoryCandidate(
        candidate_id=f"candidate-{conversation_id}-{turn_id}-{memory_type.value}",
        target_memory_class=(
            MemoryTargetClass.CURATED_KNOWLEDGE
            if source is MemorySourceKind.CURATED_KNOWLEDGE
            else MemoryTargetClass.STUDENT
        ),
        memory_type=memory_type,
        payload=payload or defaults[memory_type],
        source_kind=source,
        provenance=[_provenance(source, conversation_id, turn_id)],
        confidence=confidence,
        temporal_scope=temporal_scope,
        created_at=NOW,
    )


def _cluster(
    candidate: MemoryCandidate,
    *,
    supports: tuple[ProvenanceReference, ...],
    contradictions: tuple[ProvenanceReference, ...] = (),
    maturity: float = 0.4,
) -> EvidenceClusterSnapshot:
    normalized = CandidateNormalizer().normalize(candidate)
    all_refs = supports + contradictions
    return EvidenceClusterSnapshot(
        cluster_id="cluster-1",
        semantic_key=normalized.semantic_key,
        normalized_payload=normalized.normalized_payload,
        memory_type=candidate.memory_type,
        supporting_provenance=supports,
        contradicting_provenance=contradictions,
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
            for item in supports
        ),
        source_kinds=tuple(
            sorted({item.source_kind for item in all_refs}, key=lambda item: item.value)
        ),
        first_seen=NOW,
        last_seen=NOW,
        current_evidence_maturity=maturity,
    )


def _context(
    *,
    existing: tuple[ExistingMemorySnapshot, ...] = (),
    clusters: tuple[EvidenceClusterSnapshot, ...] = (),
) -> FrozenAdmissionContext:
    return FrozenAdmissionContext(
        student_key="student-1",
        conversation_id="conv-3",
        turn_id=3,
        existing_memories=existing,
        evidence_clusters=clusters,
        captured_at=NOW,
    )


def _features(
    candidate: MemoryCandidate,
    *,
    relation: MemoryRelation | None = MemoryRelation.NEW,
    existing: tuple[ExistingMemorySnapshot, ...] = (),
    clusters: tuple[EvidenceClusterSnapshot, ...] = (),
):
    return AdmissionFeatureExtractor(AdmissionPolicyConfig()).extract(
        candidate,
        CandidateNormalizer().normalize(candidate),
        RelationMatch(relation=relation, confidence=0.8),
        _context(existing=existing, clusters=clusters),
    )


def test_future_utility_is_type_specific_not_message_length() -> None:
    preference = _features(_candidate()).future_utility
    task = _features(
        _candidate(
            memory_type=MemoryType.TASK_STATE,
            source=MemorySourceKind.OBSERVED_BEHAVIOR,
            temporal_scope=TemporalScope.SESSION,
        )
    ).future_utility

    assert preference.value == 0.9
    assert task.value == 0.15
    assert preference.source == "type_prior:preference"


def test_factual_confidence_is_provenance_aware() -> None:
    scores = {
        source: _features(_candidate(source=source)).factual_confidence.value
        for source in MemorySourceKind
    }

    assert scores[MemorySourceKind.VERIFIED_RESULT] > scores[MemorySourceKind.USER_STATED]
    assert scores[MemorySourceKind.USER_STATED] > scores[MemorySourceKind.OBSERVED_BEHAVIOR]
    assert scores[MemorySourceKind.OBSERVED_BEHAVIOR] > scores[MemorySourceKind.MODEL_INFERENCE]
    assert scores[MemorySourceKind.CURATED_KNOWLEDGE] != scores[MemorySourceKind.MODEL_INFERENCE]


def test_stability_and_temporal_validity_use_declared_scope_and_type() -> None:
    stable = _features(_candidate())
    temporary = _features(
        _candidate(
            memory_type=MemoryType.TASK_STATE,
            source=MemorySourceKind.OBSERVED_BEHAVIOR,
            temporal_scope=TemporalScope.SESSION,
        )
    )
    unknown = _features(
        _candidate(
            memory_type=MemoryType.DIFFICULTY,
            source=MemorySourceKind.MODEL_INFERENCE,
            temporal_scope=TemporalScope.UNKNOWN,
        )
    )

    assert stable.stability.value == 0.9
    assert stable.temporal_validity.value == 0.95
    assert temporary.stability.value == 0.15
    assert temporary.temporal_validity.value == 0.15
    assert unknown.temporal_validity.value == 0.3


def test_single_verified_observation_has_configured_immature_evidence() -> None:
    candidate = _candidate(
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        source=MemorySourceKind.VERIFIED_RESULT,
        temporal_scope=TemporalScope.SHORT_LIVED,
    )

    maturity = _features(candidate).evidence_maturity

    assert maturity.value == AdmissionPolicyConfig().single_verified_evidence_maturity
    assert maturity.value < AdmissionPolicyConfig().evidence_admit_threshold


def test_repeated_distinct_verified_evidence_reaches_maturity_threshold() -> None:
    candidate = _candidate(
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        source=MemorySourceKind.VERIFIED_RESULT,
        temporal_scope=TemporalScope.SHORT_LIVED,
    )
    cluster = _cluster(
        candidate,
        supports=(
            _provenance(MemorySourceKind.VERIFIED_RESULT, "conv-1", 1),
            _provenance(MemorySourceKind.VERIFIED_RESULT, "conv-2", 2),
        ),
    )

    maturity = _features(candidate, clusters=(cluster,)).evidence_maturity.value

    assert maturity >= AdmissionPolicyConfig().evidence_admit_threshold


def test_duplicate_candidate_provenance_does_not_inflate_maturity() -> None:
    candidate = _candidate(
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        source=MemorySourceKind.VERIFIED_RESULT,
        temporal_scope=TemporalScope.SHORT_LIVED,
    )
    cluster = _cluster(candidate, supports=tuple(candidate.provenance))

    duplicated = _features(candidate, clusters=(cluster,)).evidence_maturity.value
    single = _features(candidate).evidence_maturity.value

    assert duplicated == single


def test_committed_topic_memory_never_increases_evidence_maturity() -> None:
    candidate = _candidate(
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        source=MemorySourceKind.VERIFIED_RESULT,
        temporal_scope=TemporalScope.SHORT_LIVED,
    )
    normalized = CandidateNormalizer().normalize(candidate)
    committed = ExistingMemorySnapshot(
        memory_id="baseline-just-wrote-this",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=candidate.memory_type,
        semantic_key=normalized.semantic_key,
        payload=normalized.normalized_payload,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
    )

    without_committed = _features(candidate).evidence_maturity.value
    with_committed = _features(
        candidate,
        relation=MemoryRelation.DUPLICATE,
        existing=(committed,),
    ).evidence_maturity.value

    assert with_committed == without_committed


def test_contradictory_independent_evidence_reduces_maturity_and_has_strength() -> None:
    candidate = _candidate(
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        source=MemorySourceKind.VERIFIED_RESULT,
        temporal_scope=TemporalScope.SHORT_LIVED,
    )
    support = _provenance(MemorySourceKind.VERIFIED_RESULT, "conv-1", 1)
    contradiction = _provenance(MemorySourceKind.VERIFIED_RESULT, "conv-2", 2)
    clean = _cluster(candidate, supports=(support,))
    mixed = _cluster(
        candidate,
        supports=(support,),
        contradictions=(contradiction,),
    )

    clean_features = _features(candidate, clusters=(clean,))
    mixed_features = _features(
        candidate,
        relation=MemoryRelation.CONTRADICTS,
        clusters=(mixed,),
    )

    assert mixed_features.evidence_maturity.value < clean_features.evidence_maturity.value
    assert mixed_features.contradiction_strength.value >= 0.5


@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        (MemoryRelation.NEW, 1.0),
        (MemoryRelation.DUPLICATE, 0.0),
        (MemoryRelation.SUPPORTS, 0.2),
        (MemoryRelation.REFINES, 0.55),
        (MemoryRelation.CONTRADICTS, 0.4),
        (MemoryRelation.SUPERSEDES, 0.8),
        (None, 0.25),
    ],
)
def test_novelty_has_explicit_relation_fallback(
    relation: MemoryRelation | None,
    expected: float,
) -> None:
    assert _features(_candidate(), relation=relation).novelty.value == expected
