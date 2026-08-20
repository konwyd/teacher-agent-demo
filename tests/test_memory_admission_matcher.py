from __future__ import annotations

from datetime import datetime, timezone

from app.memory.admission_matcher import MemoryRelationMatcher
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


def _candidate(
    *,
    candidate_id: str = "candidate-new",
    memory_type: MemoryType = MemoryType.PREFERENCE,
    payload: dict | None = None,
    source_kind: MemorySourceKind = MemorySourceKind.USER_STATED,
    conversation_id: str = "conv-new",
    turn_id: int = 3,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=candidate_id,
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=memory_type,
        payload=payload or {"teaching_preferences": ["图示讲解"]},
        source_kind=source_kind,
        provenance=[
            ProvenanceReference(
                conversation_id=conversation_id,
                turn_id=turn_id,
                source_kind=source_kind,
                event_reference=f"turn:{turn_id}",
            )
        ],
        confidence=0.9,
        temporal_scope=TemporalScope.PERSISTENT,
        created_at=NOW,
    )


def _existing(
    *,
    memory_id: str = "memory-existing",
    memory_type: MemoryType = MemoryType.PREFERENCE,
    semantic_key: str = "student:preference:teaching_style",
    payload: dict | None = None,
    source_kind: MemorySourceKind | None = MemorySourceKind.USER_STATED,
) -> ExistingMemorySnapshot:
    return ExistingMemorySnapshot(
        memory_id=memory_id,
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=memory_type,
        semantic_key=semantic_key,
        payload=payload or {"teaching_preferences": ["图示讲解"]},
        source_kind=source_kind,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
    )


def _context(
    *,
    existing: tuple[ExistingMemorySnapshot, ...] = (),
    clusters: tuple[EvidenceClusterSnapshot, ...] = (),
) -> FrozenAdmissionContext:
    return FrozenAdmissionContext(
        student_key="student-1",
        conversation_id="conv-new",
        turn_id=3,
        existing_memories=existing,
        evidence_clusters=clusters,
        captured_at=NOW,
    )


def test_exact_active_memory_match_is_duplicate() -> None:
    candidate = _candidate()
    normalized = CandidateNormalizer().normalize(candidate)

    match = MemoryRelationMatcher().match(
        candidate,
        normalized,
        _context(existing=(_existing(),)),
    )

    assert normalized.semantic_key == "student:preference:teaching_style"
    assert match.relation is MemoryRelation.DUPLICATE
    assert match.matched_memory_id == "memory-existing"


def test_same_independent_shadow_evidence_supports_cluster() -> None:
    candidate = _candidate(conversation_id="conv-new", turn_id=3)
    normalized = CandidateNormalizer().normalize(candidate)
    cluster = EvidenceClusterSnapshot(
        cluster_id="cluster-existing",
        semantic_key=normalized.semantic_key,
        memory_type=MemoryType.PREFERENCE,
        normalized_payload=normalized.normalized_payload,
        supporting_provenance=(
            ProvenanceReference(
                conversation_id="conv-old",
                turn_id=1,
                source_kind=MemorySourceKind.USER_STATED,
                event_reference="turn:1",
            ),
        ),
        support_count=1,
        distinct_turn_count=1,
        distinct_conversation_count=1,
        source_kinds=(MemorySourceKind.USER_STATED,),
        first_seen=NOW,
        last_seen=NOW,
        current_evidence_maturity=0.4,
    )

    match = MemoryRelationMatcher().match(
        candidate,
        normalized,
        _context(clusters=(cluster,)),
    )

    assert match.relation is MemoryRelation.SUPPORTS
    assert match.matched_cluster_id == "cluster-existing"


def test_same_semantic_key_with_different_shadow_value_is_contradiction() -> None:
    candidate = _candidate(payload={"teaching_preferences": ["图示讲解"]})
    normalized = CandidateNormalizer().normalize(candidate)
    cluster = EvidenceClusterSnapshot(
        cluster_id="cluster-existing",
        semantic_key=normalized.semantic_key,
        memory_type=MemoryType.PREFERENCE,
        normalized_payload={"teaching_preferences": ["详细讲解"]},
        supporting_provenance=(),
        support_count=1,
        distinct_turn_count=1,
        distinct_conversation_count=1,
        source_kinds=(MemorySourceKind.USER_STATED,),
        first_seen=NOW,
        last_seen=NOW,
        current_evidence_maturity=0.4,
    )

    match = MemoryRelationMatcher().match(
        candidate, normalized, _context(clusters=(cluster,))
    )

    assert match.relation is MemoryRelation.CONTRADICTS
    assert match.matched_cluster_id == "cluster-existing"



def test_specialized_preference_is_structured_refinement() -> None:
    candidate = _candidate(
        payload={
            "teaching_preferences": ["图示讲解"],
            "subject": "几何",
        }
    )
    normalized = CandidateNormalizer().normalize(candidate)

    match = MemoryRelationMatcher().match(
        candidate,
        normalized,
        _context(existing=(_existing(),)),
    )

    assert match.relation is MemoryRelation.REFINES
    assert match.matched_memory_id == "memory-existing"


def test_different_profile_fact_is_contradiction_without_explicit_correction() -> None:
    candidate = _candidate(
        memory_type=MemoryType.PROFILE_FACT,
        payload={"grade": "六年级"},
    )
    normalized = CandidateNormalizer().normalize(candidate)
    existing = _existing(
        memory_type=MemoryType.PROFILE_FACT,
        semantic_key="student:profile_fact:grade",
        payload={"grade": "五年级"},
    )

    match = MemoryRelationMatcher().match(
        candidate,
        normalized,
        _context(existing=(existing,)),
    )

    assert match.relation is MemoryRelation.CONTRADICTS
    assert match.matched_memory_id == "memory-existing"


def test_explicit_user_correction_can_supersede_one_unambiguous_match() -> None:
    candidate = _candidate(
        payload={
            "teaching_preferences": ["详细讲解"],
            "explicit_correction": True,
        }
    )
    normalized = CandidateNormalizer().normalize(candidate)
    existing = _existing(payload={"teaching_preferences": ["简洁讲解"]})

    match = MemoryRelationMatcher().match(
        candidate,
        normalized,
        _context(existing=(existing,)),
    )

    assert normalized.explicit_correction is True
    assert match.relation is MemoryRelation.SUPERSEDES
    assert match.matched_memory_id == "memory-existing"


def test_behavior_observation_cannot_supersede_user_stated_preference() -> None:
    candidate = _candidate(
        payload={
            "teaching_preferences": ["详细讲解"],
            "explicit_correction": True,
        },
        source_kind=MemorySourceKind.OBSERVED_BEHAVIOR,
    )
    normalized = CandidateNormalizer().normalize(candidate)

    match = MemoryRelationMatcher().match(
        candidate,
        normalized,
        _context(
            existing=(
                _existing(payload={"teaching_preferences": ["简洁讲解"]}),
            )
        ),
    )

    assert match.relation is MemoryRelation.CONTRADICTS


def test_multiple_possible_correction_targets_leave_relation_unset() -> None:
    candidate = _candidate(
        payload={
            "teaching_preferences": ["详细讲解"],
            "explicit_correction": True,
        }
    )
    normalized = CandidateNormalizer().normalize(candidate)

    match = MemoryRelationMatcher().match(
        candidate,
        normalized,
        _context(
            existing=(
                _existing(memory_id="memory-1", payload={"teaching_preferences": ["简洁讲解"]}),
                _existing(memory_id="memory-2", payload={"teaching_preferences": ["一步一步"]}),
            )
        ),
    )

    assert match.relation is None
    assert match.matched_memory_id == ""


def test_structured_candidate_with_no_match_is_new() -> None:
    candidate = _candidate(
        memory_type=MemoryType.LEARNING_GOAL,
        payload={"subject": "数学", "goal": "这学期补上代数基础"},
    )
    normalized = CandidateNormalizer().normalize(candidate)

    match = MemoryRelationMatcher().match(candidate, normalized, _context())

    assert normalized.semantic_key == "student:learning_goal:数学"
    assert match.relation is MemoryRelation.NEW


def test_unstructured_candidate_keeps_relation_unset_instead_of_guessing() -> None:
    candidate = _candidate(
        memory_type=MemoryType.DIFFICULTY,
        payload={"statement": "something vague"},
        source_kind=MemorySourceKind.MODEL_INFERENCE,
    )
    normalized = CandidateNormalizer().normalize(candidate)

    match = MemoryRelationMatcher().match(candidate, normalized, _context())

    assert normalized.semantic_key == ""
    assert match.relation is None
