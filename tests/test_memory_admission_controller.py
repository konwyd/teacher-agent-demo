from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.memory.admission_controller import MemoryAdmissionController
from app.memory.contracts import (
    AdmissionAction,
    AdmissionReasonCode,
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
    memory_type: MemoryType,
    payload: dict,
    source: MemorySourceKind = MemorySourceKind.USER_STATED,
    scope: TemporalScope = TemporalScope.PERSISTENT,
    target: MemoryTargetClass = MemoryTargetClass.STUDENT,
    conversation_id: str = "conv-3",
    turn_id: int = 3,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=f"candidate-{memory_type.value}-{conversation_id}-{turn_id}",
        target_memory_class=target,
        memory_type=memory_type,
        payload=payload,
        source_kind=source,
        provenance=[
            ProvenanceReference(
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
        ],
        confidence=0.9,
        temporal_scope=scope,
        created_at=NOW,
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


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(
            memory_type=MemoryType.PREFERENCE,
            payload={"teaching_preferences": ["图示讲解"]},
        ),
        _candidate(
            memory_type=MemoryType.PROFILE_FACT,
            payload={"grade": "五年级"},
        ),
        _candidate(
            memory_type=MemoryType.LEARNING_GOAL,
            payload={"subject": "数学", "goal": "这学期补上代数基础"},
        ),
    ],
)
def test_stable_direct_student_information_is_proposed_admit(
    candidate: MemoryCandidate,
) -> None:
    evaluation = MemoryAdmissionController().evaluate(candidate, _context())

    assert evaluation.decision.decision is AdmissionAction.ADMIT
    assert evaluation.decision.lifecycle_status is MemoryLifecycleStatus.ACTIVE
    assert evaluation.decision.policy_version == "v1"


def test_temporary_task_state_is_rejected_for_persistent_memory() -> None:
    candidate = _candidate(
        memory_type=MemoryType.TASK_STATE,
        payload={"subject": "数学", "requested_target": "完成第12题"},
        source=MemorySourceKind.OBSERVED_BEHAVIOR,
        scope=TemporalScope.SESSION,
    )

    evaluation = MemoryAdmissionController().evaluate(candidate, _context())

    assert evaluation.decision.decision is AdmissionAction.REJECT
    assert AdmissionReasonCode.TEMPORARY_STATE in evaluation.decision.reason_codes


def test_explicit_short_lived_difficulty_is_strong_pending() -> None:
    candidate = _candidate(
        memory_type=MemoryType.DIFFICULTY,
        payload={"subject": "数学", "concept": "配方法"},
        scope=TemporalScope.SHORT_LIVED,
    )

    evaluation = MemoryAdmissionController().evaluate(candidate, _context())

    assert evaluation.decision.decision is AdmissionAction.PENDING
    assert evaluation.features.future_utility.value == 0.75


def test_single_verified_error_is_pending_not_persistent_misconception() -> None:
    candidate = _candidate(
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        payload={
            "subject": "数学",
            "summary": "移项符号错误",
            "verification_status": "verified_incorrect",
        },
        source=MemorySourceKind.VERIFIED_RESULT,
        scope=TemporalScope.SHORT_LIVED,
    )

    evaluation = MemoryAdmissionController().evaluate(candidate, _context())

    assert evaluation.decision.decision is AdmissionAction.PENDING
    assert AdmissionReasonCode.SINGLE_OBSERVATION in evaluation.decision.reason_codes
    assert evaluation.evidence_summary.support_count == 1


def test_repeated_independent_verified_errors_may_promote_to_admit() -> None:
    candidate = _candidate(
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        payload={
            "subject": "数学",
            "summary": "移项符号错误",
            "verification_status": "verified_incorrect",
        },
        source=MemorySourceKind.VERIFIED_RESULT,
        scope=TemporalScope.SHORT_LIVED,
    )
    semantic_key = "student:behavior_evidence:数学:移项符号错误"
    supports = tuple(
        ProvenanceReference(
            conversation_id=f"conv-{index}",
            turn_id=index,
            source_kind=MemorySourceKind.VERIFIED_RESULT,
            event_reference=f"turn:{index}",
            verifier_reference=f"checker:{index}",
        )
        for index in (1, 2)
    )
    cluster = EvidenceClusterSnapshot(
        cluster_id="cluster-errors",
        semantic_key=semantic_key,
        normalized_payload={
            "subject": "数学",
            "summary": "移项符号错误",
            "verification_status": "verified_incorrect",
        },
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        supporting_provenance=supports,
        support_count=2,
        distinct_turn_count=2,
        distinct_conversation_count=2,
        verified_support_count=2,
        source_kinds=(MemorySourceKind.VERIFIED_RESULT,),
        first_seen=NOW,
        last_seen=NOW,
        current_evidence_maturity=0.7,
        last_proposed_decision=AdmissionAction.PENDING,
    )

    evaluation = MemoryAdmissionController().evaluate(
        candidate,
        _context(clusters=(cluster,)),
    )

    assert evaluation.match.relation is MemoryRelation.SUPPORTS
    assert evaluation.decision.decision is AdmissionAction.ADMIT
    assert AdmissionReasonCode.MATURE_EVIDENCE in evaluation.decision.reason_codes
    assert evaluation.evidence_summary.support_count == 3


def test_unsupported_model_inference_never_directly_admits() -> None:
    candidate = _candidate(
        memory_type=MemoryType.DIFFICULTY,
        payload={"subject": "数学", "concept": "配方法"},
        source=MemorySourceKind.MODEL_INFERENCE,
        scope=TemporalScope.UNKNOWN,
    )

    evaluation = MemoryAdmissionController().evaluate(candidate, _context())

    assert evaluation.decision.decision is AdmissionAction.PENDING
    assert AdmissionReasonCode.UNSUPPORTED_INFERENCE in evaluation.decision.reason_codes


def test_exact_active_duplicate_is_rejected_not_readmitted() -> None:
    candidate = _candidate(
        memory_type=MemoryType.PREFERENCE,
        payload={"teaching_preferences": ["图示讲解"]},
    )
    existing = ExistingMemorySnapshot(
        memory_id="memory-pref",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=MemoryType.PREFERENCE,
        semantic_key="student:preference:teaching_style",
        payload={"teaching_preferences": ["图示讲解"]},
        source_kind=MemorySourceKind.USER_STATED,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
    )

    evaluation = MemoryAdmissionController().evaluate(
        candidate,
        _context(existing=(existing,)),
    )

    assert evaluation.match.relation is MemoryRelation.DUPLICATE
    assert evaluation.decision.decision is AdmissionAction.REJECT
    assert AdmissionReasonCode.DUPLICATE in evaluation.decision.reason_codes


def test_unambiguous_explicit_user_correction_proposes_supersede() -> None:
    candidate = _candidate(
        memory_type=MemoryType.PREFERENCE,
        payload={
            "teaching_preferences": ["详细讲解"],
            "explicit_correction": True,
        },
    )
    existing = ExistingMemorySnapshot(
        memory_id="memory-pref-old",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=MemoryType.PREFERENCE,
        semantic_key="student:preference:teaching_style",
        payload={"teaching_preferences": ["简洁讲解"]},
        source_kind=MemorySourceKind.USER_STATED,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
    )

    evaluation = MemoryAdmissionController().evaluate(
        candidate,
        _context(existing=(existing,)),
    )

    assert evaluation.match.relation is MemoryRelation.SUPERSEDES
    assert evaluation.decision.decision is AdmissionAction.SUPERSEDE
    assert evaluation.decision.matched_memory_id == "memory-pref-old"
    assert AdmissionReasonCode.EXPLICIT_SUPERSESSION in evaluation.decision.reason_codes


def test_unresolved_contradiction_stays_pending_not_blind_supersede() -> None:
    candidate = _candidate(
        memory_type=MemoryType.PREFERENCE,
        payload={"teaching_preferences": ["详细讲解"]},
        source=MemorySourceKind.OBSERVED_BEHAVIOR,
    )
    existing = ExistingMemorySnapshot(
        memory_id="memory-pref-old",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=MemoryType.PREFERENCE,
        semantic_key="student:preference:teaching_style",
        payload={"teaching_preferences": ["简洁讲解"]},
        source_kind=MemorySourceKind.USER_STATED,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
    )

    evaluation = MemoryAdmissionController().evaluate(
        candidate,
        _context(existing=(existing,)),
    )

    assert evaluation.match.relation is MemoryRelation.CONTRADICTS
    assert evaluation.decision.decision is AdmissionAction.PENDING
    assert AdmissionReasonCode.CONTRADICTION in evaluation.decision.reason_codes


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (
            _candidate(
                memory_type=MemoryType.KNOWLEDGE_REFERENCE,
                payload={"reference": "配方法定义"},
                source=MemorySourceKind.CURATED_KNOWLEDGE,
                target=MemoryTargetClass.CURATED_KNOWLEDGE,
            ),
            AdmissionReasonCode.SOURCE_SEPARATION,
        ),
        (
            _candidate(
                memory_type=MemoryType.TEACHING_EXPERIENCE,
                payload={"summary": "历史高质量回答"},
                source=MemorySourceKind.OBSERVED_BEHAVIOR,
                target=MemoryTargetClass.AGENT_EXPERIENCE,
            ),
            AdmissionReasonCode.QUALITY_QA_SEPARATION,
        ),
    ],
)
def test_non_student_memory_classes_stay_outside_student_policy(
    candidate: MemoryCandidate,
    reason: AdmissionReasonCode,
) -> None:
    evaluation = MemoryAdmissionController().evaluate(candidate, _context())

    assert evaluation.decision.decision is AdmissionAction.REJECT
    assert reason in evaluation.decision.reason_codes


def test_one_verified_correct_progress_event_is_pending_not_mastery() -> None:
    candidate = _candidate(
        memory_type=MemoryType.TOPIC_PROGRESS,
        payload={
            "subject": "数学",
            "topic": "一次方程",
            "verification_status": "verified_correct",
        },
        source=MemorySourceKind.VERIFIED_RESULT,
        scope=TemporalScope.SHORT_LIVED,
    )

    evaluation = MemoryAdmissionController().evaluate(candidate, _context())

    assert evaluation.decision.decision is AdmissionAction.PENDING
    assert AdmissionReasonCode.SINGLE_OBSERVATION in evaluation.decision.reason_codes
