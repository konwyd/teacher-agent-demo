from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.memory.admission_config import AdmissionPolicyConfig
from app.memory.contracts import (
    AdmissionAction,
    AdmissionFeatures,
    AdmissionReasonCode,
    BaselineMemoryAction,
    EvidenceClusterSnapshot,
    EvidenceSummary,
    ExistingMemorySnapshot,
    FrozenAdmissionContext,
    MemoryAdmissionShadowRecord,
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


def _candidate() -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id="candidate-contract-phase2a",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=MemoryType.PREFERENCE,
        payload={"teaching_preferences": ["图示讲解"]},
        source_kind=MemorySourceKind.USER_STATED,
        provenance=[
            ProvenanceReference(
                conversation_id="conv-contract",
                turn_id=2,
                source_kind=MemorySourceKind.USER_STATED,
                event_reference="turn:2",
            )
        ],
        confidence=0.9,
        temporal_scope=TemporalScope.PERSISTENT,
        created_at=NOW,
    )


def test_admission_features_have_bounded_values_and_public_sources() -> None:
    features = AdmissionFeatures.model_validate(
        {
            name: {"value": 0.5, "source": f"test:{name}"}
            for name in (
                "future_utility",
                "factual_confidence",
                "novelty",
                "stability",
                "temporal_validity",
                "evidence_maturity",
                "contradiction_strength",
                "content_type_prior",
            )
        }
    )

    assert features.future_utility.value == 0.5
    assert features.future_utility.source == "test:future_utility"
    with pytest.raises(ValidationError):
        features.model_copy(
            update={"future_utility": {"value": 1.01, "source": "invalid"}}
        ).__class__.model_validate(
            {
                **features.model_dump(),
                "future_utility": {"value": 1.01, "source": "invalid"},
            }
        )


def test_existing_memory_snapshot_is_immutable_and_never_evidence() -> None:
    snapshot = ExistingMemorySnapshot(
        memory_id="memory-existing",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=MemoryType.PREFERENCE,
        semantic_key="student:preference:teaching_style",
        payload={"teaching_preferences": ["图示讲解"]},
        source_kind=MemorySourceKind.USER_STATED,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
    )

    assert snapshot.evidence_eligible is False
    with pytest.raises(ValidationError):
        ExistingMemorySnapshot(
            **{**snapshot.model_dump(), "evidence_eligible": True}
        )
    with pytest.raises(ValidationError):
        snapshot.memory_id = "changed"


def test_frozen_context_round_trip_contains_only_snapshot_values() -> None:
    cluster = EvidenceClusterSnapshot(
        cluster_id="cluster-1",
        semantic_key="student:preference:teaching_style",
        memory_type=MemoryType.PREFERENCE,
        supporting_provenance=tuple(_candidate().provenance),
        contradicting_provenance=(),
        support_count=1,
        contradiction_count=0,
        distinct_turn_count=1,
        distinct_conversation_count=1,
        verified_support_count=0,
        source_kinds=(MemorySourceKind.USER_STATED,),
        first_seen=NOW,
        last_seen=NOW,
        current_evidence_maturity=0.4,
    )
    context = FrozenAdmissionContext(
        student_key="student-1",
        conversation_id="conv-contract",
        turn_id=2,
        known_subject="数学",
        known_topic="一次方程",
        existing_memories=(),
        evidence_clusters=(cluster,),
        captured_at=NOW,
    )

    restored = FrozenAdmissionContext.model_validate_json(context.model_dump_json())

    assert restored == context
    assert isinstance(restored.evidence_clusters, tuple)
    with pytest.raises(ValidationError):
        context.turn_id = 3


def test_phase15_shadow_json_remains_backward_compatible() -> None:
    phase15 = {
        "schema_version": "1.0.0",
        "observer_version": "memory-v2-foundation-1",
        "record_id": "shadow-phase15",
        "candidate": _candidate().model_dump(mode="json"),
        "baseline_action": BaselineMemoryAction.ENQUEUE.value,
        "baseline_skip_reason": "",
        "proposed_decision": None,
        "policy_name": "none",
        "policy_version": "unimplemented",
        "matched_existing_memory_id": "",
        "relation": None,
        "conversation_id": "conv-contract",
        "turn_id": 2,
        "timestamp": NOW.isoformat(),
    }

    record = MemoryAdmissionShadowRecord.model_validate(phase15)

    assert record.features is None
    assert record.evidence_summary is None
    assert record.divergence is None
    assert record.decision_latency_seconds == 0.0
    assert record.controller_error == ""


def test_evidence_summary_and_phase2a_reason_families_serialize() -> None:
    summary = EvidenceSummary(
        support_count=3,
        contradiction_count=1,
        distinct_turn_count=3,
        distinct_conversation_count=2,
        verified_support_count=2,
        source_diversity_count=2,
        maturity=0.8,
    )

    assert summary.model_dump(mode="json")["maturity"] == 0.8
    assert {
        AdmissionReasonCode.SOURCE_SEPARATION,
        AdmissionReasonCode.STABLE_USER_PREFERENCE,
        AdmissionReasonCode.MATURE_EVIDENCE,
        AdmissionReasonCode.TEMPORARY_STATE,
        AdmissionReasonCode.UNSUPPORTED_INFERENCE,
        AdmissionReasonCode.EXPLICIT_SUPERSESSION,
    }.issubset(set(AdmissionReasonCode))


def test_policy_configuration_centralizes_all_priors_and_thresholds() -> None:
    config = AdmissionPolicyConfig()

    assert config.policy_name == "teacher-memory-admission"
    assert config.policy_version == "v1"
    assert set(config.source_reliability_priors) == set(MemorySourceKind)
    assert set(config.temporal_validity_priors) == set(TemporalScope)
    assert set(config.type_utility_priors) >= {
        MemoryType.PREFERENCE,
        MemoryType.PROFILE_FACT,
        MemoryType.TASK_STATE,
        MemoryType.DIFFICULTY,
        MemoryType.BEHAVIOR_EVIDENCE,
        MemoryType.TOPIC_PROGRESS,
        MemoryType.LEARNING_GOAL,
    }
    assert 0.0 < config.evidence_admit_threshold <= 1.0
    assert 0.0 < config.contradiction_pending_threshold <= 1.0
    assert config.single_verified_evidence_maturity < config.evidence_admit_threshold
    assert AdmissionPolicyConfig.model_fields["evidence_admit_threshold"].description


def test_contract_can_represent_all_phase2a_decision_and_relation_outputs() -> None:
    assert {item.value for item in AdmissionAction} == {
        "reject",
        "pending",
        "admit",
        "supersede",
    }
    assert {item.value for item in MemoryRelation} == {
        "new",
        "duplicate",
        "supports",
        "refines",
        "contradicts",
        "supersedes",
    }
