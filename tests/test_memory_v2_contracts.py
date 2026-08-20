"""Contract tests for Memory v2 Phase 1.5 (no admission policy)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.memory.contracts import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionReasonCode,
    BaselineMemoryAction,
    MemoryAdmissionShadowRecord,
    MemoryCandidate,
    MemoryLifecycleStatus,
    MemoryOutcomeObservation,
    MemoryRelation,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    ProvenanceReference,
    TemporalScope,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "memory_v2_candidate_cases.json"
NOW = datetime(2026, 8, 20, tzinfo=timezone.utc)


def _provenance(
    source_kind: MemorySourceKind = MemorySourceKind.USER_STATED,
    *,
    turn_id: int = 3,
    verifier_reference: str = "",
) -> ProvenanceReference:
    return ProvenanceReference(
        conversation_id="conv_contract",
        turn_id=turn_id,
        source_kind=source_kind,
        event_reference=f"turn:{turn_id}",
        verifier_reference=verifier_reference,
    )


def _candidate(**updates: object) -> MemoryCandidate:
    values: dict[str, object] = {
        "candidate_id": "candidate_contract_1",
        "target_memory_class": MemoryTargetClass.STUDENT,
        "memory_type": MemoryType.PREFERENCE,
        "payload": {"preference": "图像化解释"},
        "source_kind": MemorySourceKind.USER_STATED,
        "provenance": [_provenance()],
        "confidence": 0.9,
        "temporal_scope": TemporalScope.PERSISTENT,
        "created_at": NOW,
    }
    values.update(updates)
    return MemoryCandidate(**values)


def test_memory_candidate_and_provenance_round_trip() -> None:
    candidate = _candidate()

    restored = MemoryCandidate.model_validate_json(candidate.model_dump_json())

    assert restored == candidate
    assert restored.provenance[0].conversation_id == "conv_contract"
    assert restored.provenance[0].turn_id == 3
    assert "以后尽量" not in restored.model_dump_json()


def test_source_type_temporal_relation_and_lifecycle_spaces_are_explicit() -> None:
    assert {item.value for item in MemorySourceKind} == {
        "user_stated",
        "observed_behavior",
        "verified_result",
        "model_inference",
        "external_tool",
        "curated_knowledge",
    }
    assert {item.value for item in TemporalScope} == {
        "turn_only",
        "session",
        "short_lived",
        "persistent",
        "unknown",
    }
    assert {item.value for item in MemoryRelation} == {
        "new",
        "duplicate",
        "supports",
        "refines",
        "contradicts",
        "supersedes",
    }
    assert {item.value for item in MemoryLifecycleStatus} == {
        "pending",
        "active",
        "superseded",
        "retracted",
        "deleted",
    }


def test_model_inference_cannot_masquerade_as_user_stated_or_verified() -> None:
    inference = _provenance(MemorySourceKind.MODEL_INFERENCE)

    with pytest.raises(ValidationError, match="source_kind"):
        _candidate(
            source_kind=MemorySourceKind.USER_STATED,
            provenance=[inference],
        )
    with pytest.raises(ValidationError, match="source_kind"):
        _candidate(
            source_kind=MemorySourceKind.VERIFIED_RESULT,
            provenance=[inference],
        )


def test_curated_knowledge_cannot_be_a_student_memory_candidate() -> None:
    with pytest.raises(ValidationError, match="curated knowledge"):
        _candidate(
            target_memory_class=MemoryTargetClass.STUDENT,
            memory_type=MemoryType.KNOWLEDGE_REFERENCE,
            source_kind=MemorySourceKind.CURATED_KNOWLEDGE,
            provenance=[_provenance(MemorySourceKind.CURATED_KNOWLEDGE)],
        )


def test_quality_qa_remains_agent_experience_class() -> None:
    candidate = _candidate(
        target_memory_class=MemoryTargetClass.AGENT_EXPERIENCE,
        memory_type=MemoryType.TEACHING_EXPERIENCE,
        source_kind=MemorySourceKind.OBSERVED_BEHAVIOR,
        provenance=[_provenance(MemorySourceKind.OBSERVED_BEHAVIOR)],
    )

    assert candidate.target_memory_class is MemoryTargetClass.AGENT_EXPERIENCE
    assert candidate.target_memory_class is not MemoryTargetClass.STUDENT


def test_multiple_verified_evidence_refs_support_one_candidate() -> None:
    candidate = _candidate(
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        source_kind=MemorySourceKind.VERIFIED_RESULT,
        provenance=[
            _provenance(
                MemorySourceKind.VERIFIED_RESULT,
                turn_id=2,
                verifier_reference="linear_equation:2",
            ),
            _provenance(
                MemorySourceKind.VERIFIED_RESULT,
                turn_id=5,
                verifier_reference="linear_equation:5",
            ),
        ],
    )

    assert [item.turn_id for item in candidate.provenance] == [2, 5]


def test_admission_decision_round_trip_and_supersession_are_representable() -> None:
    decision = AdmissionDecision(
        candidate_id="candidate_contract_1",
        decision=AdmissionAction.SUPERSEDE,
        matched_memory_id="memory_old",
        relation=MemoryRelation.SUPERSEDES,
        reason_codes=[AdmissionReasonCode.USER_CORRECTION],
        policy_name="future-admission-policy",
        policy_version="future",
        resulting_confidence=0.4,
        lifecycle_status=MemoryLifecycleStatus.SUPERSEDED,
        created_at=NOW,
    )

    restored = AdmissionDecision.model_validate_json(decision.model_dump_json())

    assert restored == decision
    assert restored.resulting_confidence == 0.4
    assert restored.resulting_confidence < _candidate(confidence=0.95).confidence


def test_shadow_record_has_unset_proposal_and_baseline_outcome() -> None:
    candidate = _candidate()
    record = MemoryAdmissionShadowRecord(
        record_id="shadow_1",
        candidate=candidate,
        baseline_action=BaselineMemoryAction.ENQUEUE,
        baseline_skip_reason="",
        proposed_decision=None,
        policy_name="none",
        policy_version="unimplemented",
        matched_existing_memory_id="",
        relation=None,
        conversation_id="conv_contract",
        turn_id=3,
        timestamp=NOW,
    )

    restored = MemoryAdmissionShadowRecord.model_validate_json(
        record.model_dump_json()
    )

    assert restored == record
    assert restored.proposed_decision is None
    assert restored.schema_version == "1.0.0"
    assert "chain_of_thought" not in restored.model_dump_json()


def test_outcome_contract_contains_observations_not_reward() -> None:
    observation = MemoryOutcomeObservation(
        observation_id="outcome_1",
        conversation_id="conv_contract",
        turn_id=3,
        verified_student_step_status="verified_incorrect",
        tool_success_count=1,
        tool_failure_count=0,
        retrieval_hit_count=2,
        top_similarity=0.81,
        input_tokens=100,
        output_tokens=50,
        latency_seconds=0.25,
        teacher_turn_count=1,
        explicit_correction=False,
        timestamp=NOW,
    )

    serialized = observation.model_dump(mode="json")

    assert "reward" not in serialized
    assert serialized["verified_student_step_status"] == "verified_incorrect"


def test_teacher_specific_fixture_expresses_all_required_separations() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert {item["case_id"] for item in fixtures} == {
        "explicit_stable_preference",
        "temporary_task_state",
        "single_verified_error",
        "model_inference",
        "explicit_correction",
        "verified_repeated_evidence",
        "curated_graphrag_knowledge",
        "quality_qa_experience",
    }
    by_id = {item["case_id"]: item for item in fixtures}
    assert by_id["explicit_stable_preference"]["temporal_scope"] == "persistent"
    assert by_id["temporary_task_state"]["memory_type"] == "task_state"
    assert by_id["single_verified_error"]["memory_type"] == "behavior_evidence"
    assert by_id["model_inference"]["source_kind"] == "model_inference"
    assert by_id["explicit_correction"]["possible_relation"] == "supersedes"
    assert by_id["verified_repeated_evidence"]["evidence_count"] == 2
    assert by_id["curated_graphrag_knowledge"]["target_memory_class"] == (
        "curated_knowledge"
    )
    assert by_id["quality_qa_experience"]["target_memory_class"] == (
        "agent_experience"
    )
