from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.memory.admission_ledger import ShadowEvidenceLedger
from app.memory.admission_normalization import CandidateNormalizer
from app.memory.contracts import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionReasonCode,
    MemoryCandidate,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    ProvenanceReference,
    TemporalScope,
)
from app.memory.learner_memory import LearnerMemoryStore


NOW = datetime(2026, 8, 20, tzinfo=timezone.utc)


def _candidate(
    *,
    conversation_id: str,
    turn_id: int,
    value: str = "图示讲解",
    source_kind: MemorySourceKind = MemorySourceKind.USER_STATED,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=f"candidate-{conversation_id}-{turn_id}-{value}",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=MemoryType.PREFERENCE,
        payload={"teaching_preferences": [value]},
        source_kind=source_kind,
        provenance=[
            ProvenanceReference(
                conversation_id=conversation_id,
                turn_id=turn_id,
                source_kind=source_kind,
                event_reference=f"turn:{turn_id}",
                verifier_reference=(
                    f"verifier:{turn_id}"
                    if source_kind is MemorySourceKind.VERIFIED_RESULT
                    else ""
                ),
            )
        ],
        confidence=0.9,
        temporal_scope=TemporalScope.PERSISTENT,
        created_at=NOW,
    )


def _decision(candidate: MemoryCandidate, action: AdmissionAction) -> AdmissionDecision:
    return AdmissionDecision(
        candidate_id=candidate.candidate_id,
        decision=action,
        reason_codes=[AdmissionReasonCode.INSUFFICIENT_EVIDENCE],
        policy_name="teacher-memory-admission",
        policy_version="v1",
        created_at=NOW,
    )


def _append(
    ledger: ShadowEvidenceLedger,
    candidate: MemoryCandidate,
    *,
    maturity: float,
    action: AdmissionAction = AdmissionAction.PENDING,
) -> None:
    ledger.append(
        student_key="student-1",
        candidate=candidate,
        normalized=CandidateNormalizer().normalize(candidate),
        decision=_decision(candidate, action),
        evidence_maturity=maturity,
    )


def test_first_evidence_creates_persistent_shadow_cluster(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    ledger = ShadowEvidenceLedger(path)
    candidate = _candidate(conversation_id="conv-1", turn_id=1)

    _append(ledger, candidate, maturity=0.35)

    [cluster] = ledger.inspect("student-1")
    assert cluster.semantic_key == "student:preference:teaching_style"
    assert cluster.normalized_payload == {"teaching_preferences": ["图示讲解"]}
    assert cluster.support_count == 1
    assert cluster.contradiction_count == 0
    assert cluster.distinct_turn_count == 1
    assert cluster.distinct_conversation_count == 1
    assert cluster.current_evidence_maturity == 0.35
    assert cluster.last_proposed_decision is AdmissionAction.PENDING

    reloaded = ShadowEvidenceLedger(path)
    assert reloaded.inspect("student-1") == (cluster,)


def test_distinct_evidence_accumulates_turn_conversation_and_source_counts(
    tmp_path: Path,
) -> None:
    ledger = ShadowEvidenceLedger(tmp_path / "shadow.sqlite3")
    first = _candidate(conversation_id="conv-1", turn_id=1)
    second = _candidate(
        conversation_id="conv-2",
        turn_id=2,
        source_kind=MemorySourceKind.VERIFIED_RESULT,
    )

    _append(ledger, first, maturity=0.35)
    _append(ledger, second, maturity=0.8, action=AdmissionAction.ADMIT)

    [cluster] = ledger.snapshot(
        "student-1", semantic_keys=("student:preference:teaching_style",)
    )
    assert cluster.support_count == 2
    assert cluster.distinct_turn_count == 2
    assert cluster.distinct_conversation_count == 2
    assert cluster.verified_support_count == 1
    assert set(cluster.source_kinds) == {
        MemorySourceKind.USER_STATED,
        MemorySourceKind.VERIFIED_RESULT,
    }
    assert cluster.current_evidence_maturity == 0.8
    assert cluster.last_proposed_decision is AdmissionAction.ADMIT


def test_duplicate_provenance_does_not_double_count_or_change_maturity(
    tmp_path: Path,
) -> None:
    ledger = ShadowEvidenceLedger(tmp_path / "shadow.sqlite3")
    candidate = _candidate(conversation_id="conv-1", turn_id=1)

    _append(ledger, candidate, maturity=0.35)
    _append(ledger, candidate, maturity=0.9, action=AdmissionAction.ADMIT)

    [cluster] = ledger.inspect("student-1")
    assert cluster.support_count == 1
    assert cluster.current_evidence_maturity == 0.35
    assert cluster.last_proposed_decision is AdmissionAction.PENDING


def test_different_value_on_same_semantic_key_is_tracked_as_contradiction(
    tmp_path: Path,
) -> None:
    ledger = ShadowEvidenceLedger(tmp_path / "shadow.sqlite3")
    _append(
        ledger,
        _candidate(conversation_id="conv-1", turn_id=1, value="图示讲解"),
        maturity=0.35,
    )

    _append(
        ledger,
        _candidate(conversation_id="conv-2", turn_id=2, value="文字讲解"),
        maturity=0.25,
    )

    [cluster] = ledger.inspect("student-1")
    assert cluster.support_count == 1
    assert cluster.contradiction_count == 1
    assert cluster.distinct_turn_count == 2
    assert cluster.distinct_conversation_count == 2
    assert [item.conversation_id for item in cluster.contradicting_provenance] == [
        "conv-2"
    ]


def test_reset_is_scoped_to_one_student(tmp_path: Path) -> None:
    ledger = ShadowEvidenceLedger(tmp_path / "shadow.sqlite3")
    first = _candidate(conversation_id="conv-1", turn_id=1)
    _append(ledger, first, maturity=0.35)
    other = _candidate(conversation_id="conv-2", turn_id=1)
    ledger.append(
        student_key="student-2",
        candidate=other,
        normalized=CandidateNormalizer().normalize(other),
        decision=_decision(other, AdmissionAction.PENDING),
        evidence_maturity=0.35,
    )

    ledger.reset("student-1")

    assert ledger.inspect("student-1") == ()
    assert len(ledger.inspect("student-2")) == 1


def test_ledger_schema_is_isolated_from_active_student_memory(tmp_path: Path) -> None:
    ledger_path = tmp_path / "shadow.sqlite3"
    learner_path = tmp_path / "learner.sqlite3"
    ledger = ShadowEvidenceLedger(ledger_path)
    _append(
        ledger,
        _candidate(conversation_id="conv-1", turn_id=1),
        maturity=0.35,
    )
    learner = LearnerMemoryStore(learner_path)

    assert learner.list_topic_memories("student-1", "数学") == []
    with sqlite3.connect(ledger_path) as connection:
        ledger_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    with sqlite3.connect(learner_path) as connection:
        learner_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "student_topic_memories" not in ledger_tables
    assert "shadow_evidence_clusters" not in learner_tables
