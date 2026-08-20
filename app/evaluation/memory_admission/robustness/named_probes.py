"""Named misconception trajectories A–E, excluded from classification metrics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel

from app.memory.admission_controller import MemoryAdmissionController
from app.memory.admission_ledger import ShadowEvidenceLedger
from app.memory.admission_normalization import CandidateNormalizer
from app.memory.contracts import (
    AdmissionAction,
    FrozenAdmissionContext,
    MemoryCandidate,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    ProvenanceReference,
    TemporalScope,
)


BASE = datetime(2026, 8, 20, tzinfo=timezone.utc)


class NamedProbeResult(BaseModel):
    name: str
    decisions: list[str]
    relations: list[str]
    semantic_keys: list[str]
    evidence_maturity: list[float]
    support_count: int
    contradiction_count: int
    distinct_causal_turns: int
    append_count: int
    event_references: list[str]
    gap_code: str = ""


def _candidate(
    name: str,
    index: int,
    *,
    memory_type: MemoryType,
    payload: dict,
    source: MemorySourceKind,
    conversation_id: str,
    turn_id: int,
    event_reference: str,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=f"probe-{name}-{index}",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=memory_type,
        payload=payload,
        source_kind=source,
        provenance=[ProvenanceReference(
            conversation_id=conversation_id,
            turn_id=turn_id,
            source_kind=source,
            event_reference=event_reference,
        )],
        confidence=0.95,
        temporal_scope=TemporalScope.PERSISTENT,
        created_at=BASE + timedelta(minutes=index),
    )


def _specs(name: str) -> list[MemoryCandidate]:
    if name == "A":
        return [_candidate(
            name, index,
            memory_type=MemoryType.BEHAVIOR_EVIDENCE,
            payload={"subject": "数学", "evidence_pattern": "sign_error"},
            source=MemorySourceKind.VERIFIED_RESULT,
            conversation_id=f"session-{index}", turn_id=1,
            event_reference=f"verified:{index}",
        ) for index in range(1, 4)]
    if name == "B":
        results = ["incorrect", "correct", "correct", "correct"]
        return [_candidate(
            name, index,
            memory_type=MemoryType.TOPIC_PROGRESS,
            payload={"subject": "数学", "topic": "fractions", "result": result},
            source=MemorySourceKind.VERIFIED_RESULT,
            conversation_id=f"session-{index}", turn_id=1,
            event_reference=f"verified:{index}",
        ) for index, result in enumerate(results, 1)]
    if name == "C":
        values = [
            ("incorrect", MemorySourceKind.VERIFIED_RESULT, "verified-error"),
            ("misread", MemorySourceKind.USER_STATED, "student-correction"),
            ("correct", MemorySourceKind.VERIFIED_RESULT, "verified-correct"),
        ]
        return [_candidate(
            name, index,
            memory_type=MemoryType.TOPIC_PROGRESS,
            payload={"subject": "数学", "topic": "fractions", "result": value},
            source=source,
            conversation_id=f"session-{index}", turn_id=1,
            event_reference=reference,
        ) for index, (value, source, reference) in enumerate(values, 1)]
    if name == "D":
        patterns = ["sign_error", "arithmetic_error", "copying_error"]
        return [_candidate(
            name, index,
            memory_type=MemoryType.BEHAVIOR_EVIDENCE,
            payload={"subject": "数学", "evidence_pattern": pattern},
            source=MemorySourceKind.VERIFIED_RESULT,
            conversation_id="session-1", turn_id=index,
            event_reference=f"verified:{pattern}",
        ) for index, pattern in enumerate(patterns, 1)]
    references = ["tool-call", "retry", "outbox-retry"]
    return [_candidate(
        name, index,
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        payload={"subject": "数学", "evidence_pattern": "sign_error"},
        source=MemorySourceKind.VERIFIED_RESULT,
        conversation_id="same-student-turn", turn_id=42,
        event_reference=reference,
    ) for index, reference in enumerate(references, 1)]


def _run(name: str, workspace: Path) -> NamedProbeResult:
    ledger = ShadowEvidenceLedger(workspace / f"probe-{name}.sqlite3")
    controller = MemoryAdmissionController()
    normalizer = CandidateNormalizer()
    student_key = f"probe-{name}"
    decisions, relations, keys, maturity, event_references = [], [], [], [], []
    for candidate in _specs(name):
        normalized = normalizer.normalize(candidate)
        context = FrozenAdmissionContext(
            student_key=student_key,
            conversation_id=candidate.provenance[0].conversation_id,
            turn_id=candidate.provenance[0].turn_id,
            evidence_clusters=ledger.snapshot(student_key),
            captured_at=candidate.created_at,
        )
        result = controller.evaluate(candidate, context)
        decisions.append(result.decision.decision.value)
        relations.append(result.match.relation.value if result.match.relation else "unset")
        keys.append(normalized.semantic_key)
        maturity.append(result.features.evidence_maturity.value)
        event_references.append(candidate.provenance[0].event_reference)
        ledger.append(
            student_key=student_key,
            candidate=candidate,
            normalized=normalized,
            decision=result.decision,
            evidence_maturity=result.features.evidence_maturity.value,
        )
    clusters = ledger.inspect(student_key)
    support = sum(item.support_count for item in clusters)
    contradictions = sum(item.contradiction_count for item in clusters)
    distinct_turns = sum(item.distinct_turn_count for item in clusters)
    return NamedProbeResult(
        name=name,
        decisions=decisions,
        relations=relations,
        semantic_keys=keys,
        evidence_maturity=maturity,
        support_count=support,
        contradiction_count=contradictions,
        distinct_causal_turns=distinct_turns,
        append_count=support + contradictions,
        event_references=event_references,
        gap_code=(
            "ROBUSTNESS_GAP_CAUSAL_DEDUP"
            if name == "E" and support > distinct_turns
            else ""
        ),
    )


def run_named_misconception_probes(
    workspace: Path,
) -> dict[str, NamedProbeResult]:
    workspace.mkdir(parents=True, exist_ok=True)
    return {name: _run(name, workspace) for name in "ABCDE"}
