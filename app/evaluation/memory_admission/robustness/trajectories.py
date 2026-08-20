"""Causal-duplicate attacks and deterministic long-trajectory simulation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel

from app.evaluation.memory_admission.robustness.layer_a import LayerAEvaluator
from app.evaluation.memory_admission.robustness.models import RobustnessRecord
from app.memory.admission_controller import MemoryAdmissionController
from app.memory.admission_ledger import ShadowEvidenceLedger
from app.memory.admission_normalization import CandidateNormalizer
from app.memory.contracts import (
    AdmissionAction,
    ExistingMemorySnapshot,
    FrozenAdmissionContext,
    MemoryCandidate,
    MemoryLifecycleStatus,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    ProvenanceReference,
    TemporalScope,
)


START = datetime(2026, 8, 20, tzinfo=timezone.utc)


class CausalDuplicateReport(BaseModel):
    records: list[RobustnessRecord]
    inflated_count: int
    gap_code: str


class LongTrajectoryResult(BaseModel):
    turn_count: int
    decisions: list[str]
    decision_counts: dict[str, int]
    final_ledger_size: int
    max_pending_ledger_size: int
    pending_survival_distribution: dict[str, int]
    promotion_points: list[int]
    promotion_latency_independent_events: list[int]
    contradiction_count: int
    stale_pending_items: int
    premature_admits: int
    missed_promotions: int
    contradiction_recovery_behavior: str
    independent_evidence_maturity: list[float]
    sqlite_path: str


class LongTrajectoryReport(BaseModel):
    trajectories: list[LongTrajectoryResult]
    llm_calls: int = 0


def run_causal_duplicate_attacks(
    data_path: Path, workspace: Path
) -> CausalDuplicateReport:
    report = LayerAEvaluator(data_path).run(workspace / "causal")
    records = [row for row in report.records if "causal_duplicate" in row.labels]
    inflated = sum(row.support_count > row.distinct_turn_count for row in records)
    return CausalDuplicateReport(
        records=records,
        inflated_count=inflated,
        gap_code="ROBUSTNESS_GAP_CAUSAL_DEDUP" if inflated else "",
    )


def _candidate(length: int, turn: int) -> MemoryCandidate:
    session = (turn - 1) // 10 + 1
    position = turn % 5
    if position == 0:
        memory_type = MemoryType.BEHAVIOR_EVIDENCE
        source = MemorySourceKind.VERIFIED_RESULT
        scope = TemporalScope.PERSISTENT
        payload = {"subject": "数学", "evidence_pattern": "sign_error"}
    elif position == 1:
        memory_type = MemoryType.TASK_STATE
        source = MemorySourceKind.OBSERVED_BEHAVIOR
        scope = TemporalScope.SESSION
        payload = {"subject": "数学", "requested_target": f"turn-{turn}"}
    elif position == 2:
        memory_type = MemoryType.TOPIC_PROGRESS
        source = MemorySourceKind.VERIFIED_RESULT
        scope = TemporalScope.PERSISTENT
        payload = {
            "subject": "数学",
            "topic": "fractions",
            "result": "incorrect" if turn <= length // 2 else "correct",
        }
    elif position == 3:
        memory_type = MemoryType.PREFERENCE
        source = MemorySourceKind.USER_STATED
        scope = TemporalScope.PERSISTENT
        payload = {"teaching_preferences": ["分步骤讲解"]}
    else:
        memory_type = MemoryType.LEARNING_GOAL
        source = MemorySourceKind.USER_STATED
        scope = TemporalScope.SHORT_LIVED
        payload = {"subject": "数学", "goal": "本阶段复习代数"}
    return MemoryCandidate(
        candidate_id=f"trajectory-{length}-{turn}",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=memory_type,
        payload=payload,
        source_kind=source,
        provenance=[ProvenanceReference(
            conversation_id=f"trajectory-{length}-session-{session}",
            turn_id=turn,
            source_kind=source,
            event_reference=f"turn:{turn}",
        )],
        confidence=0.95,
        temporal_scope=scope,
        created_at=START + timedelta(minutes=turn),
    )


def _run_one(root: Path, length: int) -> LongTrajectoryResult:
    path = root / f"trajectory-{length}.sqlite3"
    ledger = ShadowEvidenceLedger(path)
    controller = MemoryAdmissionController()
    normalizer = CandidateNormalizer()
    student_key = f"long-{length}"
    decisions: list[str] = []
    max_pending = 0
    pending_since: dict[str, int] = {}
    survival: list[int] = []
    promotions: list[int] = []
    promotion_latency: list[int] = []
    evidence_maturity: list[float] = []
    premature = 0
    missed = 0
    contradiction_seen = False
    admit_after_contradiction = False

    for turn in range(1, length + 1):
        candidate = _candidate(length, turn)
        normalized = normalizer.normalize(candidate)
        existing = ()
        if turn % 20 == 9 and candidate.memory_type is MemoryType.LEARNING_GOAL:
            candidate = candidate.model_copy(update={
                "memory_type": MemoryType.PREFERENCE,
                "payload": {"teaching_preferences": ["详细讲解"], "explicit_correction": True},
                "temporal_scope": TemporalScope.PERSISTENT,
            })
            normalized = normalizer.normalize(candidate)
            existing = (ExistingMemorySnapshot(
                memory_id=f"existing-correction-{turn}",
                target_memory_class=MemoryTargetClass.STUDENT,
                memory_type=MemoryType.PREFERENCE,
                semantic_key=normalized.semantic_key,
                payload={"teaching_preferences": ["简短讲解"]},
                source_kind=MemorySourceKind.USER_STATED,
                lifecycle_status=MemoryLifecycleStatus.ACTIVE,
            ),)
        elif turn % 20 == 14 and candidate.memory_type is MemoryType.LEARNING_GOAL:
            candidate = candidate.model_copy(update={
                "memory_type": MemoryType.PREFERENCE,
                "payload": {"teaching_preferences": ["分步骤讲解"]},
                "temporal_scope": TemporalScope.PERSISTENT,
            })
            normalized = normalizer.normalize(candidate)
            existing = (ExistingMemorySnapshot(
                memory_id=f"existing-duplicate-{turn}",
                target_memory_class=MemoryTargetClass.STUDENT,
                memory_type=MemoryType.PREFERENCE,
                semantic_key=normalized.semantic_key,
                payload=normalized.normalized_payload,
                source_kind=MemorySourceKind.USER_STATED,
                lifecycle_status=MemoryLifecycleStatus.ACTIVE,
            ),)
        context = FrozenAdmissionContext(
            student_key=student_key,
            conversation_id=candidate.provenance[0].conversation_id,
            turn_id=candidate.provenance[0].turn_id,
            existing_memories=existing,
            evidence_clusters=ledger.snapshot(student_key),
            captured_at=candidate.created_at,
        )
        result = controller.evaluate(candidate, context)
        decisions.append(result.decision.decision.value)
        if result.match.relation and result.match.relation.value == "contradicts":
            contradiction_seen = True
        elif contradiction_seen and result.decision.decision is AdmissionAction.ADMIT:
            admit_after_contradiction = True
        if candidate.memory_type is MemoryType.BEHAVIOR_EVIDENCE:
            evidence_maturity.append(result.features.evidence_maturity.value)
            if result.decision.decision is AdmissionAction.ADMIT:
                promotions.append(turn)
                promotion_latency.append(result.evidence_summary.distinct_turn_count)
            elif result.evidence_summary.distinct_turn_count >= 2:
                missed += 1
        if (
            result.decision.decision is AdmissionAction.ADMIT
            and candidate.memory_type in {MemoryType.BEHAVIOR_EVIDENCE, MemoryType.TOPIC_PROGRESS}
            and result.evidence_summary.distinct_turn_count < 2
        ):
            premature += 1
        ledger.append(
            student_key=student_key,
            candidate=candidate,
            normalized=normalized,
            decision=result.decision,
            evidence_maturity=result.features.evidence_maturity.value,
        )
        clusters = ledger.inspect(student_key)
        pending_keys = {
            item.semantic_key
            for item in clusters
            if item.last_proposed_decision is AdmissionAction.PENDING
        }
        for key in pending_keys:
            pending_since.setdefault(key, turn)
        for key in tuple(pending_since):
            if key not in pending_keys:
                survival.append(turn - pending_since.pop(key))
        max_pending = max(max_pending, len(pending_keys))

    final_clusters = ledger.inspect(student_key)
    for key, since in pending_since.items():
        survival.append(length - since + 1)
    survival_counts = Counter(
        "1-5" if value <= 5 else "6-20" if value <= 20 else "21+"
        for value in survival
    )
    return LongTrajectoryResult(
        turn_count=length,
        decisions=decisions,
        decision_counts=dict(Counter(decisions)),
        final_ledger_size=len(final_clusters),
        max_pending_ledger_size=max_pending,
        pending_survival_distribution=dict(survival_counts),
        promotion_points=promotions,
        promotion_latency_independent_events=promotion_latency,
        contradiction_count=sum(item.contradiction_count for item in final_clusters),
        stale_pending_items=len(pending_since),
        premature_admits=premature,
        missed_promotions=missed,
        contradiction_recovery_behavior=(
            "admit_observed_after_contradiction"
            if admit_after_contradiction
            else "no_admit_observed_after_contradiction"
            if contradiction_seen
            else "no_contradiction_observed"
        ),
        independent_evidence_maturity=evidence_maturity,
        sqlite_path=str(path),
    )


def run_long_trajectories(
    workspace: Path, *, lengths: tuple[int, ...] = (20, 50, 100)
) -> LongTrajectoryReport:
    workspace.mkdir(parents=True, exist_ok=True)
    return LongTrajectoryReport(
        trajectories=[_run_one(workspace, length) for length in lengths]
    )


from app.evaluation.memory_admission.robustness.named_probes import (  # noqa: E402
    run_named_misconception_probes,
)
