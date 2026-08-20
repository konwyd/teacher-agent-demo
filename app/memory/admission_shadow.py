"""Two-stage shadow-only admission facade with pre-write frozen snapshots."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.memory.admission_controller import MemoryAdmissionController
from app.memory.admission_ledger import ShadowEvidenceLedger
from app.memory.admission_normalization import CandidateNormalizer
from app.memory.contracts import (
    AdmissionAction,
    BaselineMemoryAction,
    ExistingMemorySnapshot,
    FrozenAdmissionContext,
    MemoryAdmissionShadowRecord,
    MemoryCandidate,
    MemoryLifecycleStatus,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
)
from app.memory.learner_memory import (
    StudentMemoryIdentity,
    StudentMemoryService,
    StudentProfileSnapshot,
)


DEFAULT_SHADOW_LEDGER_PATH = "data/runtime/memory_admission_shadow.sqlite3"


class _Ledger(Protocol):
    def snapshot(self, student_key: str, *, semantic_keys=None): ...

    def append(self, **kwargs) -> None: ...


class PreparedAdmissionItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate: MemoryCandidate
    context: FrozenAdmissionContext


class PreparedAdmissionBatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    items: tuple[PreparedAdmissionItem, ...] = ()
    preparation_error: str = Field(default="", max_length=240)


def _existing_snapshots(
    *,
    profile: StudentProfileSnapshot,
    memory_service: StudentMemoryService,
    candidates: tuple[MemoryCandidate, ...],
) -> tuple[ExistingMemorySnapshot, ...]:
    snapshots: list[ExistingMemorySnapshot] = []
    global_profile = profile.global_profile
    if global_profile.grade:
        snapshots.append(
            ExistingMemorySnapshot(
                memory_id=f"profile:{profile.student_profile_id}:grade",
                target_memory_class=MemoryTargetClass.STUDENT,
                memory_type=MemoryType.PROFILE_FACT,
                semantic_key="student:profile_fact:grade",
                payload={"grade": global_profile.grade},
                lifecycle_status=MemoryLifecycleStatus.ACTIVE,
            )
        )
    if global_profile.preferred_name:
        snapshots.append(
            ExistingMemorySnapshot(
                memory_id=f"profile:{profile.student_profile_id}:preferred_name",
                target_memory_class=MemoryTargetClass.STUDENT,
                memory_type=MemoryType.PROFILE_FACT,
                semantic_key="student:profile_fact:preferred_name",
                payload={"preferred_name": global_profile.preferred_name},
                lifecycle_status=MemoryLifecycleStatus.ACTIVE,
            )
        )
    if global_profile.teaching_preferences:
        snapshots.append(
            ExistingMemorySnapshot(
                memory_id=f"profile:{profile.student_profile_id}:teaching_preferences",
                target_memory_class=MemoryTargetClass.STUDENT,
                memory_type=MemoryType.PREFERENCE,
                semantic_key="student:preference:teaching_style",
                payload={
                    "teaching_preferences": sorted(
                        global_profile.teaching_preferences
                    )
                },
                lifecycle_status=MemoryLifecycleStatus.ACTIVE,
            )
        )

    subjects = sorted(
        {
            str(candidate.payload.get("subject") or "").strip()
            for candidate in candidates
            if candidate.payload.get("subject")
        }
    )
    for subject in subjects:
        for memory in memory_service.store.list_topic_memories(
            profile.student_profile_id,
            subject,
        ):
            snapshots.append(
                ExistingMemorySnapshot(
                    memory_id=memory.memory_id,
                    target_memory_class=MemoryTargetClass.STUDENT,
                    memory_type=MemoryType.TOPIC_PROGRESS,
                    semantic_key=(
                        f"student:topic_progress:{memory.subject.lower()}:"
                        f"{memory.topic.lower()}"
                    ),
                    payload={
                        "subject": memory.subject.lower(),
                        "topic": memory.topic.lower(),
                        "progress": memory.progress,
                    },
                    lifecycle_status=(
                        MemoryLifecycleStatus.ACTIVE
                        if memory.status == "active"
                        else MemoryLifecycleStatus.PENDING
                    ),
                )
            )
    return tuple(snapshots)


def prepare_shadow_admission(
    payload,
    *,
    identity: StudentMemoryIdentity,
    profile: StudentProfileSnapshot,
    memory_service: StudentMemoryService,
    ledger: _Ledger,
    flags,
) -> PreparedAdmissionBatch:
    """Capture all admission-side state before baseline enqueue can mutate it."""

    if not flags.shadow_observation_enabled:
        return PreparedAdmissionBatch()
    # Local import avoids changing the established Phase 1.5 module boundaries.
    from app.memory.foundation import build_memory_candidates

    candidates = tuple(
        build_memory_candidates(payload, conversation_id=identity.conversation_id)
    )
    if not candidates:
        return PreparedAdmissionBatch(enabled=True)
    try:
        existing = _existing_snapshots(
            profile=profile,
            memory_service=memory_service,
            candidates=candidates,
        )
        normalizer = CandidateNormalizer()
        semantic_keys = tuple(
            sorted(
                {
                    normalized.semantic_key
                    for candidate in candidates
                    if (normalized := normalizer.normalize(candidate)).semantic_key
                }
            )
        )
        clusters = tuple(
            ledger.snapshot(identity.user_id, semantic_keys=semantic_keys)
        )
        context = FrozenAdmissionContext(
            student_key=identity.user_id,
            conversation_id=identity.conversation_id,
            turn_id=max(1, int(payload.get("turn_id") or 1)),
            known_subject=str(
                dict(payload.get("classification") or {}).get("subject") or ""
            )[:40],
            known_topic=str(payload.get("problem_context") or "")[:160],
            existing_memories=existing,
            evidence_clusters=clusters,
        )
        return PreparedAdmissionBatch(
            enabled=True,
            items=tuple(
                PreparedAdmissionItem(candidate=candidate, context=context)
                for candidate in candidates
            ),
        )
    except Exception as exc:  # noqa: BLE001 - preparation must fail open
        empty_context = FrozenAdmissionContext(
            student_key=identity.user_id,
            conversation_id=identity.conversation_id,
            turn_id=max(1, int(payload.get("turn_id") or 1)),
        )
        return PreparedAdmissionBatch(
            enabled=True,
            items=tuple(
                PreparedAdmissionItem(candidate=candidate, context=empty_context)
                for candidate in candidates
            ),
            preparation_error=f"snapshot:{type(exc).__name__}",
        )


def _diverges(
    baseline: BaselineMemoryAction,
    decision: AdmissionAction,
) -> bool:
    semantic_write = decision in {AdmissionAction.ADMIT, AdmissionAction.SUPERSEDE}
    baseline_write = baseline is BaselineMemoryAction.ENQUEUE
    return semantic_write != baseline_write


def finalize_shadow_admission(
    prepared: PreparedAdmissionBatch,
    *,
    baseline_job_id: str,
    ledger: _Ledger,
    controller: MemoryAdmissionController | None = None,
) -> list[MemoryAdmissionShadowRecord]:
    """Evaluate only frozen context, then append evidence after the decision."""

    if not prepared.enabled:
        return []
    active_controller = controller or MemoryAdmissionController()
    baseline_action = (
        BaselineMemoryAction.ENQUEUE
        if baseline_job_id
        else BaselineMemoryAction.SKIP
    )
    records: list[MemoryAdmissionShadowRecord] = []
    for item in prepared.items:
        base = MemoryAdmissionShadowRecord(
            observer_version="memory-v2-admission-shadow-1",
            candidate=item.candidate,
            baseline_action=baseline_action,
            proposed_decision=None,
            policy_name=active_controller.config.policy_name,
            policy_version=active_controller.config.policy_version,
            conversation_id=item.context.conversation_id,
            turn_id=item.context.turn_id,
            controller_error=prepared.preparation_error,
        )
        if prepared.preparation_error:
            records.append(base)
            continue
        try:
            evaluation = active_controller.evaluate(item.candidate, item.context)
        except Exception as exc:  # noqa: BLE001 - controller must fail open
            base.controller_error = f"controller:{type(exc).__name__}"
            records.append(base)
            continue
        record = base.model_copy(
            update={
                "proposed_decision": evaluation.decision,
                "matched_existing_memory_id": evaluation.match.matched_memory_id,
                "relation": evaluation.match.relation,
                "features": evaluation.features,
                "evidence_summary": evaluation.evidence_summary,
                "divergence": _diverges(
                    baseline_action,
                    evaluation.decision.decision,
                ),
                "decision_latency_seconds": evaluation.latency_seconds,
            }
        )
        try:
            ledger.append(
                student_key=item.context.student_key,
                candidate=item.candidate,
                normalized=evaluation.normalized,
                decision=evaluation.decision,
                evidence_maturity=evaluation.features.evidence_maturity.value,
            )
        except Exception as exc:  # noqa: BLE001 - ledger must fail open
            record.controller_error = f"ledger:{type(exc).__name__}"
        records.append(record)
    return records


_LEDGERS: dict[str, ShadowEvidenceLedger] = {}
_LEDGERS_LOCK = threading.Lock()


def get_shadow_evidence_ledger(
    path: str | Path | None = None,
) -> ShadowEvidenceLedger:
    resolved = str(
        path
        or os.getenv("TEACHER_MEMORY_ADMISSION_LEDGER_DB")
        or DEFAULT_SHADOW_LEDGER_PATH
    )
    with _LEDGERS_LOCK:
        ledger = _LEDGERS.get(resolved)
        if ledger is None:
            ledger = ShadowEvidenceLedger(resolved)
            _LEDGERS[resolved] = ledger
        return ledger


def reset_shadow_evidence_ledgers() -> None:
    with _LEDGERS_LOCK:
        _LEDGERS.clear()
