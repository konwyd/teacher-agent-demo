"""Non-authoritative Memory v2 observation helpers for Phase 1.5.

The adapter describes what the current 8502 write path attempted.  It neither
selects an admission outcome nor writes semantic memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Mapping

from pydantic import BaseModel

from app.memory.contracts import (
    AdmissionReasonCode,
    BaselineMemoryAction,
    MemoryAdmissionShadowRecord,
    MemoryCandidate,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    ProvenanceReference,
    TemporalScope,
)


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


class MemoryV2FeatureFlags(BaseModel):
    memory_admission_enabled: bool = False
    memory_admission_shadow_mode: bool = False
    memory_controller_enabled: bool = False
    memory_controller_shadow_mode: bool = False

    @classmethod
    def from_env(cls) -> "MemoryV2FeatureFlags":
        return cls(
            memory_admission_enabled=_enabled(
                os.getenv("TEACHER_MEMORY_ADMISSION_ENABLED")
            ),
            memory_admission_shadow_mode=_enabled(
                os.getenv("TEACHER_MEMORY_ADMISSION_SHADOW_MODE")
            ),
            memory_controller_enabled=_enabled(
                os.getenv("TEACHER_MEMORY_CONTROLLER_ENABLED")
            ),
            memory_controller_shadow_mode=_enabled(
                os.getenv("TEACHER_MEMORY_CONTROLLER_SHADOW_MODE")
            ),
        )

    @property
    def shadow_observation_enabled(self) -> bool:
        return self.memory_admission_shadow_mode or self.memory_controller_shadow_mode

    @property
    def authoritative_admission(self) -> bool:
        # Phase 1.5 deliberately has no code path that can authorize a write.
        return False


def _text(value: object, limit: int = 240) -> str:
    return str(value or "").strip()[:limit]


def _candidate_id(
    conversation_id: str, turn_id: int, memory_type: MemoryType, payload: Mapping[str, Any]
) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(
        f"{conversation_id}\n{turn_id}\n{memory_type.value}\n{canonical}".encode("utf-8")
    ).hexdigest()[:24]
    return f"candidate_{digest}"


def _candidate(
    *,
    conversation_id: str,
    turn_id: int,
    memory_type: MemoryType,
    payload: dict[str, Any],
    source_kind: MemorySourceKind,
    confidence: float,
    temporal_scope: TemporalScope,
    verifier_reference: str = "",
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=_candidate_id(conversation_id, turn_id, memory_type, payload),
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=memory_type,
        payload=payload,
        source_kind=source_kind,
        provenance=[
            ProvenanceReference(
                conversation_id=conversation_id,
                turn_id=turn_id,
                source_kind=source_kind,
                event_reference=f"turn:{turn_id}",
                verifier_reference=verifier_reference,
            )
        ],
        confidence=confidence,
        temporal_scope=temporal_scope,
    )


def build_memory_candidates(
    payload: Mapping[str, Any], *, conversation_id: str
) -> list[MemoryCandidate]:
    """Map existing bounded write inputs to typed candidates without policy."""

    turn_id = max(1, int(payload.get("turn_id") or 1))
    message = _text(payload.get("student_message"), 1200)
    classification = dict(payload.get("classification") or {})
    verification = dict(payload.get("verification") or {})
    contract = dict(payload.get("teaching_contract") or {})
    subject = _text(classification.get("subject"), 20)
    candidates: list[MemoryCandidate] = []

    grade = re.search(r"(?:我是|我上)?([一二三四五六七八九1-9])年级", message)
    name = re.search(r"(?:我叫|叫我)([A-Za-z0-9_\-\u4e00-\u9fff]{1,12})", message)
    if grade or name:
        fact = {
            key: value
            for key, value in {
                "grade": f"{grade.group(1)}年级" if grade else "",
                "preferred_name": name.group(1) if name else "",
            }.items()
            if value
        }
        candidates.append(
            _candidate(
                conversation_id=conversation_id,
                turn_id=turn_id,
                memory_type=MemoryType.PROFILE_FACT,
                payload=fact,
                source_kind=MemorySourceKind.USER_STATED,
                confidence=1.0,
                temporal_scope=TemporalScope.PERSISTENT,
            )
        )

    preferences = [
        label
        for cue, label in (("讲慢点", "慢节奏讲解"), ("一步一步", "分步骤讲解"), ("举例", "示例讲解"))
        if cue in message
    ]
    if preferences:
        candidates.append(
            _candidate(
                conversation_id=conversation_id,
                turn_id=turn_id,
                memory_type=MemoryType.PREFERENCE,
                payload={"teaching_preferences": preferences},
                source_kind=MemorySourceKind.USER_STATED,
                confidence=1.0,
                temporal_scope=TemporalScope.PERSISTENT,
            )
        )

    status = _text(verification.get("status"), 80)
    if status in {"verified_correct", "verified_incorrect"}:
        candidates.append(
            _candidate(
                conversation_id=conversation_id,
                turn_id=turn_id,
                memory_type=MemoryType.BEHAVIOR_EVIDENCE,
                payload={
                    "subject": subject,
                    "verification_status": status,
                    "summary": _text(verification.get("summary"), 180),
                },
                source_kind=MemorySourceKind.VERIFIED_RESULT,
                confidence=0.95,
                temporal_scope=TemporalScope.SHORT_LIVED,
                verifier_reference=_text(verification.get("verifier_name"), 160),
            )
        )

    if any(cue in message for cue in ("不会", "不懂", "不明白", "没懂")):
        candidates.append(
            _candidate(
                conversation_id=conversation_id,
                turn_id=turn_id,
                memory_type=MemoryType.DIFFICULTY,
                payload={"subject": subject, "statement": "student_expressed_difficulty"},
                source_kind=MemorySourceKind.USER_STATED,
                confidence=0.85,
                temporal_scope=TemporalScope.SHORT_LIVED,
            )
        )

    requested_target = _text(contract.get("requested_target"), 180)
    if requested_target:
        candidates.append(
            _candidate(
                conversation_id=conversation_id,
                turn_id=turn_id,
                memory_type=MemoryType.TASK_STATE,
                payload={"subject": subject, "requested_target": requested_target},
                source_kind=MemorySourceKind.OBSERVED_BEHAVIOR,
                confidence=0.75,
                temporal_scope=TemporalScope.SESSION,
            )
        )
    return candidates


def _baseline_skip_reason(payload: Mapping[str, Any], baseline_job_id: str) -> str:
    if baseline_job_id:
        return ""
    classification = dict(payload.get("classification") or {})
    if classification.get("scope") != "k9":
        return AdmissionReasonCode.BASELINE_SKIPPED_SCOPE.value
    if payload.get("memory_resolution_status") == "ambiguous":
        return AdmissionReasonCode.BASELINE_SKIPPED_AMBIGUOUS.value
    if int(payload.get("turn_id") or 0) < 1:
        return AdmissionReasonCode.BASELINE_SKIPPED_INVALID_TURN.value
    return AdmissionReasonCode.INSUFFICIENT_EVIDENCE.value


def observe_baseline_memory_action(
    payload: Mapping[str, Any],
    *,
    conversation_id: str,
    baseline_job_id: str,
    flags: MemoryV2FeatureFlags | None = None,
) -> list[MemoryAdmissionShadowRecord]:
    active_flags = flags or MemoryV2FeatureFlags.from_env()
    if not active_flags.shadow_observation_enabled:
        return []
    action = BaselineMemoryAction.ENQUEUE if baseline_job_id else BaselineMemoryAction.SKIP
    reason = _baseline_skip_reason(payload, baseline_job_id)
    turn_id = max(1, int(payload.get("turn_id") or 1))
    return [
        MemoryAdmissionShadowRecord(
            candidate=candidate,
            baseline_action=action,
            baseline_skip_reason=reason,
            proposed_decision=None,
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
        for candidate in build_memory_candidates(payload, conversation_id=conversation_id)
    ]

# Phase 2A two-stage facade; imported after Phase 1.5 helpers to avoid cycles.
from app.memory.admission_shadow import (  # noqa: E402
    PreparedAdmissionBatch,
    finalize_shadow_admission,
    prepare_shadow_admission,
)
