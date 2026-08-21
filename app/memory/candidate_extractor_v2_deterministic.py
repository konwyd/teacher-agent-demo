"""Deterministic CandidateExtractorV2 Route A.

This module only describes typed memory candidates. It does not make
admission decisions or participate in the authoritative memory write path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.memory.contracts import (
    MemoryCandidate,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    ProvenanceReference,
    TemporalScope,
)


_PREFERENCE_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("分步骤讲解", ("一步一步", "分步骤", "分步")),
    ("示例讲解", ("举例", "例子", "示例")),
    ("图示讲解", ("画图", "画成图", "图解", "可视化", "看图", "有图", "用图")),
    ("慢节奏讲解", ("讲慢", "慢点", "慢些", "慢一些")),
    ("快节奏讲解", ("讲快", "快点", "快些", "快一些")),
    ("简短讲解", ("短一点", "简短", "精简")),
    ("表格讲解", ("表格", "列表整理")),
)

_HARD_NEGATIVE_PATTERNS = (
    re.compile(r"(?:不要|别).{0,6}(?:记录|保存|记住).{0,6}(?:名字|姓名)"),
    re.compile(r"(?:课文|课本|故事).{0,10}(?:人物|主人公).{0,4}(?:叫|名为)"),
    re.compile(r"[“‘\"].{0,30}(?:一步一步|分步骤|举例|画图).{0,30}(?:什么|意思|含义)"),
    re.compile(r"(?:举例说明|举个例子说明|举一个例子说明).{0,20}(?:什么叫|定义|含义)"),
)


def _text(value: object, limit: int = 1200) -> str:
    return str(value or "").strip()[:limit]


def _records(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _candidate_id(
    conversation_id: str,
    turn_id: int,
    memory_type: MemoryType,
    payload: Mapping[str, Any],
) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        f"route-a\n{conversation_id}\n{turn_id}\n{memory_type.value}\n{canonical}".encode()
    ).hexdigest()[:24]
    return f"candidate_route_a_{digest}"


def _candidate(
    *,
    conversation_id: str,
    turn_id: int,
    memory_type: MemoryType,
    payload: dict[str, Any],
    source_kind: MemorySourceKind,
    temporal_scope: TemporalScope,
    confidence: float,
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


def _is_hard_negative(message: str) -> bool:
    return any(pattern.search(message) for pattern in _HARD_NEGATIVE_PATTERNS)


def _profile_payload(message: str) -> dict[str, str]:
    grade = re.search(r"(?:我是|我上)?([一二三四五六七八九1-9])年级", message)
    name = re.search(r"(?:我叫|叫我)([A-Za-z0-9_\-\u4e00-\u9fff]{1,12})", message)
    result: dict[str, str] = {}
    if grade:
        result["grade"] = f"{grade.group(1)}年级"
    if name:
        result["preferred_name"] = name.group(1).casefold()
    return result


def _preference_values(message: str) -> list[str]:
    return [
        label
        for label, cues in _PREFERENCE_FAMILIES
        if any(cue in message for cue in cues)
    ]


def _explicit_task_target(message: str, requested_target: str) -> bool:
    if not requested_target:
        return False
    compact_target = re.sub(r"\s+", "", requested_target)
    compact_message = re.sub(r"\s+", "", message)
    return compact_target in compact_message or any(
        cue in compact_message for cue in ("上一题", "当前题", "这题", "本题")
    )


def build_deterministic_memory_candidates(
    payload: Mapping[str, Any],
    *,
    conversation_id: str,
) -> list[MemoryCandidate]:
    """Extract Route A candidates through deterministic semantic families."""

    turn_id = max(1, int(payload.get("turn_id") or 1))
    message = _text(payload.get("student_message"))
    classification = dict(payload.get("classification") or {})
    contract = dict(payload.get("teaching_contract") or {})
    verification = dict(payload.get("verification") or {})
    observations = _records(payload.get("observations"))
    model_inferences = _records(payload.get("model_inferences"))
    subject = _text(classification.get("subject"), 40)
    candidates: list[MemoryCandidate] = []

    if message and not _is_hard_negative(message):
        profile = _profile_payload(message)
        if profile:
            candidates.append(
                _candidate(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    memory_type=MemoryType.PROFILE_FACT,
                    payload=profile,
                    source_kind=MemorySourceKind.USER_STATED,
                    temporal_scope=TemporalScope.PERSISTENT,
                    confidence=1.0,
                )
            )

        preferences = _preference_values(message)
        if preferences:
            candidates.append(
                _candidate(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    memory_type=MemoryType.PREFERENCE,
                    payload={"teaching_preferences": preferences},
                    source_kind=MemorySourceKind.USER_STATED,
                    temporal_scope=TemporalScope.PERSISTENT,
                    confidence=1.0,
                )
            )

    has_topic_observation = any(
        _text(item.get("topic")) and _text(item.get("status"))
        for item in observations
    )
    status = _text(verification.get("status"), 80)
    if status in {"verified_correct", "verified_incorrect"} and not has_topic_observation:
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
                temporal_scope=TemporalScope.SHORT_LIVED,
                confidence=0.95,
                verifier_reference=_text(verification.get("verifier_name"), 160),
            )
        )

    for observation in observations:
        topic = _text(observation.get("topic"), 160)
        observed_status = _text(observation.get("status"), 80)
        episode = _text(observation.get("episode"), 240)
        pattern = _text(observation.get("pattern"), 160)
        if topic and observed_status:
            source = (
                MemorySourceKind.VERIFIED_RESULT
                if status in {"verified_correct", "verified_incorrect"}
                else MemorySourceKind.OBSERVED_BEHAVIOR
            )
            candidates.append(
                _candidate(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    memory_type=MemoryType.TOPIC_PROGRESS,
                    payload={"subject": subject, "topic": topic, "status": observed_status},
                    source_kind=source,
                    temporal_scope=TemporalScope.SHORT_LIVED,
                    confidence=0.95 if source is MemorySourceKind.VERIFIED_RESULT else 0.8,
                    verifier_reference=_text(verification.get("verifier_name"), 160),
                )
            )
        if episode:
            candidates.append(
                _candidate(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    memory_type=MemoryType.LEARNING_EPISODE,
                    payload={"subject": subject, "episode": episode},
                    source_kind=MemorySourceKind.OBSERVED_BEHAVIOR,
                    temporal_scope=TemporalScope.SHORT_LIVED,
                    confidence=0.8,
                )
            )
        if pattern and int(observation.get("count") or 0) >= 2:
            candidates.append(
                _candidate(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    memory_type=MemoryType.BEHAVIOR_EVIDENCE,
                    payload={"subject": subject, "evidence_pattern": pattern},
                    source_kind=MemorySourceKind.OBSERVED_BEHAVIOR,
                    temporal_scope=TemporalScope.SHORT_LIVED,
                    confidence=0.8,
                )
            )

    for inference in model_inferences:
        if _text(inference.get("claim"), 80) != "difficulty":
            continue
        topic = _text(inference.get("topic"), 160)
        if topic:
            candidates.append(
                _candidate(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    memory_type=MemoryType.DIFFICULTY,
                    payload={
                        "subject": subject,
                        "topic": topic,
                        "uncertainty": "model_inference",
                    },
                    source_kind=MemorySourceKind.MODEL_INFERENCE,
                    temporal_scope=TemporalScope.UNKNOWN,
                    confidence=0.5,
                )
            )

    requested_target = _text(contract.get("requested_target"), 180)
    if requested_target:
        explicit = _explicit_task_target(message, requested_target)
        candidates.append(
            _candidate(
                conversation_id=conversation_id,
                turn_id=turn_id,
                memory_type=MemoryType.TASK_STATE,
                payload={"subject": subject, "requested_target": requested_target},
                source_kind=(
                    MemorySourceKind.USER_STATED
                    if explicit
                    else MemorySourceKind.OBSERVED_BEHAVIOR
                ),
                temporal_scope=(
                    TemporalScope.TURN_ONLY if explicit else TemporalScope.SESSION
                ),
                confidence=0.9 if explicit else 0.75,
            )
        )

    return candidates
