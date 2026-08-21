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
    (
        "图示讲解",
        ("画图", "画成图", "画一下", "图解", "可视化", "看图", "有图", "用图"),
    ),
    ("慢节奏讲解", ("讲慢", "慢点", "慢些", "慢一些")),
    ("快节奏讲解", ("讲快", "快点", "快些")),
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
    name = re.search(
        r"(?:我(?:(?:可能|也许))?叫|叫我)([A-Za-z0-9_\-\u4e00-\u9fff]{1,12})", message
    )
    result: dict[str, str] = {}
    if grade:
        result["grade"] = f"{grade.group(1)}年级"
    if name:
        result["preferred_name"] = name.group(1).casefold()
    return result


def _preference_values(message: str) -> list[str]:
    active_message = message
    if "现在" in message and any(cue in message for cue in ("以前", "之前")):
        active_message = message.split("现在", 1)[1]

    if re.search(r"(?:难题|复杂题).*(?:详细|展开)", message) and re.search(
        r"简单题.*(?:直接|要点|思路|简短)", message
    ):
        return ["按难度调整"]

    if (
        "复习" in message
        and any(cue in message for cue in ("第一次", "初次"))
        and "详细" in message
    ):
        return ["按学习阶段调整"]

    return [
        label
        for label, cues in _PREFERENCE_FAMILIES
        if any(cue in active_message for cue in cues)
    ]


def _is_negated_imperative(message: str) -> bool:
    return bool(re.match(r"^(?:请)?(?:不要|别)(?!.*(?:不是不|并不是不要))", message))


def _preference_qualifiers(
    message: str,
) -> tuple[dict[str, Any], TemporalScope]:
    qualifiers: dict[str, Any] = {}
    temporal_scope = TemporalScope.PERSISTENT

    if any(cue in message for cue in ("不是不喜欢", "并不是不要", "并非不喜欢")):
        qualifiers.update({"negated": False, "polarity": "positive"})
    elif any(cue in message for cue in ("不喜欢", "不想要", "讨厌")):
        qualifiers.update({"negated": True, "polarity": "negative"})

    conditional = re.search(r"如果(?:是)?([^，,。]{1,16}?)(?:请|就)", message)
    if conditional:
        qualifiers["condition"] = conditional.group(1).strip()

    scoped = re.search(r"只在([^，,。]{1,12}?)(?:讲|用|画|举|分)", message)
    if scoped:
        condition = scoped.group(1).strip()
        qualifiers["condition"] = condition
        qualifiers["scope"] = {
            "物理课": "physics",
            "化学课": "chemistry",
            "数学课": "mathematics",
        }.get(condition, condition.removesuffix("课").removesuffix("题"))

    only_subject = re.search(r"只有(几何|代数|物理|化学)(?:题|课)", message)
    if only_subject:
        qualifiers["subject"] = only_subject.group(1)
    if (
        "几何" in message
        and "condition" not in qualifiers
        and "subject" not in qualifiers
    ):
        qualifiers["subject"] = "几何"

    if re.search(r"(?:难题|复杂题).*(?:详细|展开)", message) and re.search(
        r"简单题.*(?:直接|要点|思路|简短)", message
    ):
        qualifiers["condition_map"] = {"hard": "detailed", "simple": "concise"}

    if any(cue in message for cue in ("一般", "通常", "平时")) and re.search(
        r"(?:这|当前|本)(?:道)?题.{0,6}(?:不要|别|不用)",
        message,
    ):
        qualifiers.update({"scope": "general", "condition": "except current problem"})

    if any(cue in message for cue in ("也许", "或许", "可能", "大概")):
        qualifiers["uncertainty"] = "hedged"
        temporal_scope = TemporalScope.UNKNOWN

    if any(cue in message for cue in ("更正", "纠正", "说错了", "其实")):
        qualifiers.update(
            {"explicit_correction": True, "correction_of": "prior preference"}
        )

    if "现在" in message and any(cue in message for cue in ("以前", "之前")):
        qualifiers["explicit_correction"] = True

    if "这次" in message and "以后不用" in message:
        qualifiers["temporal_expression"] = "这次；以后不用"
        temporal_scope = TemporalScope.TURN_ONLY
    elif "这一次" in message:
        temporal_scope = TemporalScope.TURN_ONLY
    elif "今天先" in message and "明天恢复" in message:
        qualifiers["temporal_expression"] = "今天；明天恢复"
        temporal_scope = TemporalScope.SESSION
    elif "今天先" in message:
        temporal_scope = TemporalScope.TURN_ONLY
    elif "今天" in message:
        qualifiers["temporal_expression"] = "今天"
        temporal_scope = TemporalScope.SESSION
    elif "暂时" in message:
        qualifiers["temporal_expression"] = "暂时"
        temporal_scope = TemporalScope.SHORT_LIVED
    elif "从现在起" in message:
        qualifiers["temporal_expression"] = "从现在起"
        temporal_scope = TemporalScope.PERSISTENT
    elif "以后" in message and temporal_scope is TemporalScope.UNKNOWN:
        qualifiers["temporal_expression"] = "以后"

    return qualifiers, temporal_scope


def _topic_from_message(message: str) -> str:
    topic_families = (
        ("分数", ("分数", "分子", "分母")),
        ("小数", ("小数",)),
        ("方程", ("方程",)),
        ("比例", ("比例",)),
        ("几何", ("几何",)),
        ("代数", ("代数",)),
    )
    return next(
        (
            topic
            for topic, cues in topic_families
            if any(cue in message for cue in cues)
        ),
        "",
    )


def _learning_goal(
    message: str,
    subject: str,
) -> tuple[dict[str, Any], TemporalScope] | None:
    bounded = re.match(
        r"(?P<when>(?:期末|考试|开学|放假|暑假|寒假|中考|月考)前)"
        r"(?P<goal>(?:学会|掌握|重点(?:学|补))[^，,。]+)",
        message,
    )
    if bounded:
        return (
            {
                "subject": subject,
                "goal": bounded.group("goal").strip(),
                "temporal_expression": bounded.group("when"),
            },
            TemporalScope.SHORT_LIVED,
        )

    semester = re.search(
        r"这学期(?:我)?想(?P<goal>(?:重点)?(?:学|补)[^，,。]+)",
        message,
    )
    if semester:
        return (
            {
                "subject": subject,
                "goal": semester.group("goal").strip(),
                "until": "semester",
            },
            TemporalScope.SHORT_LIVED,
        )

    durable = re.search(
        r"以后(?:我)?都?想(?P<goal>(?:重点)?(?:学|补)[^，,。]+)",
        message,
    )
    if durable:
        return (
            {"subject": subject, "goal": durable.group("goal").strip()},
            TemporalScope.PERSISTENT,
        )
    return None


def _resolved_progress(message: str, subject: str) -> dict[str, Any] | None:
    match = re.search(
        r"以前(?:怕|不会|不懂)(?P<topic>[^，,。]+)[，,].*"
        r"现在(?:会|懂|掌握)了",
        message,
    )
    if not match:
        return None
    topic = _topic_from_message(match.group("topic")) or match.group("topic").strip()
    return {
        "subject": subject,
        "topic": topic,
        "status": "resolved",
        "explicit_correction": True,
        "correction_of": "difficulty",
    }


def _difficulty(
    message: str,
    subject: str,
) -> tuple[dict[str, Any], TemporalScope] | None:
    confusion = re.search(r"(?:我)?总(?P<statement>(?:弄混|搞混)[^，,。]+)", message)
    if confusion:
        statement = confusion.group("statement").strip()
        return (
            {
                "subject": subject,
                "topic": _topic_from_message(statement),
                "statement": statement,
            },
            TemporalScope.PERSISTENT,
        )

    if any(
        cue in message for cue in ("不会", "不懂", "不明白", "没懂", "很难", "困难")
    ):
        return (
            {"subject": subject, "statement": "student_expressed_difficulty"},
            TemporalScope.SHORT_LIVED,
        )
    return None


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

    has_learning_episode = any(_text(item.get("episode")) for item in observations)

    if (
        message
        and not _is_hard_negative(message)
        and not _is_negated_imperative(message)
    ):
        profile = _profile_payload(message)
        if profile:
            profile_scope = TemporalScope.PERSISTENT
            if any(cue in message for cue in ("可能", "也许", "或许")):
                profile["uncertainty"] = "uncertain"
                profile_scope = TemporalScope.UNKNOWN
            candidates.append(
                _candidate(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    memory_type=MemoryType.PROFILE_FACT,
                    payload=profile,
                    source_kind=MemorySourceKind.USER_STATED,
                    temporal_scope=profile_scope,
                    confidence=1.0,
                )
            )

        preferences = _preference_values(message)
        if preferences and not has_learning_episode:
            qualifiers, temporal_scope = _preference_qualifiers(message)
            candidates.append(
                _candidate(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    memory_type=MemoryType.PREFERENCE,
                    payload={"teaching_preferences": preferences, **qualifiers},
                    source_kind=MemorySourceKind.USER_STATED,
                    temporal_scope=temporal_scope,
                    confidence=1.0,
                )
            )

    if message and not _is_hard_negative(message):
        goal = _learning_goal(message, subject)
        if goal:
            goal_payload, goal_scope = goal
            candidates.append(
                _candidate(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    memory_type=MemoryType.LEARNING_GOAL,
                    payload=goal_payload,
                    source_kind=MemorySourceKind.USER_STATED,
                    temporal_scope=goal_scope,
                    confidence=1.0,
                )
            )

        progress_payload = _resolved_progress(message, subject)
        if progress_payload:
            candidates.append(
                _candidate(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    memory_type=MemoryType.TOPIC_PROGRESS,
                    payload=progress_payload,
                    source_kind=MemorySourceKind.USER_STATED,
                    temporal_scope=TemporalScope.SHORT_LIVED,
                    confidence=0.9,
                )
            )
        else:
            difficulty = _difficulty(message, subject)
            if difficulty:
                difficulty_payload, difficulty_scope = difficulty
                candidates.append(
                    _candidate(
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        memory_type=MemoryType.DIFFICULTY,
                        payload=difficulty_payload,
                        source_kind=MemorySourceKind.USER_STATED,
                        temporal_scope=difficulty_scope,
                        confidence=0.85,
                    )
                )

    has_topic_observation = any(
        _text(item.get("topic")) and _text(item.get("status")) for item in observations
    )
    status = _text(verification.get("status"), 80)
    if (
        status in {"verified_correct", "verified_incorrect"}
        and not has_topic_observation
    ):
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
                    payload={
                        "subject": subject,
                        "topic": topic,
                        "status": observed_status,
                    },
                    source_kind=source,
                    temporal_scope=TemporalScope.SHORT_LIVED,
                    confidence=0.95
                    if source is MemorySourceKind.VERIFIED_RESULT
                    else 0.8,
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
