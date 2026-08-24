"""Utterance-derived candidates for isolated CandidateExtractorV2 Route B."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping

from app.memory.contracts import (
    MemoryCandidate,
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    ProvenanceReference,
    TemporalScope,
)

from .embedding import (
    DEFAULT_THRESHOLDS,
    FamilyMatch,
    SemanticFamily,
    SemanticFamilyRanker,
)
from .qualifiers import (
    QualifierAnalysis,
    analyze_clause,
    resolve_temporal_scope,
    segment_clauses,
)


QUALIFIER_PAYLOAD_KEYS = frozenset(
    {
        "negated",
        "polarity",
        "condition",
        "scope",
        "temporal_expression",
        "uncertainty",
        "explicit_correction",
        "correction_of",
    }
)
EXPLICIT_SELF_PREFERENCE_RE = re.compile(
    r"(?:我(?:喜欢|更喜欢|比较喜欢|特别喜欢|不喜欢|不太喜欢|有点喜欢|"
    r"偏好|习惯|希望|不希望|不想)|我的偏好)"
)
IMPLICIT_SELF_PREFERENCE_RE = re.compile(
    r"^(?:喜欢|更喜欢|比较喜欢|特别喜欢|不喜欢|不太喜欢|有点喜欢|"
    r"偏好|习惯|希望|不希望|不想)"
)
BOUNDED_PREFERENCE_REQUEST_RE = re.compile(
    r"^(?:(?:本次|这次|本轮|这一轮|这节课|本节课|今天这次|当前会话|暂时|"
    r"这一会儿|这几天|一段时间|最近|从现在起|以后都|今后|长期|一直)\s*)?"
    r"(?:请|给|举|画|讲(?:慢|得)|用(?:图|例)|展示|说明)|"
    r"^只在[^,，。;；]{1,20}课(?:请|给|举|画|讲(?:慢|得)|用(?:图|例)|展示|说明)"
)
CURRENT_INSTRUCTIONAL_REQUEST_RE = re.compile(
    r"(?:这|本)(?:一)?(?:道)?题.*(?:例子|例题|举例|示范|画图|图示|图解)|"
    r"(?:我希望你|我想让你|请你).*(?:给(?:我)?|讲|举|画|用|展示).*(?:例子|例题|图)"
)


@dataclass(frozen=True)
class RouteBConfig:
    family_thresholds: Mapping[SemanticFamily, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )


class CandidateExtractorRouteB:
    """Use embeddings for family selection and deterministic code for fields."""

    def __init__(
        self,
        ranker: SemanticFamilyRanker,
        config: RouteBConfig | None = None,
    ):
        self.ranker = ranker
        self.config = config or RouteBConfig()

    def extract(
        self,
        utterance: str,
        upstream: Mapping[str, Any],
        *,
        conversation_id: str,
        turn_id: int = 1,
    ) -> list[MemoryCandidate]:
        analyses = tuple(
            analyze_clause(clause) for clause in segment_clauses(utterance)
        )
        eligible = tuple(
            analysis for analysis in analyses if not analysis.hard_negative_reason
        )
        rankings = self.ranker.rank(tuple(analysis.text for analysis in eligible))
        candidate_specs: list[ParsedFamily] = []
        for analysis, matches in zip(eligible, rankings, strict=True):
            for match in matches:
                if not self._admitted(match):
                    continue
                parsed = _parse_utterance_family(match.family, analysis, upstream)
                if parsed is None:
                    continue
                memory_type, payload, default_scope = parsed
                payload.update(_qualifier_payload(analysis))
                candidate_specs.append(
                    (
                        memory_type,
                        payload,
                        resolve_temporal_scope(analysis, default_scope),
                    )
                )
        return [
            _make_candidate(
                conversation_id=conversation_id,
                turn_id=turn_id,
                memory_type=memory_type,
                payload=payload,
                source_kind=MemorySourceKind.USER_STATED,
                confidence=1.0,
                temporal_scope=temporal_scope,
            )
            for memory_type, payload, temporal_scope in _merge_candidate_specs(
                candidate_specs
            )
        ]

    def _admitted(self, match: FamilyMatch) -> bool:
        threshold = self.config.family_thresholds.get(
            match.family, DEFAULT_THRESHOLDS[match.family]
        )
        return match.similarity >= threshold


ParsedFamily = tuple[MemoryType, dict[str, Any], TemporalScope]


def _parse_utterance_family(
    family: SemanticFamily,
    analysis: QualifierAnalysis,
    upstream: Mapping[str, Any],
) -> ParsedFamily | None:
    if family is SemanticFamily.PROFILE:
        return _parse_profile(analysis)
    if family is SemanticFamily.PREFERENCE:
        return _parse_preference(analysis)
    if family is SemanticFamily.LEARNING_GOAL:
        return _parse_learning_goal(analysis, _subject(upstream))
    if family is SemanticFamily.TOPIC_PROGRESS:
        return _parse_topic_progress(analysis, _subject(upstream))
    if family is SemanticFamily.TASK_STATE:
        return _parse_task_state(analysis, _subject(upstream))
    if family is SemanticFamily.LEARNING_EPISODE:
        return _parse_learning_episode(analysis, _subject(upstream))
    if family is SemanticFamily.BEHAVIOR_EVIDENCE:
        return _parse_behavior_evidence(analysis, _subject(upstream))
    if family is SemanticFamily.DIFFICULTY:
        return _parse_difficulty(analysis, _subject(upstream))
    return None


def _parse_preference(analysis: QualifierAnalysis) -> ParsedFamily | None:
    if (
        not analysis.temporal_expression
        and not analysis.condition
        and CURRENT_INSTRUCTIONAL_REQUEST_RE.search(analysis.text)
    ):
        return None
    self_statement = bool(
        EXPLICIT_SELF_PREFERENCE_RE.search(analysis.text)
        or IMPLICIT_SELF_PREFERENCE_RE.match(analysis.text)
    )
    bounded_request = bool(
        (analysis.condition or analysis.temporal_expression)
        and BOUNDED_PREFERENCE_REQUEST_RE.match(analysis.text)
    )
    if not (self_statement or bounded_request):
        return None
    preferences = [
        label
        for cues, label in (
            (("慢一些", "慢点", "讲慢", "放慢"), "慢节奏讲解"),
            (("一步一步", "分步骤", "逐步"), "分步骤讲解"),
            (("例子", "举例", "示范", "例题"), "示例讲解"),
            (("图示", "图解", "画图"), "图示讲解"),
        )
        if any(cue in analysis.text for cue in cues)
    ]
    if not preferences:
        return None
    return (
        MemoryType.PREFERENCE,
        {"teaching_preferences": preferences},
        TemporalScope.PERSISTENT,
    )


def _subject(upstream: Mapping[str, Any]) -> str:
    classification = upstream.get("classification")
    if not isinstance(classification, Mapping):
        return ""
    return str(classification.get("subject") or "").strip()[:20]


def _parse_profile(analysis: QualifierAnalysis) -> ParsedFamily | None:
    if analysis.negated is True:
        return None
    grade_match = re.match(
        r"^(?:(?:我|本人)(?:现在|今年|目前|当前)?(?:是|读|上)?|)([一二三四五六七八九1-9])年级",
        analysis.text,
    )
    name_match = re.search(
        r"(?:请)?(?:称呼我|叫我|我叫)(?:为)?([A-Za-z][A-Za-z0-9_-]{0,31}|[\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,11})",
        analysis.text,
    )
    payload: dict[str, Any] = {}
    if grade_match:
        payload["grade"] = f"{grade_match.group(1)}年级"
    if name_match:
        payload["preferred_name"] = name_match.group(1).casefold()
    if not payload:
        return None
    return (
        MemoryType.PROFILE_FACT,
        payload,
        TemporalScope.PERSISTENT,
    )


def _parse_learning_goal(
    analysis: QualifierAnalysis, subject: str
) -> ParsedFamily | None:
    if not subject:
        return None
    if not re.match(
        r"^(?:我)?(?:想|希望|打算|计划|要|我的目标是|目标是)", analysis.text
    ):
        return None
    deadline = re.search(
        r"(?:期末前|月底前|考试前|中考前|高考前|本学期内|这学期内|[一二三四五六七八九十0-9]+(?:天|周|个月)内)",
        analysis.text,
    )
    goal = re.sub(
        r"^(?:我)?(?:想|希望|打算|计划|要|我的目标是|目标是)", "", analysis.text
    ).strip()
    if deadline:
        goal = goal.replace(f"在{deadline.group(0)}", "", 1)
        goal = goal.replace(deadline.group(0), "", 1)
    goal = goal.strip(" ,，。")[:180]
    if not goal or not any(
        cue in goal
        for cue in ("掌握", "学会", "提高", "完成", "通过", "补上", "理解", "熟练")
    ):
        return None
    payload = {"subject": subject, "goal": goal}
    if deadline:
        payload["deadline_expression"] = deadline.group(0)
    return (
        MemoryType.LEARNING_GOAL,
        payload,
        TemporalScope.PERSISTENT,
    )


def _parse_topic_progress(
    analysis: QualifierAnalysis, subject: str
) -> ParsedFamily | None:
    if not subject:
        return None
    for pattern, status in (
        (
            r"(?:我)?(?:还没|尚未|还没有)掌握(?P<topic>[^，,。]{1,80})(?:了)?$",
            "not_mastered",
        ),
        (r"(?:我)?(?:已经|已)?掌握(?P<topic>[^，,。]{1,80}?)(?:了)?$", "mastered"),
        (r"(?:我)?(?:正在学|在学习)(?P<topic>[^，,。]{1,80}?)(?:了)?$", "in_progress"),
    ):
        match = re.match(pattern, analysis.text)
        if not match:
            continue
        topic = match.group("topic").strip(" 了")
        if not topic:
            continue
        return (
            MemoryType.TOPIC_PROGRESS,
            {"subject": subject, "topic": topic, "status": status},
            TemporalScope.SHORT_LIVED,
        )
    return None


def _parse_task_state(analysis: QualifierAnalysis, subject: str) -> ParsedFamily | None:
    if not subject:
        return None
    target = re.sub(
        r"^(?:我)?(?:现在|当前|这会儿)?(?:想|要|需要|打算|准备)", "", analysis.text
    )
    target = target.strip(" ,，。")[:180]
    if not target or not target.startswith(
        ("完成", "解决", "继续", "做", "解答", "学习", "复习", "练习")
    ):
        return None
    return (
        MemoryType.TASK_STATE,
        {"subject": subject, "requested_target": target},
        TemporalScope.SESSION,
    )


def _parse_learning_episode(
    analysis: QualifierAnalysis, subject: str
) -> ParsedFamily | None:
    if not subject:
        return None
    summary = analysis.text.strip(" ,，。")[:180]
    episode_markers = ("上次", "昨天", "刚才", "那次", "上节课", "上周")
    episode = re.match(
        rf"^(?:我)?(?:{'|'.join(episode_markers)})(?P<body>.*)$", summary
    )
    if not episode:
        return None
    body = episode.group("body").strip()
    body = re.sub(r"^我", "", body).strip()
    body = re.sub(r"^(?:自己|独立|认真|成功地?)", "", body).strip()
    if not body.startswith(("练习", "测验", "考试", "课堂", "做", "解", "学", "答")):
        return None
    return (
        MemoryType.LEARNING_EPISODE,
        {"subject": subject, "summary": summary},
        TemporalScope.SHORT_LIVED,
    )


def _parse_behavior_evidence(
    analysis: QualifierAnalysis, subject: str
) -> ParsedFamily | None:
    if not subject:
        return None
    pattern = re.sub(r"^我", "", analysis.text).strip(" ,，。")[:180]
    if not re.match(
        r"^(?:连续[一二三四五六七八九十0-9]+次|多次|每次|总是|经常|反复)", pattern
    ):
        return None
    if not any(
        marker in pattern
        for marker in ("做对", "答对", "算对", "解出", "出错", "算错", "弄错", "检查")
    ):
        return None
    return (
        MemoryType.BEHAVIOR_EVIDENCE,
        {"subject": subject, "evidence_pattern": pattern},
        TemporalScope.SHORT_LIVED,
    )


def _parse_difficulty(analysis: QualifierAnalysis, subject: str) -> ParsedFamily | None:
    if not subject:
        return None
    reversal = re.match(
        r"^(?:我)?(?:经常|总是|老是|常常)?把(?P<topic>[A-Za-z0-9_\-\u4e00-\u9fff]{1,80}?)(?P<verb>搞反|弄反|弄错|算错|看错)",
        analysis.text,
    )
    if reversal:
        topic = reversal.group("topic").strip()
        return (
            MemoryType.DIFFICULTY,
            {
                "subject": subject,
                "topic": topic,
                "statement": f"{reversal.group('verb')}{topic}",
            },
            TemporalScope.SHORT_LIVED,
        )
    noncomprehension = re.match(
        r"^(?:我)?(?:还是|总是|一直|有点|有些|不太|完全)?(?P<verb>不懂|不会|不明白|没懂|弄不清)(?P<topic>[^，,。]{1,80})",
        analysis.text,
    )
    if noncomprehension:
        topic = noncomprehension.group("topic").strip(" 了")
        if topic:
            return (
                MemoryType.DIFFICULTY,
                {
                    "subject": subject,
                    "topic": topic,
                    "statement": f"{noncomprehension.group('verb')}{topic}",
                },
                TemporalScope.SHORT_LIVED,
            )
    return None


def _qualifier_payload(analysis: QualifierAnalysis) -> dict[str, Any]:
    values = {
        "negated": analysis.negated,
        "polarity": analysis.polarity,
        "condition": analysis.condition,
        "scope": analysis.scope,
        "temporal_expression": analysis.temporal_expression,
        "uncertainty": analysis.uncertainty,
        "explicit_correction": analysis.explicit_correction or None,
        "correction_of": analysis.correction_of,
    }
    return {key: value for key, value in values.items() if value not in ("", None)}


def _merge_candidate_specs(candidates: list[ParsedFamily]) -> list[ParsedFamily]:
    merged: list[ParsedFamily] = []
    for memory_type, payload, temporal_scope in candidates:
        qualifiers = {
            key: value
            for key, value in payload.items()
            if key in QUALIFIER_PAYLOAD_KEYS
        }
        core = {
            key: value
            for key, value in payload.items()
            if key not in QUALIFIER_PAYLOAD_KEYS
        }
        for index, (current_type, current_payload, current_scope) in enumerate(merged):
            if current_type is not memory_type or current_scope is not temporal_scope:
                continue
            current_qualifiers = {
                key: value
                for key, value in current_payload.items()
                if key in QUALIFIER_PAYLOAD_KEYS
            }
            if current_qualifiers != qualifiers:
                continue
            current_core = {
                key: value
                for key, value in current_payload.items()
                if key not in QUALIFIER_PAYLOAD_KEYS
            }
            shared_keys = current_core.keys() & core.keys()
            if any(current_core[key] != core[key] for key in shared_keys):
                continue
            merged_payload = {**current_core, **core, **qualifiers}
            merged[index] = (memory_type, merged_payload, temporal_scope)
            break
        else:
            merged.append((memory_type, dict(payload), temporal_scope))
    return merged


def _make_candidate(
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
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(
        f"{conversation_id}\n{turn_id}\n{memory_type.value}\n{canonical}".encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    provenance = ProvenanceReference(
        conversation_id=conversation_id,
        turn_id=turn_id,
        source_kind=source_kind,
        event_reference=f"turn:{turn_id}",
        verifier_reference=verifier_reference,
    )
    return MemoryCandidate(
        candidate_id=f"route_b_{digest}",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=memory_type,
        payload=payload,
        source_kind=source_kind,
        provenance=[provenance],
        confidence=confidence,
        temporal_scope=temporal_scope,
    )
