"""Utterance-derived candidates for isolated CandidateExtractorV2 Route B."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

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
OWNER_MODIFIER_FRAGMENT = (
    r"(?:(?:真的|确实|其实|还是|仍然|依然|总是|一直|比较|特别|非常|更|"
    r"有点|有些|不太|完全|独立|自己|认真|成功地?)\s*)*"
)
PREFERENCE_IMPLICIT_PREFIX_RE = re.compile(
    r"(?:(?:真的|确实|还是|仍然|依然|比较|特别|非常|更|有点|有些|不太)\s*)*"
)
DIFFICULTY_IMPLICIT_PREFIX_RE = re.compile(
    r"(?:(?:真的|确实|还是|仍然|依然|总是|一直|有点|有些|不太|完全)\s*)*"
)
CONDITION_IMPLICIT_PREFIX_RE = re.compile(r"(?:(?:有|确实|真的)\s*)*")
CONDITION_TRIGGER_RE = re.compile(r"需要|想要|希望|想|要")
EPISODE_IMPLICIT_PREFIX_RE = re.compile(
    r"(?:(?:自己|独立|认真|成功地?)\s*)*"
    r"(?:(?:练习|测验|考试|课堂|做题?|解题?|学习|答题?)(?:时|中)?\s*)?"
    r"(?:(?:自己|独立|认真|成功地?)\s*)*"
)
EXPLICIT_SELF_OWNER_RE = re.compile(rf"(?:我|本人)(?:的)?\s*{OWNER_MODIFIER_FRAGMENT}$")
THIRD_PERSON_ROLE_FRAGMENT = (
    r"老师|同学|学生|学员|孩子|哥哥|姐姐|弟弟|妹妹|爸爸|妈妈|父亲|母亲|"
    r"爷爷|奶奶|外公|外婆|朋友"
)
THIRD_PERSON_OWNER_RE = re.compile(
    rf"(?:{THIRD_PERSON_ROLE_FRAGMENT}|[\u4e00-\u9fff]{{2,4}}|"
    rf"[A-Za-z][A-Za-z0-9_-]{{0,31}})\s*{OWNER_MODIFIER_FRAGMENT}$"
)
PREFERENCE_PREDICATE_RE = re.compile(r"不太喜欢|不喜欢|不希望|不想|喜欢|偏好|习惯|希望")
DIFFICULTY_PREDICATE_RE = re.compile(r"不明白|弄不清|不懂|不会|没懂")
EPISODE_RESULT_PREDICATE_RE = re.compile(
    r"解出|做出|答出|答对|算对|完成|掌握|学会|弄懂|解决|回答|讲"
)
REQUEST_ACTION_RE = re.compile(
    r"(?:请|给(?:我)?|举|画|讲(?:慢|得)?|用(?:图|例)|展示|说明)"
)
CURRENT_PROBLEM_INSTRUCTION_RE = re.compile(
    r"(?:这|本)(?:一)?(?:道)?题.*(?:例子|例题|举例|示范|画图|图示|图解)"
)
TEACHER_REQUEST_PREFIX_RE = re.compile(r"(?:我希望|我想让你|请你)")
INSTRUCTIONAL_ACTION_RE = re.compile(r"(?:给(?:我)?|讲|举|画|用|展示|演示|做)")
INSTRUCTIONAL_OBJECT_RE = re.compile(r"(?:例子|例题|示范|画图|图示|图解)")
REQUEST_RECIPIENT_MODIFIER_FRAGMENT = r"(?:(?:比较|特别|非常|更|耐心地?)\s*)*"
REQUEST_RECIPIENT_RE = re.compile(
    rf"(?:给|向|为)\s*(?P<recipient>我|本人|{THIRD_PERSON_ROLE_FRAGMENT}|"
    rf"[\u4e00-\u9fff]{{2,4}}?|[A-Za-z][A-Za-z0-9_-]{{0,31}})\s*"
    rf"{REQUEST_RECIPIENT_MODIFIER_FRAGMENT}"
    r"(?=(?:一|几|些|个|讲|展示|说明|看|用|画|举|听|做|解))"
)
MAX_MALFORMED_UPSTREAM_DIAGNOSTICS = 10_000


@dataclass(frozen=True)
class RouteBConfig:
    family_thresholds: Mapping[SemanticFamily, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )


@dataclass
class RouteBDiagnostics:
    malformed_upstream_items: int = 0

    def record_malformed_upstream_items(self, count: int) -> None:
        self.malformed_upstream_items = min(
            MAX_MALFORMED_UPSTREAM_DIAGNOSTICS,
            self.malformed_upstream_items + max(0, count),
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
        self.diagnostics = RouteBDiagnostics()

    def extract(
        self,
        utterance: str,
        upstream: Mapping[str, Any],
        *,
        conversation_id: str,
        turn_id: int = 1,
    ) -> list[MemoryCandidate]:
        sanitized_upstream, malformed_count = _sanitize_optional_upstream(upstream)
        self.diagnostics.record_malformed_upstream_items(malformed_count)
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
                parsed = _parse_utterance_family(
                    match.family, analysis, sanitized_upstream
                )
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
        utterance_candidates = [
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
        upstream_candidates = [
            *_verification_candidates(
                sanitized_upstream,
                conversation_id=conversation_id,
                turn_id=turn_id,
            ),
            *_observation_candidates(
                sanitized_upstream,
                conversation_id=conversation_id,
                turn_id=turn_id,
            ),
            *_model_inference_candidates(
                sanitized_upstream,
                conversation_id=conversation_id,
                turn_id=turn_id,
            ),
        ]
        return _merge_candidates((*utterance_candidates, *upstream_candidates))

    def _admitted(self, match: FamilyMatch) -> bool:
        threshold = self.config.family_thresholds.get(
            match.family, DEFAULT_THRESHOLDS[match.family]
        )
        return match.similarity >= threshold


ParsedFamily = tuple[MemoryType, dict[str, Any], TemporalScope]


def _required_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _normalize_observation(observation: Mapping[str, Any]) -> dict[str, Any] | None:
    if "topic" in observation or "status" in observation:
        topic = _required_text(observation.get("topic"), 160)
        status = _required_text(observation.get("status"), 80)
        if not topic or not status:
            return None
        return {"topic": topic, "status": status}
    if "episode" in observation:
        episode = _required_text(observation.get("episode"), 180)
        return {"episode": episode} if episode else None
    if "pattern" in observation:
        pattern = _required_text(observation.get("pattern"), 180)
        return {"pattern": pattern} if pattern else None
    if "evidence_pattern" in observation:
        pattern = _required_text(observation.get("evidence_pattern"), 180)
        return {"evidence_pattern": pattern} if pattern else None
    return None


def _sanitize_optional_upstream(
    upstream: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    sanitized = dict(upstream)
    malformed_count = 0

    if "verification" not in upstream:
        sanitized["verification"] = {}
    else:
        verification = upstream["verification"]
        if not isinstance(verification, Mapping):
            malformed_count += 1
            sanitized["verification"] = {}
        elif not verification:
            sanitized["verification"] = {}
        else:
            status = _required_text(verification.get("status"), 80)
            if status not in {"verified_correct", "verified_incorrect"}:
                malformed_count += 1
                sanitized["verification"] = {}
            else:
                sanitized["verification"] = {
                    "status": status,
                    "summary": _required_text(verification.get("summary"), 180),
                    "verifier_name": _required_text(
                        verification.get("verifier_name"), 160
                    ),
                }

    valid_observations: list[dict[str, Any]] = []
    if "observations" in upstream:
        observations = upstream["observations"]
        if isinstance(observations, (tuple, list)):
            for observation in observations:
                if not isinstance(observation, Mapping):
                    malformed_count += 1
                    continue
                normalized = _normalize_observation(observation)
                if normalized is None:
                    malformed_count += 1
                    continue
                valid_observations.append(normalized)
        else:
            malformed_count += 1
    sanitized["observations"] = tuple(valid_observations)

    valid_inferences: list[dict[str, Any]] = []
    if "model_inferences" in upstream:
        model_inferences = upstream["model_inferences"]
        if isinstance(model_inferences, (tuple, list)):
            for inference in model_inferences:
                if not isinstance(inference, Mapping):
                    malformed_count += 1
                    continue
                claim = _required_text(inference.get("claim"), 80)
                memory_type = _required_text(inference.get("memory_type"), 80)
                if (
                    "claim" in inference
                    and "memory_type" in inference
                    and claim
                    and memory_type
                    and claim != memory_type
                ):
                    malformed_count += 1
                    continue
                if not claim and not memory_type:
                    malformed_count += 1
                    continue
                if "claim" in inference and not claim:
                    malformed_count += 1
                    continue
                if "memory_type" in inference and not memory_type:
                    malformed_count += 1
                    continue
                if claim == "difficulty" or memory_type == "difficulty":
                    topic = _required_text(inference.get("topic"), 160)
                    if not topic:
                        malformed_count += 1
                        continue
                    valid_inferences.append(
                        {"claim": claim, "memory_type": memory_type, "topic": topic}
                    )
                    continue
                valid_inferences.append(dict(inference))
        else:
            malformed_count += 1
    sanitized["model_inferences"] = tuple(valid_inferences)
    return sanitized, malformed_count


class PredicateOwner(str, Enum):
    SELF = "self"
    THIRD_PERSON = "third_person"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FamilyPredicate:
    text: str
    start: int
    end: int
    owner: PredicateOwner


def _owned_family_predicates(
    text: str,
    pattern: re.Pattern[str],
    *,
    implicit_prefix_re: re.Pattern[str],
) -> tuple[FamilyPredicate, ...]:
    return tuple(
        FamilyPredicate(
            text=match.group(0),
            start=match.start(),
            end=match.end(),
            owner=_resolve_predicate_owner(
                text, match.start(), implicit_prefix_re=implicit_prefix_re
            ),
        )
        for match in pattern.finditer(text)
    )


def _resolve_predicate_owner(
    text: str,
    predicate_start: int,
    *,
    implicit_prefix_re: re.Pattern[str],
) -> PredicateOwner:
    prefix = text[:predicate_start]
    if EXPLICIT_SELF_OWNER_RE.search(prefix):
        return PredicateOwner.SELF
    if implicit_prefix_re.fullmatch(prefix):
        return PredicateOwner.SELF
    if THIRD_PERSON_OWNER_RE.search(prefix):
        return PredicateOwner.THIRD_PERSON
    return PredicateOwner.UNKNOWN


def _is_current_instructional_request(text: str) -> bool:
    if CURRENT_PROBLEM_INSTRUCTION_RE.search(text):
        return True
    requester = TEACHER_REQUEST_PREFIX_RE.search(text)
    if requester is None:
        return False
    action = INSTRUCTIONAL_ACTION_RE.search(text, requester.end())
    if action is None:
        return False
    return INSTRUCTIONAL_OBJECT_RE.search(text, action.end()) is not None


def _request_recipient_owners(text: str) -> tuple[PredicateOwner, ...]:
    return tuple(
        PredicateOwner.SELF
        if match.group("recipient") in {"我", "本人"}
        else PredicateOwner.THIRD_PERSON
        for match in REQUEST_RECIPIENT_RE.finditer(text)
    )


def _is_bounded_preference_request(analysis: QualifierAnalysis) -> bool:
    if not (analysis.condition or analysis.temporal_expression):
        return False
    text = analysis.text
    if PredicateOwner.THIRD_PERSON in _request_recipient_owners(text):
        return False
    if text.startswith("如果"):
        action = REQUEST_ACTION_RE.search(text, len("如果"))
        if action is None:
            return False
        condition = text[len("如果") : action.start()].strip()
        condition_triggers = _owned_family_predicates(
            condition,
            CONDITION_TRIGGER_RE,
            implicit_prefix_re=CONDITION_IMPLICIT_PREFIX_RE,
        )
        return bool(condition_triggers) and all(
            trigger.owner is PredicateOwner.SELF for trigger in condition_triggers
        )
    if analysis.condition:
        prefix = f"只在{analysis.condition}"
        if text.startswith(prefix):
            return REQUEST_ACTION_RE.match(text[len(prefix) :].lstrip()) is not None
    for expression in analysis.temporal_expression.split("；"):
        if expression and text.startswith(expression):
            return REQUEST_ACTION_RE.match(text[len(expression) :].lstrip()) is not None
    return False


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
        and _is_current_instructional_request(analysis.text)
    ):
        return None
    stances = _owned_family_predicates(
        analysis.text,
        PREFERENCE_PREDICATE_RE,
        implicit_prefix_re=PREFERENCE_IMPLICIT_PREFIX_RE,
    )
    has_self_stance = any(stance.owner is PredicateOwner.SELF for stance in stances)
    has_foreign_or_ambiguous_stance = any(
        stance.owner is not PredicateOwner.SELF for stance in stances
    )
    self_statement = has_self_stance and not has_foreign_or_ambiguous_stance
    bounded_request = (
        _is_bounded_preference_request(analysis) and not has_foreign_or_ambiguous_stance
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


def _verification_candidates(
    upstream: Mapping[str, Any], *, conversation_id: str, turn_id: int
) -> list[MemoryCandidate]:
    verification = dict(upstream.get("verification") or {})
    status = str(verification.get("status") or "").strip()
    if status not in {"verified_correct", "verified_incorrect"}:
        return []
    verifier_reference = str(verification.get("verifier_name") or "")[:160]
    topic_progress = [
        observation
        for observation in tuple(upstream.get("observations") or ())
        if "topic" in observation and "status" in observation
    ]
    if topic_progress:
        return [
            _make_candidate(
                conversation_id=conversation_id,
                turn_id=turn_id,
                memory_type=MemoryType.TOPIC_PROGRESS,
                payload={
                    "subject": _subject(upstream),
                    "topic": observation["topic"],
                    "status": observation["status"],
                },
                source_kind=MemorySourceKind.VERIFIED_RESULT,
                confidence=0.95,
                temporal_scope=TemporalScope.SHORT_LIVED,
                verifier_reference=verifier_reference,
            )
            for observation in topic_progress
        ]
    return [
        _make_candidate(
            conversation_id=conversation_id,
            turn_id=turn_id,
            memory_type=MemoryType.BEHAVIOR_EVIDENCE,
            payload={
                "subject": _subject(upstream),
                "verification_status": status,
                "summary": str(verification.get("summary") or "")[:180],
            },
            source_kind=MemorySourceKind.VERIFIED_RESULT,
            confidence=0.95,
            temporal_scope=TemporalScope.SHORT_LIVED,
            verifier_reference=verifier_reference,
        )
    ]


def _observation_candidates(
    upstream: Mapping[str, Any], *, conversation_id: str, turn_id: int
) -> list[MemoryCandidate]:
    verification = upstream.get("verification")
    has_verification = isinstance(verification, Mapping) and bool(verification)
    candidates: list[MemoryCandidate] = []
    for observation in tuple(upstream.get("observations") or ()):
        if "topic" in observation and "status" in observation:
            if has_verification:
                continue
            memory_type = MemoryType.TOPIC_PROGRESS
            payload = {
                "subject": _subject(upstream),
                "topic": observation["topic"],
                "status": observation["status"],
            }
        elif "episode" in observation:
            memory_type = MemoryType.LEARNING_EPISODE
            payload = {
                "subject": _subject(upstream),
                "episode": observation["episode"],
            }
        else:
            memory_type = MemoryType.BEHAVIOR_EVIDENCE
            pattern = observation.get("pattern") or observation.get("evidence_pattern")
            payload = {
                "subject": _subject(upstream),
                "evidence_pattern": pattern,
            }
        candidates.append(
            _make_candidate(
                conversation_id=conversation_id,
                turn_id=turn_id,
                memory_type=memory_type,
                payload=payload,
                source_kind=MemorySourceKind.OBSERVED_BEHAVIOR,
                confidence=0.75,
                temporal_scope=TemporalScope.SHORT_LIVED,
            )
        )
    return candidates


def _model_inference_candidates(
    upstream: Mapping[str, Any], *, conversation_id: str, turn_id: int
) -> list[MemoryCandidate]:
    return [
        _make_candidate(
            conversation_id=conversation_id,
            turn_id=turn_id,
            memory_type=MemoryType.DIFFICULTY,
            payload={
                "subject": _subject(upstream),
                "topic": inference["topic"],
                "uncertainty": "model_inference",
            },
            source_kind=MemorySourceKind.MODEL_INFERENCE,
            confidence=0.4,
            temporal_scope=TemporalScope.UNKNOWN,
        )
        for inference in tuple(upstream.get("model_inferences") or ())
        if inference.get("claim") == "difficulty"
        or inference.get("memory_type") == "difficulty"
    ]


def _merge_candidates(
    candidates: Iterable[MemoryCandidate],
) -> list[MemoryCandidate]:
    unique: dict[tuple[str, str, str], MemoryCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.memory_type.value,
            candidate.source_kind.value,
            json.dumps(
                candidate.payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        unique.setdefault(key, candidate)
    return list(unique.values())


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
    result_predicates = _owned_family_predicates(
        body,
        EPISODE_RESULT_PREDICATE_RE,
        implicit_prefix_re=EPISODE_IMPLICIT_PREFIX_RE,
    )
    if not result_predicates:
        return None
    decisive_result = result_predicates[-1]
    if decisive_result.owner is not PredicateOwner.SELF:
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
    noncomprehension = _owned_family_predicates(
        analysis.text,
        DIFFICULTY_PREDICATE_RE,
        implicit_prefix_re=DIFFICULTY_IMPLICIT_PREFIX_RE,
    )
    for predicate in noncomprehension:
        if predicate.owner is not PredicateOwner.SELF:
            continue
        topic = analysis.text[predicate.end :].strip(" 了,，。")[:80]
        if topic:
            return (
                MemoryType.DIFFICULTY,
                {
                    "subject": subject,
                    "topic": topic,
                    "statement": f"{predicate.text}{topic}",
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
        f"{conversation_id}\n{turn_id}\n{memory_type.value}\n{source_kind.value}\n{canonical}".encode(
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
