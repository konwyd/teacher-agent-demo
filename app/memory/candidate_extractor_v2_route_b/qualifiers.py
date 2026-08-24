"""Deterministic qualifier analysis for isolated CandidateExtractorV2 Route B."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from app.memory.contracts import TemporalScope


CONJUNCTION_SPLIT_RE = re.compile(
    r"(?:[，,；;。]\s*(?:并且|而且|然后)?|并且|而且|然后)+"
)
THIRD_PERSON_PATTERNS = ("书中的人物", "课文中的人物", "故事中的人物")
THIRD_PERSON_CONTEXT_RE = re.compile(
    r"(?<![\w\u4e00-\u9fff])(?:他们|她们|它们|他|她|它)"
)
KINSHIP_REFERENT_RE = re.compile(
    r"^我(?:的)?(?:哥哥|姐姐|弟弟|妹妹|兄弟|姐妹|爸爸|妈妈|父亲|母亲|"
    r"爷爷|奶奶|外公|外婆|朋友|同学|老师)"
)
EXPLICIT_THIRD_PERSON_OWNER_RE = re.compile(
    r"^(?:(?:小|阿)[\u4e00-\u9fff]{1,3}|"
    r"(?:这位|那位|那个|一名|一位)?(?:学生|同学|老师|朋友|孩子|男生|女生)|"
    r"[\u4e00-\u9fff]{1,3}(?:同学|老师|同桌))"
)
META_QUOTED_RE = re.compile(r"[‘'\"“].+?[’'\"”].*(?:这个词|含义|是什么意思|定义)")
PRIVACY_PROHIBITION_RE = re.compile(r"(?:不要|别|请勿).{0,8}(?:保存|记住|记录|存储)")
INSTRUCTIONAL_EXAMPLE_PATTERNS = ("举例说明", "例题中的", "假设学生")
NEGATION_MARKERS = ("不是", "不", "没", "没有", "勿", "别")
CORRECTION_MARKERS = ("改一下", "更正", "纠正", "我其实", "其实")
HEDGE_MARKERS = ("可能", "也许", "大概", "或许", "不一定", "应该")
SCOPE_MARKERS = (
    ("化学", "chemistry"),
    ("数学", "mathematics"),
    ("物理", "physics"),
    ("语文", "chinese"),
    ("英语", "english"),
)
CONDITION_PATTERNS = (
    re.compile(r"只在(?P<condition>[^，。；;]+?课)"),
    re.compile(r"如果(?P<condition>[^，。；;]+)(?:[，。；;]|$)"),
)
TEMPORAL_RULES = (
    (TemporalScope.TURN_ONLY, ("本次", "这次", "本轮", "这一轮", "之后恢复默认")),
    (TemporalScope.SESSION, ("这节课", "本节课", "今天这次", "当前会话")),
    (TemporalScope.SHORT_LIVED, ("暂时", "这一会儿", "这几天", "一段时间", "最近")),
    (TemporalScope.PERSISTENT, ("从现在起", "以后都", "今后", "长期", "一直")),
)


@dataclass(frozen=True)
class QualifierAnalysis:
    text: str
    hard_negative_reason: str = ""
    negated: bool | None = None
    polarity: str = ""
    condition: str = ""
    scope: str = ""
    temporal_expression: str = ""
    temporal_hint: TemporalScope | None = None
    uncertainty: str = ""
    explicit_correction: bool = False
    correction_of: str = ""


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).strip().split())


def segment_clauses(text: str) -> tuple[str, ...]:
    """Split only bounded conjunctions, retaining non-empty normalized clauses."""
    normalized = _normalize_text(text)
    if _is_bound_qualifier_clause(normalized):
        return (normalized,)
    return tuple(
        part.strip() for part in CONJUNCTION_SPLIT_RE.split(normalized) if part.strip()
    )


def _is_bound_qualifier_clause(text: str) -> bool:
    return bool(
        re.match(r"(?:改一下|更正|纠正)[，,]\s*(?:我其实|其实)", text)
        or re.match(r"(?:本次|这次|本轮|这一轮)[^，,]*[，,]\s*之后恢复默认$", text)
    )


def _hard_negative_reason(text: str) -> str:
    if PRIVACY_PROHIBITION_RE.search(text):
        return "privacy_prohibition"
    if META_QUOTED_RE.search(text):
        return "meta_quoted"
    if (
        any(marker in text for marker in THIRD_PERSON_PATTERNS)
        or THIRD_PERSON_CONTEXT_RE.search(text)
        or _has_third_person_owner(text)
    ):
        return "third_person"
    if any(marker in text for marker in INSTRUCTIONAL_EXAMPLE_PATTERNS):
        return "instructional_example"
    return ""


def _has_third_person_owner(text: str) -> bool:
    return bool(
        KINSHIP_REFERENT_RE.match(text) or EXPLICIT_THIRD_PERSON_OWNER_RE.match(text)
    )


def _negation_parity(text: str) -> tuple[bool | None, str]:
    count = 0
    index = 0
    while index < len(text):
        marker = next(
            (item for item in NEGATION_MARKERS if text.startswith(item, index)), ""
        )
        if marker:
            count += 1
            index += len(marker)
        else:
            index += 1
    if not count:
        return None, ""
    negated = bool(count % 2)
    return negated, "negative" if negated else "positive"


def _condition_and_scope(text: str) -> tuple[str, str]:
    condition = ""
    for pattern in CONDITION_PATTERNS:
        match = pattern.search(text)
        if match:
            condition = match.group("condition").strip()
            break
    scope = next(
        (scope for marker, scope in SCOPE_MARKERS if marker in (condition or text)), ""
    )
    return condition, scope


def _temporal_analysis(text: str, uncertainty: str) -> tuple[str, TemporalScope | None]:
    expressions: list[str] = []
    temporal_hint: TemporalScope | None = None
    for scope, markers in TEMPORAL_RULES:
        found = [marker for marker in markers if marker in text]
        if found:
            expressions.extend(found)
            temporal_hint = scope
            break
    if "以后" in text and "以后都" not in text:
        expressions.append("以后")
    return "；".join(expressions), temporal_hint


def analyze_clause(text: str) -> QualifierAnalysis:
    """Return clause-local qualifiers without semantic-family or similarity input."""
    normalized = _normalize_text(text)
    hard_negative_reason = _hard_negative_reason(normalized)
    if hard_negative_reason:
        return QualifierAnalysis(
            text=normalized, hard_negative_reason=hard_negative_reason
        )

    negated, polarity = _negation_parity(normalized)
    condition, scope = _condition_and_scope(normalized)
    uncertainty = (
        "hedged" if any(marker in normalized for marker in HEDGE_MARKERS) else ""
    )
    temporal_expression, temporal_hint = _temporal_analysis(normalized, uncertainty)
    explicit_correction = any(marker in normalized for marker in CORRECTION_MARKERS)
    return QualifierAnalysis(
        text=normalized,
        negated=negated,
        polarity=polarity,
        condition=condition,
        scope=scope,
        temporal_expression=temporal_expression,
        temporal_hint=temporal_hint,
        uncertainty=uncertainty,
        explicit_correction=explicit_correction,
        correction_of="prior claim" if explicit_correction else "",
    )


def resolve_temporal_scope(
    analysis: QualifierAnalysis, default: TemporalScope
) -> TemporalScope:
    if analysis.temporal_hint is not None:
        return analysis.temporal_hint
    if analysis.uncertainty:
        return TemporalScope.UNKNOWN
    return default
