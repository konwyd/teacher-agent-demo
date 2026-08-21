from __future__ import annotations

import itertools
import json
import re
import time
import unicodedata
from collections.abc import Callable, Iterable
from typing import Any

from app.memory.admission_normalization import CandidateNormalizer
from app.memory.contracts import MemoryCandidate

from .models import BenchmarkCase, BenchmarkReport, OperationalMetrics, RateMetric

Extractor = Callable[[BenchmarkCase], list[MemoryCandidate]]
QUALIFIERS = ("negated", "condition", "scope", "polarity", "temporal_expression", "uncertainty", "explicit_correction", "correction_of")
PERSISTENCE = {"turn_only": 0, "session": 1, "short_lived": 2, "persistent": 3, "unknown": 0}


def _canon(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip()).casefold()
    if isinstance(value, dict):
        return {key: _canon(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return sorted((_canon(item) for item in value), key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    return value


def _summary(candidate: Any) -> dict[str, Any]:
    def val(item: Any) -> Any:
        return item.value if hasattr(item, "value") else item
    if isinstance(candidate, dict):
        raw = candidate
    elif hasattr(candidate, "model_dump"):
        raw = candidate.model_dump(mode="json")
    else:
        raw = vars(candidate)
    return _canon({
        "target_memory_class": val(raw.get("target_memory_class", "student")),
        "memory_type": val(raw["memory_type"]), "payload": raw["payload"],
        "source_kind": val(raw["source_kind"]), "temporal_scope": val(raw["temporal_scope"]),
    })


def _rate(numerator: int | float, support: int) -> RateMetric:
    return RateMetric(value=numerator / support if support else None, support=support, numerator=numerator)


def _score(expected: dict[str, Any], actual: dict[str, Any]) -> int:
    return sum(expected[key] == actual[key] for key in ("memory_type", "source_kind", "temporal_scope", "payload"))


def _pairs(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not expected or not actual:
        return []
    if len(expected) <= len(actual):
        choices = (list(zip(expected, perm)) for perm in itertools.permutations(actual, len(expected)))
    else:
        choices = (list(zip(perm, actual)) for perm in itertools.permutations(expected, len(actual)))
    return max(choices, key=lambda rows: (sum(_score(e, a) for e, a in rows), json.dumps(rows, ensure_ascii=False, sort_keys=True)))


def evaluate_cases(cases: Iterable[BenchmarkCase], extractor: Extractor, *, split: str, operational: OperationalMetrics | None = None) -> BenchmarkReport:
    rows = []
    tp = fp = fn = exact = field_support = 0
    field_hits = {key: 0 for key in ("memory_type", "source_kind", "temporal_scope", "payload")}
    qualifier_support = qualifier_hits = hard_support = false_persistent = 0
    neg_support = neg_loss = temporal_support = temporal_inflation = condition_support = condition_loss = correction_support = correction_loss = 0
    normalization_support = normalization_hits = expected_total = actual_total = gap_count = 0
    latencies = []
    for case in cases:
        start = time.perf_counter()
        raw_actual = extractor(case)
        latencies.append((time.perf_counter() - start) * 1000)
        expected = [_summary(item) for item in case.expected_candidates]
        actual = [_summary(item) for item in raw_actual]
        expected_total += len(expected); actual_total += len(actual); gap_count += len(case.representation_gaps)
        tp += min(len(expected), len(actual)); fp += max(0, len(actual)-len(expected)); fn += max(0, len(expected)-len(actual))
        is_exact = sorted(expected, key=str) == sorted(actual, key=str); exact += is_exact
        paired = _pairs(expected, actual); field_support += len(paired)
        for e, a in paired:
            for key in field_hits: field_hits[key] += e[key] == a[key]
            expected_q = {key: e["payload"].get(key) for key in QUALIFIERS if key in e["payload"]}
            if expected_q:
                qualifier_support += 1; qualifier_hits += all(a["payload"].get(key) == value for key, value in expected_q.items())
            if e["payload"].get("negated") is True:
                neg_support += 1; neg_loss += a["payload"].get("negated") is not True
            if "condition" in e["payload"]:
                condition_support += 1; condition_loss += a["payload"].get("condition") != e["payload"]["condition"]
            if e["payload"].get("explicit_correction") or "correction_of" in e["payload"]:
                correction_support += 1; correction_loss += not all(a["payload"].get(k) == e["payload"].get(k) for k in ("explicit_correction", "correction_of") if k in e["payload"])
            temporal_support += 1; temporal_inflation += PERSISTENCE.get(a["temporal_scope"], 0) > PERSISTENCE.get(e["temporal_scope"], 0)
        if "HARD_NEGATIVE" in case.labels:
            hard_support += 1; false_persistent += any(a["temporal_scope"] == "persistent" for a in actual)
        for item in raw_actual:
            normalization_support += 1
            try:
                normalized = CandidateNormalizer().normalize(item)
                normalization_hits += bool(normalized.semantic_key)
            except Exception:
                pass
        rows.append({"case_id": case.case_id, "labels": list(case.labels), "exact": is_exact, "expected": expected, "actual": actual})
    precision = tp/(tp+fp) if tp+fp else 1.0; recall = tp/(tp+fn) if tp+fn else 1.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    case_count = len(rows)
    op = operational or OperationalMetrics()
    if latencies:
        ordered = sorted(latencies); op.latency_p50_ms = ordered[len(ordered)//2]; op.latency_p95_ms = ordered[min(len(ordered)-1, int(len(ordered)*.95))]
    return BenchmarkReport(split=split, case_count=case_count, expected_candidate_count=expected_total, actual_candidate_count=actual_total,
        detection_precision=_rate(precision, 1), detection_recall=_rate(recall, 1), detection_f1=_rate(f1, 1), exact_candidate_set_accuracy=_rate(exact, case_count),
        memory_type_accuracy=_rate(field_hits["memory_type"], field_support), source_kind_accuracy=_rate(field_hits["source_kind"], field_support), temporal_scope_accuracy=_rate(field_hits["temporal_scope"], field_support), payload_accuracy=_rate(field_hits["payload"], field_support),
        qualifier_preservation=_rate(qualifier_hits, qualifier_support), false_persistent_hard_negative_rate=_rate(false_persistent, hard_support), negation_loss_rate=_rate(neg_loss, neg_support), temporal_inflation_rate=_rate(temporal_inflation, temporal_support), conditional_corruption_rate=_rate(condition_loss, condition_support), correction_loss_rate=_rate(correction_loss, correction_support), normalization_compatibility=_rate(normalization_hits, normalization_support), representation_gap_count=gap_count, operational=op, case_results=rows)
