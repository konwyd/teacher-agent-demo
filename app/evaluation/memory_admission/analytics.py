"""Runtime-style reducers and bounded human-review export."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from app.evaluation.memory_admission.models import EvaluationRecord


def reduce_shadow_analytics(records: Iterable[EvaluationRecord]) -> dict:
    rows = list(records)
    return {
        "candidate_count": len(rows),
        "by_decision": dict(Counter(row.predicted_action.value for row in rows)),
        "by_memory_type": dict(Counter(row.candidate["memory_type"] for row in rows)),
        "by_source": dict(Counter(row.candidate["source_kind"] for row in rows)),
        "by_scope": dict(Counter(row.candidate["temporal_scope"] for row in rows)),
        "divergence_count": sum(row.divergence for row in rows),
        "controller_error_count": sum(bool(row.controller_error) for row in rows),
        "latency_seconds": {
            "max": max((row.latency_seconds for row in rows), default=0.0),
            "mean": sum(row.latency_seconds for row in rows) / len(rows) if rows else 0.0,
        },
    }


def export_shadow_review(
    records: Iterable[EvaluationRecord], path: Path, *, max_records: int = 100
) -> int:
    allowed = []
    for row in list(records)[:max(0, max_records)]:
        allowed.append({
            "case_id": row.case_id,
            "candidate": row.candidate,
            "provenance": row.provenance,
            "features": row.features,
            "relation": row.predicted_relation.value if row.predicted_relation else None,
            "evidence": row.evidence,
            "decision": row.predicted_action.value,
            "reasons": row.reasons,
            "baseline_action": row.baseline_action.value,
            "divergence": row.divergence,
            "controller_error": row.controller_error,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in allowed))
    return len(allowed)
