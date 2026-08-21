#!/usr/bin/env python3
"""Run deterministic CandidateExtractorV2 Route A without loading holdout."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.candidate_extractor.baseline import (  # noqa: E402
    DATA,
    load_cases,
    load_layer_b,
)
from app.evaluation.candidate_extractor.evaluator import evaluate_cases  # noqa: E402
from app.evaluation.candidate_extractor.models import (  # noqa: E402
    BenchmarkCase,
    OperationalMetrics,
)
from app.memory.candidate_extractor_v2_deterministic import (  # noqa: E402
    build_deterministic_memory_candidates,
)


ALLOWED_SPLITS = ("development", "layer_b")


def route_a_extractor(case: BenchmarkCase):
    context = case.frozen_upstream_context
    return build_deterministic_memory_candidates(
        {
            "turn_id": 1,
            "student_message": case.utterance,
            "classification": context.classification,
            "teaching_contract": context.teaching_contract,
            "verification": context.verification,
            "observations": context.observations,
            "model_inferences": context.model_inferences,
        },
        conversation_id=f"candidate-route-a-{case.case_id}",
    )


def _peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load_split(split: str) -> list[BenchmarkCase]:
    if split == "development":
        return load_cases(DATA / "development.jsonl")
    if split == "layer_b":
        return load_layer_b()
    raise ValueError(f"unsupported Route A split: {split}")


def run_benchmark(splits: Iterable[str] = ALLOWED_SPLITS) -> dict[str, Any]:
    selected = tuple(splits)
    invalid = [split for split in selected if split not in ALLOWED_SPLITS]
    if invalid:
        raise ValueError(f"unsupported Route A split: {invalid[0]}")

    reports: dict[str, Any] = {}
    for split in selected:
        cases = _load_split(split)
        operational = OperationalMetrics(
            additional_model_calls=0,
            dependency_footprint=["python-stdlib", "pydantic"],
            offline_capability=True,
            peak_vram_mb=0.0,
        )
        started = time.perf_counter()
        report = evaluate_cases(
            cases,
            route_a_extractor,
            split=split,
            operational=operational,
        )
        elapsed = time.perf_counter() - started
        replay = evaluate_cases(cases, route_a_extractor, split=split)
        report.operational.throughput_cases_per_second = (
            len(cases) / elapsed if elapsed else None
        )
        report.operational.peak_ram_mb = _peak_rss_mb()
        report.operational.deterministic_replay = json.dumps(
            report.case_results, ensure_ascii=False, sort_keys=True
        ) == json.dumps(replay.case_results, ensure_ascii=False, sort_keys=True)
        reports[split] = report.model_dump(mode="json")

    return {
        "benchmark_version": "candidate-extractor-v2-comparative-v1",
        "route": "A",
        "extractor": "candidate_extractor_v2_deterministic",
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("development", "layer_b", "all"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    splits = ALLOWED_SPLITS if args.split == "all" else (args.split,)
    result = run_benchmark(splits)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    for split, report in result["reports"].items():
        exact = report["exact_candidate_set_accuracy"]["value"]
        print(f"{split} {report['case_count']} exact_set={exact}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
