"""Read-only benchmark adapter for CandidateExtractorV2 Route B."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

from app.evaluation.candidate_extractor.baseline import DATA, load_cases, load_layer_b
from app.evaluation.candidate_extractor.evaluator import evaluate_cases
from app.evaluation.candidate_extractor.models import BenchmarkCase, OperationalMetrics

from .extractor import CandidateExtractorRouteB


ALLOWED_SPLITS = ("development", "layer_b")
_DEPENDENCIES = ("pydantic", "sentence-transformers", "transformers", "torch")


def _rate(rows: Sequence[dict[str, Any]]) -> dict[str, int | float | None]:
    support = len(rows)
    numerator = sum(bool(row["exact"]) for row in rows)
    return {
        "value": numerator / support if support else None,
        "support": support,
        "numerator": numerator,
    }


def _slice_metrics(case_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({label for row in case_results for label in row["labels"]})
    by_label = {
        label: _rate([row for row in case_results if label in row["labels"]])
        for label in labels
    }
    multi_rows = [row for row in case_results if "MULTI_CANDIDATE" in row["labels"]]
    return {
        "multi_candidate_exact_match": _rate(multi_rows),
        "exact_match_by_label": by_label,
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _implementation_line_count() -> int:
    repository = Path(__file__).resolve().parents[3]
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths.append(repository / "scripts" / "run_candidate_extractor_route_b.py")
    return sum(
        1
        for path in paths
        if path.is_file()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _peak_ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _selected_cuda_device(extractor: CandidateExtractorRouteB) -> str | None:
    backend = getattr(extractor.ranker, "_backend", None)
    device = str(getattr(backend, "device", ""))
    return device if device.startswith("cuda") else None


def _reset_cuda_peak(device: str | None) -> None:
    if device is None:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram_mb(device: str | None) -> float | None:
    if device is None:
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def _counter(extractor: CandidateExtractorRouteB, name: str) -> int | None:
    value = getattr(extractor.ranker, name, None)
    return value if isinstance(value, int) else None


def _counter_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _load_split(split: str) -> list[BenchmarkCase]:
    if split == "development":
        return load_cases(DATA / "development.jsonl")
    return load_layer_b()


def run_route_b_benchmark(
    extractor: CandidateExtractorRouteB,
    splits: tuple[str, ...] = ALLOWED_SPLITS,
    *,
    cases_by_split: Mapping[str, Sequence[BenchmarkCase]] | None = None,
    provisional: bool = True,
) -> dict[str, Any]:
    """Evaluate Route B without changing the frozen evaluator or its metrics."""

    if not splits or any(split not in ALLOWED_SPLITS for split in splits):
        raise ValueError("Route B runner supports development and layer_b only")
    if len(set(splits)) != len(splits):
        raise ValueError("Route B runner splits must be unique")

    selected_cases: dict[str, tuple[BenchmarkCase, ...]] = {}
    for split in splits:
        if cases_by_split is None:
            cases = tuple(_load_split(split))
        elif split not in cases_by_split:
            raise ValueError(f"Route B cases missing for split: {split}")
        else:
            cases = tuple(cases_by_split[split])
        if any(case.split != split for case in cases):
            raise ValueError("Route B case split must match selected split")
        selected_cases[split] = cases

    dependency_footprint = [
        "python-stdlib",
        "pydantic",
        "sentence-transformers",
        "transformers",
        "torch",
        "bge-m3-local",
    ]
    device = _selected_cuda_device(extractor)
    _reset_cuda_peak(device)
    encode_calls_before = _counter(extractor, "encode_call_count")
    encoded_texts_before = _counter(extractor, "encoded_text_count")
    malformed_before = extractor.diagnostics.malformed_upstream_items
    counter_supported = encode_calls_before is not None
    embedding_warmed = bool(encode_calls_before) if counter_supported else False
    fallback_seen = False
    cold_load_inclusive_extract_ms: float | None = None
    cold_load_case_id: str | None = None
    warm_latencies_ms: list[float] = []

    def extract_case(case: BenchmarkCase):
        nonlocal embedding_warmed, fallback_seen
        nonlocal cold_load_inclusive_extract_ms, cold_load_case_id
        encode_calls_at_start = _counter(extractor, "encode_call_count")
        started = time.perf_counter()
        context = case.frozen_upstream_context.model_dump(mode="python")
        candidates = extractor.extract(
            case.utterance,
            context,
            conversation_id=f"candidate-route-b-{case.case_id}",
            turn_id=1,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        encode_calls_at_end = _counter(extractor, "encode_call_count")
        if counter_supported:
            consumed_embedding = (
                encode_calls_at_start is not None
                and encode_calls_at_end is not None
                and encode_calls_at_end > encode_calls_at_start
            )
            if not embedding_warmed and consumed_embedding:
                cold_load_inclusive_extract_ms = elapsed_ms
                cold_load_case_id = case.case_id
                embedding_warmed = True
            elif embedding_warmed:
                warm_latencies_ms.append(elapsed_ms)
        elif not fallback_seen:
            cold_load_inclusive_extract_ms = elapsed_ms
            cold_load_case_id = case.case_id
            fallback_seen = True
        else:
            warm_latencies_ms.append(elapsed_ms)
        return candidates

    reports = {}
    started_all = time.perf_counter()
    for split in splits:
        started_split = time.perf_counter()
        operational = OperationalMetrics(
            additional_model_calls=0,
            dependency_footprint=list(dependency_footprint),
            offline_capability=True,
        )
        report = evaluate_cases(
            selected_cases[split], extract_case, split=split, operational=operational
        )
        elapsed = time.perf_counter() - started_split
        report.operational.throughput_cases_per_second = (
            len(selected_cases[split]) / elapsed if elapsed else None
        )
        report.operational.peak_ram_mb = _peak_ram_mb()
        reports[split] = report
    primary_elapsed = time.perf_counter() - started_all
    encode_calls_after = _counter(extractor, "encode_call_count")
    encoded_texts_after = _counter(extractor, "encoded_text_count")
    malformed_after = extractor.diagnostics.malformed_upstream_items
    replay_encode_calls_before = encode_calls_after
    replay_encoded_texts_before = encoded_texts_after

    replay_equal: dict[str, bool] = {}
    for split in splits:

        def replay_extract(case: BenchmarkCase):
            context = case.frozen_upstream_context.model_dump(mode="python")
            return extractor.extract(
                case.utterance,
                context,
                conversation_id=f"candidate-route-b-{case.case_id}",
                turn_id=1,
            )

        replay = evaluate_cases(selected_cases[split], replay_extract, split=split)
        replay_equal[split] = json.dumps(
            reports[split].case_results,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) == json.dumps(
            replay.case_results,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        reports[split].operational.deterministic_replay = replay_equal[split]

    replay_encode_calls_after = _counter(extractor, "encode_call_count")
    replay_encoded_texts_after = _counter(extractor, "encoded_text_count")
    warm = sorted(warm_latencies_ms)
    warm_p50 = warm[len(warm) // 2] if warm else None
    warm_p95 = warm[min(len(warm) - 1, int(len(warm) * 0.95))] if warm else None
    total_cases = sum(len(selected_cases[split]) for split in splits)
    dumped_reports = {split: reports[split].model_dump(mode="json") for split in splits}
    return {
        "benchmark_version": "candidate-extractor-v2-comparative-v1",
        "extractor": "candidate_extractor_route_b",
        "reports": dumped_reports,
        "slice_metrics": {
            split: _slice_metrics(dumped_reports[split]["case_results"])
            for split in splits
        },
        "operational": {
            "provisional": provisional,
            "offline_capability": True,
            "remote_llm_calls": 0,
            "remote_api_calls": 0,
            "cold_load_inclusive_extract_ms": cold_load_inclusive_extract_ms,
            "cold_load_case_id": cold_load_case_id,
            "warm_latency_p50_ms": warm_p50,
            "warm_latency_p95_ms": warm_p95,
            "wall_time_seconds": primary_elapsed,
            "throughput_cases_per_second": (
                total_cases / primary_elapsed if primary_elapsed else None
            ),
            "peak_ram_mb": _peak_ram_mb(),
            "peak_vram_mb": _peak_vram_mb(device),
            "encode_calls": _counter_delta(
                encode_calls_before, replay_encode_calls_after
            ),
            "encoded_text_count": _counter_delta(
                encoded_texts_before, replay_encoded_texts_after
            ),
            "primary_encode_calls": _counter_delta(
                encode_calls_before, encode_calls_after
            ),
            "primary_encoded_text_count": _counter_delta(
                encoded_texts_before, encoded_texts_after
            ),
            "replay_encode_calls": _counter_delta(
                replay_encode_calls_before, replay_encode_calls_after
            ),
            "replay_encoded_text_count": _counter_delta(
                replay_encoded_texts_before, replay_encoded_texts_after
            ),
            "malformed_upstream_items": malformed_after - malformed_before,
            "dependency_versions": _dependency_versions(),
            "deterministic_replay": all(replay_equal.values()),
            "implementation_line_count": _implementation_line_count(),
            "selected_splits": list(splits),
        },
    }
