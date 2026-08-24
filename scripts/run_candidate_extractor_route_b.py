#!/usr/bin/env python3
"""Run the offline development/Layer B benchmark for Route B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory.candidate_extractor_v2_route_b import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    BgeM3Embedder,
    CandidateExtractorRouteB,
    SemanticFamilyRanker,
)
from app.memory.candidate_extractor_v2_route_b.benchmark import (  # noqa: E402
    ALLOWED_SPLITS,
    run_route_b_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALLOWED_SPLITS,
        default=list(ALLOWED_SPLITS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/candidate_extractor_route_b.json"),
    )
    parser.add_argument(
        "--provisional",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    embedder = BgeM3Embedder(args.model_path, device=args.device)
    ranker = SemanticFamilyRanker(embedder)
    extractor = CandidateExtractorRouteB(ranker)
    report = run_route_b_benchmark(
        extractor,
        splits=tuple(args.splits),
        provisional=args.provisional,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for split, result in report["reports"].items():
        print(
            split, result["case_count"], result["exact_candidate_set_accuracy"]["value"]
        )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
