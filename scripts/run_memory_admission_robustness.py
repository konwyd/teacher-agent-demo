#!/usr/bin/env python3
"""Freeze or run the Memory v2 Phase 2A.5 robustness evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.memory_admission.robustness.manifest import capture_manifest
from app.evaluation.memory_admission.robustness.report import (
    render_markdown,
    run_robustness_evaluation,
)
from app.memory.admission_config import AdmissionPolicyConfig


ROBUSTNESS_ROOT = Path("app/evaluation/memory_admission/robustness")
MANIFEST = ROBUSTNESS_ROOT / "policy_v1_manifest.json"
RELEVANT_POLICY_PATHS = (
    Path("app/memory/admission_policy.py"),
    Path("app/memory/admission_config.py"),
    Path("app/memory/admission_matcher.py"),
    Path("app/memory/admission_normalization.py"),
    Path("app/memory/admission_features.py"),
    Path("app/memory/admission_controller.py"),
    Path("app/memory/admission_ledger.py"),
    Path("app/memory/contracts.py"),
)
ROBUSTNESS_DATASETS = (
    ROBUSTNESS_ROOT / "data/layer_a_core.jsonl",
    ROBUSTNESS_ROOT / "data/layer_a_evidence.jsonl",
    ROBUSTNESS_ROOT / "data/layer_b.jsonl",
)


def freeze() -> int:
    target = ROOT / MANIFEST
    if target.exists():
        raise SystemExit("manifest already exists; refusing to overwrite frozen labels")
    config = AdmissionPolicyConfig()
    manifest = capture_manifest(
        ROOT,
        relevant_paths=RELEVANT_POLICY_PATHS,
        policy_v1_path=Path("app/memory/admission_policy.py"),
        dataset_paths=ROBUSTNESS_DATASETS,
        policy_identity={
            "policy_name": config.policy_name,
            "policy_version": config.policy_version,
        },
        config=config.model_dump(mode="json"),
    )
    target.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"frozen manifest: {target}")
    print(f"dataset label seal: {manifest.dataset_label_sha256}")
    return 0


def run(
    output: Path | None,
    markdown_output: Path | None,
    existing_full_test_results: str,
    existing_evaluation_results: str,
) -> int:
    with tempfile.TemporaryDirectory(prefix="memory-admission-robustness-") as temp:
        report = run_robustness_evaluation(
            ROOT,
            ROOT / MANIFEST,
            Path(temp),
            live_smoke_status="DEFERRED",
            live_smoke_reason=(
                "local quantized 32B live inference excluded by bounded-cost rule; "
                "Layers A/B and long trajectories are complete"
            ),
            existing_full_test_results=existing_full_test_results,
            existing_evaluation_results=existing_evaluation_results,
        )
    payload = report.model_dump(mode="json")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(rendered)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--existing-full-test-results", default="PENDING_FINAL_REGRESSION")
    parser.add_argument("--existing-evaluation-results", default="PENDING_FINAL_REGRESSION")
    args = parser.parse_args()
    if args.freeze:
        if args.output or args.markdown_output:
            parser.error("--freeze cannot be combined with output options")
        return freeze()
    return run(
        args.output,
        args.markdown_output,
        args.existing_full_test_results,
        args.existing_evaluation_results,
    )


if __name__ == "__main__":
    raise SystemExit(main())
