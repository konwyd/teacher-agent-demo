"""Manifest-first Phase 2A.5 orchestration and markdown reporting."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from app.evaluation.memory_admission.robustness.layer_a import LayerAEvaluator
from app.evaluation.memory_admission.robustness.layer_b import LayerBEvaluator
from app.evaluation.memory_admission.robustness.manifest import verify_manifest
from app.evaluation.memory_admission.robustness.metrics import (
    RobustnessMetric,
    compute_robustness_metrics,
)
from app.evaluation.memory_admission.robustness.models import FailureAttribution
from app.evaluation.memory_admission.robustness.named_probes import (
    NamedProbeResult,
    run_named_misconception_probes,
)
from app.evaluation.memory_admission.robustness.trajectories import (
    LongTrajectoryResult,
    run_long_trajectories,
)
from app.memory.admission_config import AdmissionPolicyConfig


class LiveSmokeResult(BaseModel):
    status: str
    reason: str
    max_trajectories: int = 5
    wall_latency_seconds: float | None = None
    model_call_count: int = 0
    tool_call_count: int = 0
    duplicate_call_count: int = 0
    extraction_failure_count: int = 0
    findings: str = "not executed"


class RobustnessEvaluationReport(BaseModel):
    policy_v1_frozen: bool
    frozen_git_head: str
    frozen_files: list[dict]
    dataset_label_sha256: str
    layer_a_cases: int
    layer_b_cases: int
    long_trajectories: list[int]
    metrics: dict[str, RobustnessMetric]
    layer_b_extraction_accuracy: RobustnessMetric
    failure_counts_by_layer: dict[str, dict[str, int]]
    failure_taxonomy: dict[str, list[str]]
    long_trajectory_results: list[LongTrajectoryResult]
    named_probe_results: dict[str, NamedProbeResult]
    live_smoke: LiveSmokeResult
    phase_2b_blockers: list[str]
    existing_full_test_results: str = "PENDING_FINAL_REGRESSION"
    existing_evaluation_results: str = "PENDING_FINAL_REGRESSION"
    policy_v1_modified: bool = False
    authoritative_admission_implemented: bool = False
    memcon_implemented: bool = False
    online_reward_implemented: bool = False
    phase_2a_5_complete: bool = True
    ready_for_phase_2b: str = "REVIEW_REQUIRED"


def run_robustness_evaluation(
    repo_root: Path,
    manifest_path: Path,
    workspace: Path,
    *,
    live_smoke_status: str,
    live_smoke_reason: str,
    existing_full_test_results: str = "PENDING_FINAL_REGRESSION",
    existing_evaluation_results: str = "PENDING_FINAL_REGRESSION",
) -> RobustnessEvaluationReport:
    config = AdmissionPolicyConfig().model_dump(mode="json")
    manifest = verify_manifest(
        repo_root,
        manifest_path,
        current_config=config,
    )
    data = repo_root / "app/evaluation/memory_admission/robustness/data"
    layer_a_evaluator = LayerAEvaluator(data)
    layer_a_cases = layer_a_evaluator.load_cases(data)
    layer_a = layer_a_evaluator.run(workspace / "layer-a")
    layer_b_evaluator = LayerBEvaluator(data / "layer_b.jsonl")
    layer_b = layer_b_evaluator.run()
    long_report = run_long_trajectories(workspace / "long")
    named_probes = run_named_misconception_probes(workspace / "named-probes")
    taxonomy = {item.value: [] for item in FailureAttribution}
    for row in layer_a.failures:
        taxonomy[row.failure_attribution.value].append(row.case_id)
    for row in layer_b.failures:
        taxonomy[row.failure_attribution.value].append(row.case_id)
    layer_a_counts = Counter(
        row.failure_attribution.value for row in layer_a.failures
    )
    layer_b_counts = Counter(
        row.failure_attribution.value for row in layer_b.failures
    )
    causal_gap = bool(taxonomy[FailureAttribution.CAUSAL_DEDUP_GAP.value])
    blockers = [
        "Human review of every false ADMIT and missed promotion",
        "Production shadow divergence and privacy/retention review",
        "Stale-belief and correction lifecycle semantics",
    ]
    if causal_gap:
        blockers.insert(
            0,
            "ROBUSTNESS_GAP_CAUSAL_DEDUP: contracts lack a causal-root identity",
        )
    return RobustnessEvaluationReport(
        policy_v1_frozen=True,
        frozen_git_head=manifest.git_head,
        frozen_files=[item.model_dump() for item in manifest.files],
        dataset_label_sha256=manifest.dataset_label_sha256,
        layer_a_cases=len(layer_a.records),
        layer_b_cases=len(layer_b.records),
        long_trajectories=[item.turn_count for item in long_report.trajectories],
        metrics=compute_robustness_metrics(layer_a_cases, layer_a.records),
        layer_b_extraction_accuracy=layer_b.extraction_accuracy,
        failure_counts_by_layer={
            "LAYER_A": dict(layer_a_counts),
            "LAYER_B": dict(layer_b_counts),
            "LAYER_C": {},
        },
        failure_taxonomy=taxonomy,
        long_trajectory_results=long_report.trajectories,
        named_probe_results=named_probes,
        live_smoke=LiveSmokeResult(
            status=live_smoke_status,
            reason=live_smoke_reason,
            findings=(
                "LIVE_SMOKE_CONFOUNDED_BY_AGENT_LOOP = UNKNOWN; deferred before model invocation"
                if live_smoke_status == "DEFERRED"
                else "bounded live smoke completed"
            ),
        ),
        phase_2b_blockers=blockers,
        existing_full_test_results=existing_full_test_results,
        existing_evaluation_results=existing_evaluation_results,
    )


def _pct(metric: RobustnessMetric) -> str:
    return f"{metric.value:.4f} ({metric.numerator}/{metric.support})"


def render_markdown(report: RobustnessEvaluationReport) -> str:
    metric_lines = "\n".join(
        f"- `{name}`: {_pct(metric)}"
        for name, metric in sorted(report.metrics.items())
    )
    file_lines = "\n".join(
        f"- `{item['path']}` — SHA-256 `{item['sha256']}`, dirty=`{str(item['dirty']).lower()}`"
        for item in report.frozen_files
    )
    failure_lines = "\n".join(
        f"- `{name}`: {len(cases)}" + (f" — {', '.join(cases)}" if cases else "")
        for name, cases in report.failure_taxonomy.items()
    )
    trajectory_lines = "\n".join(
        f"- {item.turn_count} turns: final ledger={item.final_ledger_size}, "
        f"max pending={item.max_pending_ledger_size}, promotions={item.promotion_points}, "
        f"latency={item.promotion_latency_independent_events}, contradictions={item.contradiction_count}, "
        f"stale pending={item.stale_pending_items}, recovery={item.contradiction_recovery_behavior}"
        for item in report.long_trajectory_results
    )
    probe_lines = "\n".join(
        f"- {name}: decisions={item.decisions}, relations={item.relations}, "
        f"keys={item.semantic_keys}, support={item.support_count}, "
        f"causal_turns={item.distinct_causal_turns}, events={item.event_references}, "
        f"gap={item.gap_code or 'none'}"
        for name, item in report.named_probe_results.items()
    )
    blocker_lines = "\n".join(f"- {item}" for item in report.phase_2b_blockers)
    return f"""# Memory v2 Phase 2A.5 — Admission Robustness

## PolicyV1 frozen manifest

`POLICY_V1_FROZEN = {'YES' if report.policy_v1_frozen else 'NO'}`  
Git HEAD: `{report.frozen_git_head}`  
Dataset label seal: `{report.dataset_label_sha256}`

{file_lines}

The manifest verifies only listed paths; unrelated dirty worktree state is ignored.

## Evaluation layers

- Layer A: {report.layer_a_cases} direct candidate/context/ledger attacks; no LLM, agent loop, GraphRAG, quality-QA, or StudentMemoryService writes.
- Layer B: {report.layer_b_cases} frozen-upstream extraction attacks; accuracy {_pct(report.layer_b_extraction_accuracy)}.
- Layer C: `{report.live_smoke.status}` — {report.live_smoke.reason}. It is excluded from PolicyV1 metrics.

## Dataset construction and attack taxonomy

The new sealed set is disjoint from the frozen 30 development and 18 holdout cases. It covers paraphrase/deceptive-neighbor, negation/correction, conditional and temporal preference, source mutation, misconception trajectories A–E, same/cross-session evidence, stale beliefs, noise, capability-independent preferences, relations, and metamorphic mutations.

## Layer A robustness results

{metric_lines}

## Layer B extraction results

Extraction failures and normalization failures are reported separately from Policy failures. Layer B accuracy: {_pct(report.layer_b_extraction_accuracy)}.

## Long-trajectory results

{trajectory_lines}

## Named misconception trajectories A–E

{probe_lines}

Pending survival distributions and decision timelines are present in the machine-readable report.

## Causal duplicate findings

`CAUSAL_DUPLICATE_INFLATION_RATE = {_pct(report.metrics['causal_duplicate_inflation_rate'])}`. Any observed inflation is classified as `CAUSAL_DEDUP_GAP`, not `POLICY_FAILURE`; PolicyV1, ledger, and contracts remain unchanged.

## Failure attribution

{failure_lines}

## Live-smoke findings

Status: `{report.live_smoke.status}`. {report.live_smoke.findings}. Wall latency is `{report.live_smoke.wall_latency_seconds}` and no live calls are included in Layer A/B scores.

## Existing regression results

- Full pytest: `{report.existing_full_test_results}`
- Existing offline evaluation: `{report.existing_evaluation_results}`

## Known limitations

This is a deterministic adversarial set, not a production prevalence estimate. Layer B evaluates the current lexical adapter with frozen upstream fixtures. Legacy memory lacks causal-root and claim-level provenance.

## Phase 2B blockers

{blocker_lines}

## PolicyV2 recommendations (not implemented)

- Add an explicit causal-root observation identity before evidence accumulation.
- Preserve conditional and temporal qualifiers in a future extraction contract.
- Define stale-belief recovery and retraction semantics with a new sealed holdout.
- Any behavior change must be named `AdmissionPolicyV2`; no V1 tuning is permitted.

## Stop boundary

`POLICY_V1_MODIFIED = NO`  
`AUTHORITATIVE_ADMISSION_IMPLEMENTED = NO`  
`MEMCON_IMPLEMENTED = NO`  
`ONLINE_REWARD_IMPLEMENTED = NO`  
`PHASE_2A_5_COMPLETE = YES`  
`READY_FOR_PHASE_2B = REVIEW_REQUIRED`
"""
