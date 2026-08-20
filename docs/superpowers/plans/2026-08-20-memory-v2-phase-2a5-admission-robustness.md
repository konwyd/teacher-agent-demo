# Memory v2 Phase 2A.5 Admission Robustness Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sealed, deterministic adversarial robustness evaluation for frozen AdmissionPolicyV1 with separate policy/extraction/live layers and long-trajectory analysis.

**Architecture:** A new robustness package verifies an immutable policy/dataset manifest, evaluates candidate-level attacks through the real controller and ledger, evaluates the existing extraction adapter with frozen upstream fixtures, simulates 20/50/100-turn histories, and reports metrics and failure attribution without modifying V1.

**Tech Stack:** Python 3.10, Pydantic v2, SQLite, JSON/JSONL, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-memory-v2-phase-2a5-admission-robustness-design.md`

## Global Constraints

- Do not modify policy/config/threshold/prior/matcher/ledger/contracts.
- Do not alter the existing development or holdout JSONL files.
- Freeze new case IDs, ground-truth decisions/relations, labels before first formal run.
- Attribute causal duplication to `CAUSAL_DEDUP_GAP`, never `POLICY_FAILURE`.
- No live LLM in Layers A/B or long trajectories.
- No authority, MemCon, reward, PolicyV2, or migration.

---

### Task 1: Frozen manifest and dataset seal

**Files:**
- Create: `app/evaluation/memory_admission/robustness/manifest.py`
- Create: `app/evaluation/memory_admission/robustness/policy_v1_manifest.json`
- Test: `tests/test_memory_admission_robustness_manifest.py`

**Interfaces:**
- Produces `create_manifest(repo_root, dataset_paths)`, `verify_manifest(repo_root)`, and `dataset_label_digest(paths)`.

- [ ] Write tests proving relevant byte/config/HEAD/dataset changes fail while unrelated dirty files do not.
- [ ] Run the tests and verify RED due to missing module.
- [ ] Implement canonical hashes, git path-dirty detection, config serialization, and dataset-label sealing.
- [ ] Run tests GREEN, then create the manifest once before any formal robustness run.

### Task 2: Layer A contracts, attack set, and evaluator

**Files:**
- Create: `app/evaluation/memory_admission/robustness/models.py`
- Create: `app/evaluation/memory_admission/robustness/layer_a.py`
- Create: `app/evaluation/memory_admission/robustness/data/layer_a.jsonl`
- Test: `tests/test_memory_admission_robustness_layer_a.py`

**Interfaces:**
- Produces `RobustnessCase`, `RobustnessRecord`, `LayerAReport`, and `LayerAEvaluator.run()`.

- [ ] Write RED tests for unique IDs, attack coverage, no overlap with 30/18 IDs, expected labels, and controller/ledger-only imports.
- [ ] Author and seal at least 60 new adversarial cases before formal execution.
- [ ] Implement real frozen-context/ledger evaluation and case-level failure attribution.
- [ ] Run tests GREEN without changing expectations after observing outputs.

### Task 3: Metrics and metamorphic evaluation

**Files:**
- Create: `app/evaluation/memory_admission/robustness/metrics.py`
- Test: `tests/test_memory_admission_robustness_metrics.py`

**Interfaces:**
- Produces `compute_robustness_metrics(cases, records)` with every required classification, consistency, safety, inflation, and support metric.

- [ ] Write literal confusion-matrix and metamorphic RED tests.
- [ ] Implement zero-safe metrics and distinct system-gap accounting.
- [ ] Verify all metric tests GREEN.

### Task 4: Layer B extraction evaluation

**Files:**
- Create: `app/evaluation/memory_admission/robustness/layer_b.py`
- Create: `app/evaluation/memory_admission/robustness/data/layer_b.jsonl`
- Test: `tests/test_memory_admission_robustness_layer_b.py`

**Interfaces:**
- Produces `LayerBEvaluator.run()` over frozen upstream fixtures and the existing `build_memory_candidates()` adapter.

- [ ] Write RED tests for at least 30 paraphrase, negation, conditional, temporal, noisy, and capability-independent fixtures.
- [ ] Implement exact candidate comparison and extraction/normalization attribution.
- [ ] Verify Layer B GREEN without invoking the agent loop or LLM.

### Task 5: Causal duplication and long trajectories

**Files:**
- Create: `app/evaluation/memory_admission/robustness/trajectories.py`
- Test: `tests/test_memory_admission_robustness_trajectories.py`

**Interfaces:**
- Produces `run_causal_duplicate_attacks()` and `run_long_trajectories(lengths=(20,50,100))`.

- [ ] Write RED tests for five duplicate-noise variants and 20/50/100-turn reports.
- [ ] Implement simulations with the unchanged real SQLite ledger/controller.
- [ ] Assert inflation is reported as `CAUSAL_DEDUP_GAP`, not repaired.
- [ ] Verify long-trajectory metrics and deterministic replay GREEN.

### Task 6: CLI, live-smoke status, and final report

**Files:**
- Create: `app/evaluation/memory_admission/robustness/report.py`
- Create: `scripts/run_memory_admission_robustness.py`
- Create: `docs/memory/MEMORY_V2_ADMISSION_ROBUSTNESS.md`
- Test: `tests/test_memory_admission_robustness_report.py`

**Interfaces:**
- Produces one JSON artifact, the required console fields, bounded Layer C status, and markdown report rendering.

- [ ] Write RED tests for every required output field and failure taxonomy.
- [ ] Implement manifest-first orchestration and a bounded/deferred live-smoke section.
- [ ] Run the first formal robustness evaluation exactly once against the sealed expectations.
- [ ] Render results without changing PolicyV1 or ground truth.

### Task 7: Completion audit and regressions

**Files:**
- Modify only generated robustness result documentation if required.

- [ ] Verify manifest and dataset seal remain valid.
- [ ] Run all Phase 2A.5 and Phase 2A focused tests.
- [ ] Run full `python3 -m pytest -q` and `python3 scripts/run_eval.py`.
- [ ] Compile modified Python, run `git diff --check`, inspect policy-file hashes and worktree artifacts.
- [ ] Map all 23 requirements to direct evidence and stop with `READY_FOR_PHASE_2B = REVIEW_REQUIRED`.
