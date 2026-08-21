# CandidateExtractorV2 Route A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an offline deterministic CandidateExtractorV2 Route A without changing frozen evaluation assets or shared contracts.

**Architecture:** A route-local Memory module performs semantic-family matching followed by independent qualifier parsing and existing-contract candidate construction. A separate CLI adapts frozen development and Layer B cases to that production interface and reports existing evaluator metrics without loading holdout.

**Tech Stack:** Python standard library, Pydantic-backed existing Memory contracts, pytest, frozen CandidateExtractorV2 evaluator.

**Spec:** `docs/superpowers/specs/2026-08-21-candidate-extractor-v2-route-a-design.md`

## Global Constraints

- Work only on Route A in `spike/candidate-extractor-v2-deterministic`.
- Do not read sealed-holdout labels or run Route A against holdout.
- Do not modify benchmark, evaluator, PolicyV1, matcher, ledger, baseline writer, shared contracts, causal identity, existing labels, or Memory V2 authority.
- Use no embeddings, model inference calls, remote APIs, or LLM judges.
- Use only reusable lexicons, semantic families, and deterministic parsing.

---

### Task 1: Public extractor and core semantic families

**Files:**
- Create: `app/memory/candidate_extractor_v2_deterministic.py`
- Create: `tests/test_candidate_extractor_v2_deterministic.py`

**Interfaces:**
- Consumes: existing `MemoryCandidate`, enum, and provenance contracts.
- Produces: `build_deterministic_memory_candidates(payload: Mapping[str, Any], *, conversation_id: str) -> list[MemoryCandidate]`.

- [ ] Write literal table-driven tests for profile, preference paraphrases,
  hard negatives, upstream verification/observation/inference, and
  multi-candidate behavior.
- [ ] Run `python3 -m pytest tests/test_candidate_extractor_v2_deterministic.py -q`
  and confirm collection fails because the Route A module is absent.
- [ ] Add the minimal candidate factory, semantic-family lexicons, hard-negative
  guards, and upstream converters needed to make those tests pass.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Qualifier parser and development benchmark

**Files:**
- Modify: `app/memory/candidate_extractor_v2_deterministic.py`
- Modify: `tests/test_candidate_extractor_v2_deterministic.py`

**Interfaces:**
- Consumes: the Task 1 public extractor.
- Produces: explicit negation, polarity, temporal, conditional, scoped,
  uncertainty, correction, and supersession fields in existing candidate
  payloads.

- [ ] Add focused failing tests for single/double negation, temporal bounds,
  condition/scope, uncertainty, correction, and stable output order.
- [ ] Run focused tests and confirm failures identify missing qualifier behavior.
- [ ] Add minimal reusable qualifier parsing and compatible-field merging.
- [ ] Run focused tests, then evaluate the frozen development split through
  `evaluate_cases()`; fix only general semantic-family or parser gaps exposed by
  development cases.
- [ ] Repeat the full focused suite after refactoring.

### Task 3: Route-local benchmark runner and regression gates

**Files:**
- Create: `scripts/run_candidate_extractor_v2_route_a.py`
- Modify: `tests/test_candidate_extractor_v2_deterministic.py`

**Interfaces:**
- Consumes: frozen case loaders/evaluator and the Route A public extractor.
- Produces: JSON reports for `development`, `layer_b`, or both; never holdout.

- [ ] Add a failing integration test that development and Layer B can be
  evaluated through the real extractor and that operational metadata declares
  zero model/VRAM use, offline capability, and deterministic replay.
- [ ] Implement the CLI with `--split {development,layer_b,all}` and optional
  `--output`, using `resource` for peak RSS and existing evaluator latency.
- [ ] Run Route A tests and the CLI for development and Layer B.
- [ ] Run existing Layer B regression and confirm Route A does not regress below
  the frozen current-extractor exact-set baseline of 0.4375.

### Task 4: Repository verification, progress, and commit

**Files:**
- Modify: `PROGRESS.md` (Memory System section only)

**Interfaces:**
- Consumes: verified Route A reports.
- Produces: coordinator-ready branch commit and handoff.

- [ ] Run Route A tests and development/Layer B benchmark commands.
- [ ] Run `python3 -m pytest` and `python3 scripts/run_eval.py`.
- [ ] Run the existing Memory admission robustness command from the repository
  harness and direct `verify_manifest()` without invoking the baseline runner.
- [ ] Record measured quality, safety, contamination, latency, RAM, dependency,
  offline, and determinism results in the Memory System progress section.
- [ ] Run `git diff --check`, inspect `git diff --name-only`, verify protected
  files are unchanged, and confirm no secrets/runtime artifacts are staged.
- [ ] Commit only Route A implementation, tests, runner, design/plan, and the
  scoped `PROGRESS.md` update to the assigned branch.
