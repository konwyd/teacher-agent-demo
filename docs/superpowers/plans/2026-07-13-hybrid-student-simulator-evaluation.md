# Hybrid Student Simulator Evaluation Implementation Plan

**Goal:** Implement the approved 18-scenario deterministic student simulator, run it against all three Teacher modes, persist 54 auditable traces before scoring, and produce deterministic per-conversation and aggregate metrics.

**Architecture:** Add an evaluation-only simulator, runner, scorer, scenario dataset, and CLI. Reuse the existing Teacher mode behavior through a small evaluation adapter; do not modify workflow, math_checker, or retrieval semantics. Implement each slice test-first and keep the existing golden evaluator as a regression gate.

**Tech Stack:** Python 3.10+, Pydantic 2, pytest, existing DeepSeek client, existing math_checker and mode runner.

## Global Constraints

- Exactly 18 default scenarios: six concepts times low, medium, and high mastery.
- Exactly 54 traces and scores for an unfiltered full run.
- Student state transitions are deterministic and receive Teacher response text only.
- DeepSeek may reword a fixed student intent but may not change its mathematical meaning.
- Maximum eight Teacher turns; only the four approved termination reasons are valid.
- Persist and close traces before scoring from disk.
- Do not change core workflow, math_checker, or retriever behavior.
- Final gates: `python -m pytest`, `python scripts/run_eval.py`, and `python scripts/run_hybrid_eval.py`.

## Task 1: Scenario and simulator domain

**Files:**

- Create `app/evaluation/student_simulator.py`
- Create `data/eval/student_simulator_scenarios.jsonl`
- Create `tests/test_student_simulator.py`

**Deliverable:** Pydantic domain models, strict 6x3 loader validation, math_checker-backed intent validation, deterministic observations/transitions, four termination reasons, template wording, and guarded optional DeepSeek wording.

**TDD cycle:**

1. Add failing loader/model/math-intent tests.
2. Run `python -m pytest tests/test_student_simulator.py -q` and confirm missing API failures.
3. Add minimal models and 18 scenarios.
4. Add failing transition and wording-safety tests.
5. Implement deterministic transition and wording fallback.
6. Re-run the focused tests after each behavior.

## Task 2: Dynamic Teacher mode runner

**Files:**

- Create `app/evaluation/hybrid_runner.py`
- Modify `app/evaluation/mode_runner.py` only if a stable public single-turn adapter is required
- Create `tests/test_hybrid_runner.py`

**Deliverable:** Isolated scenario-mode sessions, deep-copied initial state, dynamic turn exchange, maximum-eight-turn enforcement, full transcript/state trace models, and JSONL trace persistence without scoring.

**TDD cycle:**

1. Add failing isolation, no-mode-leakage, max-turn, terminal-state, and trace-shape tests.
2. Implement an injected Teacher responder interface for unit tests.
3. Implement the existing three Teacher mode adapters.
4. Verify full unfiltered generation returns 54 trace objects offline.

## Task 3: Disk-first scoring

**Files:**

- Create `app/evaluation/hybrid_scorer.py`
- Create `tests/test_hybrid_scorer.py`

**Deliverable:** Read traces from disk, apply observable deterministic Teacher checks, calculate trajectory metrics, write score JSONL, and aggregate by mode/concept/mastery combinations.

**TDD cycle:**

1. Add fixed-trace tests with exact metric expectations.
2. Add failing tests proving scoring reads the persisted file rather than in-memory traces.
3. Implement per-conversation metrics and summary grouping.
4. Verify trace/score counts and key uniqueness.

## Task 4: CLI and end-to-end pipeline

**Files:**

- Create `scripts/run_hybrid_eval.py`
- Extend `tests/test_hybrid_runner.py` or `tests/test_hybrid_scorer.py` with CLI/pipeline coverage

**Deliverable:** Filterable CLI, independent Teacher/Student LLM switches, run manifest, trace-first pipeline, score and summary outputs, non-zero exit on incomplete data, and exactly 54 offline records by default.

**TDD cycle:**

1. Add parser/filter/count and trace-before-score tests.
2. Implement run directory and manifest lifecycle.
3. Run the complete offline Hybrid evaluation.

## Task 5: Regression verification

1. Run focused Hybrid tests.
2. Run `python -m pytest`.
3. Run `python scripts/run_eval.py`.
4. Run `python scripts/run_hybrid_eval.py`.
5. Inspect output counts, termination distribution, and `git diff --check`.

## Implementation Result

- Full repository tests: 179 passed with one pre-existing Starlette deprecation warning.
- Existing strict golden evaluation: 26/26 passed.
- Full offline Hybrid evaluation: 54 traces and 54 scores, 54 unique scenario-mode keys, zero initial-state mismatches, maximum eight Teacher turns.
- Real DeepSeek Teacher runs: 18 completed traces and scores for each Agent mode; a static comparison report is generated from their persisted summaries.
