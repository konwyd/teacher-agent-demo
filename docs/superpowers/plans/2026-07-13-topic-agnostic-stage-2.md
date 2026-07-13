# Topic-agnostic teaching harness — Stage 2 execution plan

> **Execution note:** this plan implements only the approved Stage 2 in
> `docs/topic-agnostic-teaching-harness-audit.md`. It preserves the current
> one-variable-linear-equation workflow, prompts, checker, RAG paths, and
> externally consumed legacy response fields.

## Objective

Introduce a neutral teaching-harness contract alongside the existing equation
teacher contract. The new contract must represent an ordinary tutoring turn
without an equation, a concept identifier, or a graph result, while the current
runtime continues to select and execute its legacy actions unchanged.

## Scope and boundaries

In scope:

1. Add neutral state and action schemas in `app/agent/state.py`.
2. Add a deterministic projection from `TeacherState` to the neutral schema.
3. Expose the projected state as additive API response fields.
4. Add focused contract tests before each implementation change.

Out of scope:

1. Do not alter `app/agent/workflow.py`, `app/agent/socratic_policy.py`,
   prompts, `math_checker.py`, retrievers, graph data, or evaluation fixtures.
2. Do not attach an LLM-based `MathAnalysis` pass to production decisions.
3. Do not replace legacy `TeachingAction`; its enum values remain the runtime
   compatibility surface until Stage 3 migrates callers.

## Implementation sequence

### 1. Neutral schemas and legacy action mapping

Add a separate `TeachingActionKind` because `TeachingAction` is already the
legacy runtime enum. Its values are `ASSESS`, `DIAGNOSE`, `GIVE_HINT`,
`EXPLAIN_STEP`, `CHECK_UNDERSTANDING`, `SHOW_WORKED_EXAMPLE`,
`OFFER_PRACTICE`, and `FINISH`.

Define `TeachingActionPlan`, `MathAnalysis`, `VerificationResult`,
`GraphContext`, and `TeachingState`. Each model has defaults that allow an
empty, non-equation tutoring turn. The verifier and graph context must both
express unavailable/not-applicable status without causing validation failure.

Create a complete mapping from every legacy action to one neutral action. An
unmapped legacy action raises a clear `ValueError`; contract tests enumerate the
legacy enum so later additions cannot silently lose a neutral projection.

### 2. Compatibility projection

Implement `project_teacher_state(legacy_state)` in `app/agent/state.py` after
`TeacherState` is declared. It copies stable student-facing context and
translates legacy diagnostic/checker/retrieval/action data into the neutral
models. It does not call an LLM, verifier, or retriever and does not mutate the
legacy object.

The projected data intentionally contains summaries and evidence, not legacy
equation-specific state names such as `target_equation` or
`detected_concept_id`.

### 3. Additive API output

Extend `TurnResponse` in `app/api/main.py` with optional/additive
`teaching_action_kind` and `teaching_state` fields. After `run_turn` returns its
legacy `TeacherState`, project it and serialize the projection into those fields.
All existing response fields, values, HTTP behavior, and workflow invocation
remain unchanged.

### 4. Verification

Run, in this order:

```powershell
python -m pytest tests/test_topic_agnostic_state.py
python -m pytest tests/test_api.py
python -m pytest
python scripts/run_eval.py
python scripts/run_eval_modes.py --mode deepseek_only
python scripts/run_eval_modes.py --mode agent_no_graph
python scripts/run_eval_modes.py --mode agent_with_graph
```

The first two focused commands prove the new contract and compatibility
boundary. The full suite and mock evaluation protect the existing equation
workflow. Mode results are recorded; the known `deepseek_only` 6/26 baseline is
not a Stage 2 regression if it remains unchanged, and the two agent modes must
remain 26/26.

## Acceptance criteria

1. A neutral `TeachingState` validates without an equation, concept ID, graph
   context, or verifier result.
2. Every legacy action projects deterministically to one neutral action.
3. API responses add the neutral projection without removing or changing legacy
   fields.
4. The runtime invokes no new model, verifier, RAG, or graph operation.
5. Full offline tests and the mock evaluation are green; mode behavior is
   recorded without fixing pre-existing deepseek-only failures.
