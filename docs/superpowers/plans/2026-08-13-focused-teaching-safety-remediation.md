# Focused Teaching Safety Remediation Plan

**Goal:** Close the four residual merge blockers found after the teaching-contract Agent Loop final review, without expanding the feature surface.

**Baseline:** `f24ab4e` on `codex/teaching-contract-8502`.

## Constraints

- Preserve login, history, pending-message recovery, token/cache/context metrics, real Agent trace, MathJax output, and the existing 21G Chroma data.
- Do not modify or commit Chroma files, runtime databases, logs, caches, or environment files.
- Use TDD for every behavior change.
- Keep web fallback policy unchanged: provider-error deterministic fallback may call local RAG only, never web.
- Full suite and offline evaluation must remain green.

## Task 1: Clause-aware full-solution intent

**Files:** `prototypes/llm_led_teacher/teaching_policy.py`, `tests/prototypes/test_teaching_policy.py`.

- Split intent on sentence punctuation and contrast boundaries.
- Evaluate each full-solution marker in its own clause.
- The last explicit full-solution mention determines positive versus negated intent.
- Cover `不想要完整解答`, `不需要完整解答`, long negations, postposed negation, and `不要解释为什么，请完整解答`.
- No-answer and no-application requests remain higher-priority safety constraints.

## Task 2: Universal post-repair audit and public-data redaction

**Files:** `prototypes/llm_led_teacher/logic.py`, `prototypes/llm_led_teacher/streamlit_app.py`, relevant turn/UI/archive tests.

- Every Agent-path repair, regardless of why repair was triggered, receives a second `SemanticBoundaryAuditor.audit`.
- A failed/error second audit or hard-check hit always selects the fixed contract-safe fallback.
- Never expose or persist rejected-draft evidence. Persist public audit copies with rejected evidence removed; violation categories, audit status, error type, and safe fallback reason remain visible.
- Count the second audit in usage metadata.
- Preserve legacy record compatibility.

## Task 3: Strict linear-verifier support domain

**Files:** `app/agent/math_checker.py`, `app/agent/optional_verifier.py`, relevant verifier tests.

- Stop progressively trimming a failed equation candidate into another equation.
- Reject powers, variable denominators, variable-by-variable multiplication, adjacent factor parentheses, and variable/parenthesis products before parsing.
- Keep valid scalar-distribution linear equations supported.
- Unsupported expressions such as `1/x=2`, `x(x+1)=2`, and `(x+1)(x-1)=0` return neutral `not_applicable`.

## Task 4: Strict local-RAG source provenance

**Files:** `llm-test/rag_retriever.py`, `prototypes/llm_led_teacher/rag_hook.py`, relevant RAG/tool tests.

- Accept only explicit `source=textbook` or `source=exercise`.
- Drop missing or unknown source values at both retrieval normalization and adapter boundaries.
- Continue exact subject/stage filtering; do not silently cross subject or stage.

## Verification

1. Focused tests for all four tasks.
2. `python3 -m pytest`.
3. `python3 scripts/run_eval.py` → 38/38.
4. Scoped Ruff on changed files, `git diff --check`, manifest/secret hygiene.
5. Read-only code review.
6. Real local-model acceptance: concept, explicit full solution, Physics local RAG, and off-topic zero-tool.
