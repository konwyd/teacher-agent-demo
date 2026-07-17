# LLM-led Schema Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a recoverable `LLMTeachingTurn` enum-placement error from reaching the student by retrying structured generation exactly once.

**Architecture:** Add a prototype-local wrapper around `StructuredTeacherClient.chat_json`. It catches only Pydantic validation failures, adds the structured error details to a corrective prompt, and performs one final schema-validated call.

**Tech Stack:** Python 3.13, Pydantic 2, pytest, existing DeepSeek JSON mode client.

## Global Constraints

- Do not call a real paid API.
- Do not modify the global DeepSeek client or legacy Teacher Agent flow.
- Retry only `pydantic.ValidationError`, at most once.

---

### Task 1: Add prototype-local schema repair retry

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py`

**Interfaces:**
- Consumes: `StructuredTeacherClient.chat_json(...) -> LLMTeachingTurn`
- Produces: `_chat_teaching_turn(...) -> LLMTeachingTurn`

- [ ] **Step 1: Write the failing tests**

Add a fake client whose first call raises `ValidationError` from
`LLMTeachingTurn.model_validate`, whose second call returns a valid direct-fact turn,
and assert that `run_turn` succeeds after exactly two calls. Add a fake client that
raises `DeepSeekError` and assert that only one call occurs.

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/prototypes/test_llm_led_turn_contract.py -k "schema_validation or transport_error" -v
```

Expected: the validation test fails because the first `ValidationError` escapes.

- [ ] **Step 3: Implement the minimal wrapper**

Catch `pydantic.ValidationError`, serialize `exc.errors(include_url=False)` with the
original turn payload, and call `chat_json` once more with a schema-correction system
message. Route both initial generation and `_repair` through the wrapper. Do not catch
`DeepSeekError` or the second validation failure.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```powershell
python -m pytest tests/prototypes/test_llm_led_turn_contract.py -v
python -m pytest
python scripts/run_eval.py
```

Expected: all commands exit 0; no real API is used.

- [ ] **Step 5: Commit**

```powershell
git add docs/superpowers/specs/2026-07-17-llm-led-schema-retry-design.md docs/superpowers/plans/2026-07-17-llm-led-schema-retry.md prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_turn_contract.py
git commit -m "fix: retry invalid prototype teaching schemas"
```

