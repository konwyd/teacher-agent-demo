# LLM Turn Observer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-blocking pre-teaching LLM observer that detects topic switches and selects single-move versus complete-response pacing.

**Architecture:** Call `chat_json` with a three-field `TurnGuidance` schema before the existing plain-text `chat`. Apply successful guidance to active-problem state and the teaching prompt; catch observer failures and continue with deterministic fallback guidance.

**Tech Stack:** Python 3.13, Pydantic 2, existing DeepSeek JSON/text client, pytest, Streamlit.

## Global Constraints

- Observer failure must never block the main teacher response.
- Observer never judges mathematical correctness.
- The student message, not an LLM rewrite, becomes the new active problem.
- No real paid API calls in automated tests.
- Keep unsupported math responses non-blocking and linear-equation checks exact.

---

### Task 1: Add observer contract and cross-topic behavior

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py`

**Interfaces:**
- Produces: `TurnGuidance(starts_new_problem, pacing, reason)`
- Consumes: `StructuredTeacherClient.chat_json(...)` for observation and `.chat(...)` for teaching

- [ ] Write a failing two-turn test that starts with probability and switches to `3x-5=10`.
- [ ] Assert that the second guidance changes active problem, sends `single_move` to the teacher payload, and catches a leaked `x=5` through the equation verifier.
- [ ] Write a failing observer-error test asserting that `DeepSeekError` does not prevent plain-text teaching.
- [ ] Implement `_observe_turn`, deterministic fallback, active-problem switching, and guidance persistence.
- [ ] Run `python -m pytest tests/prototypes/test_llm_led_turn_contract.py -v`.

### Task 2: Expose the experiment and verify regressions

**Files:**
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Modify: `tests/prototypes/test_llm_led_streamlit_session.py`
- Modify: `prototypes/llm_led_teacher/README.md`

**Interfaces:**
- UI reads persisted observer guidance and observer error from each turn record.

- [ ] Bump prototype session schema to 4 and update the stale-session test.
- [ ] Display pacing, topic-switch decision, reason, and fallback error in verification details.
- [ ] Document the extra-call cost and non-blocking fallback.
- [ ] Run syntax checks, complete pytest, `scripts/run_eval.py`, and `git diff --check`.
- [ ] Commit with `feat: add non-blocking LLM turn observer`.
