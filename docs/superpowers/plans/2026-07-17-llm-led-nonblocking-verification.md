# LLM-led Non-blocking Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make unsupported math verification non-blocking by generating the teacher response as plain text and auditing only equations the existing checker can parse.

**Architecture:** Replace the prototype's structured teaching generation with `DeepSeekClient.chat()`. Construct the persisted turn record locally, extract equations with `math_checker.find_equations`, and retain only deterministic hard audits.

**Tech Stack:** Python 3.13, Pydantic 2 for local records, existing DeepSeek client, existing math checker, pytest, Streamlit.

## Global Constraints

- No real paid API calls in automated tests.
- No topic-specific branches for opposite numbers, geometry, probability, or other knowledge points.
- Missing verifier returns `unverified` and never blocks the teacher response.
- Preserve exact linear-equation checking and one meaningful move per normal multi-step turn.

---

### Task 1: Replace structured teacher generation with plain text

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py`

**Interfaces:**
- Consumes: `StructuredTeacherClient.chat(system, user, temperature, max_tokens) -> str`
- Produces: `run_turn(...) -> PrototypeTurnRecord` with a locally constructed `TeacherResponse`

- [ ] **Step 1: Write RED tests**

Add a plain-text-only fake client. Verify an opposite-number response succeeds with
`evidence_level == "unverified"` and no `chat_json` call. Convert the linear-equation
tests to return text and verify final-form leakage still triggers one repair.

- [ ] **Step 2: Run target tests and confirm failure**

```powershell
python -m pytest tests/prototypes/test_llm_led_turn_contract.py -v
```

Expected: failure because current `run_turn` requires `chat_json` and structured enums.

- [ ] **Step 3: Implement plain-text generation**

Replace the structured protocol with `chat`, reduce the output record to `response`,
remove JSON schema retry, and make the repair path return plain text. Reject only an
empty response.

- [ ] **Step 4: Extract and audit supported equations**

Call `math_checker.find_equations(response)` after each generation. Feed the resulting
list to `_audit_claims`; unsupported prose produces an empty list and remains
`unverified`. Preserve deterministic conflict and final-form audits.

- [ ] **Step 5: Run target tests**

```powershell
python -m pytest tests/prototypes/test_llm_led_turn_contract.py -v
```

Expected: all prototype turn tests pass offline.

### Task 2: Update UI, session compatibility, and regressions

**Files:**
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Modify: `tests/prototypes/test_llm_led_streamlit_session.py`
- Modify: `prototypes/llm_led_teacher/README.md`

**Interfaces:**
- Consumes: simplified `PrototypeTurnRecord`
- Produces: UI details showing verification status, extracted equations, audits, and warnings without LLM-generated research enums

- [ ] **Step 1: Update the stale-session test for schema version 3**

Expect legacy structured turns to be cleared before rendering.

- [ ] **Step 2: Simplify UI diagnostics**

Remove action, task-shape, disclosure, confidence, and rationale fields. Display extracted
equations and explicitly show `UNVERIFIED` as non-blocking.

- [ ] **Step 3: Document the B-lite boundary**

Update the prototype README to state that unsupported claims are displayed with
`unverified` evidence and that only parseable equations receive deterministic checks.

- [ ] **Step 4: Run all verification**

```powershell
python -m py_compile prototypes/llm_led_teacher/logic.py prototypes/llm_led_teacher/streamlit_app.py
python -m pytest
python scripts/run_eval.py
git diff --check
```

Expected: all commands exit 0 without a real API call.

- [ ] **Step 5: Commit**

```powershell
git add docs/superpowers/specs/2026-07-17-llm-led-nonblocking-verification-design.md docs/superpowers/plans/2026-07-17-llm-led-nonblocking-verification.md prototypes/llm_led_teacher/logic.py prototypes/llm_led_teacher/streamlit_app.py prototypes/llm_led_teacher/README.md tests/prototypes/test_llm_led_turn_contract.py tests/prototypes/test_llm_led_streamlit_session.py
git commit -m "refactor: make prototype verification non-blocking"
```
