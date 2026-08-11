# Safe Streaming Teacher Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give LLM-led Teacher users immediate stage feedback and stream only the final audited teacher response.

**Architecture:** Keep `run_turn()` synchronous and preserve its audit/repair boundary, adding only an optional best-effort progress callback with stable stage identifiers. Keep presentation concerns in a small Streamlit-independent helper that losslessly chunks audited text; the Streamlit page maps stages to Chinese labels and streams the returned final response once.

**Tech Stack:** Python 3.11, Pydantic, Streamlit, pytest, existing DeepSeek wrapper.

## Global Constraints

- Never expose the teacher draft before deterministic audit and optional repair finish.
- Do not change observer, RAG, verifier, repair policy, model, temperature, base URL, or active-problem semantics.
- Do not call a real paid API in tests.
- Preserve all unrelated working-tree changes.

---

### Task 1: Core progress contract

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Test: `tests/prototypes/test_llm_led_turn_contract.py`

**Interfaces:**
- Consumes: existing `run_turn(episode, message, *, client=None)` behavior.
- Produces: `run_turn(..., progress_callback: Callable[[str], None] | None = None)` and best-effort stage notifications.

- [ ] **Step 1: Write failing normal-path and repair-path tests**

Add fake-client tests that collect stage strings and assert the normal path is exactly `observing, retrieving, generating, auditing, complete`; the existing leaking-answer repair case must additionally include `repairing` before `complete`. Add a callback that raises and assert the final teacher response still returns.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
```

Expected: FAIL because `run_turn()` does not accept `progress_callback`.

- [ ] **Step 3: Implement the minimum callback support**

Add:

```python
ProgressCallback = Callable[[str], None]

def _report_progress(callback: ProgressCallback | None, stage: str) -> None:
    if callback is None:
        return
    try:
        callback(stage)
    except Exception:
        return
```

Call it immediately before observation, retrieval, draft generation, audit, optional repair, and after appending the final record. Do not send response text through this callback.

- [ ] **Step 4: Run target tests to verify GREEN**

Run the target test file and expect all tests to pass.

### Task 2: Lossless audited-response chunking

**Files:**
- Create: `prototypes/llm_led_teacher/streaming.py`
- Test: `tests/prototypes/test_llm_led_streaming.py`

**Interfaces:**
- Consumes: final `str` from `PrototypeTurnRecord.teacher_output.response`.
- Produces: `iter_display_chunks(text: str, chunk_size: int = 12) -> Iterator[str]`.

- [ ] **Step 1: Write failing chunking tests**

Cover empty text, invalid non-positive chunk size, short text, and Chinese/Markdown/LaTeX text. Assert `''.join(iter_display_chunks(text)) == text` and every non-final chunk is no longer than `chunk_size`.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python3 -m pytest tests/prototypes/test_llm_led_streaming.py -q
```

Expected: collection failure because the module/function does not exist.

- [ ] **Step 3: Implement fixed-width lossless chunking**

Implement a pure iterator using string slicing. Raise `ValueError` for `chunk_size <= 0`; yield nothing for empty input. Do not parse or rewrite Markdown/LaTeX.

- [ ] **Step 4: Run streaming tests to verify GREEN**

Run the new test file and expect all tests to pass.

### Task 3: Streamlit integration

**Files:**
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Test: `tests/prototypes/test_llm_led_streaming.py`

**Interfaces:**
- Consumes: progress stages and final audited response.
- Produces: visible status progression and one-time final-response stream for the new turn.

- [ ] **Step 1: Add a source-level protection test**

Read `streamlit_app.py` and assert it imports `iter_display_chunks`, passes `progress_callback` to `run_turn`, streams `turn.teacher_output.response` only after `run_turn` returns, and no call streams a draft value.

- [ ] **Step 2: Run the test to verify RED**

Expected: FAIL because the page still wraps `run_turn` in a spinner and reruns immediately.

- [ ] **Step 3: Implement the UI integration**

Render the submitted user message immediately, create an assistant chat container, map stable stages to Chinese text using `st.status`, run the turn with the callback, then call `st.write_stream(iter_display_chunks(turn.teacher_output.response))`. On success mark the status complete; on failure mark it error and retain existing exception display. Do not `st.rerun()` after success; sidebar/history state updates naturally on the next Streamlit interaction.

- [ ] **Step 4: Run both prototype test files**

Run:

```bash
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py tests/prototypes/test_llm_led_streaming.py -q
```

Expected: all tests pass.

### Task 4: Regression verification

**Files:**
- Verify only; no production changes unless a regression directly caused by Tasks 1–3 is found.

**Interfaces:**
- Consumes: completed implementation.
- Produces: fresh offline verification evidence.

- [ ] **Step 1: Run the required target suite**

```bash
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
```

- [ ] **Step 2: Run complete pytest**

```bash
python3 -m pytest -q
```

- [ ] **Step 3: Run offline evaluation**

```bash
TEACHER_FORCE_MOCK=1 python3 scripts/run_eval.py
```

- [ ] **Step 4: Inspect the final diff and status**

Confirm only the planned files changed for this feature, excluding pre-existing user modifications. Report exact pass/fail counts and do not push or merge.
