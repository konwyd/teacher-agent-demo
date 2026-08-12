# Stable Bottom Context Control and Pending Messages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the context indicator permanently beside the chat send button and preserve a submitted student message if generation is interrupted.

**Architecture:** Render the indicator and chat input in Streamlit 1.59's public `st.bottom` root, with only a small absolute offset inside that stable root. Add `pending_student_message` to `PrototypeEpisode`; save it before calling the model, clear it only after a successful turn, and render a retry action after interrupted runs.

**Tech Stack:** Python 3.10, Streamlit 1.59, Pydantic 2, SQLite conversation archive, pytest.

## Global Constraints

- Preserve authentication and per-user conversation isolation.
- Preserve all completed `PrototypeTurnRecord` values and token/cache metadata.
- Do not depend on page-level CSS anchors or `st.empty()` replacement for placement.
- Do not call the real model in tests.

---

### Task 1: Stable bottom control

**Files:**
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Test: `tests/prototypes/test_llm_led_streaming.py`
- Test: `tests/prototypes/test_llm_led_auth_ui.py`

**Interfaces:**
- Consumes: Streamlit's public `st.bottom` container and `st.popover`.
- Produces: `render_context_usage(episode)` inside the same bottom root as `st.chat_input`.

- [ ] Write a failing AppTest/source-contract regression proving the control uses `st.bottom`, has the stable keyed container, and no longer uses CSS anchors or the replaceable context slot.
- [ ] Run the focused tests and confirm failure because the current page still uses `.context-usage-float` and `context_usage_slot`.
- [ ] Replace the custom HTML `<details>` control with a native popover in `st.bottom`; position its keyed wrapper directly left of the send button.
- [ ] Run the focused UI and authentication tests and confirm they pass.

### Task 2: Durable in-flight student message

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Test: `tests/prototypes/test_llm_led_turn_contract.py`
- Test: `tests/prototypes/test_llm_led_auth_ui.py`
- Test: `tests/test_conversation_archive.py`

**Interfaces:**
- Produces: `PrototypeEpisode.pending_student_message: str` with an empty default.
- Consumes: `_save_active_conversation(episode)` before and after `run_turn`.

- [ ] Write failing tests proving the pending message survives Pydantic/SQLite round trips and is saved before model generation.
- [ ] Run those tests and confirm failure because the field and pre-generation save do not exist.
- [ ] Set and save the pending message before `run_turn`; clear and save it after success; leave it stored after failure.
- [ ] Render an interrupted pending message with a retry action without duplicating the user bubble.
- [ ] Run focused persistence, turn-contract, and UI tests.

### Task 3: Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes: all changes from Tasks 1 and 2.
- Produces: fresh regression evidence.

- [ ] Run `python3 -m pytest` and require zero failures.
- [ ] Run `python3 scripts/run_eval.py` and require all offline evaluation cases to pass.
- [ ] Run `python3 -m py_compile prototypes/llm_led_teacher/streamlit_app.py prototypes/llm_led_teacher/logic.py` and `git diff --check`.
