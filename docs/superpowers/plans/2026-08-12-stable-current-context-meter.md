# Stable Current Context Meter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the context meter always calculate against the currently running model limit while preserving any different limit saved by an older conversation as clearly labeled historical detail.

**Architecture:** Add a small pure formatting module that converts current runtime capacity plus last-turn metadata into an immutable display model. The Streamlit page renders only that display model, so historical metadata cannot silently override runtime configuration. Existing episode JSON remains unchanged.

**Tech Stack:** Python 3.10, dataclasses, Streamlit, pytest.

## Global Constraints

- Current model capacity comes only from `DEEPSEEK_CONTEXT_WINDOW_TOKENS` for the running Streamlit process.
- Historical `context_window_tokens` remains read-only diagnostic data and never participates in the primary percentage.
- Existing conversation records are not migrated or overwritten.
- The meter describes the latest model request, not cumulative conversation tokens or GPU memory usage.

---

### Task 1: Pure context display model

**Files:**
- Create: `prototypes/llm_led_teacher/context_usage.py`
- Create: `tests/prototypes/test_context_usage.py`

**Interfaces:**
- Consumes: `metadata: dict[str, object]`, `current_window_tokens: int`
- Produces: `ContextUsageDisplay` and `build_context_usage_display(metadata, current_window_tokens) -> ContextUsageDisplay`

- [ ] **Step 1: Write failing tests for old history versus current capacity**

```python
def test_current_model_limit_wins_over_historical_conversation_limit():
    display = build_context_usage_display(
        {"context_used_tokens": 65536, "context_window_tokens": 524288},
        current_window_tokens=131072,
    )
    assert display.used_tokens == 65536
    assert display.current_window_tokens == 131072
    assert display.usage_ratio == 0.5
    assert display.historical_window_tokens == 524288
    assert display.historical_limit_changed is True


def test_missing_history_does_not_invent_historical_limit():
    display = build_context_usage_display({}, current_window_tokens=131072)
    assert display.used_tokens == 0
    assert display.historical_window_tokens is None
    assert display.historical_limit_changed is False
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/prototypes/test_context_usage.py -q`

Expected: collection fails because `prototypes.llm_led_teacher.context_usage` does not exist.

- [ ] **Step 3: Implement the immutable display model**

```python
@dataclass(frozen=True)
class ContextUsageDisplay:
    used_tokens: int
    current_window_tokens: int
    historical_window_tokens: int | None
    usage_ratio: float
    historical_limit_changed: bool


def build_context_usage_display(
    metadata: Mapping[str, object],
    current_window_tokens: int,
) -> ContextUsageDisplay:
    current_window = max(int(current_window_tokens), 0)
    used = max(int(metadata.get("context_used_tokens", 0) or 0), 0)
    historical_raw = metadata.get("context_window_tokens")
    historical = int(historical_raw) if historical_raw else None
    ratio = used / current_window if current_window else 0.0
    return ContextUsageDisplay(
        used_tokens=used,
        current_window_tokens=current_window,
        historical_window_tokens=historical,
        usage_ratio=ratio,
        historical_limit_changed=(
            historical is not None and historical != current_window
        ),
    )
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_context_usage.py -q`

Expected: all tests pass.

---

### Task 2: Render current and historical values separately

**Files:**
- Modify: `prototypes/llm_led_teacher/streamlit_app.py:298-326`
- Modify: `tests/prototypes/test_llm_led_streaming.py:120-145`

**Interfaces:**
- Consumes: `build_context_usage_display()` from Task 1
- Produces: primary label based on runtime capacity and separate historical diagnostic rows

- [ ] **Step 1: Add a UI contract test**

The test must assert that the Streamlit source calls `build_context_usage_display`, renders the labels `本轮请求占用`, `当前模型上限`, and `该对话历史上限`, includes `旧配置记录`, and no longer assigns `context_window` from `metadata.get("context_window_tokens", ...)`.

- [ ] **Step 2: Run the UI contract test and verify RED**

Run: `python3 -m pytest tests/prototypes/test_llm_led_streaming.py::test_streamlit_shows_context_occupancy_and_kv_cache_hits -q`

Expected: failure because the new labels and display-builder call are absent.

- [ ] **Step 3: Update `render_context_usage`**

Read `DEEPSEEK_CONTEXT_WINDOW_TOKENS`, call `build_context_usage_display`, calculate the primary label from `display.used_tokens`, `display.current_window_tokens`, and `display.usage_ratio`, then render separate detail rows. When `display.historical_limit_changed` is true, render the historical value with `（旧配置记录）`; when no historical value exists, render `暂无记录`. Add a caption explaining that this is the latest request rather than cumulative conversation or VRAM usage.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_context_usage.py tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py -q`

Expected: all focused tests pass.

---

### Task 3: Full verification

**Files:**
- Verify only; no production changes expected.

**Interfaces:**
- Consumes: completed Tasks 1–2
- Produces: fresh evidence for correctness and regression safety

- [ ] **Step 1: Run the full test suite**

Run: `python3 -m pytest`

Expected: zero failures.

- [ ] **Step 2: Run the offline teaching evaluation**

Run: `python3 scripts/run_eval.py`

Expected: `38/38` cases pass.

- [ ] **Step 3: Run syntax and patch checks**

Run: `python3 -m py_compile prototypes/llm_led_teacher/context_usage.py prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_context_usage.py`

Run: `git diff --check -- prototypes/llm_led_teacher/context_usage.py prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_context_usage.py tests/prototypes/test_llm_led_streaming.py`

Expected: both commands exit 0 with no diagnostics.

- [ ] **Step 4: Review the final diff**

Confirm that no conversation database, historical episode JSON, or unrelated file is changed by this implementation.
