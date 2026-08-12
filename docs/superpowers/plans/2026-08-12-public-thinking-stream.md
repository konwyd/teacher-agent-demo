# Public Thinking Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persisted, real-time, DeepSeek-style public thinking panel to the 8502 teacher UI using only whitelisted Agent, retrieval, verification, and audit events.

**Architecture:** Introduce a focused `public_thinking.py` boundary that owns the serializable event contract and deterministic whitelist mapping. `run_turn()` records and emits those events through a new callback without changing the legacy progress callback, then Streamlit renders the same event model live and from conversation history.

**Tech Stack:** Python 3.11, Pydantic v2, Streamlit, pytest, existing `PrototypeTurnRecord` conversation serialization.

## Global Constraints

- Do not retain or render DeepSeek `reasoning_content`, system prompts, full model inputs, unaudited drafts, full RAG text, API keys, stack traces, or arbitrary Agent `details` payloads.
- Do not modify model selection, temperature, context window, reasoning budget, RAG policy, web fallback policy, math checker, teacher prompt, or final-answer audit rules.
- Keep the legacy `progress_callback` stage contract intact for existing callers and tests.
- Keep final-answer streaming after `run_turn()` returns so only audited text reaches the user.
- New persisted fields must default safely so old conversations continue to load.
- All tests and evaluation remain offline; do not call a real model or paid API.

---

## File Structure

- Create `prototypes/llm_led_teacher/public_thinking.py`: serializable event enums/model, deterministic whitelist recorder, and display formatting helpers.
- Modify `prototypes/llm_led_teacher/logic.py`: capture public events at real execution boundaries, emit them through a dedicated callback, and persist them on each turn.
- Modify `prototypes/llm_led_teacher/live_progress.py`: render public events in a live Streamlit-like status panel without technical Agent labels or payloads.
- Modify `prototypes/llm_led_teacher/streamlit_app.py`: show the live panel, persist the newly generated expanded state across the immediate rerun, and render historical panels collapsed.
- Create `tests/prototypes/test_public_thinking.py`: unit tests for the event contract, whitelist mapper, sequencing, formatting, and sensitive-data exclusion.
- Modify `tests/prototypes/test_llm_led_turn_contract.py`: integration tests for real event order, callback isolation, persistence, repair, and old-record compatibility.
- Modify `tests/prototypes/test_llm_led_streaming.py`: live/history UI source-contract and panel-renderer tests.

---

### Task 1: Public event contract and whitelist recorder

**Files:**
- Create: `prototypes/llm_led_teacher/public_thinking.py`
- Create: `tests/prototypes/test_public_thinking.py`

**Interfaces:**
- Consumes: `AgentTraceEvent` from `prototypes.llm_led_teacher.orchestration_models` and `VerificationResult` / `VerificationStatus` from `app.agent.state`.
- Produces: `PublicThinkingKind`, `PublicThinkingStatus`, `PublicThinkingEvent`, `PublicThinkingRecorder`, and `format_public_thinking_event(event: PublicThinkingEvent) -> str`.
- `PublicThinkingRecorder(callback: Callable[[PublicThinkingEvent], None] | None = None)` exposes `events: list[PublicThinkingEvent]`, `stage(name: str) -> None`, `agent(event: AgentTraceEvent) -> None`, `verification(result: VerificationResult) -> None`, `retrieval(*, source: str, hit_count: int, online_triggered: bool, online_status: str) -> None`, `audit(*, repair_performed: bool, remaining_warning_count: int) -> None`, and `complete(elapsed_seconds: float) -> None`.

- [ ] **Step 1: Write failing model and sequence tests**

```python
from prototypes.llm_led_teacher.public_thinking import (
    PublicThinkingKind,
    PublicThinkingRecorder,
    PublicThinkingStatus,
)


def test_recorder_assigns_strict_sequence_and_emits_callback() -> None:
    emitted = []
    recorder = PublicThinkingRecorder(callback=emitted.append)

    recorder.stage("observing")
    recorder.stage("retrieving")
    recorder.complete(2.5)

    assert [event.sequence for event in recorder.events] == [1, 2, 3]
    assert emitted == recorder.events
    assert recorder.events[0].kind is PublicThinkingKind.CLASSIFICATION
    assert recorder.events[-1].status is PublicThinkingStatus.COMPLETED
    assert recorder.events[-1].elapsed_seconds == 2.5
```

- [ ] **Step 2: Write failing whitelist and formatting tests**

```python
from prototypes.llm_led_teacher.orchestration_models import AgentTraceEvent
from prototypes.llm_led_teacher.public_thinking import (
    PublicThinkingRecorder,
    format_public_thinking_event,
)


def test_agent_mapper_uses_only_whitelisted_fields() -> None:
    recorder = PublicThinkingRecorder()
    recorder.agent(
        AgentTraceEvent(
            step=2,
            event_type="tool_result",
            summary="arbitrary model reason containing SECRET",
            tool_name="search_local_knowledge",
            ok=True,
            elapsed_seconds=0.25,
            details={
                "hit_count": 4,
                "reasoning_content": "SECRET COT",
                "system_prompt": "SECRET PROMPT",
            },
        )
    )

    dumped = repr(recorder.events)
    assert "找到 4 条可用的本地参考" in dumped
    assert "SECRET" not in dumped
    assert "reasoning_content" not in dumped
    assert "system_prompt" not in dumped
    assert "Agent 2" not in format_public_thinking_event(recorder.events[0])
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `python3 -m pytest tests/prototypes/test_public_thinking.py -q`

Expected: FAIL during import because `public_thinking.py` does not exist.

- [ ] **Step 4: Implement the minimal public event module**

Implement enums with exact persisted values and a Pydantic event model:

```python
class PublicThinkingKind(str, Enum):
    CLASSIFICATION = "classification"
    STRATEGY = "strategy"
    SKILL = "skill"
    RETRIEVAL = "retrieval"
    VERIFICATION = "verification"
    GENERATION = "generation"
    AUDIT = "audit"
    REPAIR = "repair"
    COMPLETE = "complete"
    WARNING = "warning"


class PublicThinkingStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    WARNING = "warning"


class PublicThinkingEvent(BaseModel):
    sequence: int = Field(ge=1)
    kind: PublicThinkingKind
    status: PublicThinkingStatus
    summary: str = Field(min_length=1, max_length=240)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
```

Implement `_append()` as the only constructor path so sequence numbers cannot diverge. Catch callback exceptions after appending so UI failures never alter business state.

Map stable stage names to fixed copy:

```python
_STAGE_COPY = {
    "observing": (PublicThinkingKind.CLASSIFICATION, "正在识别题目和学生意图……"),
    "retrieving": (PublicThinkingKind.RETRIEVAL, "正在查找可用的教材与题目参考……"),
    "generating": (PublicThinkingKind.GENERATION, "正在根据题目、学生情况和参考资料组织回答……"),
    "auditing": (PublicThinkingKind.AUDIT, "正在核验解题内容和教学节奏……"),
    "repairing": (PublicThinkingKind.REPAIR, "发现需要调整的内容，正在进行安全修正……"),
}
```

For `AgentTraceEvent`, ignore `event.summary` entirely. Use only `event_type`, an allowlisted `tool_name`, `ok`, `elapsed_seconds`, and explicit scalar allowlist entries such as `hit_count` from `details`. Unknown events produce no public event. Map classification from `details` only when `scope`, `subject`, `stage`, and `is_new_problem` have expected scalar types; otherwise emit the generic “题目类型已经确认。”.

Implement fixed tool copy for `load_skill`, `inspect_student_context`, `search_local_knowledge`, and `search_web`; never render raw tool arguments or error strings. Implement verification, final retrieval, audit, and completion methods using typed/scalar parameters instead of passing arbitrary dictionaries.

Format status icons only:

```python
def format_public_thinking_event(event: PublicThinkingEvent) -> str:
    icon = {
        PublicThinkingStatus.IN_PROGRESS: "◦",
        PublicThinkingStatus.COMPLETED: "✓",
        PublicThinkingStatus.WARNING: "⚠️",
    }[event.status]
    elapsed = f" · {event.elapsed_seconds:.3f} 秒" if event.elapsed_seconds > 0 else ""
    return f"{icon} {event.summary}{elapsed}"
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_public_thinking.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add prototypes/llm_led_teacher/public_thinking.py tests/prototypes/test_public_thinking.py
git commit -m "feat: add public thinking event contract"
```

---

### Task 2: Record and persist real turn events

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py:53-98`
- Modify: `prototypes/llm_led_teacher/logic.py:500-727`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py:100-170`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py:225-340`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py:490-550`

**Interfaces:**
- Consumes: `PublicThinkingEvent` and `PublicThinkingRecorder` from Task 1.
- Produces: `PrototypeTurnRecord.public_thinking_events: list[PublicThinkingEvent]` and new optional `thinking_callback: Callable[[PublicThinkingEvent], None]` on `run_turn()`.
- Preserves: existing `progress_callback: Callable[[str | AgentTraceEvent], None]` ordering and best-effort behavior.

- [ ] **Step 1: Write failing persistence and compatibility tests**

```python
def test_turn_persists_public_thinking_without_raw_reasoning(monkeypatch) -> None:
    _no_rag(monkeypatch)
    emitted = []
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
    )

    turn = run_turn(
        PrototypeEpisode(),
        "2x+9=14，这道题怎么解",
        client=client,
        thinking_callback=emitted.append,
    )

    assert emitted == turn.public_thinking_events
    assert [event.sequence for event in emitted] == list(range(1, len(emitted) + 1))
    assert emitted[-1].summary == "最终回答已经通过教学内容与节奏审计。"
    assert "reasoning_content" not in turn.model_dump_json()


def test_old_turn_without_public_thinking_events_still_loads() -> None:
    payload = {
        "turn_id": 1,
        "student_message": "旧问题",
        "teacher_output": {"response": "旧回答"},
        "student_verification": {
            "status": "unable_to_verify",
            "summary": "旧记录没有公开思考字段。",
            "verifier_name": "none",
        },
        "evidence_level": "unverified",
    }
    restored = PrototypeTurnRecord.model_validate(payload)
    assert restored.public_thinking_events == []
```

- [ ] **Step 2: Write failing repair, verification, and callback-isolation tests**

Add assertions that a repair path contains a `repair` event, an unsupported verifier produces a warning rather than a success claim, and a `thinking_callback` that raises still leaves the persisted event list complete and returns the audited teacher response.

Use explicit assertions:

```python
assert any(event.kind.value == "repair" for event in turn.public_thinking_events)
assert any(
    event.kind.value == "verification" and event.status.value == "warning"
    for event in turn.public_thinking_events
)
assert turn.public_thinking_events[-1].kind.value == "complete"
```

- [ ] **Step 3: Run turn-contract tests and verify RED**

Run: `python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q`

Expected: FAIL because the new field and callback do not exist.

- [ ] **Step 4: Add the schema field and dedicated callback**

Import the Task 1 types and add:

```python
public_thinking_events: list[PublicThinkingEvent] = Field(default_factory=list)

ThinkingCallback = Callable[[PublicThinkingEvent], None]
```

Extend `run_turn(..., thinking_callback: ThinkingCallback | None = None, ...)` without changing the existing progress callback signature.

- [ ] **Step 5: Wire the recorder to actual execution boundaries**

Create `thinking = PublicThinkingRecorder(callback=thinking_callback)` after timing initialization. Emit `thinking.stage("observing")` beside the existing observing progress report. Update `enter_stage()` to call both the legacy progress callback and `thinking.stage(stage)`.

Wrap Agent callbacks without exposing them directly:

```python
def report_agent_event(event: AgentTraceEvent) -> None:
    _report_progress(progress_callback, event)
    thinking.agent(event)
```

Pass `report_agent_event` to the orchestrator. After it returns, feed only trace events not already observed through the recorder so test orchestrators and old adapters still produce persisted public events without duplicating live events.

After verification and retrieval are known, call typed recorder methods. After `remaining_warnings` is known, call `thinking.audit(...)`. Once `backend_total_seconds` is calculated, call `thinking.complete(backend_total_seconds)` before constructing the record and assign `public_thinking_events=thinking.events`.

The final completion copy must depend only on audited state: use a completed event when `remaining_warnings` is empty and a warning event when warnings remain.

- [ ] **Step 6: Run focused integration tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_public_thinking.py tests/prototypes/test_llm_led_turn_contract.py -q`

Expected: PASS, including unchanged legacy stage-order tests.

- [ ] **Step 7: Commit Task 2**

```bash
git add prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_turn_contract.py
git commit -m "feat: persist public thinking events"
```

---

### Task 3: DeepSeek-style live and historical Streamlit panels

**Files:**
- Modify: `prototypes/llm_led_teacher/live_progress.py:1-90`
- Modify: `prototypes/llm_led_teacher/streamlit_app.py:541-647`
- Modify: `tests/prototypes/test_llm_led_streaming.py:31-180`

**Interfaces:**
- Consumes: `PublicThinkingEvent` and `format_public_thinking_event()` from Task 1; `PrototypeTurnRecord.public_thinking_events` from Task 2.
- Produces: `LiveThinkingPanel.append(event)`, `LiveThinkingPanel.complete(total_seconds)`, and `render_public_thinking(turn, *, expanded: bool = False)`.
- Preserves: final audited `st.write_stream(iter_display_chunks(turn.teacher_output.response))`, token metrics, validation details, pending-message retry, and immediate rerun.

- [ ] **Step 1: Replace technical live-trace tests with failing public-panel tests**

```python
def test_live_thinking_panel_renders_public_copy_and_updates_elapsed_label() -> None:
    from prototypes.llm_led_teacher.live_progress import LiveThinkingPanel
    from prototypes.llm_led_teacher.public_thinking import (
        PublicThinkingEvent,
        PublicThinkingKind,
        PublicThinkingStatus,
    )

    status = FakeStatus()
    clock = iter([10.0, 12.4]).__next__
    panel = LiveThinkingPanel(status, clock=clock)
    panel.append(PublicThinkingEvent(
        sequence=1,
        kind=PublicThinkingKind.RETRIEVAL,
        status=PublicThinkingStatus.COMPLETED,
        summary="找到 4 条可用的本地参考。",
    ))
    panel.complete(6.8)

    assert status.lines == ["✓ 找到 4 条可用的本地参考。"]
    assert status.updates[0]["label"] == "🧠 思考中…（已用时 2 秒）"
    assert status.updates[-1] == {
        "label": "🧠 已思考（用时 7 秒）",
        "state": "complete",
        "expanded": True,
    }
```

`FakeStatus.update(**kwargs)` must append each update dictionary. Add a failure test for `panel.fail()` that expects `state="error"` and does not render exception contents.

- [ ] **Step 2: Write failing Streamlit source-contract tests**

Assert the page:

```python
assert "thinking_callback=live_thinking.append" in source
assert "render_public_thinking(turn, expanded=" in source
assert '_JUST_GENERATED_TURN_KEY = "public_thinking_just_generated_turn_id"' in source
assert "st.session_state.pop(_JUST_GENERATED_TURN_KEY, None)" in source
assert "st.session_state[_JUST_GENERATED_TURN_KEY] = turn.turn_id" in source
assert "AgentTraceEvent" not in source
assert "Observer 结论" not in source
assert "st.write_stream(iter_display_chunks(turn.teacher_output.response))" in source
assert source.index("turn = run_turn(") < source.index(
    "st.write_stream(iter_display_chunks(turn.teacher_output.response))"
)
```

Also assert that empty `public_thinking_events` returns before opening an expander and that the history label uses `turn.backend_total_seconds` rounded to whole seconds.

- [ ] **Step 3: Run UI-focused tests and verify RED**

Run: `python3 -m pytest tests/prototypes/test_llm_led_streaming.py -q`

Expected: FAIL because `LiveThinkingPanel` and historical rendering do not exist.

- [ ] **Step 4: Implement the focused live panel**

Replace technical rendering in `live_progress.py` with:

```python
class LiveThinkingPanel:
    def __init__(self, status: StatusWriter, *, clock=perf_counter) -> None:
        self._status = status
        self._clock = clock
        self._started_at = clock()

    def append(self, event: PublicThinkingEvent) -> None:
        self._status.write(format_public_thinking_event(event))
        elapsed = max(round(self._clock() - self._started_at), 0)
        self._status.update(label=f"🧠 思考中…（已用时 {elapsed} 秒）")

    def complete(self, total_seconds: float) -> None:
        self._status.update(
            label=f"🧠 已思考（用时 {max(round(total_seconds), 0)} 秒）",
            state="complete",
            expanded=True,
        )

    def fail(self) -> None:
        self._status.update(label="🧠 思考中断", state="error", expanded=True)
```

Use a protocol that includes both `write()` and `update()`.

- [ ] **Step 5: Render saved events and preserve one-rerun expansion**

In `streamlit_app.py`, add:

```python
_JUST_GENERATED_TURN_KEY = "public_thinking_just_generated_turn_id"


def render_public_thinking(turn: PrototypeTurnRecord, *, expanded: bool = False) -> None:
    if not turn.public_thinking_events:
        return
    seconds = max(round(turn.backend_total_seconds), 0)
    with st.expander(f"🧠 已思考（用时 {seconds} 秒）", expanded=expanded):
        for event in turn.public_thinking_events:
            st.markdown(format_public_thinking_event(event))
```

Before the history loop, pop the just-generated turn id once. Render the thinking panel before each historical final answer, setting `expanded=True` only for that id.

For the current turn, create `st.status("🧠 思考中…（已用时 0 秒）", expanded=True)`, construct `LiveThinkingPanel`, and pass `thinking_callback=live_thinking.append` to `run_turn()`. Keep the legacy progress callback either disconnected from visible technical logs or as a no-op needed by tests; never render raw `AgentTraceEvent`.

On failure call `live_thinking.fail()`. On success call `live_thinking.complete(turn.backend_total_seconds)`, remove the existing Observer/RAG/audit debug writes, stream the audited response, preserve token/details rendering, set `st.session_state[_JUST_GENERATED_TURN_KEY] = turn.turn_id`, then call the existing `st.rerun()`.

- [ ] **Step 6: Run UI and auth regression tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add prototypes/llm_led_teacher/live_progress.py prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_llm_led_streaming.py
git commit -m "feat: show DeepSeek-style public thinking"
```

---

### Task 4: Security, serialization, and full regression verification

**Files:**
- Modify only if a real defect is found: files changed in Tasks 1-3 and their tests.

**Interfaces:**
- Consumes: completed feature from Tasks 1-3.
- Produces: verified offline feature with unchanged teaching/evaluation behavior.

- [ ] **Step 1: Run the focused public-thinking suite**

Run:

```bash
python3 -m pytest \
  tests/prototypes/test_public_thinking.py \
  tests/prototypes/test_llm_led_turn_contract.py \
  tests/prototypes/test_llm_led_streaming.py \
  tests/prototypes/test_llm_led_auth_ui.py -q
```

Expected: PASS.

- [ ] **Step 2: Scan the persisted and rendered paths for forbidden raw reasoning**

Run:

```bash
rg -n "reasoning_content|chain_of_thought|system_prompt|secret_prompt" \
  prototypes/llm_led_teacher/public_thinking.py \
  prototypes/llm_led_teacher/live_progress.py \
  prototypes/llm_led_teacher/streamlit_app.py
```

Expected: no matches. The low-level DeepSeek client may still detect and discard `reasoning_content`; it is intentionally outside this rendering-path scan.

- [ ] **Step 3: Run syntax and patch hygiene checks**

Run:

```bash
python3 -m py_compile \
  prototypes/llm_led_teacher/public_thinking.py \
  prototypes/llm_led_teacher/live_progress.py \
  prototypes/llm_led_teacher/logic.py \
  prototypes/llm_led_teacher/streamlit_app.py
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 4: Run the full offline pytest suite**

Run: `python3 -m pytest`

Expected: PASS with no real model or network calls.

- [ ] **Step 5: Run the mandatory offline evaluation**

Run: `TEACHER_FORCE_MOCK=1 python3 scripts/run_eval.py`

Expected: all configured golden and smoke cases pass; no real model or network calls occur.

- [ ] **Step 6: Review the final diff against the design**

Run:

```bash
git diff --stat HEAD~3..HEAD
git log --oneline -4
```

Confirm that changes are limited to the public event contract, turn persistence/callback, 8502 rendering, tests, and the already committed design/plan documents.

- [ ] **Step 7: Commit any verification-only fixes**

If Steps 1-6 exposed a real implementation defect, stage only the affected files and commit:

```bash
git add \
  prototypes/llm_led_teacher/public_thinking.py \
  prototypes/llm_led_teacher/live_progress.py \
  prototypes/llm_led_teacher/logic.py \
  prototypes/llm_led_teacher/streamlit_app.py \
  tests/prototypes/test_public_thinking.py \
  tests/prototypes/test_llm_led_turn_contract.py \
  tests/prototypes/test_llm_led_streaming.py
git commit -m "fix: harden public thinking stream"
```

If no defect was found, do not create an empty commit.
