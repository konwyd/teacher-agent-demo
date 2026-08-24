# Dynamic Reasoning Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed four-slot public thinking display with a dynamic, source-attributed AI teaching decision trace derived from real Agent runtime facts without exposing private Chain of Thought or adding LLM calls.

**Architecture:** Extend the existing `PublicThinkingRecorder` as a deterministic projection layer: it keeps legacy events/sections for rollback while producing an ordered `reasoning_trace` whose evidence items point to whitelisted runtime fields. Reuse the current queue-based streaming bridge and episode JSON persistence; the UI renders new traces first and falls back to legacy sections for old turns.

**Tech Stack:** Python 3, Pydantic v2, Streamlit, HTML/CSS, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-dynamic-reasoning-trace-design.md`

## Global Constraints

- Active subsystem: Agent Runtime / Product UI; acting owner: 3322.
- 8502 is the only active runtime; do not modify frozen 8501/8000 code.
- Do not expose `reasoning_content`, prompts, drafts, free-form ActionProvider reasons, or private Memory text.
- Do not add an LLM call or change routing, RAG, Memory, verifier, audit, repair, or final-answer semantics.
- Trace stages are dynamic projections of facts that actually occurred; do not create a fixed four-stage workflow.
- Every visible evidence item identifies a whitelisted `source` and exact `source_field`.
- UI copy uses `🧠 AI教学决策过程` / `🧠 Agent Reasoning Trace`; never label it “思维链” or Chain of Thought.
- Keep `public_thinking_events` and `sections` backward compatible; new fields use empty-list defaults.
- Do not commit, merge, push, reset, restore, or clean as part of this plan. Each task ends with targeted tests and a reviewable `git diff` checkpoint.

---

## File Map

- `prototypes/llm_led_teacher/public_thinking.py`: trace schemas, whitelist mappings, dynamic recorder projection, legacy compatibility.
- `prototypes/llm_led_teacher/logic.py`: passes authoritative runtime facts into the recorder and persists `reasoning_trace` on `PrototypeTurnRecord`.
- `prototypes/llm_led_teacher/live_progress.py`: live panel consumes trace snapshots and reports teaching-decision status.
- `prototypes/llm_led_teacher/desk_theme.py`: escaped `<details>` stage renderer and motion-safe CSS.
- `prototypes/llm_led_teacher/streamlit_app.py`: live/history renderer selection and fallback order.
- `tests/prototypes/test_public_thinking.py`: schema, source attribution, dynamic stage, redaction, and deep-copy behavior.
- `tests/prototypes/test_llm_led_turn_contract.py`: persisted turn schema and end-to-end runtime facts.
- `tests/prototypes/test_llm_led_streaming.py`: queue timing, snapshot compatibility, panel state, and dynamic renderer wiring.
- `tests/prototypes/test_llm_led_auth_ui.py`: history rendering, escaping, labels, independent disclosure controls, and legacy fallback.
- `PROGRESS.md`: only the Agent Runtime section, after verification.

---

### Task 1: Add Backward-Compatible Trace Data Models

**Files:**
- Modify: `prototypes/llm_led_teacher/public_thinking.py`
- Modify: `prototypes/llm_led_teacher/logic.py`
- Test: `tests/prototypes/test_public_thinking.py`
- Test: `tests/prototypes/test_llm_led_turn_contract.py`

**Interfaces:**
- Produces: `ReasoningTraceSource`, `ReasoningTraceEvidence`, `ReasoningTraceStageId`, `ReasoningTraceStage`.
- Extends: `PublicThinkingUpdate.reasoning_trace: list[ReasoningTraceStage]` and `PrototypeTurnRecord.reasoning_trace: list[ReasoningTraceStage]`.
- Compatibility: all new list fields use `Field(default_factory=list)`; no existing field changes.

- [ ] **Step 1: Write failing model and old-record compatibility tests**

Add literal expectations that prove evidence is field-attributed and missing fields remain loadable:

```python
def test_reasoning_trace_evidence_names_the_real_decision_field():
    stage = ReasoningTraceStage(
        sequence=1,
        stage_id=ReasoningTraceStageId.REQUEST_ANALYSIS,
        stage="问题分析",
        icon="🔍",
        summary="已识别本轮任务。",
        content=[ReasoningTraceEvidence(
            label="学科",
            value="数学",
            source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
            source_field="RequestClassification.subject",
        )],
        status=PublicThinkingStatus.COMPLETED,
    )
    assert stage.model_dump(mode="json")["content"][0] == {
        "label": "学科", "value": "数学",
        "source": "RequestClassification",
        "source_field": "RequestClassification.subject",
    }
```

Extend the turn-contract fixture without `reasoning_trace` and assert it loads as `[]`.

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
python3 -m pytest tests/prototypes/test_public_thinking.py -q
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
```

Expected: import/field failures for the new trace types and turn field.

- [ ] **Step 3: Implement the minimal schemas**

Add enum-backed sources and bounded evidence/stage models. Extend `PublicThinkingUpdate` and `PrototypeTurnRecord` only with defaulted lists. Do not alter recorder behavior yet.

- [ ] **Step 4: Run targeted tests and review the data-model diff**

```bash
python3 -m pytest tests/prototypes/test_public_thinking.py -q
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
git diff --check
git diff -- prototypes/llm_led_teacher/public_thinking.py prototypes/llm_led_teacher/logic.py tests/prototypes/test_public_thinking.py tests/prototypes/test_llm_led_turn_contract.py
```

Expected: PASS; only additive schema changes and their tests.

---

### Task 2: Project Real Agent Decisions in the Recorder

**Files:**
- Modify: `prototypes/llm_led_teacher/public_thinking.py`
- Test: `tests/prototypes/test_public_thinking.py`

**Interfaces:**
- Consumes: Task 1 trace models.
- Produces: `PublicThinkingRecorder.reasoning_trace`, `_upsert_trace_stage(...)`, and safe projection calls for classification, Memory resolution, retrieval, tools, TeachingContract, verifier, audit, repair, and completion.
- Invariant: legacy `events` and `sections` continue to be populated.

- [ ] **Step 1: Write failing behavior tests for dynamic stages and provenance**

Cover these observable cases with hand-derived stage-id lists:

```python
assert [item.stage_id.value for item in recorder.reasoning_trace] == [
    "request_analysis", "knowledge_decision", "teaching_strategy",
    "answer_generation", "quality_validation",
]
assert all(item.stage_id.value != "tool_execution" for item in recorder.reasoning_trace)
```

Add separate tests proving:

- repeated updates upsert one stage and preserve its first sequence;
- retrieval evidence points to `RetrievalDecision.mode`;
- contract evidence points to `TeachingContract.response_mode` and permission fields;
- verifier stage appears only when deterministic verification is applicable;
- Memory/tool/repair/final-recheck stages appear only after corresponding real events;
- arbitrary `reasoning_content`, prompt text, exception text, and ActionProvider `reason` never appear in `model_dump_json()`.

- [ ] **Step 2: Run recorder tests and confirm RED**

```bash
python3 -m pytest tests/prototypes/test_public_thinking.py -q
```

Expected: no `reasoning_trace` projection/upsert behavior yet.

- [ ] **Step 3: Implement a whitelist registry and upsert primitive**

Define a local stage registry for title/icon and implement `_upsert_trace_stage()` so first appearance assigns `sequence`, subsequent updates retain it, and all incoming content is already typed `ReasoningTraceEvidence`. Never accept arbitrary stage title/icon/source strings from the model.

- [ ] **Step 4: Connect existing recorder methods to safe field projections**

Update `stage()`, `classification()`, `agent()`, `verification()`, `retrieval_decision()`, `composition()`, `audit()`, and `complete()` so they update only stages backed by actual events. Add a structured teaching-contract recorder method. Map enum/boolean/tool-result values through local copy tables; ignore unknown/free-form values.

- [ ] **Step 5: Preserve legacy output and deep-copy update snapshots**

Continue generating existing events/sections. Extend `_append()` to copy each trace stage deeply into `PublicThinkingUpdate`; a captured first update must remain unchanged after a later recorder call.

- [ ] **Step 6: Run tests and review only the recorder diff**

```bash
python3 -m pytest tests/prototypes/test_public_thinking.py -q
git diff --check
git diff -- prototypes/llm_led_teacher/public_thinking.py tests/prototypes/test_public_thinking.py
```

Expected: PASS; no runtime wiring or UI edits in this checkpoint.

---

### Task 3: Keep the Existing Streaming Bridge Compatible

**Files:**
- Modify: `prototypes/llm_led_teacher/live_progress.py`
- Test: `tests/prototypes/test_llm_led_streaming.py`

**Interfaces:**
- Consumes: `PublicThinkingUpdate.reasoning_trace` snapshots.
- Preserves: `LiveThinkingPanel.append(PublicThinkingEvent | PublicThinkingUpdate)`, queue/thread bridge, and fail-open callback behavior.
- Produces: teaching-decision running/completed/failed labels.

- [ ] **Step 1: Write failing stream/panel tests**

Construct a real `PublicThinkingUpdate` with one trace stage and assert the injected render callback receives it before task completion. Assert label behavior uses:

```text
🧠 AI教学决策中…
🧠 AI教学决策完成
🧠 AI教学决策中断
```

Keep the existing “collapse only once after completion” and conversation/turn isolation assertions.

- [ ] **Step 2: Run streaming tests and confirm RED**

```bash
python3 -m pytest tests/prototypes/test_llm_led_streaming.py -q
```

Expected: old public-thinking labels/render expectations fail.

- [ ] **Step 3: Make the minimal panel compatibility change**

Keep the callback and queue shapes unchanged. Update only user-visible status labels and ensure `PublicThinkingUpdate` is forwarded intact; do not simulate token streaming or introduce another thread.

- [ ] **Step 4: Run tests and review the streaming diff**

```bash
python3 -m pytest tests/prototypes/test_llm_led_streaming.py -q
git diff --check
git diff -- prototypes/llm_led_teacher/live_progress.py tests/prototypes/test_llm_led_streaming.py
```

Expected: PASS; no renderer implementation in this checkpoint.

---

### Task 4: Render Dynamic, Escaped, Collapsible Trace Stages

**Files:**
- Modify: `prototypes/llm_led_teacher/desk_theme.py`
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Test: `tests/prototypes/test_llm_led_auth_ui.py`
- Test: `tests/prototypes/test_llm_led_streaming.py`

**Interfaces:**
- Consumes: `list[ReasoningTraceStage]`.
- Produces: `render_reasoning_trace(trace, *, live=False, container=None) -> str`.
- Fallback order: persisted trace, legacy sections, then legacy public events.

- [ ] **Step 1: Write failing UI behavior tests**

Assert rendered HTML contains independent `<details>` elements, `🧠 AI教学决策过程`, and an evidence badge such as `RequestClassification.subject`. Use hostile values like `<script>alert(1)</script>` and assert only escaped output appears. Assert CSS includes stage entrance/status animation and `prefers-reduced-motion: reduce`.

Add history tests for:

- new turn renders `turn.reasoning_trace`;
- old turn with `reasoning_trace=[]` still renders legacy sections;
- live callback renders the current trace snapshot;
- removed caption `备课批注 · 题意 / 写法 / 资料 / 成稿` is absent.

- [ ] **Step 2: Run UI tests and confirm RED**

```bash
python3 -m pytest tests/prototypes/test_llm_led_auth_ui.py -q
python3 -m pytest tests/prototypes/test_llm_led_streaming.py -q
```

Expected: renderer/import/title/fallback assertions fail.

- [ ] **Step 3: Implement the renderer and CSS**

Add `render_reasoning_trace()` using `html.escape()` on stage, summary, evidence label/value/source/source_field. Render one native `<details>` per stage, open the latest live stage and the first history stage by default, show status-specific classes, and add reduced-motion-safe CSS with no new dependency.

- [ ] **Step 4: Wire live and history selection**

In `streamlit_app.py`, render `reasoning_trace` when non-empty. Otherwise call the existing legacy section/event renderer. Preserve existing per-turn outer panel state and do not rerun the backend during history rendering.

- [ ] **Step 5: Run tests and review only the UI diff**

```bash
python3 -m pytest tests/prototypes/test_llm_led_auth_ui.py -q
python3 -m pytest tests/prototypes/test_llm_led_streaming.py -q
git diff --check
git diff -- prototypes/llm_led_teacher/desk_theme.py prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_llm_led_auth_ui.py tests/prototypes/test_llm_led_streaming.py
```

Expected: PASS; HTML is escaped and old histories remain visible.

---

### Task 5: Wire Authoritative Runtime Facts and Persistence

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Test: `tests/prototypes/test_llm_led_turn_contract.py`
- Test: `tests/prototypes/test_public_thinking.py`

**Interfaces:**
- Consumes: recorder structured methods from Task 2.
- Produces: populated `PrototypeTurnRecord.reasoning_trace` using the same runtime objects that drive behavior.
- Sources: `RequestClassification`, retrieval decision, orchestration trace, `TeachingContract`, `VerificationResult`, semantic/hard boundary audit facts, repair flags.

- [ ] **Step 1: Write failing end-to-end turn tests**

Use existing offline turn fixtures and assert exact stage presence/absence for a simple concept turn, an actual retrieval/tool turn, a verifier-applicable continuation, and a repair path. Assert every evidence `source_field` begins with the corresponding real schema name and serialized trace excludes private markers.

- [ ] **Step 2: Run turn tests and confirm RED**

```bash
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
```

Expected: recorder trace is not persisted or lacks structured contract/audit facts.

- [ ] **Step 3: Pass structured facts and persist the trace**

Call recorder methods at existing decision points; do not add orchestration steps. Pass enum/boolean/count/timing fields only. Add `reasoning_trace=[stage.model_copy(deep=True) ...]` to the final turn record. Do not use `AgentDecision.reason` or model text as an explanation source.

- [ ] **Step 4: Run all four focused suites and inspect the integration diff**

```bash
python3 -m pytest tests/prototypes/test_public_thinking.py -q
python3 -m pytest tests/prototypes/test_llm_led_streaming.py -q
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
python3 -m pytest tests/prototypes/test_llm_led_auth_ui.py -q
git diff --check
git diff -- prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_turn_contract.py
```

Expected: PASS; LLM/provider call assertions remain unchanged.

---

### Task 6: Full Verification and Progress Reconstruction

**Files:**
- Modify: `PROGRESS.md` (Agent Runtime section only)

**Interfaces:**
- Consumes: verified implementation and current git/worktree facts.
- Produces: current Agent Runtime progress without architecture-decision duplication.

- [ ] **Step 1: Run required repository gates**

```bash
python3 -m pytest
python3 scripts/run_eval.py
git diff --check
```

Expected: complete pytest pass, evaluation pass, clean whitespace check.

- [ ] **Step 2: Run a local 8502 health/manual smoke if the service remains active**

Check `http://127.0.0.1:8502/_stcore/health`. If available, verify a simple concept question and an evidence/tool question show different dynamic stage subsets. Mark browser-only observations as manual evidence, never as automated test results.

- [ ] **Step 3: Update only Agent Runtime progress**

Record owner 3322, current phase/status, verified completed work, findings, current work/blockers/next, branch/worktree, and materially updated date. Use `UNKNOWN / NEEDS CONFIRMATION` for facts the repository cannot establish. Do not rewrite another subsystem or duplicate `DECISIONS.md`.

- [ ] **Step 4: Final scope and worktree audit**

```bash
git status --short
git diff --name-only
git diff --stat
git diff --check
```

Confirm no production path outside Agent Runtime/Product approved scope was edited, unrelated dirty files remain preserved, and no commit/push/merge occurred.

