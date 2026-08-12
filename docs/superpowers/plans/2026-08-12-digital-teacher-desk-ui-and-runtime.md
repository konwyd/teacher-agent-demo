# Digital Teacher Desk UI and Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved “数字教师书桌” interface and fix context capacity, public-thinking depth/state, and unnecessary slow RAG in the same user request path.

**Architecture:** Keep Streamlit and the existing conversation archive, but extract pure presentation and routing decisions into focused modules. Replace the multi-step free-form retrieval loop with one typed retrieval decision plus deterministic guards, emit a four-section whitelisted public summary, and drive the live thinking panel with a one-way state machine. The UI consumes these contracts without reading private model reasoning or historical conversation state from another conversation ID.

**Tech Stack:** Python 3.11, Streamlit, Pydantic v2, pytest, existing DeepSeek client, existing Chroma/Tavily adapters, CSS.

## Global Constraints

- Preserve the unique `conversation_id` isolation already implemented; no migration or deletion of current conversations.
- Preserve math-checker precedence, original-problem grounding, memory soft constraints, and the full evaluation suite.
- Do not display private chain-of-thought, system prompts, raw `reasoning_content`, API keys, stack traces, or arbitrary model-authored details.
- The local launcher default is exactly `524288`; the UI must display the actual current runtime value rather than a hard-coded “512K”.
- Web retrieval has a hard 12-second tool timeout and may run at most once per turn.
- Ordinary rewriting, polishing, brainstorming, and user-text explanation do not invoke RAG.
- During one request the thought panel transitions once from expanded running state to collapsed completed state; reruns do not reopen it.
- Use lightweight CSS textures only; do not add a new frontend framework or a bitmap page background.
- All code changes follow TDD and end with `python -m pytest` and `python scripts/run_eval.py`.

---

## File Structure

- Create `prototypes/llm_led_teacher/retrieval_policy.py`: typed retrieval modes, decision validation, and deterministic safety guards.
- Create `prototypes/llm_led_teacher/thinking_state.py`: pure per-conversation/per-turn expansion state transitions.
- Create `prototypes/llm_led_teacher/desk_theme.py`: digital-desk CSS and small safe HTML helpers.
- Modify `prototypes/llm_led_teacher/orchestration_models.py`: carry the typed retrieval decision in the orchestration result.
- Modify `prototypes/llm_led_teacher/orchestrator.py`: make one retrieval decision and execute at most the allowed tools.
- Modify `prototypes/llm_led_teacher/orchestration_tools.py`: use separate local/web timeout budgets and expose real tool timings.
- Modify `prototypes/llm_led_teacher/public_thinking.py`: build detailed, four-section, whitelisted summaries.
- Modify `prototypes/llm_led_teacher/live_progress.py`: complete by collapsing exactly once.
- Modify `prototypes/llm_led_teacher/logic.py`: persist retrieval decision and correct timing categories; skip irrelevant repair work.
- Modify `prototypes/llm_led_teacher/streamlit_app.py`: compose the new desk layout and remove absolute-positioned capacity/status rerun hacks.
- Modify `scripts/run_llm_led_teacher.py` and `.env.example`: use and document the actual 512K local context target.
- Extend the existing `tests/prototypes/` test modules; do not create browser-dependent tests for pure behavior.

---

### Task 1: Typed Retrieval Policy and Deterministic Guards

**Files:**
- Create: `prototypes/llm_led_teacher/retrieval_policy.py`
- Modify: `prototypes/llm_led_teacher/orchestration_models.py`
- Test: `tests/prototypes/test_retrieval_policy.py`

**Interfaces:**
- Produces: `RetrievalMode`, `RetrievalDecision`, `apply_retrieval_guards(decision, *, message, classification) -> RetrievalDecision`.
- Consumes: existing `RequestClassification`, `RequestScope`, and `Subject`.

- [ ] **Step 1: Write failing policy tests**

```python
from prototypes.llm_led_teacher.orchestration_models import (
    AnswerMode, RequestClassification, RequestScope, SchoolStage, Subject,
)
from prototypes.llm_led_teacher.retrieval_policy import (
    RetrievalDecision, RetrievalMode, apply_retrieval_guards,
)


def chinese_classification() -> RequestClassification:
    return RequestClassification(
        scope=RequestScope.K9,
        subject=Subject.CHINESE,
        stage=SchoolStage.MIDDLE,
        is_new_problem=True,
        answer_mode=AnswerMode.GUIDED,
        reason="语文写作指导",
    )


def test_rewriting_is_forced_to_no_retrieval() -> None:
    proposed = RetrievalDecision(
        mode=RetrievalMode.LOCAL_THEN_WEB,
        reason="模型提出搜索",
        confidence=0.8,
        needs_current_information=False,
    )
    guarded = apply_retrieval_guards(
        proposed,
        message="把这句话润色成一个有画面感的作文开头",
        classification=chinese_classification(),
    )
    assert guarded.mode is RetrievalMode.NONE
    assert guarded.reason == "这是基于用户文本的创作指导，无需外部资料"


def test_explicit_current_information_may_use_web() -> None:
    proposed = RetrievalDecision(
        mode=RetrievalMode.LOCAL_THEN_WEB,
        reason="需要核对最新信息",
        confidence=0.9,
        needs_current_information=True,
    )
    guarded = apply_retrieval_guards(
        proposed,
        message="查一下今天教育部最新发布的通知",
        classification=chinese_classification(),
    )
    assert guarded.mode is RetrievalMode.LOCAL_THEN_WEB
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `python -m pytest tests/prototypes/test_retrieval_policy.py -q`

Expected: FAIL with `ModuleNotFoundError: prototypes.llm_led_teacher.retrieval_policy`.

- [ ] **Step 3: Implement the typed contract and guards**

```python
class RetrievalMode(str, Enum):
    NONE = "none"
    LOCAL = "local"
    LOCAL_THEN_WEB = "local_then_web"


class RetrievalDecision(BaseModel):
    mode: RetrievalMode
    reason: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_current_information: bool = False


_NO_RAG_PATTERNS = ("润色", "改写", "仿写", "续写", "作文开头", "换一种说法", "头脑风暴")
_CURRENT_PATTERNS = ("今天", "最新", "刚刚", "目前", "发布", "新闻", "通知")


def apply_retrieval_guards(decision, *, message, classification):
    if classification.scope is RequestScope.OFF_TOPIC:
        return RetrievalDecision(mode="none", reason="当前请求无需课程资料", confidence=1.0)
    if any(marker in message for marker in _NO_RAG_PATTERNS):
        return RetrievalDecision(
            mode="none",
            reason="这是基于用户文本的创作指导，无需外部资料",
            confidence=1.0,
        )
    if decision.mode is RetrievalMode.LOCAL_THEN_WEB and not (
        decision.needs_current_information or any(marker in message for marker in _CURRENT_PATTERNS)
    ):
        return decision.model_copy(update={"mode": RetrievalMode.LOCAL})
    return decision
```

Add `retrieval_decision: RetrievalDecision` to `AgentOrchestrationResult`, using a safe default of `mode=none` so old serialized data remains valid.

- [ ] **Step 4: Run policy and schema tests**

Run: `python -m pytest tests/prototypes/test_retrieval_policy.py tests/prototypes/test_llm_led_orchestration_models.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the policy contract**

```bash
git add prototypes/llm_led_teacher/retrieval_policy.py prototypes/llm_led_teacher/orchestration_models.py tests/prototypes/test_retrieval_policy.py tests/prototypes/test_llm_led_orchestration_models.py
git commit -m "feat: add guarded retrieval decisions"
```

---

### Task 2: Replace the Slow Free-Form RAG Loop

**Files:**
- Modify: `prototypes/llm_led_teacher/orchestrator.py`
- Modify: `prototypes/llm_led_teacher/orchestration_tools.py`
- Modify: `tests/prototypes/test_llm_led_orchestrator.py`
- Modify: `tests/prototypes/test_llm_led_orchestration_tools.py`

**Interfaces:**
- Consumes: `RetrievalDecision` and `RetrievalMode` from Task 1.
- Produces: `RetrievalDecisionProvider.decide(payload) -> RetrievalDecision`; `MultisubjectAgentLoop.run(...)` with one classification call, one route call, and at most one local plus one web tool call.

- [ ] **Step 1: Add failing call-budget tests**

Replace the existing `_Decisions` fake with this typed route fake and extend `_Registry` with per-tool counters:

```python
class _Routes:
    def __init__(self, decision: RetrievalDecision) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, payload) -> RetrievalDecision:
        self.calls += 1
        return self.decision


class _Registry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name, arguments, runtime) -> ToolResult:
        self.calls.append((name, arguments))
        if name == "search_local_knowledge":
            runtime.local_search_succeeded = True
        return ToolResult(
            name=name,
            ok=True,
            data={"text": f"{name} evidence", "hits": []},
            elapsed_seconds=0.25,
        )


def test_creative_guidance_skips_all_retrieval_tools() -> None:
    registry = _Registry()
    routes = _Routes(RetrievalDecision(
        mode="local_then_web", reason="模型提出搜索", confidence=0.8,
    ))
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("语文", new=True)),
        retrieval_provider=routes,
        registry=registry,
    )
    result = loop.run("", "把这句话润色成作文开头", "", "")
    assert result.retrieval_decision.mode.value == "none"
    assert routes.calls == 1
    assert registry.calls == []


def test_local_then_web_executes_each_tool_at_most_once() -> None:
    registry = _Registry()
    routes = _Routes(RetrievalDecision(
        mode="local_then_web", reason="需要核对最新政策", confidence=0.9,
        needs_current_information=True,
    ))
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("道法", new=True)),
        retrieval_provider=routes,
        registry=registry,
    )
    result = loop.run("", "查询今天最新教材政策", "", "")
    assert [name for name, _ in registry.calls] == [
        "search_local_knowledge", "search_web"
    ]
    assert [event.tool_name for event in result.trace if event.event_type == "tool_result"] == [
        "search_local_knowledge", "search_web"
    ]
```

Update tool tests to assert `OrchestrationToolRegistry(local_timeout_seconds=3, web_timeout_seconds=12)` passes the correct timeout to `timeout_runner`.

- [ ] **Step 2: Run targeted tests and verify current loop violates the call budget**

Run: `python -m pytest tests/prototypes/test_llm_led_orchestrator.py tests/prototypes/test_llm_led_orchestration_tools.py -q`

Expected: FAIL because the loop still asks `DeepSeekActionProvider` on each iteration and the registry has one shared 45-second timeout.

- [ ] **Step 3: Implement one route decision and bounded execution**

Replace `DeepSeekActionProvider` with a provider whose JSON schema is `RetrievalDecision`. The route prompt must ask only whether external evidence is needed; it must not generate an answer or free-form plan. After `apply_retrieval_guards`:

```python
decision = apply_retrieval_guards(
    self.retrieval_provider.decide(payload),
    message=student_message,
    classification=classification,
)
if decision.mode in {RetrievalMode.LOCAL, RetrievalMode.LOCAL_THEN_WEB}:
    local = self.registry.execute("search_local_knowledge", local_arguments, runtime)
    record(local)
if decision.mode is RetrievalMode.LOCAL_THEN_WEB:
    web = self.registry.execute("search_web", web_arguments, runtime)
    record(web)
```

Do not call the route provider for off-topic requests. Generate deterministic local arguments from classification plus the current student message; use `top_k=5` and both textbook/exercise sources. Set web `evidence_gap` from the public route reason.

Change the registry constructor to:

```python
def __init__(..., local_timeout_seconds: float = 3.0, web_timeout_seconds: float = 12.0):
    self.local_timeout_seconds = local_timeout_seconds
    self.web_timeout_seconds = web_timeout_seconds
```

- [ ] **Step 4: Run orchestration tests**

Run: `python -m pytest tests/prototypes/test_llm_led_orchestrator.py tests/prototypes/test_llm_led_orchestration_tools.py -q`

Expected: PASS; no test permits duplicate tool calls.

- [ ] **Step 5: Commit the bounded orchestration**

```bash
git add prototypes/llm_led_teacher/orchestrator.py prototypes/llm_led_teacher/orchestration_tools.py tests/prototypes/test_llm_led_orchestrator.py tests/prototypes/test_llm_led_orchestration_tools.py
git commit -m "fix: bound retrieval routing and latency"
```

---

### Task 3: Persist Accurate Retrieval and Timing Metadata

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py`

**Interfaces:**
- Consumes: `AgentOrchestrationResult.retrieval_decision` and tool `elapsed_seconds` from Task 2.
- Produces: `PrototypeTurnRecord.retrieval_decision`, timing keys `observing`, `routing`, `local_retrieval`, `web_retrieval`, `generating`, `auditing`, and optional `repairing`.

- [ ] **Step 1: Add failing turn-contract tests**

```python
class _FixedOrchestrator:
    def __init__(self, result: AgentOrchestrationResult) -> None:
        self.result = result

    def run(self, *args, **kwargs) -> AgentOrchestrationResult:
        return self.result


def test_creative_turn_records_no_rag_and_no_repair_call(monkeypatch) -> None:
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.COMPLETE_RESPONSE),
        ["可以从画面、情绪和悬念三个方向来写。你更喜欢哪一种？"],
    )
    orchestration = AgentOrchestrationResult(
        classification=RequestClassification(
            scope="k9", subject="语文", stage="初中", is_new_problem=True,
            answer_mode="guided", reason="作文创作指导",
        ),
        rag_context=RagContext(source="none"),
        retrieval_decision=RetrievalDecision(
            mode="none", reason="创作指导无需外部资料", confidence=1.0,
        ),
    )
    turn = run_turn(
        PrototypeEpisode(),
        "把这句话润色成作文开头",
        client=client,
        orchestrator=_FixedOrchestrator(orchestration),
    )
    assert turn.retrieval_decision.mode.value == "none"
    assert turn.rag_hits == []
    assert "local_retrieval" not in turn.stage_timings_seconds
    assert "web_retrieval" not in turn.stage_timings_seconds
    assert turn.repair_performed is False


def test_tool_timings_are_not_hidden_inside_routing() -> None:
    observations = [
        ToolResult(name="search_local_knowledge", ok=True, elapsed_seconds=1.25),
        ToolResult(name="search_web", ok=True, elapsed_seconds=2.5),
    ]
    timings = build_orchestration_timings(total_seconds=4.5, observations=observations)
    assert timings["routing"] == 0.75
    assert timings["local_retrieval"] == 1.25
    assert timings["web_retrieval"] == 2.5
```

- [ ] **Step 2: Run the contract tests and observe missing fields/categories**

Run: `python -m pytest tests/prototypes/test_llm_led_turn_contract.py -q`

Expected: FAIL because `PrototypeTurnRecord` has no retrieval decision and currently labels the entire orchestration block as `retrieving`.

- [ ] **Step 3: Implement metadata persistence and relevant audits**

Add a defaulted `retrieval_decision` field to `PrototypeTurnRecord`. Add the pure helper `build_orchestration_timings(total_seconds: float, observations: list[ToolResult]) -> dict[str, float]`; map local and web tool names to their own keys and compute `routing = max(total_seconds - sum(tool elapsed), 0.0)` without double-counting.

Restrict `_collect_warnings(..., include_advisory=True)` and `_repair(...)` to responses where deterministic checks produced an actionable warning. A Chinese composition response with no extracted equation and no policy violation must not enter the repair model call. Preserve all current math warning and repair behavior.

- [ ] **Step 4: Run turn, policy, and math regression tests**

Run: `python -m pytest tests/prototypes/test_llm_led_turn_contract.py tests/test_math_checker.py tests/test_flexible_steps.py -q`

Expected: PASS.

- [ ] **Step 5: Commit accurate runtime metadata**

```bash
git add prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_turn_contract.py
git commit -m "fix: report retrieval and model timings accurately"
```

---

### Task 4: Rich Safe Public Thinking Summary

**Files:**
- Modify: `prototypes/llm_led_teacher/public_thinking.py`
- Modify: `tests/prototypes/test_public_thinking.py`

**Interfaces:**
- Consumes: classification scalars, guarded retrieval decision, whitelisted tool counts/timings, verification status, repair flag, and answer structure labels.
- Produces: `PublicThinkingSection` with `kind`, `title`, `summary`, `details`, `status`, and `elapsed_seconds`; `PublicThinkingRecorder` exposes a maximum of four consolidated sections.

- [ ] **Step 1: Write failing summary and privacy tests**

```python
def test_public_summary_contains_four_useful_sections() -> None:
    recorder = PublicThinkingRecorder()
    recorder.classification(subject="语文", stage="未知", is_new_problem=True,
                            intent="把普通问句改成作文开头")
    recorder.strategy("从画面、情绪、悬念三个方向改写")
    recorder.retrieval_decision(mode="none", reason="创作指导无需外部资料")
    recorder.composition("先解释方法，再给三个示例和迁移提示")
    assert [(s.title, s.details) for s in recorder.sections] == [
        ("题意", ["语文新题", "把普通问句改成作文开头"]),
        ("写法", ["从画面、情绪、悬念三个方向改写"]),
        ("资料", ["未使用 RAG", "创作指导无需外部资料"]),
        ("成稿", ["先解释方法，再给三个示例和迁移提示"]),
    ]


def test_public_summary_rejects_private_fields() -> None:
    recorder = PublicThinkingRecorder()
    recorder.agent(AgentTraceEvent(
        step=1, event_type="tool_result", tool_name="search_web", ok=True,
        summary="SECRET", details={"hit_count": 2, "system_prompt": "SECRET", "reasoning_content": "SECRET"},
    ))
    assert "SECRET" not in repr(recorder.sections)
```

- [ ] **Step 2: Run tests and verify the current flat event list is insufficient**

Run: `python -m pytest tests/prototypes/test_public_thinking.py -q`

Expected: FAIL because sections and explicit strategy/composition APIs do not exist.

- [ ] **Step 3: Consolidate public events into four sections**

Keep the existing callback compatibility by emitting a section update when details change. Accept only enumerated subjects/stages, fixed route modes, integer hit counts, public route reasons capped at 120 characters, predefined strategy labels, and predefined answer-structure labels. Never copy `AgentTraceEvent.summary` or arbitrary `details` into the UI.

Define `PublicThinkingSection` in `public_thinking.py` as a Pydantic model with `kind: PublicThinkingKind`, `title: Literal["题意", "写法", "资料", "成稿"]`, `summary: str`, `details: list[str]`, `status: PublicThinkingStatus`, and `elapsed_seconds: float`. Add `sections: list[PublicThinkingSection]` to `PrototypeTurnRecord` with `default_factory=list`; retain `public_thinking_events` as the compatibility field for old saved conversations.

For unsupported data, use neutral copy such as `"处理方式已确定"`; do not expose the raw value. Keep old `PublicThinkingEvent` deserialization working through default fields or a compatibility adapter.

- [ ] **Step 4: Run public-thinking and turn serialization tests**

Run: `python -m pytest tests/prototypes/test_public_thinking.py tests/prototypes/test_llm_led_turn_contract.py -q`

Expected: PASS, including old-conversation loading.

- [ ] **Step 5: Commit the safe detailed summary**

```bash
git add prototypes/llm_led_teacher/public_thinking.py tests/prototypes/test_public_thinking.py tests/prototypes/test_llm_led_turn_contract.py
git commit -m "feat: show detailed public teaching summaries"
```

---

### Task 5: One-Way Thinking Panel State Machine

**Files:**
- Create: `prototypes/llm_led_teacher/thinking_state.py`
- Modify: `prototypes/llm_led_teacher/live_progress.py`
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Modify: `tests/prototypes/test_llm_led_streaming.py`
- Create: `tests/prototypes/test_thinking_state.py`

**Interfaces:**
- Produces: `ThinkingPanelState`, `thinking_key(conversation_id, turn_id)`, `begin_thinking`, `complete_thinking`, `fail_thinking`, and `set_user_expanded`.
- Consumes: stable conversation ID and turn ID; does not inspect an episode from another conversation.

- [ ] **Step 1: Write failing transition tests**

```python
def test_running_updates_never_toggle_expansion() -> None:
    state = begin_thinking(None)
    assert state.phase == "running"
    assert state.expanded is True
    assert begin_thinking(state) == state


def test_completion_collapses_once_and_rerun_is_idempotent() -> None:
    running = begin_thinking(None)
    completed = complete_thinking(running)
    assert completed.phase == "complete"
    assert completed.expanded is False
    assert complete_thinking(completed) == completed


def test_keys_are_isolated_by_conversation_and_turn() -> None:
    assert thinking_key("conv_a", 1) != thinking_key("conv_b", 1)
```

- [ ] **Step 2: Run the state tests and verify the module is absent**

Run: `python -m pytest tests/prototypes/test_thinking_state.py -q`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the pure state reducer and live panel behavior**

Use an immutable Pydantic model or frozen dataclass with phases `idle`, `running`, `complete`, `failed`. `LiveThinkingPanel.append` updates only label/content; `complete` must call:

```python
self._status.update(
    label=f"备课完成 · {seconds}秒",
    state="complete",
    expanded=False,
)
```

Remove `_JUST_GENERATED_TURN_KEY`, its `pop`, and the unconditional `st.rerun()` after the answer. Render the saved answer in the same run. On future natural reruns, historical turns default to completed/collapsed unless the user changed their keyed state.

- [ ] **Step 4: Run state and Streamlit source-contract tests**

Run: `python -m pytest tests/prototypes/test_thinking_state.py tests/prototypes/test_llm_led_streaming.py -q`

Expected: PASS; source tests assert `_JUST_GENERATED_TURN_KEY` is absent, completion uses `expanded=False`, and the success path contains no unconditional `st.rerun()`.

- [ ] **Step 5: Commit the state fix**

```bash
git add prototypes/llm_led_teacher/thinking_state.py prototypes/llm_led_teacher/live_progress.py prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_thinking_state.py tests/prototypes/test_llm_led_streaming.py
git commit -m "fix: make thinking panel transitions stable"
```

---

### Task 6: Real 512K Context and Non-Overlapping Composer

**Files:**
- Modify: `scripts/run_llm_led_teacher.py`
- Modify: `.env.example`
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Modify: `tests/prototypes/test_llm_led_launcher.py`
- Modify: `tests/prototypes/test_llm_led_streaming.py`
- Modify: `tests/prototypes/test_context_usage.py`

**Interfaces:**
- Consumes: `DEEPSEEK_CONTEXT_WINDOW_TOKENS` set by the launcher.
- Produces: default `DEFAULT_CTX_SIZE = 524288`; a composer footer whose context control remains in normal flex/grid flow.

- [ ] **Step 1: Update tests first**

Change the launcher expectation to:

```python
def test_local_launcher_defaults_to_512k_context() -> None:
    assert DEFAULT_CTX_SIZE == 524288
```

Add a source-contract assertion that `_CONTEXT_USAGE_STYLE` contains no `position: absolute`, `right: 4.35rem`, or `transform: translateY`, and that the capacity control is rendered inside the composer-actions container before the send control.

- [ ] **Step 2: Run tests and confirm the old 131K/absolute-position behavior fails**

Run: `python -m pytest tests/prototypes/test_llm_led_launcher.py tests/prototypes/test_context_usage.py tests/prototypes/test_llm_led_streaming.py -q`

Expected: FAIL on `131072` and absolute positioning.

- [ ] **Step 3: Implement actual capacity and flow layout**

Set `DEFAULT_CTX_SIZE = 524288` and update `.env.example` to explain that hosted endpoints may still report 128K. Keep `build_runtime_env(... context_window_tokens=args.ctx_size)` as the single source passed to Streamlit.

Replace the absolute capacity overlay with a normal composer footer row:

```css
.teacher-composer-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  padding-right: 64px;
  min-width: 0;
}
.st-key-context_usage_control { position: static; transform: none; }
@media (max-width: 640px) {
  .teacher-context-long { display: none; }
  .teacher-composer-actions { gap: 8px; padding-right: 56px; }
}
```

If the local model process exits while starting at 524288, preserve and display its failure; do not relabel another capacity as 512K.

- [ ] **Step 4: Run launcher/context/UI contract tests**

Run: `python -m pytest tests/prototypes/test_llm_led_launcher.py tests/prototypes/test_context_usage.py tests/prototypes/test_llm_led_streaming.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the context fix**

```bash
git add scripts/run_llm_led_teacher.py .env.example prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_llm_led_launcher.py tests/prototypes/test_context_usage.py tests/prototypes/test_llm_led_streaming.py
git commit -m "fix: use real 512k context in the composer"
```

---

### Task 7: Build the Digital Teacher Desk Theme

**Files:**
- Create: `prototypes/llm_led_teacher/desk_theme.py`
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Modify: `prototypes/llm_led_teacher/conversation_ui.py`
- Modify: `tests/prototypes/test_conversation_ui.py`
- Modify: `tests/prototypes/test_llm_led_streaming.py`

**Interfaces:**
- Produces: `DESK_THEME_CSS`, `render_brand_header()`, `render_thinking_sections(turn, expanded)`, and semantic CSS class names used by Streamlit composition.
- Consumes: conversation groups, public-thinking sections, context display, and the pure thought state from prior tasks.

- [ ] **Step 1: Add failing visual contract tests**

```python
def test_desk_theme_has_required_responsive_tokens() -> None:
    assert "--desk-paper" in DESK_THEME_CSS
    assert "--desk-ink" in DESK_THEME_CSS
    assert "--desk-vermilion" in DESK_THEME_CSS
    assert "@media (max-width: 760px)" in DESK_THEME_CSS
    assert "background-image: url(" not in DESK_THEME_CSS


def test_sidebar_copy_uses_lesson_language() -> None:
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()
    assert "新建一课" in source
    assert "备课批注" in source
    assert "Deploy" not in source
```

Extend conversation UI tests to ensure current titles and timestamps survive grouping and selection.

- [ ] **Step 2: Run UI contract tests and verify missing theme**

Run: `python -m pytest tests/prototypes/test_conversation_ui.py tests/prototypes/test_llm_led_streaming.py -q`

Expected: FAIL because `desk_theme.py` and the new semantic copy do not exist.

- [ ] **Step 3: Implement the desktop layout**

Use CSS variables and Streamlit keyed containers to build:

- a 288px ivory conversation directory with date-index accents;
- a centered paper-like main sheet with a thin vermilion margin rule;
- a pinned-note user message;
- the left-margin “备课批注” sections with ink labels and status stamps;
- an editorial answer column with readable Chinese line height;
- a two-row bottom composer with normal-flow controls and a separate circular send area.

Hide only Streamlit chrome selectors, never functional application elements. Escape every user- or model-authored string passed to custom HTML with `html.escape`; prefer normal `st.markdown` for answer bodies.

- [ ] **Step 4: Implement responsive degradation**

At `760px`, collapse the directory, remove decorative margin rules, move thinking sections above the answer, and keep the composer full-width. At `390px`, ensure the context short label, toggle controls, and send button fit without horizontal scrolling.

- [ ] **Step 5: Run UI and conversation tests**

Run: `python -m pytest tests/prototypes/test_conversation_ui.py tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the digital desk UI**

```bash
git add prototypes/llm_led_teacher/desk_theme.py prototypes/llm_led_teacher/streamlit_app.py prototypes/llm_led_teacher/conversation_ui.py tests/prototypes/test_conversation_ui.py tests/prototypes/test_llm_led_streaming.py
git commit -m "feat: build the digital teacher desk interface"
```

---

### Task 8: Full Regression and Live Acceptance

**Files:**
- Modify only if a real regression is found; do not weaken assertions.

**Interfaces:**
- Consumes all prior tasks.
- Produces a verified runnable project with evidence for all five user-reported problems.

- [ ] **Step 1: Run the full offline test suite**

Run: `python -m pytest`

Expected: all tests pass with no skipped regression caused by this work.

- [ ] **Step 2: Run the golden evaluation suite**

Run: `python scripts/run_eval.py`

Expected: every golden case passes; math checker precedence and teaching invariants remain unchanged.

- [ ] **Step 3: Start the local stack with the real configured context**

Run: `python scripts/run_llm_led_teacher.py`

Expected: launcher reports `--ctx-size 524288`, Chroma/llama/Streamlit become healthy, and the page returns HTTP 200. If the model cannot allocate 512K, capture the explicit startup error and do not claim success or fake the UI value.

- [ ] **Step 4: Perform the five acceptance scenarios**

1. Create a new conversation, ask a rewriting question, and verify no RAG tool call plus a substantially shorter total time.
2. During generation, verify the thought area remains open continuously; on completion it closes once and stays closed across reruns.
3. Reopen the thought area and verify useful `题意 / 写法 / 资料 / 成稿` content with no private raw reasoning.
4. Verify the context indicator matches the launcher value and never overlaps the send button at desktop and narrow widths.
5. Ask a second follow-up, switch conversations twice, and verify both turns remain in the correct conversation.

- [ ] **Step 5: Inspect the latest persisted metadata**

Use the Python standard library `sqlite3` in read-only diagnostic code to confirm the newest turn records `retrieval_decision`, separated tool timings, `context_window_tokens=524288`, and four-section public summaries. Do not print API keys or raw model reasoning.

- [ ] **Step 6: Close evidence-backed regressions in their owning task**

Run `git status --short`. If no files changed, record that no regression-fix commit is needed. If verification exposed a defect, return to the task that owns that behavior, add a focused failing test beside its existing tests, implement the smallest fix, rerun that task's targeted command plus Steps 1 and 2 above, and commit only that task's explicit files using `fix: close digital teacher desk regression`.
