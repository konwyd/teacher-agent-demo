# DeepSeek-Style Conversation UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DeepSeek-style conversation interface whose new-chat, history switching, persistence, and follow-up behavior cannot mix or silently replace conversation context.

**Architecture:** Keep Streamlit and the existing Teacher Agent, but replace the selectbox-driven dual-ID state with one authoritative nullable conversation ID. Put history display semantics in a pure module, keep persistence explicitly targeted by ID, and cover the real Streamlit rerun behavior with `AppTest` plus temporary SQLite databases.

**Tech Stack:** Python 3.10+, Streamlit, Pydantic, SQLite, pytest, Streamlit `AppTest`.

## Global Constraints

- Keep Streamlit, current model calls, RAG, public thinking, verification details, token display, and the in-progress context-capacity meter.
- `prototype_conversation_id: str | None` is the only authoritative active-conversation target; do not reintroduce `conversation_selector`.
- A new-chat draft must not create a SQLite row until its first student message is submitted.
- Every save must receive an explicit immutable target conversation ID and remain scoped by `user_id + conversation_id`.
- Main UI technical information belongs under a collapsed `调试详情` section.
- Do not alter Teacher Agent prompts, teaching policy, RAG decisions, math verification, or evaluation behavior.
- Do not add React/Vue, conversation search, sharing, manual rename, or per-conversation deletion.
- Do not expose private chain-of-thought.
- Do not stage or commit Chroma/vector-index runtime file changes.
- Existing conversation cleanup is a separate one-time destructive operation; never add automatic startup deletion.

---

### Task 1: Preserve and checkpoint the existing context-capacity meter

**Files:**
- Existing uncommitted: `prototypes/llm_led_teacher/context_usage.py`
- Existing uncommitted: `tests/prototypes/test_context_usage.py`
- Existing uncommitted portions: `prototypes/llm_led_teacher/streamlit_app.py:21,271-339,572-582`
- Existing uncommitted: `docs/superpowers/plans/2026-08-12-stable-current-context-meter.md`
- Test: `tests/prototypes/test_llm_led_streaming.py`

**Interfaces:**
- Consumes: last-turn model metadata and `DEEPSEEK_CONTEXT_WINDOW_TOKENS`.
- Produces: `ContextUsageDisplay` and the bottom input-area context-capacity popover that later UI work must preserve.

- [ ] **Step 1: Run the focused context-meter tests before touching shared UI code**

Run:

```bash
python3 -m pytest tests/prototypes/test_context_usage.py tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py -q
```

Expected: all tests pass. If not, stop and repair this pre-existing work without changing its documented semantics.

- [ ] **Step 2: Check syntax and patch hygiene for only the context-meter files**

Run:

```bash
python3 -m py_compile prototypes/llm_led_teacher/context_usage.py prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_context_usage.py
git diff --check -- prototypes/llm_led_teacher/context_usage.py prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_context_usage.py tests/prototypes/test_llm_led_streaming.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Commit the context-meter checkpoint without runtime vector files**

```bash
git add docs/superpowers/plans/2026-08-12-stable-current-context-meter.md prototypes/llm_led_teacher/context_usage.py prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_context_usage.py tests/prototypes/test_llm_led_streaming.py
git commit -m "feat: stabilize current context usage meter"
```

Before committing, inspect `git diff --cached --stat` and confirm no path under `data/china_textbook_k9_vectordb/` is staged.

---

### Task 2: Add pure conversation-history display semantics

**Files:**
- Create: `prototypes/llm_led_teacher/conversation_ui.py`
- Create: `tests/prototypes/test_conversation_ui.py`

**Interfaces:**
- Consumes: `ConversationRecord`, episode dictionaries, and an aware UTC `datetime` supplied by the caller.
- Produces:
  - `ConversationHistoryItem(conversation_id: str, title: str, group: str, updated_at: str)`
  - `episode_has_content(episode: Mapping[str, object]) -> bool`
  - `derive_conversation_title(record: ConversationRecord) -> str`
  - `build_conversation_history(records: Sequence[ConversationRecord], *, now: datetime) -> list[ConversationHistoryItem]`

- [ ] **Step 1: Write failing tests for empty filtering, fallback titles, ordering, and date groups**

Add literal fixtures and assertions such as:

```python
def test_history_omits_empty_records_and_uses_first_student_message_as_title():
    records = [
        record("conv_empty", "新对话", episode={"turns": [], "pending_student_message": ""}, hours_ago=0),
        record(
            "conv_math",
            "新对话",
            episode={"turns": [{"student_message": "  杨辉三角   是什么？  "}]},
            hours_ago=2,
        ),
    ]

    items = build_conversation_history(records, now=NOW)

    assert [(item.conversation_id, item.title, item.group) for item in items] == [
        ("conv_math", "杨辉三角 是什么？", "今天")
    ]


def test_pending_message_is_visible_and_history_groups_use_fixed_boundaries():
    records = [
        record("today", "今天", pending="今天", days_ago=0),
        record("week", "一周", pending="一周", days_ago=3),
        record("month", "一月", pending="一月", days_ago=12),
        record("older", "更早", pending="更早", days_ago=40),
    ]

    items = build_conversation_history(records, now=NOW)

    assert [(item.conversation_id, item.group) for item in items] == [
        ("today", "今天"),
        ("week", "7 天内"),
        ("month", "30 天内"),
        ("older", "更早"),
    ]
```

Use real `ConversationRecord` instances. Derive all expected titles and groups as literals rather than calling production helpers in assertions.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
python3 -m pytest tests/prototypes/test_conversation_ui.py -q
```

Expected: test collection fails because `prototypes.llm_led_teacher.conversation_ui` does not exist.

- [ ] **Step 3: Implement the immutable history item and pure functions**

Implement the following behavior:

```python
@dataclass(frozen=True)
class ConversationHistoryItem:
    conversation_id: str
    title: str
    group: str
    updated_at: str


def episode_has_content(episode: Mapping[str, object]) -> bool:
    turns = episode.get("turns") or []
    pending = str(episode.get("pending_student_message") or "").strip()
    return bool(turns or pending)
```

`derive_conversation_title()` must retain a meaningful stored title, replace empty/`新对话` with the first turn's `student_message` or pending message, normalize whitespace, and return at most 80 characters. `build_conversation_history()` must omit empty records, retain store order, parse ISO timestamps, and assign the four exact group labels.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/prototypes/test_conversation_ui.py -q
```

Expected: all new tests pass.

- [ ] **Step 5: Commit the pure display model**

```bash
git add prototypes/llm_led_teacher/conversation_ui.py tests/prototypes/test_conversation_ui.py
git commit -m "feat: add conversation history display model"
```

---

### Task 3: Replace dual-ID selection with a stable draft/current state machine

**Files:**
- Modify: `prototypes/llm_led_teacher/streamlit_app.py:29-120,173-230,550-630`
- Modify: `tests/prototypes/test_llm_led_auth_ui.py`

**Interfaces:**
- Consumes: `episode_has_content()`, `derive_conversation_title()`, `build_conversation_history()`, `LocalConversationStore`, `PrototypeEpisode`.
- Produces:
  - `_activate_draft(user_id: str) -> None`
  - `_activate_conversation(record: ConversationRecord) -> None`
  - `_activate_conversation_by_id(conversation_id: str) -> None`
  - `_restore_current_or_latest(user_id: str) -> None`
  - `_ensure_target_conversation(episode: PrototypeEpisode, first_message: str) -> str`
  - `_save_conversation(target_conversation_id: str, episode: PrototypeEpisode, *, fallback_title: str = "") -> None`
  - `ConversationStateError`

- [ ] **Step 1: Add an AppTest regression test for the observed rerun bug**

The production mutation this test catches is reintroducing a second selector-owned ID or recomputing current state from “latest” during ordinary reruns.

```python
def test_new_chat_remains_an_unpersisted_draft_across_reruns(...):
    app, account, conversation_db = logged_in_app_with_history(..., messages=["你好"])
    old_id = app.session_state.prototype_conversation_id

    next(button for button in app.button if button.label == "开启新对话").click().run()
    assert app.session_state.prototype_conversation_id is None
    assert app.session_state.prototype_episode.turns == []
    assert len(LocalConversationStore(conversation_db).list_for_user(account.user_id)) == 1

    app.run()
    app.run()

    assert app.session_state.prototype_conversation_id is None
    assert app.session_state.prototype_episode.turns == []
    assert old_id != app.session_state.prototype_conversation_id
```

- [ ] **Step 2: Add an AppTest regression test for first-message isolation and stable follow-up ID**

Use a local successful `run_turn` fake that appends real `PrototypeTurnRecord` values but performs no network call. Submit `数学有什么非常毒的定理吗`, capture the new ID, submit `我想让你给我讲一下`, then assert:

```python
assert new_id is not None
assert app.session_state.prototype_conversation_id == new_id
saved = LocalConversationStore(conversation_db).load(account.user_id, new_id)
assert [turn["student_message"] for turn in saved.episode["turns"]] == [
    "数学有什么非常毒的定理吗",
    "我想让你给我讲一下",
]
old = LocalConversationStore(conversation_db).load(account.user_id, old_id)
assert [turn["student_message"] for turn in old.episode["turns"]] == ["你好"]
```

- [ ] **Step 3: Add tests for history switching and hot-reload preservation**

Create two real stored episodes, click the history button belonging to the non-active conversation, and assert the activity ID and rendered user messages both change together. Set `prototype_session_schema_version` to the previous value and run again; assert the same current ID remains active rather than the newest record being selected.

- [ ] **Step 4: Run the new AppTest cases and verify RED**

Run each newly added test explicitly:

```bash
python3 -m pytest \
  tests/prototypes/test_llm_led_auth_ui.py::test_new_chat_remains_an_unpersisted_draft_across_reruns \
  tests/prototypes/test_llm_led_auth_ui.py::test_first_message_and_follow_up_stay_isolated_in_the_new_conversation \
  tests/prototypes/test_llm_led_auth_ui.py::test_history_switch_and_schema_reload_preserve_one_authoritative_id \
  -q
```

Expected: failures demonstrate that current code creates an empty row, jumps back on rerun, or lacks direct history buttons.

- [ ] **Step 5: Implement draft activation and remove selector state**

Increment `_SESSION_SCHEMA_VERSION`. Remove all reads and writes of `conversation_selector`. `_activate_draft()` must set a new empty `PrototypeEpisode`, set `_PENDING_STUDENT_MESSAGE_KEY` to `""`, set `prototype_conversation_id` to `None`, and preserve authenticated user/schema fields.

`_restore_current_or_latest()` must follow this order:

```python
current_id = st.session_state.get("prototype_conversation_id")
if current_id:
    try:
        _activate_conversation(store.load(user_id, current_id))
        return
    except ConversationArchiveError:
        pass
for record in _visible_conversation_records(user_id):
    _activate_conversation(record)
    return
_activate_draft(user_id)
```

Ordinary reruns with the current schema must not call this function at all.

- [ ] **Step 6: Implement explicit target creation, saving, and mismatch protection**

Add:

```python
class ConversationStateError(RuntimeError):
    pass


def _assert_active_target(target_conversation_id: str) -> None:
    if st.session_state.get("prototype_conversation_id") != target_conversation_id:
        raise ConversationStateError("会话状态已变化，请重试。")
```

For a draft, `_ensure_target_conversation()` creates one record with the current episode payload and `title=first_message`, stores its ID as active, and returns it. For an existing conversation it returns the active ID unchanged. `_save_conversation()` accepts the explicit target ID, calls `_assert_active_target()` immediately before writing, and derives the title from the first turn/pending/fallback message.

In the send path, set pending first, call `_ensure_target_conversation()`, assign the result once to `target_conversation_id`, save pending to that ID, run the agent, then save completion to that same ID. Catch `ConversationStateError` with `ConversationArchiveError` and stop rather than selecting another record.

- [ ] **Step 7: Run the regression tests and existing auth/UI suite**

Run:

```bash
python3 -m pytest tests/prototypes/test_llm_led_auth_ui.py tests/prototypes/test_conversation_ui.py -q
```

Expected: all tests pass, including interrupted generation and public-thinking expansion behavior.

- [ ] **Step 8: Commit the stable state machine**

```bash
git add prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_llm_led_auth_ui.py
git commit -m "fix: keep conversation context on one stable id"
```

---

### Task 4: Render a DeepSeek-style history sidebar and clean chat surface

**Files:**
- Modify: `prototypes/llm_led_teacher/streamlit_app.py:32,188-250,271-339,550-630`
- Modify: `tests/prototypes/test_llm_led_auth_ui.py`

**Interfaces:**
- Consumes: history items and stable callbacks from Tasks 2–3.
- Produces: direct history buttons, current-item highlight, grouped headings, collapsed debug details, centered chat column, right-aligned user bubbles, and preserved bottom context meter.

- [ ] **Step 1: Write an AppTest for the user-visible sidebar contract**

Create conversations whose updated timestamps land in the four groups and assert observable UI behavior:

```python
assert any(button.label == "开启新对话" for button in app.button)
assert not app.selectbox
assert {heading.value for heading in app.subheader} >= {"今天", "7 天内", "30 天内", "更早"}
assert any(expander.label == "调试详情" for expander in app.expander)
assert any(button.label == "杨辉三角是什么" for button in app.button)
```

Click the `杨辉三角是什么` history button and assert the active ID and rendered message correspond to that record. Do not assert CSS source text.

- [ ] **Step 2: Run the UI test and verify RED**

Run:

```bash
python3 -m pytest tests/prototypes/test_llm_led_auth_ui.py::test_sidebar_renders_grouped_direct_history_and_collapsed_debug_details -q
```

Expected: failure because the current page still exposes a selectbox and technical fields directly.

- [ ] **Step 3: Implement grouped direct history buttons**

Load visible records, call `build_conversation_history()`, emit each non-empty group heading once, and render one button per item with a unique key containing the conversation ID. Use `type="primary"` for the active ID and `type="tertiary"` otherwise. Callback arguments must contain the exact conversation ID; labels are display-only and never identify records.

- [ ] **Step 4: Move technical information into `调试详情`**

Inside a collapsed expander render student ID, current conversation ID or `尚未保存`, RAG state, active problem, and turn count. Keep username and logout visible near the sidebar bottom. The main sidebar above it contains only new-chat and history navigation.

- [ ] **Step 5: Apply scoped presentation CSS and main layout**

Add one `_DEEPSEEK_STYLE` block that:

- sets the sidebar to a light neutral background;
- reduces history-button chrome and gives the active primary button a pale-blue treatment;
- limits the main content column to a readable centered width;
- right-aligns user chat messages and uses a light rounded bubble;
- leaves assistant output open and borderless;
- keeps the bottom input and context meter usable on desktop and narrower screens.

CSS may style Streamlit DOM but must not encode conversation IDs or control state. Retain `_CONTEXT_USAGE_STYLE` behavior.

- [ ] **Step 6: Run UI, thinking, and context-meter tests**

Run:

```bash
python3 -m pytest \
  tests/prototypes/test_llm_led_auth_ui.py \
  tests/prototypes/test_llm_led_streaming.py \
  tests/prototypes/test_public_thinking.py \
  tests/prototypes/test_context_usage.py \
  -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit the DeepSeek-style UI**

```bash
git add prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_llm_led_auth_ui.py
git commit -m "feat: add DeepSeek-style conversation navigation"
```

---

### Task 5: Full regression and live-page verification

**Files:**
- Verify only; production changes occur only if a failing check identifies a real defect.

**Interfaces:**
- Consumes: completed conversation UI and state machine.
- Produces: fresh automated and live evidence that no context mixing, regression, or startup failure remains.

- [ ] **Step 1: Run syntax and patch checks**

```bash
python3 -m py_compile \
  prototypes/llm_led_teacher/conversation_ui.py \
  prototypes/llm_led_teacher/context_usage.py \
  prototypes/llm_led_teacher/streamlit_app.py \
  tests/prototypes/test_conversation_ui.py \
  tests/prototypes/test_llm_led_auth_ui.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 2: Run the full test suite**

```bash
python3 -m pytest
```

Expected: zero failures.

- [ ] **Step 3: Run the offline teaching evaluation**

```bash
TEACHER_FORCE_MOCK=1 python3 scripts/run_eval.py
```

Expected: 38/38 cases pass.

- [ ] **Step 4: Restart the one-click service to avoid Streamlit hot-reload module skew**

Send SIGINT to the exact parent `python3 scripts/run_llm_led_teacher.py` process after confirming its PID and children. Then start:

```bash
python3 scripts/run_llm_led_teacher.py
```

Wait until the model health check, Chroma, RAG warmup, and Streamlit 8502 startup all report ready.

- [ ] **Step 5: Verify the live endpoint and import contract**

```bash
curl -sS -I --max-time 5 http://127.0.0.1:8502
python3 -c "from prototypes.llm_led_teacher.conversation_ui import build_conversation_history; print(build_conversation_history.__name__)"
```

Expected: HTTP 200 and `build_conversation_history` printed.

- [ ] **Step 6: Verify repository scope**

Run `git status --short` and confirm only Chroma/vector-index runtime files remain modified. No SQLite runtime database, account data, `.env`, secret, or model artifact may be staged or committed.

- [ ] **Step 7: Perform the separately authorized history cleanup**

Only after destructive-command approval is available, count the exact rows in `data/runtime/conversations.sqlite3`, delete rows only from the `conversations` table, commit the SQLite transaction, and recount to prove zero remain. Do not delete the database file or any account data. If approval remains unavailable, stop and give the user the exact pending status; do not add automatic deletion code or use an indirect workaround.
