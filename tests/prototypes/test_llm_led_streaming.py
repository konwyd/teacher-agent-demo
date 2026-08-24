from pathlib import Path

import pytest

from prototypes.llm_led_teacher import streaming
from prototypes.llm_led_teacher.streaming import iter_display_chunks


@pytest.mark.parametrize(
    ("text", "chunk_size"),
    [
        ("", 12),
        ("2x=5", 12),
        ("先做一步：**两边减 9**。\n\n得到 $2x=5$，你来继续？", 7),
        ("$$\n\\frac{x}{2}=3\n$$", 4),
    ],
)
def test_display_chunks_reconstruct_audited_text_exactly(text, chunk_size):
    chunks = list(iter_display_chunks(text, chunk_size=chunk_size))

    assert "".join(chunks) == text
    assert all(0 < len(chunk) <= chunk_size for chunk in chunks)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_display_chunks_reject_non_positive_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        list(iter_display_chunks("教师回复", chunk_size=chunk_size))


def test_default_display_stream_yields_one_character_with_visible_delay(monkeypatch):
    pauses = []
    monkeypatch.setattr(streaming, "_sleep", pauses.append, raising=False)

    chunks = list(
        iter_display_chunks("数学", delay_seconds=0.02)
    )

    assert chunks == ["数", "学"]
    assert pauses == [0.02]


def test_streamlit_streams_only_final_audited_turn_response():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()
    run_turn_call = "return run_turn("
    final_stream_call = (
        "st.write_stream(iter_display_chunks(turn.teacher_output.response))"
    )

    assert "from prototypes.llm_led_teacher.streaming import iter_display_chunks" in source
    assert "thinking_update_callback=emit_thinking" in source
    assert "run_with_event_stream(" in source
    assert "live_thinking.append" in source
    assert run_turn_call in source
    assert final_stream_call in source
    assert source.index(run_turn_call) < source.index(final_stream_call)
    assert "st.write_stream(iter_display_chunks(draft" not in source
    assert "st.write_stream(iter_display_chunks(output" not in source


def test_current_turn_details_render_immediately_after_streaming():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()
    expected_current_turn_block = (
        "st.write_stream(iter_display_chunks(turn.teacher_output.response))\n"
        "            turn.display_seconds = round("
    )
    immediate_details_call = "            render_turn_details(turn)"

    assert "def render_turn_details(" in source
    assert expected_current_turn_block in source
    assert source.index(expected_current_turn_block) < source.rindex(
        immediate_details_call
    )


def test_live_thinking_panel_appends_content_without_rebuilding_header():
    from prototypes.llm_led_teacher.live_progress import LiveThinkingPanel
    from prototypes.llm_led_teacher.public_thinking import (
        PublicThinkingEvent,
        PublicThinkingKind,
        PublicThinkingStatus,
    )

    class FakeStatus:
        def __init__(self):
            self.lines = []
            self.updates = []

        def write(self, value):
            self.lines.append(value)

        def update(self, **kwargs):
            self.updates.append(kwargs)

    status = FakeStatus()
    panel = LiveThinkingPanel(status)
    panel.append(
        PublicThinkingEvent(
            sequence=1,
            kind=PublicThinkingKind.RETRIEVAL,
            status=PublicThinkingStatus.COMPLETED,
            summary="找到 4 条可用的本地参考。",
        )
    )

    assert status.lines == ["✓ 找到 4 条可用的本地参考。"]
    assert status.updates == []

    panel.complete(6.8)

    assert status.updates == [{
        "label": "🧠 AI教学决策完成（用时 7 秒）",
        "state": "complete",
    }]


def test_live_thinking_panel_replaces_one_structured_snapshot_in_place():
    from prototypes.llm_led_teacher.live_progress import LiveThinkingPanel
    from prototypes.llm_led_teacher.public_thinking import (
        PublicThinkingEvent,
        PublicThinkingKind,
        PublicThinkingSection,
        PublicThinkingStatus,
        PublicThinkingUpdate,
        ReasoningTraceEvidence,
        ReasoningTraceSource,
        ReasoningTraceStage,
        ReasoningTraceStageId,
    )

    rendered = []

    class FakeStatus:
        def write(self, value):
            raise AssertionError("structured updates should use the renderer")

        def update(self, **kwargs):
            pass

    panel = LiveThinkingPanel(FakeStatus(), render_update=rendered.append)
    update = PublicThinkingUpdate(
        event=PublicThinkingEvent(
            sequence=1,
            kind=PublicThinkingKind.CLASSIFICATION,
            status=PublicThinkingStatus.COMPLETED,
            summary="已确认题目与作答目标。",
        ),
        sections=[
            PublicThinkingSection(
                kind=PublicThinkingKind.CLASSIFICATION,
                title="题意",
                summary="已确认题目与作答目标。",
                details=["初中数学新题"],
                status=PublicThinkingStatus.COMPLETED,
            )
        ],
        reasoning_trace=[
            ReasoningTraceStage(
                sequence=1,
                stage_id=ReasoningTraceStageId.REQUEST_ANALYSIS,
                stage="问题分析",
                icon="🔍",
                summary="已确认题目与作答目标。",
                content=[
                    ReasoningTraceEvidence(
                        label="学科",
                        value="数学",
                        source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                        source_field="RequestClassification.subject",
                    )
                ],
                status=PublicThinkingStatus.COMPLETED,
            )
        ],
    )

    panel.append(update)

    assert rendered == [update]
    assert rendered[0].reasoning_trace[0].content[0].source_field == (
        "RequestClassification.subject"
    )


def test_live_thinking_panel_completion_updates_status_only_once():
    from prototypes.llm_led_teacher.live_progress import LiveThinkingPanel

    class FakeStatus:
        def __init__(self):
            self.updates = []

        def write(self, value):
            raise AssertionError("complete should not append content")

        def update(self, **kwargs):
            self.updates.append(kwargs)

    status = FakeStatus()
    panel = LiveThinkingPanel(status, clock=lambda: 1.0)

    panel.complete(6.8)
    panel.complete(9.2)

    assert status.updates == [
        {
            "label": "🧠 AI教学决策完成（用时 7 秒）",
            "state": "complete",
        }
    ]


def test_live_timer_runs_in_flat_surface_before_answer_stream():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()
    theme = Path("prototypes/llm_led_teacher/desk_theme.py").read_text()

    assert "desk-live-thinking-marker" not in source
    assert "desk-reasoning-runtime" in source
    assert '.desk-reasoning-runtime[data-state="running"]::after' in theme
    assert "desk-thinking-clock" in theme
    assert "steps(900, end)" in theme
    answer_stream = source.index(
        "st.write_stream(iter_display_chunks(turn.teacher_output.response))"
    )
    completion = source.index("live_thinking.complete(turn.backend_total_seconds)")
    assert completion < answer_stream


def test_live_thinking_panel_failure_never_renders_exception_details():
    from prototypes.llm_led_teacher.live_progress import LiveThinkingPanel

    class FakeStatus:
        def __init__(self):
            self.lines = []
            self.updates = []

        def write(self, value):
            self.lines.append(value)

        def update(self, **kwargs):
            self.updates.append(kwargs)

    status = FakeStatus()
    panel = LiveThinkingPanel(status, clock=lambda: 1.0)

    panel.fail()

    assert status.lines == []
    assert status.updates == [
        {"label": "🧠 AI教学决策中断", "state": "error"}
    ]


def test_streamlit_shows_generation_and_online_fallback_metadata_not_raw_thoughts():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()

    assert "model_generation_metadata" in source
    assert "online_status" in source
    assert "online_reason" in source
    assert "原始逐字思维链不会展示" in source
    assert "reasoning_content" not in source


def test_streamlit_shows_context_occupancy_and_kv_cache_hits():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()

    assert "def render_context_usage(" in source
    assert "build_conversation_context_usage_display(" in source
    assert "with st.bottom:" in source
    assert 'key="context_usage_control"' in source
    assert "st-key-context_usage_control" in source
    assert "context_usage_slot" not in source
    assert "position-anchor" not in source
    assert "当前对话上下文" in source
    assert "并发生成" in source
    assert "当前模型上限" in source
    assert "该对话历史上限" in source
    assert "旧配置记录" in source
    assert 'metadata.get("context_window_tokens", configured_window)' not in source
    assert "def render_turn_token_usage(" in source
    assert "本轮 KV 缓存命中" in source
    assert "prompt_cache_hit_tokens" in source
    assert "prompt_cache_miss_tokens" in source
    assert "KvCacheIdentity(" in source
    assert "authenticated_user[\"user_id\"]" in source
    assert "target_conversation_id" in source
    assert "KV 持久化" in source
    assert "build_kv_persistence_display(metadata)" in source
    assert "st.progress(" in source
    assert source.count("render_context_usage(episode)") == 1
    assert source.count("render_turn_token_usage(turn)") == 2
    assert "_render_auth_gate()" in source
    assert "_save_conversation(" in source
    assert "target_conversation_id" in source
    assert "_get_pending_student_message(episode)" in source
    assert "enable_agent_loop=True" in source


def test_streamlit_uses_keyed_flat_thinking_surface_without_generation_rerun():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()

    assert 'key=f"reasoning_trace_{current_conversation_id}_{turn.turn_id}"' in source
    assert "on_change=_remember_thinking_expansion" not in source
    assert "st.status(" not in source
    assert "begin_thinking(" in source
    assert "complete_thinking(" in source
    assert "fail_thinking(" in source
    assert "_JUST_GENERATED_TURN_KEY" not in source
    assert "public_thinking_just_generated_turn_id" not in source
    assert "st.session_state.pop(_JUST_GENERATED_TURN_KEY" not in source
    assert "st.session_state[_JUST_GENERATED_TURN_KEY]" not in source
    assert "            st.rerun()\n" not in source[source.index("if message:") :]


def test_ui_extra_requires_streamlit_with_keyed_expander_support():
    project = Path("pyproject.toml").read_text()

    assert 'ui = ["streamlit>=1.55", "requests>=2.31"]' in project


def test_digital_teacher_desk_theme_has_paper_ink_and_responsive_contract():
    from prototypes.llm_led_teacher.desk_theme import DESK_THEME_CSS

    assert "--desk-paper" in DESK_THEME_CSS
    assert "--desk-ink" in DESK_THEME_CSS
    assert "--desk-vermilion" in DESK_THEME_CSS
    assert "@media (max-width: 760px)" in DESK_THEME_CSS
    assert "@media (max-width: 390px)" in DESK_THEME_CSS
    assert "background-image: url(" not in DESK_THEME_CSS.lower()
    assert ".desk-directory" in DESK_THEME_CSS
    assert ".desk-paper-sheet" in DESK_THEME_CSS
    assert ".desk-user-note" in DESK_THEME_CSS
    assert ".desk-thinking-rail" in DESK_THEME_CSS
    assert ".desk-answer" in DESK_THEME_CSS
    assert ".desk-composer" in DESK_THEME_CSS
    assert '[data-testid="stHeader"]' in DESK_THEME_CSS
    assert '[data-testid="stBottom"]' in DESK_THEME_CSS
    assert '[data-testid="stExpandSidebarButton"]' in DESK_THEME_CSS
    assert '[data-testid="stToolbar"],\n' not in DESK_THEME_CSS
    assert ".st-key-teacher_composer_controls" in DESK_THEME_CSS
    assert ".desk-auth-page-marker" in DESK_THEME_CSS
    assert ".desk-auth-panel-header" in DESK_THEME_CSS
    assert ".desk-auth-collage" in DESK_THEME_CSS
    assert ".desk-auth-bookmark" in DESK_THEME_CSS
    assert ":has(.desk-auth-page-marker)" in DESK_THEME_CSS
    assert "grid-template-columns" in DESK_THEME_CSS


def test_streamlit_source_composes_the_desk_without_hiding_functional_controls():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()

    assert "from prototypes.llm_led_teacher.desk_theme import" in source
    assert "DESK_THEME_CSS" in source
    assert source.index("st.markdown(DESK_THEME_CSS") < source.index(
        'if "authenticated_user" not in st.session_state:'
    )
    assert "新建一课" in source
    assert "def render_thinking_sections(" in source
    assert "turn.sections" in source
    assert "escape(section.summary)" in source
    assert "escape(detail)" in source
    assert "st.markdown(turn.teacher_output.response)" in source
    assert 'key="teacher_composer_controls"' in source
    assert "Deploy" not in source



def test_event_bridge_delivers_updates_before_background_task_returns():
    from threading import Event

    from prototypes.llm_led_teacher.live_event_stream import run_with_event_stream

    ui_received = Event()
    worker_finished = Event()
    observed = []

    def task(emit):
        emit("题意已确认")
        assert ui_received.wait(timeout=1)
        worker_finished.set()
        return "回答完成"

    def on_event(event):
        assert not worker_finished.is_set()
        observed.append(event)
        ui_received.set()

    result = run_with_event_stream(
        task,
        on_event,
        poll_interval_seconds=0.005,
    )

    assert result == "回答完成"
    assert observed == ["题意已确认"]
    assert worker_finished.is_set()
