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
    run_turn_call = "turn = run_turn("
    final_stream_call = (
        "st.write_stream(iter_display_chunks(turn.teacher_output.response))"
    )

    assert "from prototypes.llm_led_teacher.streaming import iter_display_chunks" in source
    assert "progress_callback=update_progress" in source
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


def test_live_progress_trace_appends_work_and_elapsed_time():
    from prototypes.llm_led_teacher.live_progress import LiveProgressTrace

    class FakeStatus:
        def __init__(self):
            self.lines = []

        def write(self, value):
            self.lines.append(value)

    status = FakeStatus()
    clock = iter([10.0, 12.5, 15.0, 19.0]).__next__
    trace = LiveProgressTrace(status, clock=clock)

    trace.update("retrieving")
    trace.update("generating")
    trace.update("complete")

    rendered = "\n".join(status.lines)
    assert "正在识别题目与教学节奏" in rendered
    assert "完成题目与节奏判断" in rendered
    assert "2.500 秒" in rendered
    assert "正在检索教材与题库" in rendered
    assert "正在让本地模型组织教师初稿" in rendered
    assert "4.000 秒" in rendered


def test_streamlit_uses_expanded_live_progress_trace():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()

    assert "from prototypes.llm_led_teacher.live_progress import LiveProgressTrace" in source
    assert "expanded=True" in source
    assert "live_trace = LiveProgressTrace(progress_status)" in source
    assert "live_trace.update(stage)" in source
    assert "progress_status.write(" in source


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
    assert "with st.bottom:" in source
    assert 'key="context_usage_control"' in source
    assert "st-key-context_usage_control" in source
    assert "context_usage_slot" not in source
    assert "position-anchor" not in source
    assert "上下文已使用" in source
    assert "def render_turn_token_usage(" in source
    assert "本轮 KV 缓存命中" in source
    assert "prompt_cache_hit_tokens" in source
    assert "prompt_cache_miss_tokens" in source
    assert "st.progress(" in source
    assert source.count("render_context_usage(episode)") == 1
    assert source.count("render_turn_token_usage(turn)") == 2
    assert "_render_auth_gate()" in source
    assert "_save_active_conversation(episode)" in source
    assert "_get_pending_student_message(episode)" in source
    assert "enable_agent_loop=True" in source


def test_live_progress_renders_public_agent_event_without_private_payload() -> None:
    from prototypes.llm_led_teacher.live_progress import LiveProgressTrace
    from prototypes.llm_led_teacher.orchestration_models import AgentTraceEvent

    class FakeStatus:
        def __init__(self):
            self.lines = []

        def write(self, value):
            self.lines.append(value)

    status = FakeStatus()
    trace = LiveProgressTrace(status, clock=lambda: 1.0)
    trace.update_agent_event(
        AgentTraceEvent(
            step=2,
            event_type="tool_result",
            summary="本地物理教材命中 4 条",
            tool_name="search_local_knowledge",
            ok=True,
            details={"secret_prompt": "must never render"},
        )
    )

    rendered = "\n".join(status.lines)
    assert "本地物理教材命中 4 条" in rendered
    assert "search_local_knowledge" in rendered
    assert "must never render" not in rendered
