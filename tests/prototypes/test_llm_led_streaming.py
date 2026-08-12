from pathlib import Path
from types import SimpleNamespace

import pytest

from prototypes.llm_led_teacher.live_progress import agent_trace_rows
from prototypes.llm_led_teacher import live_progress
from prototypes.llm_led_teacher.orchestration_models import AgentTraceEvent
from prototypes.llm_led_teacher import streaming
from prototypes.llm_led_teacher.streaming import (
    iter_display_chunks,
    iter_markdown_display_chunks,
)


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


def test_display_math_is_emitted_as_one_closed_chunk():
    chunks = list(
        iter_markdown_display_chunks(
            "说明\n\n$$x=\\frac{1}{2}$$\n\n结束", delay_seconds=0
        )
    )

    assert "$$x=\\frac{1}{2}$$" in chunks
    assert all(chunk.count("$$") in (0, 2) for chunk in chunks)
    assert "".join(chunks) == "说明\n\n$$x=\\frac{1}{2}$$\n\n结束"


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
        "st.write_stream(iter_markdown_display_chunks(turn.teacher_output.response))"
    )

    assert (
        "from prototypes.llm_led_teacher.streaming import iter_markdown_display_chunks"
        in source
    )
    assert "progress_callback=update_progress" in source
    assert run_turn_call in source
    assert final_stream_call in source
    assert source.index(run_turn_call) < source.index(final_stream_call)
    assert "st.write_stream(iter_markdown_display_chunks(draft" not in source
    assert "st.write_stream(iter_markdown_display_chunks(output" not in source
    assert "st.markdown(turn.teacher_output.response)" in source


def test_current_turn_details_render_immediately_after_streaming():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()
    expected_current_turn_block = (
        "st.write_stream(iter_markdown_display_chunks(turn.teacher_output.response))\n"
        "            turn.display_seconds = round("
    )
    immediate_details_call = "            render_turn_details(turn)"

    assert "def render_turn_details(" in source
    assert expected_current_turn_block in source
    assert source.index(expected_current_turn_block) < source.rindex(
        immediate_details_call
    )


def test_live_progress_trace_shows_agent_and_answer_guardrail_stages():
    from prototypes.llm_led_teacher.live_progress import LiveProgressTrace

    class FakeStatus:
        def __init__(self):
            self.lines = []

        def write(self, value):
            self.lines.append(value)

    status = FakeStatus()
    clock = iter([10.0, 12.5, 15.0, 19.0, 20.5, 23.0]).__next__
    trace = LiveProgressTrace(status, clock=clock)

    trace.update("generating")
    trace.update("boundary_check")
    trace.update("repairing")
    trace.update("formatting")
    trace.update("complete")

    rendered = "\n".join(status.lines)
    assert "Agent · 请求分类" in rendered
    assert "Answer Guardrails · 生成回答" in rendered
    assert "2.500 秒" in rendered
    assert "Answer Guardrails · 边界检查" in rendered
    assert "Answer Guardrails · 修正" in rendered
    assert "Answer Guardrails · Markdown 格式化" in rendered
    assert "完成" in rendered


def test_streamlit_uses_expanded_live_progress_trace():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()

    assert "LiveProgressTrace," in source
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
    assert "build_context_usage_display(" in source
    assert "with st.bottom:" in source
    assert 'key="context_usage_control"' in source
    assert "st-key-context_usage_control" in source
    assert "context_usage_slot" not in source
    assert "position-anchor" not in source
    assert "本轮请求占用" in source
    assert "当前模型上限" in source
    assert "该对话历史上限" in source
    assert "旧配置记录" in source
    assert 'metadata.get("context_window_tokens", configured_window)' not in source
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


def test_streamlit_persists_public_teaching_boundary_and_real_agent_trace():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()

    assert "teaching_contract" in source
    assert "agent_trace" in source
    assert "boundary_audit" in source
    assert "iter_markdown_display_chunks" in source
    assert "观察器原始输出" not in source
    assert "processing_steps = [" not in source
    assert "可选确定性验证" in source
    assert "VerificationStatus.NOT_APPLICABLE" in source
    assert source.count("render_context_usage(episode)") == 1
    assert source.count("render_turn_token_usage(turn)") == 2


def test_agent_trace_rows_render_every_public_event_without_private_details() -> None:
    turn = SimpleNamespace(
        agent_trace=[
            AgentTraceEvent(
                step=1,
                event_type="tool_result",
                tool_name="load_skill",
                ok=True,
                elapsed_seconds=0.125,
                summary="已加载苏格拉底引导技能",
                details={"private_prompt": "never render this"},
            ),
            AgentTraceEvent(
                step=2,
                event_type="tool_result",
                tool_name="search_local_knowledge",
                ok=True,
                elapsed_seconds=0.75,
                summary="本地教材命中 4 条",
                details={"raw_documents": ["private corpus excerpt"]},
            ),
            AgentTraceEvent(
                step=3,
                event_type="teaching_contract",
                ok=None,
                summary="教学契约已确定：one_step",
                details={"hidden_reasoning": "never render this either"},
            ),
        ]
    )

    rows = agent_trace_rows(turn)

    assert rows == [
        {
            "步骤": "1",
            "类型": "tool_result",
            "Tool": "load_skill",
            "状态": "成功",
            "耗时": "0.125 秒",
            "摘要": "已加载苏格拉底引导技能",
        },
        {
            "步骤": "2",
            "类型": "tool_result",
            "Tool": "search_local_knowledge",
            "状态": "成功",
            "耗时": "0.750 秒",
            "摘要": "本地教材命中 4 条",
        },
        {
            "步骤": "3",
            "类型": "teaching_contract",
            "Tool": "—",
            "状态": "信息",
            "耗时": "—",
            "摘要": "教学契约已确定：one_step",
        },
    ]
    assert "never render" not in repr(rows)
    assert "private corpus excerpt" not in repr(rows)


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
    assert "search\\_local\\_knowledge" in rendered
    assert "must never render" not in rendered


def test_escape_markdown_text_neutralizes_links_images_and_headings() -> None:
    malicious = "[点击](https://evil.test) ![](https://evil.test/x) # 标题"

    escaped = live_progress.escape_markdown_text(malicious)

    assert "[点击](" not in escaped
    assert "![](" not in escaped
    assert "\\[点击\\]" in escaped
    assert "\\!\\[\\]" in escaped
    assert "\\# 标题" in escaped


def test_live_progress_escapes_every_dynamic_agent_event_field() -> None:
    from prototypes.llm_led_teacher.live_progress import LiveProgressTrace

    class FakeStatus:
        def __init__(self):
            self.lines = []

        def write(self, value):
            self.lines.append(value)

    malicious = "[点击](https://evil.test) ![](https://evil.test/x) # 标题"
    status = FakeStatus()
    trace = LiveProgressTrace(status, clock=lambda: 1.0)

    trace.update_agent_event(
        AgentTraceEvent(
            step=2,
            event_type=malicious,
            summary=malicious,
            tool_name=malicious,
            ok=True,
        )
    )

    rendered = "\n".join(status.lines)
    assert rendered.count("\\[点击\\]") == 3
    assert "[点击](" not in rendered
    assert "![](" not in rendered
    assert rendered.count("\\# 标题") == 3


def test_agent_event_progress_callback_escapes_unknown_event_type_label() -> None:
    from prototypes.llm_led_teacher.live_progress import LiveProgressTrace

    class FakeStatus:
        def __init__(self):
            self.lines = []
            self.updates = []

        def write(self, value):
            self.lines.append(value)

        def update(self, **kwargs):
            self.updates.append(kwargs)

    malicious = "[点击](https://evil.test)![](https://evil.test/x)# 标题"
    status = FakeStatus()
    trace = LiveProgressTrace(status, clock=lambda: 1.0)
    event = AgentTraceEvent(
        step=2,
        event_type=malicious,
        summary="公开摘要",
    )

    try:
        live_progress.update_agent_progress(status, trace, event)
    except AttributeError:
        pytest.fail("update_agent_progress must own the Streamlit status callback")

    label = status.updates[-1]["label"]
    assert label == f"Agent · {live_progress.escape_markdown_text(malicious)}"
    assert "[点击](" not in label
    assert "![](" not in label
    assert "\\# 标题" in label
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()
    assert "update_agent_progress(progress_status, live_trace, stage)" in source


def test_persisted_trace_table_keeps_untrusted_text_as_plain_cell_values() -> None:
    malicious = "[点击](https://evil.test) ![](https://evil.test/x) # 标题"
    turn = SimpleNamespace(
        agent_trace=[
            AgentTraceEvent(
                step=1,
                event_type=malicious,
                summary=malicious,
                tool_name=malicious,
            )
        ]
    )

    rows = agent_trace_rows(turn)

    assert rows[0]["类型"] == malicious
    assert rows[0]["Tool"] == malicious
    assert rows[0]["摘要"] == malicious
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()
    assert "st.table(trace_rows)" in source
