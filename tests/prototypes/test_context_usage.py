from pathlib import Path

from prototypes.llm_led_teacher.context_usage import (
    build_context_usage_display,
    build_kv_persistence_display,
)


def test_context_control_stays_in_composer_flow_before_send_area():
    source = Path("prototypes/llm_led_teacher/streamlit_app.py").read_text()
    style_start = source.index('_CONTEXT_USAGE_STYLE = """')
    style_end = source.index('"""', style_start + len('_CONTEXT_USAGE_STYLE = """'))
    context_style = source[style_start:style_end]

    assert "position: absolute" not in context_style
    assert "right: 4.35rem" not in context_style
    assert "transform: translateY" not in context_style
    assert ".st-key-teacher_composer_actions" in context_style
    assert "gap: 12px" in context_style
    assert "@media (max-width: 640px)" in context_style
    assert "gap: 8px" in context_style
    assert 'content: "上下文"' in context_style

    composer_start = source.index(
        'with st.container(key="teacher_composer_actions"):'
    )
    context_control = source.index(
        'with st.container(key="context_usage_control", width="content"):',
        composer_start,
    )
    send_area = source.index("message = st.chat_input(", composer_start)

    assert composer_start < context_control < send_area


def test_current_model_limit_wins_over_historical_conversation_limit():
    display = build_context_usage_display(
        {
            "context_used_tokens": 65_536,
            "context_window_tokens": 524_288,
        },
        current_window_tokens=131_072,
    )

    assert display.used_tokens == 65_536
    assert display.current_window_tokens == 131_072
    assert display.usage_ratio == 0.5
    assert display.historical_window_tokens == 524_288
    assert display.historical_limit_changed is True


def test_missing_history_does_not_invent_historical_limit():
    display = build_context_usage_display({}, current_window_tokens=131_072)

    assert display.used_tokens == 0
    assert display.current_window_tokens == 131_072
    assert display.usage_ratio == 0.0
    assert display.historical_window_tokens is None
    assert display.historical_limit_changed is False


def test_usage_over_current_limit_remains_visible_without_changing_capacity():
    display = build_context_usage_display(
        {
            "context_used_tokens": 216_500,
            "context_window_tokens": 524_288,
        },
        current_window_tokens=131_072,
    )

    assert display.used_tokens == 216_500
    assert display.current_window_tokens == 131_072
    assert display.usage_ratio > 1.0


def test_kv_persistence_display_reads_directory_and_error_as_metadata_keys():
    display = build_kv_persistence_display(
        {
            "kv_persistence_enabled": True,
            "kv_persistence_slot": 0,
            "kv_persistence_restore_status": "restored",
            "kv_persistence_save_status": "saved",
            "kv_persistence_filename": "kv_test.bin",
            "kv_persistence_directory": "/tmp/kv-states",
            "kv_persistence_error": "save failed safely",
        }
    )

    assert display.directory == "/tmp/kv-states"
    assert display.error == "save failed safely"
    assert display.filename == "kv_test.bin"
