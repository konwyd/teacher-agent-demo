from prototypes.llm_led_teacher.thinking_state import (
    begin_thinking,
    complete_thinking,
    fail_thinking,
    set_user_expanded,
    thinking_key,
)


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
    assert thinking_key("conv_a", 1) != thinking_key("conv_a", 2)


def test_failure_stays_expanded() -> None:
    failed = fail_thinking(begin_thinking(None))

    assert failed.phase == "failed"
    assert failed.expanded is True


def test_user_expansion_after_completion_survives_completion_reruns() -> None:
    completed = complete_thinking(begin_thinking(None))
    user_expanded = set_user_expanded(completed, True)

    assert user_expanded.expanded is True
    assert complete_thinking(user_expanded) == user_expanded
