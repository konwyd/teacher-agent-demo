import pytest

from app.agent.state import VerificationStatus
from app.agent.teaching_actions import TeachingAction
from app.agent.workflow import run_teaching_turn


pytestmark = pytest.mark.teaching_behavior


def test_linear_equation_slice_uses_new_orchestrator_and_existing_checker(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TEACHER_UNIFIED_LINEAR_SLICE", "true")
    conversation_id = "unified-linear-slice"

    first = run_teaching_turn(
        "linear-slice-student",
        "老师，2x+3=7怎么解？",
        conversation_id=conversation_id,
        analysis_allow_llm=False,
    )
    second = run_teaching_turn(
        "linear-slice-student",
        "我得到2x=4",
        conversation_id=conversation_id,
        analysis_allow_llm=False,
    )

    assert first.episode_state.orchestration_path == "unified_linear_slice"
    assert second.episode_state.orchestration_path == "unified_linear_slice"
    assert second.verification.status is VerificationStatus.VERIFIED_CORRECT
    assert second.verification.verifier_name == "linear_equation_fraction_checker"
    assert second.orchestration_trace.shadow_mode is False
    assert second.orchestration_trace.active_plan == second.action_plan
    assert second.orchestration_trace.shadow_plan is None
    assert second.teaching_brief.action_plan == second.action_plan
    assert second.action_plan.action in {
        TeachingAction.CHECK,
        TeachingAction.EXPLAIN,
        TeachingAction.CONSOLIDATE,
        TeachingAction.PRACTICE,
    }
    assert "x=2" not in first.response
    assert "x=2" not in second.response
    assert first.response.count("？") == 1
    assert second.response.count("？") == 1


def test_linear_slice_flag_off_keeps_legacy_runtime(monkeypatch) -> None:
    monkeypatch.delenv("TEACHER_UNIFIED_LINEAR_SLICE", raising=False)
    state = run_teaching_turn("legacy-student", "我不会解2x+3=7")

    assert state.episode_state.target_equation


def test_existing_legacy_episode_is_not_moved_by_flag(monkeypatch) -> None:
    conversation_id = "existing-legacy-owner"
    run_teaching_turn(
        "legacy-owner",
        "我不会解2x+3=7",
        conversation_id=conversation_id,
    )
    monkeypatch.setenv("TEACHER_UNIFIED_LINEAR_SLICE", "true")
    state = run_teaching_turn(
        "legacy-owner",
        "我得到2x=4",
        conversation_id=conversation_id,
    )

    assert state.episode_state.target_equation


def test_linear_slice_flag_does_not_capture_an_equation_system(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_UNIFIED_LINEAR_SLICE", "true")
    state = run_teaching_turn(
        "equation-system-owner",
        "老师，3x+2y=5，6x-2y=3怎么解？",
        conversation_id="equation-system-owner",
        analysis_allow_llm=False,
    )

    assert state.episode_state.orchestration_path == "generic"
    assert state.verification.status is VerificationStatus.NOT_APPLICABLE


def test_flagged_equation_system_keeps_three_turn_generic_context(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_UNIFIED_LINEAR_SLICE", "true")
    messages = [
        "老师，3x+2y=5，6x-2y=3怎么解？",
        "具体要怎么做呢？",
        "我不知道要怎么选择方法",
    ]

    states = [
        run_teaching_turn(
            "flagged-equation-system-owner",
            message,
            conversation_id="flagged-equation-system-three-turns",
            analysis_allow_llm=False,
        )
        for message in messages
    ]

    assert [state.episode_state.orchestration_path for state in states] == [
        "generic",
        "generic",
        "generic",
    ]
    assert [state.problem_context for state in states] == [messages[0]] * 3
    assert all(
        state.verification.status is VerificationStatus.NOT_APPLICABLE
        for state in states
    )
    assert all(state.response.count("？") == 1 for state in states)
    assert all(
        "x=8/9" not in state.response and "y=7/6" not in state.response
        for state in states
    )
