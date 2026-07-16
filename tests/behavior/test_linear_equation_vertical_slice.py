import pytest
from fastapi.testclient import TestClient

from app.agent.runtime_router import (
    RuntimePath,
    resolve_runtime_path,
    unified_linear_slice_enabled,
)
from app.agent.state import GenericMathReasoning, VerificationStatus
from app.agent.teaching_actions import TeachingAction
from app.agent.workflow import run_teaching_turn
from app.api.main import app
from app.memory.schemas import EpisodeState


pytestmark = pytest.mark.teaching_behavior


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        (" TRUE ", True),
        ("1", False),
        ("yes", False),
        ("on", False),
        ("", False),
        ("false", False),
    ],
)
def test_linear_slice_flag_requires_explicit_true(
    monkeypatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("TEACHER_UNIFIED_LINEAR_SLICE", value)

    assert unified_linear_slice_enabled() is expected


def test_malformed_persisted_path_falls_back_to_existing_generic_owner(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TEACHER_UNIFIED_LINEAR_SLICE", "true")
    episode = EpisodeState(
        conversation_id="corrupt-path",
        student_id="corrupt-path-student",
        problem_context="老师，2x+3=7怎么解？",
        orchestration_path="corrupt",
    )

    route = resolve_runtime_path(
        episode.student_id,
        "我得到2x=4",
        episode.conversation_id,
        episode=episode,
    )

    assert route is RuntimePath.GENERIC


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


def test_flagged_api_projects_verified_step_to_legacy_response(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_UNIFIED_LINEAR_SLICE", "true")
    client = TestClient(app)
    conversation_id = "flagged-api-verification"
    student_id = "flagged-api-student"

    first = client.post(
        "/turn",
        json={
            "student_id": student_id,
            "conversation_id": conversation_id,
            "message": "老师，2x+3=7怎么解？",
        },
    )
    second = client.post(
        "/turn",
        json={
            "student_id": student_id,
            "conversation_id": conversation_id,
            "message": "我得到2x=4",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["student_step_validity"] == ""
    data = second.json()
    assert data["teaching_state"]["verification"]["status"] == "verified_correct"
    assert data["student_step_validity"] == "correct_step"


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


class _EquationSystemReasoner:
    def analyze(self, message, problem_context, analysis):
        return GenericMathReasoning(
            problem_interpretation="需要消去一个未知数。",
            next_step_hint="先观察两式中 y 的系数关系。",
            candidate_conclusion="x=4, y=1/2",
            confidence=0.95,
        )


class _PairLeakingResponseClient:
    def chat(self, system, user, temperature=0.2, max_tokens=None):
        return "计算可得 x=4, y=1/2。你能说明先消去哪个未知数吗？"


def _normalize_pair(value: str) -> str:
    return "".join(value.replace("，", ",").split()).lower()


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
            reasoner=_EquationSystemReasoner(),
            response_client=_PairLeakingResponseClient(),
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
    assert all("x=4,y=1/2" not in _normalize_pair(state.response) for state in states)
