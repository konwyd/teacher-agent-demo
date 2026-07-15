import pytest

import app.agent.teaching_planner as teaching_planner
from app.agent.teaching_actions import SafetyConstraints, TeachingAction, TeachingActionPlan
from app.agent.teaching_planner import PlannerInput, TeachingPlanner
from app.llm.deepseek_client import DeepSeekError


pytestmark = pytest.mark.safety_contract


def _input(actions: list[TeachingAction], **updates) -> PlannerInput:
    base = PlannerInput(
        problem_context="2x+3=7",
        current_student_request="我不知道下一步",
        constraints=SafetyConstraints(allowed_actions=actions),
    )
    return base.model_copy(update=updates)


def test_single_candidate_skips_llm() -> None:
    result = TeachingPlanner().plan(_input([TeachingAction.ELICIT]))
    assert result.plan.action is TeachingAction.ELICIT
    assert result.used_llm is False
    assert result.reason == "single_allowed_action"


def test_invalid_model_action_falls_back_inside_allowed_set() -> None:
    class Client:
        def chat_json(self, **kwargs):
            return TeachingActionPlan(action=TeachingAction.MODEL)

    result = TeachingPlanner(client=Client()).plan(
        _input([TeachingAction.FOCUS, TeachingAction.SCAFFOLD])
    )
    assert result.plan.action in {TeachingAction.FOCUS, TeachingAction.SCAFFOLD}
    assert result.reason == "planner_output_rejected"


def test_correct_requires_verified_error_evidence() -> None:
    result = TeachingPlanner().validate_or_fallback(
        TeachingActionPlan(action=TeachingAction.CORRECT),
        _input([TeachingAction.CORRECT, TeachingAction.FOCUS], has_verified_error=False),
    )
    assert result.plan.action is TeachingAction.FOCUS


def test_deepseek_client_construction_failure_falls_back(monkeypatch) -> None:
    class BrokenDeepSeekClient:
        def __init__(self):
            raise DeepSeekError("invalid planner configuration")

    monkeypatch.setattr(teaching_planner, "is_configured", lambda: True)
    monkeypatch.setattr(teaching_planner, "DeepSeekClient", BrokenDeepSeekClient)

    result = TeachingPlanner().plan(
        _input([TeachingAction.FOCUS, TeachingAction.SCAFFOLD])
    )

    assert result.plan.action is TeachingAction.FOCUS
    assert result.used_llm is True
    assert result.reason == "planner_call_failed"
