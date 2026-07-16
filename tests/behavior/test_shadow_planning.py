import pytest

import app.agent.math_analysis as math_analysis
import app.agent.response_generation as response_generation
import app.agent.teaching_harness as teaching_harness
from app.agent.generic_math_reasoning import GenericMathReasoner
from app.agent.state import GenericMathReasoning, MathAnalysis
from app.agent.teaching_actions import TeachingAction, TeachingActionPlan
from app.agent.teaching_planner import PlannerResult, TeachingPlanner
from app.agent.workflow import run_teaching_turn
from app.llm.deepseek_client import DeepSeekError


pytestmark = pytest.mark.teaching_behavior


class Reasoner:
    def analyze(self, message, problem_context, analysis):
        return GenericMathReasoning(next_step_hint="先整理已知关系。", confidence=0.9)


class PlannerClient:
    def chat_json(self, **kwargs):
        return TeachingActionPlan(action=TeachingAction.EXPLAIN)


class ResponseClient:
    def chat(self, system, user, temperature=0.2, max_tokens=None):
        return "先整理题目中已经明确的关系。你准备先写哪一条关系？"


class BudgetReasoningClient:
    def chat_json(self, **kwargs):
        return GenericMathReasoning(next_step_hint="先整理已知关系。", confidence=0.9)


class MutatingPlanner:
    def __init__(self) -> None:
        self.validation_calls = 0

    def plan(self, planner_input):
        raise AssertionError("LLM planner must be skipped at the call budget")

    def validate_or_fallback(self, proposed_plan, planner_input):
        self.validation_calls += 1
        proposed_plan.scaffold_level = 0
        proposed_plan.action = TeachingAction.EXPLAIN
        planner_input.constraints.allowed_actions = [TeachingAction.EXPLAIN]
        return PlannerResult(
            plan=proposed_plan,
            used_llm=False,
            reason="malicious_mutation",
        )


class ActionSensitiveResponseClient:
    def chat(self, system, user, temperature=0.2, max_tokens=None):
        if "教学动作：explain" in user:
            return "被 shadow 改成了解释动作。你看到了吗？"
        return "主动策略仍然给出一步提示。你准备先写哪一条关系？"


class RecordingAnalysisClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, **kwargs):
        self.calls += 1
        return MathAnalysis()


class FailingResponseClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        raise DeepSeekError("response transport failed")


def test_shadow_plan_is_recorded_but_does_not_control_response() -> None:
    state = run_teaching_turn(
        "shadow-student",
        "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
        reasoner=Reasoner(),
        planner=TeachingPlanner(client=PlannerClient()),
        response_client=ResponseClient(),
        analysis_allow_llm=False,
    )
    assert state.action_plan.action is TeachingAction.SCAFFOLD
    assert state.orchestration_trace.shadow_plan.action is TeachingAction.EXPLAIN
    assert state.orchestration_trace.shadow_mode is True
    assert state.orchestration_trace.active_plan == state.action_plan
    assert state.orchestration_trace.llm_call_count <= 3
    planner_decision = next(
        decision
        for decision in state.orchestration_trace.decisions
        if decision.module == "planner"
    )
    assert planner_decision.reason == "planner_selected_action"
    assert planner_decision.llm_calls == 1
    assert state.response == "先整理题目中已经明确的关系。你准备先写哪一条关系？"


def test_budget_fallback_never_exposes_active_plan_to_injected_planner(
    monkeypatch,
) -> None:
    malicious_planner = MutatingPlanner()
    monkeypatch.setattr(teaching_harness, "is_configured", lambda: True)

    def analyze_with_recorded_call(
        message,
        problem_context,
        *,
        allow_llm,
        on_llm_call,
    ):
        on_llm_call()
        return MathAnalysis()

    monkeypatch.setattr(
        teaching_harness,
        "analyze_turn",
        analyze_with_recorded_call,
    )

    state = run_teaching_turn(
        "shadow-alias-student",
        "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
        reasoner=GenericMathReasoner(client=BudgetReasoningClient()),
        planner=malicious_planner,
        response_client=ActionSensitiveResponseClient(),
    )

    assert malicious_planner.validation_calls == 0
    assert state.action_plan.action is TeachingAction.SCAFFOLD
    assert state.current_phase == TeachingAction.SCAFFOLD.value
    assert state.response == "主动策略仍然给出一步提示。你准备先写哪一条关系？"


def test_analysis_call_is_attributed_to_analysis_not_context_resolution(
    monkeypatch,
) -> None:
    analysis_client = RecordingAnalysisClient()
    monkeypatch.setattr(math_analysis, "is_configured", lambda: True)
    monkeypatch.setattr(math_analysis, "DeepSeekClient", lambda: analysis_client)
    monkeypatch.setattr(teaching_harness, "is_configured", lambda: True)

    state = run_teaching_turn(
        "trace-accounting-student",
        "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
        reasoner=Reasoner(),
        planner=TeachingPlanner(client=PlannerClient()),
        response_client=ResponseClient(),
    )

    context_decision = next(
        decision
        for decision in state.orchestration_trace.decisions
        if decision.module == "context_resolution"
    )
    analysis_decision = next(
        decision
        for decision in state.orchestration_trace.decisions
        if decision.module == "analysis"
    )
    assert analysis_client.calls == 1
    assert context_decision.llm_calls == 0
    assert analysis_decision.called is True
    assert analysis_decision.reason == "analysis_completed"
    assert analysis_decision.llm_calls == 1
    assert state.orchestration_trace.llm_call_count <= 3


def test_response_client_construction_failure_is_zero_llm_calls(
    monkeypatch,
) -> None:
    monkeypatch.setattr(teaching_harness, "is_configured", lambda: True)
    monkeypatch.setattr(response_generation, "is_configured", lambda: True)

    class BrokenResponseClient:
        def __init__(self):
            raise DeepSeekError("invalid response configuration")

    monkeypatch.setattr(response_generation, "DeepSeekClient", BrokenResponseClient)

    state = run_teaching_turn(
        "response-construction-student",
        "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
        reasoner=Reasoner(),
        planner=TeachingPlanner(client=PlannerClient()),
        analysis_allow_llm=False,
    )

    response_decision = next(
        decision
        for decision in state.orchestration_trace.decisions
        if decision.module == "response_generation"
    )
    assert response_decision.llm_calls == 0
    assert state.response.count("？") == 1


def test_failed_response_chat_is_one_llm_call() -> None:
    response_client = FailingResponseClient()

    state = run_teaching_turn(
        "response-chat-student",
        "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
        reasoner=Reasoner(),
        planner=TeachingPlanner(client=PlannerClient()),
        response_client=response_client,
        analysis_allow_llm=False,
    )

    response_decision = next(
        decision
        for decision in state.orchestration_trace.decisions
        if decision.module == "response_generation"
    )
    assert response_client.calls == 1
    assert response_decision.llm_calls == 1
    assert state.response.count("？") == 1
