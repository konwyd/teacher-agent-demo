import pytest

from app.agent.state import GenericMathReasoning
from app.agent.teaching_actions import TeachingAction, TeachingActionPlan
from app.agent.teaching_planner import TeachingPlanner
from app.agent.workflow import run_teaching_turn


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
