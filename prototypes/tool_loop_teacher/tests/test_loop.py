from __future__ import annotations

from collections import deque

from app.agent.state import GenericMathReasoning, VerificationStatus
from app.agent.teaching_actions import TeachingAction, TeachingActionPlan
from prototypes.tool_loop_teacher.loop import AgentLoop, DeepSeekActionProvider
from prototypes.tool_loop_teacher.models import DecisionKind, LoopDecision


class ScriptedProvider:
    def __init__(self, *decisions: LoopDecision) -> None:
        self._decisions = deque(decisions)
        self.payloads: list[dict] = []

    def decide(self, payload: dict) -> LoopDecision:
        self.payloads.append(payload)
        return self._decisions.popleft()


class FakeReasoner:
    def analyze(self, message, problem_context, analysis):  # noqa: ANN001, ARG002
        return GenericMathReasoning(
            problem_interpretation=problem_context,
            next_step_hint="先把等式两边同时减去 3。",
            confidence=0.9,
        )


class FakeResponseClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def chat(self, system, user, temperature=0.2, max_tokens=None):  # noqa: ANN001, ARG002
        self.calls += 1
        return self.response


class RecordingJsonClient:
    def __init__(self) -> None:
        self.max_tokens = None

    def chat_json(
        self,
        system,
        user,
        schema,
        temperature=0.0,
        max_tokens=None,
    ):  # noqa: ANN001, ARG002
        self.max_tokens = max_tokens
        return _finish(TeachingAction.FOCUS)


def _finish(action: TeachingAction) -> LoopDecision:
    plan_kwargs = {"action": action}
    if action is TeachingAction.SCAFFOLD:
        plan_kwargs["scaffold_level"] = 1
    return LoopDecision(
        kind=DecisionKind.FINISH,
        proposed_plan=TeachingActionPlan(**plan_kwargs),
        rationale="enough evidence",
    )


def test_tools_and_skills_are_loaded_only_when_selected() -> None:
    provider = ScriptedProvider(
        LoopDecision(
            kind=DecisionKind.CALL_TOOL,
            tool_name="load_skill",
            arguments={"name": "socratic_scaffolding"},
        ),
        LoopDecision(
            kind=DecisionKind.CALL_TOOL,
            tool_name="reason_about_problem",
            arguments={"focus": "下一小步"},
        ),
        _finish(TeachingAction.SCAFFOLD),
    )
    response_client = FakeResponseClient(
        "可以先隔离含有未知数的部分。你能把等式两边同时减去 3 吗？"
    )
    result = AgentLoop(
        provider,
        response_client=response_client,
    ).run_turn(
        student_id="s1",
        problem_context="解方程 2x+3=7",
        student_message="我不会",
        reasoner=FakeReasoner(),
    )

    assert result.loaded_skills == ["socratic_scaffolding"]
    assert [item.tool_name for item in result.observations] == [
        "verify_math_attempt",
        "load_skill",
        "reason_about_problem",
    ]
    assert result.action_plan.action is TeachingAction.SCAFFOLD
    assert response_client.calls == 1
    assert result.termination_reason == "model_finished"


def test_optional_modules_are_not_called_when_model_finishes_immediately() -> None:
    provider = ScriptedProvider(_finish(TeachingAction.EXPLAIN))
    result = AgentLoop(provider).run_turn(
        student_id="s1",
        problem_context="解方程 2x+3=7",
        student_message="为什么等式两边可以同时减 3？",
    )

    assert [item.tool_name for item in result.observations] == [
        "verify_math_attempt"
    ]
    assert result.action_plan.action is TeachingAction.EXPLAIN
    assert "retrieve_pedagogical_context" not in {
        item.tool_name for item in result.observations
    }


def test_math_checker_is_mandatory_and_unsafe_plan_is_rejected() -> None:
    provider = ScriptedProvider(_finish(TeachingAction.MODEL))
    result = AgentLoop(provider).run_turn(
        student_id="s1",
        problem_context="解方程 2x+3=7",
        student_message="x=9",
    )

    assert result.verification.status is VerificationStatus.VERIFIED_INCORRECT
    assert result.observations[0].tool_name == "verify_math_attempt"
    assert result.trace[0].event == "mandatory_tool"
    assert result.action_plan.action is not TeachingAction.MODEL
    assert result.action_plan.action in result.safety_constraints.allowed_actions
    assert result.fallback_used is True
    assert any(event.event == "plan_rejected" for event in result.trace)


def test_unknown_tool_becomes_observation_instead_of_crashing() -> None:
    provider = ScriptedProvider(
        LoopDecision(
            kind=DecisionKind.CALL_TOOL,
            tool_name="invented_tool",
            arguments={},
        ),
        _finish(TeachingAction.FOCUS),
    )
    result = AgentLoop(provider).run_turn(
        student_id="s1",
        problem_context="解方程 2x+3=7",
        student_message="下一步是什么",
    )

    unknown = result.observations[-1]
    assert unknown.tool_name == "invented_tool"
    assert unknown.ok is False
    assert "unknown tool" in unknown.error
    assert result.termination_reason == "model_finished"


def test_iteration_budget_falls_back_to_safe_plan() -> None:
    provider = ScriptedProvider(
        LoopDecision(
            kind=DecisionKind.CALL_TOOL,
            tool_name="inspect_student_context",
            arguments={},
        )
    )
    result = AgentLoop(provider, max_iterations=1).run_turn(
        student_id="s1",
        problem_context="解方程 2x+3=7",
        student_message="我不知道",
    )

    assert result.termination_reason == "iteration_budget_exhausted"
    assert result.fallback_used is True
    assert result.action_plan.action in result.safety_constraints.allowed_actions


def test_deepseek_provider_forwards_local_reasoning_budget() -> None:
    client = RecordingJsonClient()
    provider = DeepSeekActionProvider(client, max_tokens=1234)

    decision = provider.decide({"test": True})

    assert decision.kind is DecisionKind.FINISH
    assert client.max_tokens == 1234
