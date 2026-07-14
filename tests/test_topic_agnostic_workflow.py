"""Tests for the topic-agnostic workflow facade and legacy adapter boundary."""

from app.agent.state import GenericMathReasoning

from app.agent.state import TeachingActionKind, VerificationStatus
from app.agent.workflow import run_teaching_turn, run_turn


def test_non_equation_turn_uses_generic_harness_without_a_legacy_concept() -> None:
    state = run_teaching_turn(
        "ratio-student",
        "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
    )

    assert state.action_plan is not None
    assert state.action_plan.action is TeachingActionKind.GIVE_HINT
    assert state.graph_context.available is False
    assert state.verification.status is VerificationStatus.NOT_APPLICABLE
    assert state.current_phase == state.action_plan.action.value
    assert state.response.count("？") == 1


class _FakeReasoner:
    def analyze(self, message, problem_context, analysis):
        return GenericMathReasoning(
            reasoning_steps=["先把3:2看成5个相等份。"],
            next_step_hint="先求总份数。",
            candidate_conclusion="男生18人，女生12人。",
            confidence=0.95,
        )


class _FakeResponseClient:
    def chat(self, system, user, temperature=0.2, max_tokens=None):
        return "先把3和2相加得到总份数。你能先写出总份数吗？"


def test_non_equation_turn_uses_injected_reasoning_without_leaking_answer() -> None:
    state = run_teaching_turn(
        "ratio-reasoning-student",
        "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
        reasoner=_FakeReasoner(),
        response_client=_FakeResponseClient(),
    )

    assert state.action_plan is not None
    assert state.action_plan.action is TeachingActionKind.GIVE_HINT
    assert state.answer_disclosure_allowed is False
    assert "18" not in state.response
    assert state.math_reasoning.reasoning_steps
    assert state.trace[:2] == ["generic_analysis", "generic_math_reasoning"]


def test_equation_turn_still_uses_the_legacy_compatibility_adapter() -> None:
    state = run_turn("equation-student", "我不会解 2x+3=7")

    assert state.detected_concept_id == "c_linear"
    assert state.target_equation == "2x + 3 = 7"


def test_legacy_return_type_projects_a_non_equation_turn_without_c_linear() -> None:
    state = run_turn("geometry-student", "三角形的内角和为什么是180度？")

    assert state.detected_concept_id is None
    assert state.target_equation == ""
    assert state.response.count("？") == 1


def test_existing_equation_episode_keeps_using_adapter_for_a_follow_up_request() -> None:
    conversation_id = "equation-follow-up"
    run_turn("follow-up-student", "老师 2x+5=11 怎么解？", conversation_id=conversation_id)

    state = run_turn("follow-up-student", "你出一道题好吗？", conversation_id=conversation_id)

    assert state.detected_concept_id == "c_linear"
    assert state.generated_practice_problem
