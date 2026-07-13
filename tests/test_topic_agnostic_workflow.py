"""Tests for the topic-agnostic workflow facade and legacy adapter boundary."""

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
