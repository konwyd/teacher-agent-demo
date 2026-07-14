"""Contract tests for the neutral teaching-harness boundary."""

import pytest

from app.agent.state import (
    DiagnosisResult,
    GenericMathReasoning,
    GraphContext,
    MathAnalysis,
    TeacherState,
    TeachingAction,
    TeachingActionKind,
    TeachingActionPlan,
    TeachingState,
    VerificationResult,
    VerificationStatus,
    project_teacher_state,
    project_teaching_action,
    project_teaching_state_to_teacher_state,
)


def test_neutral_teaching_state_requires_no_equation_or_concept_id() -> None:
    state = TeachingState(
        student_id="neutral-student",
        message="我不确定下一步怎么想。",
        action_plan=TeachingActionPlan(action=TeachingActionKind.GIVE_HINT, hint_level=1),
        math_analysis=MathAnalysis(
            attempt_summary="学生表达了不确定。",
            likely_obstacle="下一步策略尚不清楚",
            confidence=0.4,
        ),
        verification=VerificationResult(status=VerificationStatus.NOT_APPLICABLE),
        graph_context=GraphContext(),
    )

    dumped = state.model_dump(mode="json")
    assert dumped["action_plan"]["action"] == "give_hint"
    assert dumped["verification"]["status"] == "not_applicable"
    assert dumped["graph_context"]["available"] is False
    assert "target_equation" not in dumped
    assert "detected_concept_id" not in dumped


def test_teaching_state_carries_generic_reasoning_without_topic_id() -> None:
    state = TeachingState(
        student_id="ratio-student",
        message="男生和女生人数的比是3:2，总数30人。",
        math_reasoning=GenericMathReasoning(
            problem_interpretation="总份数是5份。",
            reasoning_steps=["先求总份数 3+2。"],
            next_step_hint="把30按5份平均分。",
            candidate_conclusion="男生18人，女生12人。",
            confidence=0.9,
        ),
    )

    dumped = state.model_dump(mode="json")

    assert dumped["math_reasoning"]["candidate_conclusion"] == "男生18人，女生12人。"
    assert "concept_id" not in dumped["math_reasoning"]


@pytest.mark.parametrize("legacy_action", list(TeachingAction))
def test_every_legacy_action_has_a_neutral_projection(legacy_action: TeachingAction) -> None:
    projected = project_teaching_action(legacy_action)

    assert isinstance(projected, TeachingActionKind)


def test_legacy_teacher_state_projects_without_mutating_equation_state() -> None:
    legacy = TeacherState(
        student_id="student-1",
        message="我把 x 移到右边了。",
        original_problem="2x + 3 = 7",
        target_equation="2x + 3 = 7",
        lesson_goal="求解方程",
        current_step="检查移项",
        solved_steps=["2x = 4"],
        pending_teacher_question="接下来如何处理系数？",
        diagnosis=DiagnosisResult(confidence=0.8, reason="学生已经给出一个中间步骤。"),
        action=TeachingAction.GIVE_HINT_LEVEL_1,
        student_step_validity="correct",
        retrieval_backend="legacy_json_graph",
        top_evidence_reasons=["常见卡点：移项符号"],
    )

    projected = project_teacher_state(legacy)

    assert projected.problem_context == legacy.original_problem
    assert projected.action_plan is not None
    assert projected.action_plan.action is TeachingActionKind.GIVE_HINT
    assert projected.verification.status is VerificationStatus.VERIFIED_CORRECT
    assert projected.graph_context.available is True
    assert projected.graph_context.provider == "legacy_json_graph"
    assert projected.math_analysis.confidence == 0.8
    assert legacy.target_equation == "2x + 3 = 7"
    assert "target_equation" not in projected.model_dump()


def test_generic_reasoning_survives_legacy_compatibility_projection() -> None:
    generic = TeachingState(
        student_id="geometry-student",
        message="三角形内角和为什么是180度？",
        problem_context="三角形内角和为什么是180度？",
        math_reasoning=GenericMathReasoning(
            reasoning_steps=["把三角形分割后可拼成平角。"],
            next_step_hint="观察拼接后的角。",
            confidence=0.9,
        ),
    )

    round_tripped = project_teacher_state(
        project_teaching_state_to_teacher_state(generic)
    )

    assert round_tripped.math_reasoning.reasoning_steps == ["把三角形分割后可拼成平角。"]
