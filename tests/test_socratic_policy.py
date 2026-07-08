"""Socratic advisor policy tests.

The advisor layer is a secondary teaching signal: it can shape Socratic
questioning, but it must never override math_checker or original_problem
grounding.
"""

from app.agent.socratic_policy import choose_socratic_decision
from app.agent.state import DiagnosisResult, StudentIntent, TeachingAction


def test_advisor_maps_problem_solving_hint_to_concept_probe() -> None:
    decision = choose_socratic_decision(
        diagnosis=DiagnosisResult(intent=StudentIntent.PROBLEM_SOLVING, confidence=0.8),
        action=TeachingAction.GIVE_HINT_LEVEL_1,
        student_step_validity="",
        response_kind="",
        consecutive_wrong_steps=0,
    )

    assert decision.stage == "concept_probe"
    assert decision.move == "generate_subquestion"
    assert "探查" in decision.reason
    assert decision.must_not_override_checker is True


def test_advisor_preserves_checker_feedback_for_correct_final_answer() -> None:
    decision = choose_socratic_decision(
        diagnosis=DiagnosisResult(intent=StudentIntent.PROVIDES_FINAL_ANSWER, confidence=1.0),
        action=TeachingAction.GRADE_AND_FEEDBACK,
        student_step_validity="correct_final_answer",
        response_kind="multi_step_plan",
        consecutive_wrong_steps=0,
    )

    assert decision.stage == "summary"
    assert decision.move == "ask_student_to_verify"
    assert decision.must_not_override_checker is True


def test_advisor_marks_repeated_wrong_step_as_counterexample_move() -> None:
    decision = choose_socratic_decision(
        diagnosis=DiagnosisResult(intent=StudentIntent.PROVIDES_STEP, confidence=1.0),
        action=TeachingAction.SHOW_WORKED_EXAMPLE,
        student_step_validity="incorrect_step",
        response_kind="intermediate_equation",
        consecutive_wrong_steps=1,
    )

    assert decision.stage == "rule_building"
    assert decision.move == "offer_counterexample_or_worked_example"
