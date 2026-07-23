"""Deterministic hybrid student simulator tests."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from app.agent.math_checker import StepValidity, check_step
from app.evaluation.student_simulator import (
    HelpSeekingStyle,
    MasteryLevel,
    StudentHiddenState,
    StudentIntentType,
    StudentSimulator,
    StudentWording,
    TerminationReason,
    UnderstandingStatus,
    advance_student_state,
    load_student_scenarios,
    validate_math_intent,
)


def _scenario(concept_id: str = "move_terms", mastery: str = "low"):
    return next(
        scenario
        for scenario in load_student_scenarios()
        if scenario.concept_id == concept_id
        and scenario.mastery_level.value == mastery
    )


def test_default_scenarios_cover_exact_six_by_three_grid() -> None:
    scenarios = load_student_scenarios()

    assert len(scenarios) == 18
    assert len({scenario.scenario_id for scenario in scenarios}) == 18
    assert {
        (scenario.concept_id, scenario.mastery_level.value)
        for scenario in scenarios
    } == {
        (concept, mastery.value)
        for concept in {
            "linear_equation",
            "equation_properties",
            "move_terms",
            "divide_by_coefficient",
            "remove_parentheses",
            "fractional_coefficient_equation",
        }
        for mastery in MasteryLevel
    }

    for concept in {scenario.concept_id for scenario in scenarios}:
        problems = {
            scenario.original_problem
            for scenario in scenarios
            if scenario.concept_id == concept
        }
        assert len(problems) == 1


def test_every_declared_math_intent_is_checked_by_math_checker() -> None:
    for scenario in load_student_scenarios():
        intents = [*scenario.solution_steps, scenario.verification_intent]
        if scenario.misconception_intent is not None:
            intents.append(scenario.misconception_intent)
        for intent in intents:
            assert validate_math_intent(scenario, intent)
            if intent.expected_step_validity:
                assert (
                    check_step(
                        scenario.original_problem,
                        intent.canonical_content,
                    ).validity.value
                    == intent.expected_step_validity
                )


def test_hidden_state_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        StudentHiddenState(
            mastery=1.1,
            active_misconception=None,
            current_solution_step=0,
            confidence=0.5,
            help_seeking_style=HelpSeekingStyle.DIRECT,
            consecutive_failures=0,
            hint_exposure=0,
            understanding_status=UnderstandingStatus.PARTIAL,
        )


def test_loader_rejects_mastery_baseline_drift(tmp_path) -> None:
    scenario = _scenario(mastery="low")
    row = scenario.model_dump(mode="json")
    row["initial_state"]["mastery"] = 0.35
    path = tmp_path / "drifted_scenario.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="initial state does not match"):
        load_student_scenarios(path, require_complete=False)


def test_targeted_support_clears_misconception_and_advances_correctly() -> None:
    scenario = _scenario()
    before = deepcopy(scenario.initial_state)

    transition = advance_student_state(
        scenario,
        before,
        "移项变号的本质是两边同时减去 3。你先写出移项后的式子？",
        teacher_turn_index=1,
    )

    assert transition.observation.addresses_misconception is True
    assert transition.selected_intent is not None
    assert transition.selected_intent.intent_type is StudentIntentType.CORRECT_STEP
    assert transition.state_after.active_misconception is None
    assert transition.state_after.current_solution_step == 1
    assert transition.state_after.mastery > before.mastery
    assert transition.termination_reason is None


def test_premature_answer_leak_does_not_advance_learning_state() -> None:
    scenario = _scenario()
    before = deepcopy(scenario.initial_state)

    transition = advance_student_state(
        scenario,
        before,
        "这题答案是 x=4。你记住了吗？",
        teacher_turn_index=1,
    )

    assert transition.observation.leaks_final_answer is True
    assert transition.state_after.current_solution_step == 0
    assert transition.state_after.mastery == before.mastery
    assert transition.state_after.understanding_status == before.understanding_status
    assert transition.selected_intent is not None
    assert transition.selected_intent.intent_type in {
        StudentIntentType.ASK_WHY,
        StudentIntentType.RESTATE_CONFUSION,
    }


def test_low_mastery_requires_two_generic_hints_before_correct_step() -> None:
    scenario = _scenario()

    first = advance_student_state(
        scenario,
        deepcopy(scenario.initial_state),
        "提示一下：先想想等号两边要做什么？",
        teacher_turn_index=1,
    )
    assert first.selected_intent is not None
    assert first.selected_intent.intent_type is StudentIntentType.INCORRECT_STEP
    assert first.state_after.hint_exposure == 1

    second = advance_student_state(
        scenario,
        first.state_after,
        "再给一个提示：可以先整理常数项。下一步是什么？",
        teacher_turn_index=2,
    )
    assert second.selected_intent is not None
    assert second.selected_intent.intent_type is StudentIntentType.CORRECT_STEP
    assert second.state_after.current_solution_step == 1


def test_three_incorrect_attempts_without_progress_end_stuck() -> None:
    scenario = _scenario()
    state = scenario.initial_state.model_copy(
        update={"confidence": 0.9, "mastery": 0.2}
    )

    for turn in range(1, 4):
        transition = advance_student_state(
            scenario,
            state,
            "你再猜一次下一步是什么？",
            teacher_turn_index=turn,
        )
        state = transition.state_after

    assert state.consecutive_failures == 3
    assert state.current_solution_step == 0
    assert transition.termination_reason is TerminationReason.STUDENT_STUCK


def test_low_confidence_failure_can_end_with_give_up() -> None:
    scenario = _scenario()
    state = scenario.initial_state.model_copy(
        update={"confidence": 0.15, "consecutive_failures": 2}
    )

    transition = advance_student_state(
        scenario,
        state,
        "你再猜一次下一步是什么？",
        teacher_turn_index=3,
    )

    assert transition.termination_reason is TerminationReason.STUDENT_GAVE_UP
    assert transition.selected_intent is not None
    assert transition.selected_intent.intent_type is StudentIntentType.GIVE_UP


def test_solution_requires_verification_and_teacher_confirmation() -> None:
    scenario = _scenario(mastery="high")
    state = scenario.initial_state.model_copy(
        update={
            "current_solution_step": len(scenario.solution_steps),
            "understanding_status": UnderstandingStatus.UNDERSTOOD,
        }
    )

    verification = advance_student_state(
        scenario,
        state,
        "答案正确，请代回原方程验算一下？",
        teacher_turn_index=2,
    )
    assert verification.selected_intent == scenario.verification_intent
    assert (
        verification.state_after.understanding_status
        is UnderstandingStatus.VERIFIED
    )
    assert verification.termination_reason is None

    confirmed = advance_student_state(
        scenario,
        verification.state_after,
        "验算正确，代回后两边相等，这题完成了。",
        teacher_turn_index=3,
    )
    assert confirmed.selected_intent is None
    assert confirmed.termination_reason is TerminationReason.SOLVED_AND_VERIFIED


class _BadWordingClient:
    def chat_json(self, system, user, schema, temperature=0.0):
        del system, user, temperature
        return schema(text="我直接写成 x=999。")


def test_real_wording_that_changes_math_falls_back_to_template() -> None:
    scenario = _scenario(mastery="high")
    simulator = StudentSimulator(
        scenario,
        use_real_wording_llm=True,
        llm_client=_BadWordingClient(),
    )

    transition = simulator.advance(
        "你可以直接写出移项后的等式。",
        teacher_turn_index=1,
    )

    assert transition.used_real_student_wording_llm is False
    assert transition.wording_fallback_reason
    assert "999" not in (transition.generated_student_message or "")
    assert transition.selected_intent is not None
    assert transition.selected_intent.canonical_content in (
        transition.generated_student_message or ""
    )


def test_student_wording_schema_is_bounded() -> None:
    with pytest.raises(ValidationError):
        StudentWording(text="")
