"""Temporary compatibility boundary for the retired teaching-action enum."""

from __future__ import annotations

from enum import Enum

from app.agent.teaching_actions import (
    PracticeDifficulty,
    PracticeSimilarity,
    TeachingAction,
    TeachingActionPlan,
)


class LegacyTeachingAction(str, Enum):
    """Legacy runtime actions preserved for existing callers and evaluation."""

    ASK_DIAGNOSTIC_QUESTION = "ask_diagnostic_question"
    EXPLAIN_CONCEPT = "explain_concept"
    GIVE_HINT_LEVEL_1 = "give_hint_level_1"
    GIVE_HINT_LEVEL_2 = "give_hint_level_2"
    SHOW_WORKED_EXAMPLE = "show_worked_example"
    GIVE_PRACTICE_QUESTION = "give_practice_question"
    GENERATE_PRACTICE_QUESTION = "generate_practice_question"
    GENERATE_SIMILAR_EXERCISE = "generate_similar_exercise"
    GENERATE_HARDER_EXERCISE = "generate_harder_exercise"
    GENERATE_EASIER_EXERCISE = "generate_easier_exercise"
    SUMMARIZE_THEN_PRACTICE = "summarize_then_practice"
    GRADE_AND_FEEDBACK = "grade_and_feedback"
    SCHEDULE_REVIEW = "schedule_review"


_LEGACY_TO_PLAN = {
    LegacyTeachingAction.ASK_DIAGNOSTIC_QUESTION: TeachingActionPlan(action=TeachingAction.FOCUS),
    LegacyTeachingAction.EXPLAIN_CONCEPT: TeachingActionPlan(action=TeachingAction.EXPLAIN),
    LegacyTeachingAction.GIVE_HINT_LEVEL_1: TeachingActionPlan(action=TeachingAction.SCAFFOLD, scaffold_level=1),
    LegacyTeachingAction.GIVE_HINT_LEVEL_2: TeachingActionPlan(action=TeachingAction.SCAFFOLD, scaffold_level=2),
    LegacyTeachingAction.SHOW_WORKED_EXAMPLE: TeachingActionPlan(action=TeachingAction.MODEL),
    LegacyTeachingAction.GIVE_PRACTICE_QUESTION: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.SIMILAR,
        practice_similarity=PracticeSimilarity.SAME_STRATEGY,
    ),
    LegacyTeachingAction.GENERATE_PRACTICE_QUESTION: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.SIMILAR,
        practice_similarity=PracticeSimilarity.SAME_STRATEGY,
    ),
    LegacyTeachingAction.GENERATE_SIMILAR_EXERCISE: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.SIMILAR,
        practice_similarity=PracticeSimilarity.SAME_STRATEGY,
    ),
    LegacyTeachingAction.GENERATE_HARDER_EXERCISE: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.HARDER,
        practice_similarity=PracticeSimilarity.TRANSFER,
    ),
    LegacyTeachingAction.GENERATE_EASIER_EXERCISE: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.EASIER,
        practice_similarity=PracticeSimilarity.SAME_STRATEGY,
    ),
    LegacyTeachingAction.SUMMARIZE_THEN_PRACTICE: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.SIMILAR,
        practice_similarity=PracticeSimilarity.TRANSFER,
    ),
    LegacyTeachingAction.GRADE_AND_FEEDBACK: TeachingActionPlan(action=TeachingAction.CHECK),
    LegacyTeachingAction.SCHEDULE_REVIEW: TeachingActionPlan(action=TeachingAction.CONSOLIDATE),
}


def legacy_to_plan(
    action: LegacyTeachingAction,
    *,
    verified_incorrect: bool = False,
) -> TeachingActionPlan:
    if action is LegacyTeachingAction.GRADE_AND_FEEDBACK and verified_incorrect:
        return TeachingActionPlan(action=TeachingAction.CORRECT)
    return _LEGACY_TO_PLAN[action].model_copy(deep=True)


_SIMPLE_PLAN_TO_LEGACY = {
    TeachingAction.ELICIT: LegacyTeachingAction.ASK_DIAGNOSTIC_QUESTION,
    TeachingAction.FOCUS: LegacyTeachingAction.ASK_DIAGNOSTIC_QUESTION,
    TeachingAction.EXPLAIN: LegacyTeachingAction.EXPLAIN_CONCEPT,
    TeachingAction.CORRECT: LegacyTeachingAction.GRADE_AND_FEEDBACK,
    TeachingAction.CHECK: LegacyTeachingAction.GRADE_AND_FEEDBACK,
    TeachingAction.MODEL: LegacyTeachingAction.SHOW_WORKED_EXAMPLE,
    TeachingAction.CONSOLIDATE: LegacyTeachingAction.SCHEDULE_REVIEW,
}


def plan_to_legacy_action(plan: TeachingActionPlan) -> LegacyTeachingAction:
    if plan.action is TeachingAction.SCAFFOLD:
        return (
            LegacyTeachingAction.GIVE_HINT_LEVEL_2
            if plan.scaffold_level >= 2
            else LegacyTeachingAction.GIVE_HINT_LEVEL_1
        )
    if plan.action is TeachingAction.PRACTICE:
        if plan.practice_difficulty is PracticeDifficulty.EASIER:
            return LegacyTeachingAction.GENERATE_EASIER_EXERCISE
        if plan.practice_difficulty is PracticeDifficulty.HARDER:
            return LegacyTeachingAction.GENERATE_HARDER_EXERCISE
        if plan.practice_similarity is PracticeSimilarity.TRANSFER:
            return LegacyTeachingAction.SUMMARIZE_THEN_PRACTICE
        return LegacyTeachingAction.GIVE_PRACTICE_QUESTION
    return _SIMPLE_PLAN_TO_LEGACY[plan.action]


_LEGACY_KIND_CODE = {
    TeachingAction.ELICIT: "assess",
    TeachingAction.FOCUS: "diagnose",
    TeachingAction.SCAFFOLD: "give_hint",
    TeachingAction.EXPLAIN: "explain_step",
    TeachingAction.CORRECT: "explain_step",
    TeachingAction.CHECK: "check_understanding",
    TeachingAction.MODEL: "show_worked_example",
    TeachingAction.PRACTICE: "offer_practice",
    TeachingAction.CONSOLIDATE: "finish",
}


def legacy_kind_code(plan: TeachingActionPlan) -> str:
    return _LEGACY_KIND_CODE[plan.action]
