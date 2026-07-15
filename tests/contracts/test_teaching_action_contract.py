"""Permanent contract for the sole teaching-action model."""

import pytest
from pydantic import ValidationError

from app.agent.teaching_actions import (
    PracticeDifficulty,
    PracticeSimilarity,
    TeachingAction,
    TeachingActionPlan,
)


pytestmark = pytest.mark.safety_contract


def test_unified_action_set_is_exact() -> None:
    assert {action.value for action in TeachingAction} == {
        "elicit", "focus", "scaffold", "explain", "correct",
        "check", "model", "practice", "consolidate",
    }


def test_scaffold_requires_a_nonzero_level() -> None:
    with pytest.raises(ValidationError):
        TeachingActionPlan(action=TeachingAction.SCAFFOLD, scaffold_level=0)


def test_practice_requires_difficulty_and_similarity() -> None:
    plan = TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.SIMILAR,
        practice_similarity=PracticeSimilarity.SAME_STRATEGY,
    )
    assert plan.practice_difficulty is PracticeDifficulty.SIMILAR


def test_non_practice_action_rejects_practice_parameters() -> None:
    with pytest.raises(ValidationError):
        TeachingActionPlan(
            action=TeachingAction.EXPLAIN,
            practice_difficulty=PracticeDifficulty.HARDER,
            practice_similarity=PracticeSimilarity.TRANSFER,
        )
