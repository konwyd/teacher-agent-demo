from __future__ import annotations

import pytest

from prototypes.llm_led_teacher.orchestration_models import TeachingContract
from prototypes.llm_led_teacher.skill_policy import (
    available_skill_descriptors,
    select_skill_for_contract,
)


def _contract(response_mode: str) -> TeachingContract:
    return TeachingContract(
        response_mode=response_mode,
        requested_target="test target",
        may_apply_to_active_problem=response_mode
        in {"one_step", "guided_solution", "full_solution"},
        may_reveal_final_answer=response_mode == "full_solution",
        reason="test contract",
    )


def test_available_skill_descriptors_are_sorted_and_complete() -> None:
    assert available_skill_descriptors({"z": "Z", "a": "A"}) == [
        {"name": "a", "description": "A"},
        {"name": "z", "description": "Z"},
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("concept_explanation", "concept_explanation"),
        ("one_step", "socratic_guidance"),
        ("guided_solution", "socratic_guidance"),
        ("full_solution", "complete_solution"),
        ("brief_redirect", ""),
    ],
)
def test_contract_selects_only_canonical_skill(mode: str, expected: str) -> None:
    assert select_skill_for_contract(_contract(mode)) == expected
