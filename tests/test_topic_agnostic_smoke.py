"""Data-driven non-equation smoke conversations for the teaching harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.state import VerificationStatus
from app.agent.workflow import run_teaching_turn


_CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "topic_agnostic_smoke.jsonl"


def _load_smoke_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in _CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("case", _load_smoke_cases(), ids=lambda case: case["id"])
def test_cross_domain_smoke_preserves_generic_teaching_invariants(case: dict) -> None:
    state = run_teaching_turn(case["student_id"], case["message"])

    assert state.action_plan is not None
    assert state.action_plan.action.value in case["allowed_actions"]
    assert state.problem_context == case["message"]
    assert state.verification.status is VerificationStatus.NOT_APPLICABLE
    assert state.graph_context.available is False
    assert state.current_phase == state.action_plan.action.value
    assert state.response.count("？") == 1
    assert state.response.endswith("？")
    assert all(token not in state.response for token in case["forbidden_answer_tokens"])
    assert "另一题" not in state.response
