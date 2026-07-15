"""Enforce one migration layer for every collected test."""

import pytest


pytestmark = pytest.mark.safety_contract
_CONTRACT_MARKS = {"safety_contract", "legacy_lock", "teaching_behavior"}


def test_every_collected_test_has_exactly_one_contract_layer(request) -> None:
    failures: list[str] = []
    for item in request.session.items:
        active = _CONTRACT_MARKS.intersection(item.keywords)
        if len(active) != 1:
            failures.append(f"{item.nodeid}: {sorted(active)}")

    assert failures == []
