from __future__ import annotations

from typing import Any

import pytest

from app.memory.candidate_extractor_v2_deterministic import (
    build_deterministic_memory_candidates,
)


def _preference(utterance: str) -> dict[str, Any]:
    candidates = build_deterministic_memory_candidates(
        {"turn_id": 2, "student_message": utterance},
        conversation_id="route-a-regressions",
    )
    assert len(candidates) == 1
    item = candidates[0]
    return {"payload": item.payload, "temporal_scope": item.temporal_scope.value}


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        (
            "几何题尽量用图解释",
            {
                "payload": {
                    "teaching_preferences": ["图示讲解"],
                    "subject": "几何",
                },
                "temporal_scope": "persistent",
            },
        ),
        (
            "我发现看图时理解得快一些",
            {
                "payload": {"teaching_preferences": ["图示讲解"]},
                "temporal_scope": "persistent",
            },
        ),
        (
            "我以前喜欢分步骤，现在请用图解释",
            {
                "payload": {
                    "teaching_preferences": ["图示讲解"],
                    "explicit_correction": True,
                },
                "temporal_scope": "persistent",
            },
        ),
        (
            "复杂题展开讲，简单题只给要点",
            {
                "payload": {
                    "teaching_preferences": ["按难度调整"],
                    "condition_map": {"hard": "detailed", "simple": "concise"},
                },
                "temporal_scope": "persistent",
            },
        ),
        (
            "复习时快一点，初次学习时详细一些",
            {
                "payload": {"teaching_preferences": ["按学习阶段调整"]},
                "temporal_scope": "persistent",
            },
        ),
        (
            "这一次能画一下吗",
            {
                "payload": {"teaching_preferences": ["图示讲解"]},
                "temporal_scope": "turn_only",
            },
        ),
    ],
)
def test_generalized_layer_b_regressions(
    utterance: str,
    expected: dict[str, Any],
) -> None:
    assert _preference(utterance) == expected
