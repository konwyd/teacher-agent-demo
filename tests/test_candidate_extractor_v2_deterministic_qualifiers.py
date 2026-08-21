from __future__ import annotations

from typing import Any

import pytest

from app.memory.candidate_extractor_v2_deterministic import (
    build_deterministic_memory_candidates,
)


def _extract(utterance: str):
    return build_deterministic_memory_candidates(
        {"turn_id": 8, "student_message": utterance},
        conversation_id="route-a-qualifiers",
    )


def _summaries(candidates) -> list[dict[str, Any]]:
    return [
        {
            "memory_type": item.memory_type.value,
            "payload": item.payload,
            "source_kind": item.source_kind.value,
            "temporal_scope": item.temporal_scope.value,
        }
        for item in candidates
    ]


@pytest.mark.parametrize(
    ("utterance", "payload", "temporal_scope"),
    [
        (
            "我不喜欢用图解释",
            {
                "teaching_preferences": ["图示讲解"],
                "negated": True,
                "polarity": "negative",
            },
            "persistent",
        ),
        (
            "我不是不喜欢分步骤解释",
            {
                "teaching_preferences": ["分步骤讲解"],
                "negated": False,
                "polarity": "positive",
            },
            "persistent",
        ),
        (
            "如果是代数题请分步骤讲",
            {
                "teaching_preferences": ["分步骤讲解"],
                "condition": "代数题",
            },
            "persistent",
        ),
        (
            "一般都用图解释，这道题别用",
            {
                "teaching_preferences": ["图示讲解"],
                "scope": "general",
                "condition": "except current problem",
            },
            "persistent",
        ),
        (
            "也许以后用图解释",
            {
                "teaching_preferences": ["图示讲解"],
                "uncertainty": "hedged",
                "temporal_expression": "以后",
            },
            "unknown",
        ),
        (
            "更正，我其实喜欢分步骤讲",
            {
                "teaching_preferences": ["分步骤讲解"],
                "explicit_correction": True,
                "correction_of": "prior preference",
            },
            "persistent",
        ),
        (
            "这次用图解释，以后不用",
            {
                "teaching_preferences": ["图示讲解"],
                "temporal_expression": "这次；以后不用",
            },
            "turn_only",
        ),
    ],
)
def test_preserves_preference_qualifiers(
    utterance: str,
    payload: dict[str, Any],
    temporal_scope: str,
) -> None:
    assert _summaries(_extract(utterance)) == [
        {
            "memory_type": "preference",
            "payload": payload,
            "source_kind": "user_stated",
            "temporal_scope": temporal_scope,
        }
    ]


def test_negated_imperative_does_not_become_a_positive_persistent_preference() -> None:
    assert _extract("不要分步骤讲，直接说思路") == []


def test_candidate_ids_and_order_are_repeatable() -> None:
    payload = {
        "turn_id": 4,
        "student_message": "我是八年级，叫我Nova，以后用图解释",
        "classification": {"subject": "数学"},
    }

    first = build_deterministic_memory_candidates(
        payload,
        conversation_id="deterministic-replay",
    )
    second = build_deterministic_memory_candidates(
        payload,
        conversation_id="deterministic-replay",
    )

    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ]
    assert _summaries(first) == _summaries(second)
