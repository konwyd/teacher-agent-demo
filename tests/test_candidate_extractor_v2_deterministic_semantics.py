from __future__ import annotations

from typing import Any

import pytest

from app.memory.candidate_extractor_v2_deterministic import (
    build_deterministic_memory_candidates,
)


def _summary(
    utterance: str,
    *,
    subject: str = "数学",
    observations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates = build_deterministic_memory_candidates(
        {
            "turn_id": 3,
            "student_message": utterance,
            "classification": {"subject": subject},
            "observations": observations or [],
        },
        conversation_id="route-a-semantics",
    )
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
            "暑假前掌握比例",
            {"subject": "数学", "goal": "掌握比例", "temporal_expression": "暑假前"},
            "short_lived",
        ),
        (
            "这学期我想重点补几何",
            {"subject": "数学", "goal": "重点补几何", "until": "semester"},
            "short_lived",
        ),
        (
            "以后我都想重点学代数",
            {"subject": "数学", "goal": "重点学代数"},
            "persistent",
        ),
    ],
)
def test_extracts_learning_goal_temporal_families(
    utterance: str,
    payload: dict[str, Any],
    temporal_scope: str,
) -> None:
    assert _summary(utterance) == [
        {
            "memory_type": "learning_goal",
            "payload": payload,
            "source_kind": "user_stated",
            "temporal_scope": temporal_scope,
        }
    ]


def test_stable_confusion_keeps_topic_and_statement() -> None:
    assert _summary("我总弄混分子和分母") == [
        {
            "memory_type": "difficulty",
            "payload": {
                "subject": "数学",
                "topic": "分数",
                "statement": "弄混分子和分母",
            },
            "source_kind": "user_stated",
            "temporal_scope": "persistent",
        }
    ]


def test_resolved_prior_difficulty_becomes_corrected_topic_progress() -> None:
    assert _summary("我以前怕小数，现在掌握了") == [
        {
            "memory_type": "topic_progress",
            "payload": {
                "subject": "数学",
                "topic": "小数",
                "status": "resolved",
                "explicit_correction": True,
                "correction_of": "difficulty",
            },
            "source_kind": "user_stated",
            "temporal_scope": "short_lived",
        }
    ]


def test_uncertain_profile_fact_is_not_inflated_to_persistent() -> None:
    assert _summary("我也许叫安宁", subject="") == [
        {
            "memory_type": "profile_fact",
            "payload": {"preferred_name": "安宁", "uncertainty": "uncertain"},
            "source_kind": "user_stated",
            "temporal_scope": "unknown",
        }
    ]


def test_past_learning_episode_does_not_become_a_durable_preference() -> None:
    assert _summary(
        "用数轴后我理解了",
        observations=[{"episode": "number line enabled understanding"}],
    ) == [
        {
            "memory_type": "learning_episode",
            "payload": {
                "subject": "数学",
                "episode": "number line enabled understanding",
            },
            "source_kind": "observed_behavior",
            "temporal_scope": "short_lived",
        }
    ]
