from __future__ import annotations

from typing import Any

import pytest

from app.memory.candidate_extractor_v2_deterministic import (
    build_deterministic_memory_candidates,
)


def _extract(
    utterance: str,
    *,
    classification: dict[str, Any] | None = None,
    teaching_contract: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
    observations: list[dict[str, Any]] | None = None,
    model_inferences: list[dict[str, Any]] | None = None,
):
    return build_deterministic_memory_candidates(
        {
            "turn_id": 7,
            "student_message": utterance,
            "classification": classification or {},
            "teaching_contract": teaching_contract or {},
            "verification": verification or {},
            "observations": observations or [],
            "model_inferences": model_inferences or [],
        },
        conversation_id="route-a-unit",
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
    ("utterance", "expected"),
    [
        (
            "我是七年级，叫我阿澄",
            [
                {
                    "memory_type": "profile_fact",
                    "payload": {"grade": "七年级", "preferred_name": "阿澄"},
                    "source_kind": "user_stated",
                    "temporal_scope": "persistent",
                }
            ],
        ),
        (
            "以后最好分步骤解释，也多给几个例子",
            [
                {
                    "memory_type": "preference",
                    "payload": {
                        "teaching_preferences": ["分步骤讲解", "示例讲解"]
                    },
                    "source_kind": "user_stated",
                    "temporal_scope": "persistent",
                }
            ],
        ),
        (
            "这种题能画成图就更容易理解",
            [
                {
                    "memory_type": "preference",
                    "payload": {"teaching_preferences": ["图示讲解"]},
                    "source_kind": "user_stated",
                    "temporal_scope": "persistent",
                }
            ],
        ),
    ],
)
def test_extracts_profile_and_preference_semantic_families(
    utterance: str,
    expected: list[dict[str, Any]],
) -> None:
    assert _summaries(_extract(utterance)) == expected


@pytest.mark.parametrize(
    "utterance",
    [
        "不要保存我的名字",
        "课本里的主人公叫阿澄",
        "‘分步骤讲’这个词是什么意思",
        "请举一个例子说明反例的定义",
    ],
)
def test_rejects_privacy_third_person_meta_and_instructional_hard_negatives(
    utterance: str,
) -> None:
    assert _extract(utterance) == []


def test_converts_frozen_upstream_evidence_without_model_calls() -> None:
    actual = _summaries(
        _extract(
            "我已经能自己解一元一次方程",
            classification={"subject": "数学"},
            verification={
                "status": "verified_correct",
                "summary": "checked independently",
                "verifier_name": "math-checker",
            },
            observations=[
                {"topic": "一元一次方程", "status": "independent"},
                {"episode": "number line enabled understanding"},
                {"pattern": "equation first", "count": 4},
            ],
            model_inferences=[{"claim": "difficulty", "topic": "绝对值"}],
        )
    )

    assert actual == [
        {
            "memory_type": "topic_progress",
            "payload": {
                "subject": "数学",
                "topic": "一元一次方程",
                "status": "independent",
            },
            "source_kind": "verified_result",
            "temporal_scope": "short_lived",
        },
        {
            "memory_type": "learning_episode",
            "payload": {
                "subject": "数学",
                "episode": "number line enabled understanding",
            },
            "source_kind": "observed_behavior",
            "temporal_scope": "short_lived",
        },
        {
            "memory_type": "behavior_evidence",
            "payload": {"subject": "数学", "evidence_pattern": "equation first"},
            "source_kind": "observed_behavior",
            "temporal_scope": "short_lived",
        },
        {
            "memory_type": "difficulty",
            "payload": {
                "subject": "数学",
                "topic": "绝对值",
                "uncertainty": "model_inference",
            },
            "source_kind": "model_inference",
            "temporal_scope": "unknown",
        },
    ]


def test_emits_independent_profile_preference_and_task_candidates_in_stable_order() -> None:
    actual = _summaries(
        _extract(
            "叫我星星，以后用图解释，继续当前题",
            classification={"subject": "数学"},
            teaching_contract={"requested_target": "继续当前题"},
        )
    )

    assert [item["memory_type"] for item in actual] == [
        "profile_fact",
        "preference",
        "task_state",
    ]
    assert actual[-1] == {
        "memory_type": "task_state",
        "payload": {"subject": "数学", "requested_target": "继续当前题"},
        "source_kind": "user_stated",
        "temporal_scope": "turn_only",
    }
