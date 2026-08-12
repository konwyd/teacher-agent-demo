from __future__ import annotations

import pytest
from pydantic import ValidationError

from prototypes.llm_led_teacher.orchestration_models import (
    AnswerMode,
    RequestClassification,
)


def test_k9_classification_requires_supported_subject() -> None:
    with pytest.raises(ValidationError):
        RequestClassification(
            scope="k9",
            subject="计算机",
            stage="初中",
            is_new_problem=True,
            answer_mode="guided",
            reason="invalid subject",
        )


def test_off_topic_classification_forces_brief_redirect() -> None:
    item = RequestClassification(
        scope="off_topic",
        subject=None,
        stage="未知",
        is_new_problem=True,
        answer_mode="brief_redirect",
        reason="not K9",
    )

    assert item.subject is None
    assert item.answer_mode is AnswerMode.BRIEF_REDIRECT


@pytest.mark.parametrize(
    "subject", ["数学", "语文", "英语", "物理", "化学", "生物", "历史", "地理", "道法"]
)
def test_all_supported_k9_subjects_are_valid(subject: str) -> None:
    item = RequestClassification(
        scope="k9",
        subject=subject,
        stage="初中",
        is_new_problem=True,
        answer_mode="guided",
        reason="supported subject",
    )

    assert item.subject is not None
    assert item.subject.value == subject


@pytest.mark.parametrize(
    ("subject", "answer_mode"),
    [("数学", "brief_redirect"), (None, "guided")],
)
def test_off_topic_rejects_subject_or_non_redirect_mode(
    subject: str | None,
    answer_mode: str,
) -> None:
    with pytest.raises(ValidationError):
        RequestClassification(
            scope="off_topic",
            subject=subject,
            stage="未知",
            is_new_problem=True,
            answer_mode=answer_mode,
            reason="invalid off-topic shape",
        )
