from __future__ import annotations

import pytest

from app.agent.state import VerificationResult, VerificationStatus
from app.retrieval.semantic_provider import (
    RetrievalContentKind,
    RetrievalHit,
    RetrievalStatus,
    SemanticRetrievalRequest,
    SemanticRetrievalResult,
)
from prototypes.llm_led_teacher.logic import (
    PrototypeEpisode,
    PrototypeTurnRecord,
    TeacherResponse,
)
from prototypes.llm_led_teacher.rag_adapter import (
    example_exposure_allowed,
    retrieval_payload,
    retrieve_for_turn,
)


class _CountingProvider:
    name = "counting"

    def __init__(self) -> None:
        self.requests: list[SemanticRetrievalRequest] = []

    def retrieve(
        self,
        request: SemanticRetrievalRequest,
    ) -> SemanticRetrievalResult:
        self.requests.append(request)
        hits = [
            RetrievalHit(
                reference_id="textbook:1",
                source_label="七年级教材",
                collection="k9_textbook",
                content_kind=RetrievalContentKind.REFERENCE,
                text="等式两边同时进行相同运算，等式仍成立。",
                similarity=0.9,
            )
        ]
        if RetrievalContentKind.EXAMPLE in request.allowed_content_kinds:
            hits.append(
                RetrievalHit(
                    reference_id="exercise:1",
                    source_label="练习题库",
                    collection="k9_exercises",
                    content_kind=RetrievalContentKind.EXAMPLE,
                    text="例题：解方程3x+2=8。",
                    similarity=0.8,
                )
            )
        return SemanticRetrievalResult(
            status=RetrievalStatus.SUCCESS,
            provider=self.name,
            query=request.problem_context,
            hits=hits,
        )


def _verification(status: VerificationStatus) -> VerificationResult:
    return VerificationResult(status=status, verifier_name="test")


def _wrong_turn(turn_id: int) -> PrototypeTurnRecord:
    return PrototypeTurnRecord(
        turn_id=turn_id,
        student_message="错误步骤",
        teacher_output=TeacherResponse(response="请检查这一步。"),
        student_verification=_verification(VerificationStatus.VERIFIED_INCORRECT),
        evidence_level="verified",
    )


@pytest.mark.parametrize("message", ["请举个例子", "给我一道练习", "有类似题吗"])
def test_explicit_example_or_practice_request_allows_examples(message: str) -> None:
    assert example_exposure_allowed(
        PrototypeEpisode(problem_context="2x+3=7"),
        message,
        _verification(VerificationStatus.NOT_APPLICABLE),
    )


@pytest.mark.parametrize("message", ["我完全不懂", "我一点都不会"])
def test_complete_confusion_allows_examples(message: str) -> None:
    assert example_exposure_allowed(
        PrototypeEpisode(problem_context="2x+3=7"),
        message,
        _verification(VerificationStatus.NOT_APPLICABLE),
    )


def test_two_consecutive_verified_wrong_attempts_allow_examples() -> None:
    episode = PrototypeEpisode(
        problem_context="2x+3=7",
        turns=[_wrong_turn(1)],
    )

    assert example_exposure_allowed(
        episode,
        "x=1",
        _verification(VerificationStatus.VERIFIED_INCORRECT),
    )


@pytest.mark.parametrize("message", ["我不会", "帮帮我", "怎么解"])
def test_generic_help_does_not_allow_examples(message: str) -> None:
    assert not example_exposure_allowed(
        PrototypeEpisode(problem_context="2x+3=7"),
        message,
        _verification(VerificationStatus.NOT_APPLICABLE),
    )


def test_one_verified_wrong_attempt_does_not_allow_examples() -> None:
    assert not example_exposure_allowed(
        PrototypeEpisode(problem_context="2x+3=7"),
        "x=1",
        _verification(VerificationStatus.VERIFIED_INCORRECT),
    )


def test_new_problem_retrieves_once_and_follow_up_reuses_cache() -> None:
    provider = _CountingProvider()
    episode = PrototypeEpisode(problem_context="2x+3=7")

    first = retrieve_for_turn(
        episode,
        "2x+3=7怎么解",
        _verification(VerificationStatus.UNABLE_TO_VERIFY),
        starts_new_problem=True,
        provider=provider,
    )
    second = retrieve_for_turn(
        episode,
        "为什么两边都要减3",
        _verification(VerificationStatus.NOT_APPLICABLE),
        starts_new_problem=False,
        provider=provider,
    )

    assert first is second
    assert len(provider.requests) == 1
    assert episode.active_retrieval_problem == "2x+3=7"


def test_new_active_problem_replaces_previous_cache() -> None:
    provider = _CountingProvider()
    episode = PrototypeEpisode(problem_context="2x+3=7")
    retrieve_for_turn(
        episode,
        "2x+3=7怎么解",
        _verification(VerificationStatus.UNABLE_TO_VERIFY),
        starts_new_problem=True,
        provider=provider,
    )
    episode.problem_context = "3x-5=10"

    retrieve_for_turn(
        episode,
        "3x-5=10怎么解",
        _verification(VerificationStatus.UNABLE_TO_VERIFY),
        starts_new_problem=True,
        provider=provider,
    )

    assert len(provider.requests) == 2
    assert episode.active_retrieval_problem == "3x-5=10"


def test_newly_allowed_example_request_refreshes_reference_only_cache() -> None:
    provider = _CountingProvider()
    episode = PrototypeEpisode(problem_context="2x+3=7")
    retrieve_for_turn(
        episode,
        "2x+3=7怎么解",
        _verification(VerificationStatus.UNABLE_TO_VERIFY),
        starts_new_problem=True,
        provider=provider,
    )

    refreshed = retrieve_for_turn(
        episode,
        "请举个例子",
        _verification(VerificationStatus.NOT_APPLICABLE),
        starts_new_problem=False,
        provider=provider,
    )

    assert len(provider.requests) == 2
    assert RetrievalContentKind.EXAMPLE in provider.requests[1].allowed_content_kinds
    assert any(
        hit.content_kind is RetrievalContentKind.EXAMPLE for hit in refreshed.hits
    )
    assert episode.active_retrieval_allows_examples is True


@pytest.mark.parametrize(
    "status",
    [RetrievalStatus.DISABLED, RetrievalStatus.EMPTY, RetrievalStatus.ERROR],
)
def test_non_success_result_has_no_teacher_payload(status: RetrievalStatus) -> None:
    result = SemanticRetrievalResult(
        status=status,
        provider="fake",
        query="2x+3=7",
        error_code="backend_unavailable" if status is RetrievalStatus.ERROR else "",
    )

    assert retrieval_payload(result) is None


def test_success_payload_is_bounded_and_cannot_replace_control_fields() -> None:
    result = _CountingProvider().retrieve(
        SemanticRetrievalRequest(problem_context="2x+3=7")
    )

    payload = retrieval_payload(result)

    assert payload is not None
    assert payload["status"] == "success"
    assert payload["provider"] == "counting"
    assert payload["hits"][0]["reference_id"] == "textbook:1"
    assert payload["usage"] == "untrusted_soft_reference"
    assert "problem_context" not in payload
    assert "turn_guidance" not in payload
    assert "deterministic_verification" not in payload
