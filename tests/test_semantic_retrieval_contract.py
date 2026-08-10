import pytest
from pydantic import ValidationError

from app.retrieval.semantic_provider import (
    DisabledSemanticRetrievalProvider,
    RetrievalContentKind,
    RetrievalHit,
    RetrievalStatus,
    SemanticRetrievalRequest,
    SemanticRetrievalResult,
    safe_retrieve,
)


def test_disabled_provider_returns_no_io_fallback() -> None:
    request = SemanticRetrievalRequest(problem_context="2x+3=7")

    result = DisabledSemanticRetrievalProvider().retrieve(request)

    assert result == SemanticRetrievalResult(
        status=RetrievalStatus.DISABLED,
        provider="disabled",
        query="2x+3=7",
    )


def test_safe_retrieve_sanitizes_unexpected_provider_failure() -> None:
    class ExplodingProvider:
        name = "exploding"

        def retrieve(
            self,
            request: SemanticRetrievalRequest,
        ) -> SemanticRetrievalResult:
            raise RuntimeError("secret backend detail")

    result = safe_retrieve(
        ExplodingProvider(),
        SemanticRetrievalRequest(problem_context="x+1=2"),
    )

    assert result.status is RetrievalStatus.ERROR
    assert result.provider == "exploding"
    assert result.query == "x+1=2"
    assert result.error_code == "provider_failed"
    assert "secret backend detail" not in result.model_dump_json()


def test_safe_retrieve_preserves_valid_provider_result() -> None:
    expected = SemanticRetrievalResult(
        status=RetrievalStatus.SUCCESS,
        provider="fake",
        query="三角形内角和",
        hits=[
            RetrievalHit(
                reference_id="textbook:42",
                source_label="七年级数学",
                collection="k9_textbook",
                content_kind=RetrievalContentKind.REFERENCE,
                text="三角形三个内角的和等于180度。",
                similarity=0.91,
            )
        ],
    )

    class WorkingProvider:
        name = "fake"

        def retrieve(self, request: SemanticRetrievalRequest) -> SemanticRetrievalResult:
            return expected

    assert safe_retrieve(
        WorkingProvider(),
        SemanticRetrievalRequest(problem_context="三角形内角和是多少"),
    ) is expected


@pytest.mark.parametrize("similarity", [-0.01, 1.01])
def test_retrieval_hit_rejects_out_of_range_similarity(similarity: float) -> None:
    with pytest.raises(ValidationError):
        RetrievalHit(
            reference_id="invalid",
            source_label="invalid",
            collection="k9_textbook",
            content_kind=RetrievalContentKind.REFERENCE,
            text="invalid",
            similarity=similarity,
        )
