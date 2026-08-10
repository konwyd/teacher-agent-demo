"""Topic-neutral contracts for optional, non-blocking semantic retrieval."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RetrievalStatus(str, Enum):
    DISABLED = "disabled"
    SUCCESS = "success"
    EMPTY = "empty"
    ERROR = "error"


class RetrievalContentKind(str, Enum):
    REFERENCE = "reference"
    EXAMPLE = "example"


class SemanticRetrievalRequest(BaseModel):
    problem_context: str = Field(min_length=1)
    current_message: str = ""
    allowed_content_kinds: set[RetrievalContentKind] = Field(
        default_factory=lambda: {RetrievalContentKind.REFERENCE}
    )
    subject: str = "数学"
    top_k: int = Field(default=5, ge=1, le=50)
    minimum_similarity: float = Field(default=0.44, ge=0.0, le=1.0)


class RetrievalHit(BaseModel):
    reference_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    content_kind: RetrievalContentKind = RetrievalContentKind.REFERENCE
    text: str = Field(min_length=1)
    similarity: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class SemanticRetrievalResult(BaseModel):
    status: RetrievalStatus
    provider: str = ""
    query: str = ""
    hits: list[RetrievalHit] = Field(default_factory=list)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    error_code: str = ""


@runtime_checkable
class SemanticRetrievalProvider(Protocol):
    name: str

    def retrieve(
        self,
        request: SemanticRetrievalRequest,
    ) -> SemanticRetrievalResult:
        """Return optional evidence without changing teaching control flow."""


class DisabledSemanticRetrievalProvider:
    name = "disabled"

    def retrieve(
        self,
        request: SemanticRetrievalRequest,
    ) -> SemanticRetrievalResult:
        return SemanticRetrievalResult(
            status=RetrievalStatus.DISABLED,
            provider=self.name,
            query=request.problem_context,
        )


def safe_retrieve(
    provider: SemanticRetrievalProvider,
    request: SemanticRetrievalRequest,
) -> SemanticRetrievalResult:
    """Keep unexpected provider failures outside the autonomous teaching loop."""

    try:
        return provider.retrieve(request)
    except Exception:  # noqa: BLE001 - optional retrieval must not block teaching
        return SemanticRetrievalResult(
            status=RetrievalStatus.ERROR,
            provider=getattr(provider, "name", type(provider).__name__),
            query=request.problem_context,
            error_code="provider_failed",
        )

