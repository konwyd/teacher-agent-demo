"""Policy boundary between optional semantic RAG and the autonomous teacher."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agent.state import VerificationResult, VerificationStatus
from app.retrieval.semantic_provider import (
    RetrievalContentKind,
    RetrievalStatus,
    SemanticRetrievalProvider,
    SemanticRetrievalRequest,
    SemanticRetrievalResult,
    safe_retrieve,
)

if TYPE_CHECKING:
    from prototypes.llm_led_teacher.logic import PrototypeEpisode


_EXAMPLE_REQUEST_MARKERS = (
    "举个例子",
    "举例",
    "例题",
    "类似题",
    "相似题",
    "练习",
)

_COMPLETE_CONFUSION_MARKERS = (
    "完全不懂",
    "一点都不懂",
    "完全不会",
    "一点都不会",
)


def example_exposure_allowed(
    episode: PrototypeEpisode,
    message: str,
    verification: VerificationResult,
) -> bool:
    """Gate retrieved examples without choosing the LLM's teaching move."""

    if any(marker in message for marker in _EXAMPLE_REQUEST_MARKERS):
        return True
    if any(marker in message for marker in _COMPLETE_CONFUSION_MARKERS):
        return True
    if verification.status is not VerificationStatus.VERIFIED_INCORRECT:
        return False

    consecutive_wrong = 1
    for turn in reversed(episode.turns):
        if turn.student_verification.status is not VerificationStatus.VERIFIED_INCORRECT:
            break
        consecutive_wrong += 1
        if consecutive_wrong >= 2:
            return True
    return False


def retrieve_for_turn(
    episode: PrototypeEpisode,
    message: str,
    verification: VerificationResult,
    *,
    starts_new_problem: bool,
    provider: SemanticRetrievalProvider,
) -> SemanticRetrievalResult:
    """Retrieve once per active problem and refresh only for newly allowed examples."""

    allow_examples = example_exposure_allowed(episode, message, verification)
    cache_matches = (
        episode.active_retrieval is not None
        and episode.active_retrieval_problem == episode.problem_context
    )
    newly_allows_examples = (
        allow_examples and not episode.active_retrieval_allows_examples
    )
    if cache_matches and not starts_new_problem and not newly_allows_examples:
        return episode.active_retrieval

    allowed_kinds = {RetrievalContentKind.REFERENCE}
    if allow_examples:
        allowed_kinds.add(RetrievalContentKind.EXAMPLE)
    request = SemanticRetrievalRequest(
        problem_context=episode.problem_context,
        current_message=message,
        allowed_content_kinds=allowed_kinds,
    )
    result = safe_retrieve(provider, request)
    episode.active_retrieval = result
    episode.active_retrieval_problem = episode.problem_context
    episode.active_retrieval_allows_examples = allow_examples
    return result


def retrieval_payload(
    result: SemanticRetrievalResult,
) -> dict[str, object] | None:
    """Project successful retrieval into a bounded, untrusted prompt fragment."""

    if result.status is not RetrievalStatus.SUCCESS or not result.hits:
        return None
    hits: list[dict[str, object]] = []
    used_characters = 0
    for hit in result.hits[:5]:
        text = hit.text[:900]
        if used_characters + len(text) > 3000:
            remaining = 3000 - used_characters
            if remaining <= 0:
                break
            text = text[:remaining]
        hits.append(
            {
                "reference_id": hit.reference_id,
                "source_label": hit.source_label,
                "collection": hit.collection,
                "content_kind": hit.content_kind.value,
                "text": text,
                "similarity": hit.similarity,
            }
        )
        used_characters += len(text)
    if not hits:
        return None
    return {
        "status": result.status.value,
        "provider": result.provider,
        "query": result.query,
        "usage": "untrusted_soft_reference",
        "hits": hits,
    }
