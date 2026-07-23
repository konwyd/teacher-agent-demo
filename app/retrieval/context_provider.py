"""Optional graph-context providers.

The empty provider is the normal safe fallback: it neither instantiates nor
reads graph data.  The local provider simply adapts the existing JSON graph
retrievers for callers that explicitly enable graph context.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from app.agent.state import GraphContext
from app.retrieval.graph_retriever import (
    ConceptNode,
    Exercise,
    Misconception,
    PedagogicalEvidenceResult,
    get_local_pedagogical_graph_retriever,
    get_retriever,
)


class GraphContextRequest(BaseModel):
    """Small, topic-neutral retrieval request used by the generic harness."""

    problem_context: str = ""
    concept_id: str = ""


@runtime_checkable
class GraphContextProvider(Protocol):
    """Non-blocking graph context plus the legacy retrieval compatibility API."""

    name: str
    available: bool

    def get_context(self, request: GraphContextRequest) -> GraphContext:
        """Return optional, auditable context for a neutral teaching turn."""

    def get_concept(self, name_or_id: str) -> ConceptNode | None:
        """Return a legacy graph concept, if graph context is available."""

    def get_prerequisites(self, concept_id: str, depth: int = 1) -> list[ConceptNode]:
        """Return legacy prerequisites without making them workflow gates."""

    def get_related_exercises(self, concept_id: str) -> list[Exercise]:
        """Return optional legacy exercises."""

    def get_misconceptions(self, concept_id: str) -> list[Misconception]:
        """Return optional legacy misconception descriptions."""

    def retrieve_pedagogical_evidence(
        self,
        concept_id: str,
        *,
        misconception: str = "",
        mastery_score: float = 0.5,
        request_type: str = "",
    ) -> PedagogicalEvidenceResult:
        """Return optional pedagogical GraphRAG evidence."""


class EmptyGraphContextProvider:
    """No-graph provider that performs no local data reads."""

    name = "disabled"
    available = False

    def get_context(self, request: GraphContextRequest) -> GraphContext:  # noqa: ARG002
        return GraphContext(available=False, provider=self.name)

    def get_concept(self, name_or_id: str) -> ConceptNode | None:  # noqa: ARG002
        return None

    def get_prerequisites(
        self, concept_id: str, depth: int = 1  # noqa: ARG002
    ) -> list[ConceptNode]:
        return []

    def get_related_exercises(self, concept_id: str) -> list[Exercise]:  # noqa: ARG002
        return []

    def get_misconceptions(self, concept_id: str) -> list[Misconception]:  # noqa: ARG002
        return []

    def retrieve_pedagogical_evidence(
        self,
        concept_id: str,  # noqa: ARG002
        *,
        misconception: str = "",  # noqa: ARG002
        mastery_score: float = 0.5,  # noqa: ARG002
        request_type: str = "",  # noqa: ARG002
    ) -> PedagogicalEvidenceResult:
        return PedagogicalEvidenceResult(retrieval_backend=self.name)


class LocalGraphContextProvider:
    """Adapter around the existing local JSON graph implementations."""

    name = "local_json_graph"
    available = True

    def get_context(self, request: GraphContextRequest) -> GraphContext:
        concept = self.get_concept(request.concept_id or request.problem_context)
        if concept is None:
            return GraphContext(available=False, provider=self.name)
        evidence = [concept.description] if concept.description else []
        return GraphContext(
            available=True,
            provider=self.name,
            evidence=evidence,
            source_ids=[concept.id],
        )

    def get_concept(self, name_or_id: str) -> ConceptNode | None:
        return get_retriever().get_concept(name_or_id)

    def get_prerequisites(self, concept_id: str, depth: int = 1) -> list[ConceptNode]:
        return get_retriever().get_prerequisites(concept_id, depth=depth)

    def get_related_exercises(self, concept_id: str) -> list[Exercise]:
        return get_retriever().get_related_exercises(concept_id)

    def get_misconceptions(self, concept_id: str) -> list[Misconception]:
        return get_retriever().get_misconceptions(concept_id)

    def retrieve_pedagogical_evidence(
        self,
        concept_id: str,
        *,
        misconception: str = "",
        mastery_score: float = 0.5,
        request_type: str = "",
    ) -> PedagogicalEvidenceResult:
        return get_local_pedagogical_graph_retriever().retrieve_evidence(
            concept_id,
            misconception=misconception,
            mastery_score=mastery_score,
            request_type=request_type,
        )


def graph_context_enabled() -> bool:
    """Read the explicit all-graph switch; default remains backward compatible."""

    return os.getenv("USE_GRAPH_CONTEXT", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def get_graph_context_provider() -> GraphContextProvider:
    """Resolve a provider at call time so evaluation modes do not leak state."""

    if not graph_context_enabled():
        return EmptyGraphContextProvider()
    return LocalGraphContextProvider()
