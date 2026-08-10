"""Lazy, vector-only Chroma retrieval for the autonomous Teacher Agent."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any, Callable
from urllib.parse import urlsplit

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.retrieval.rag_device import configure_cpu_threads, resolve_rag_device
from app.retrieval.semantic_provider import (
    DisabledSemanticRetrievalProvider,
    RetrievalContentKind,
    RetrievalHit,
    RetrievalStatus,
    SemanticRetrievalProvider,
    SemanticRetrievalRequest,
    SemanticRetrievalResult,
)


class SemanticRagSettings(BaseSettings):
    enabled: bool = False
    chroma_server: str = Field(default="", validation_alias="CHROMA_SERVER")
    vectordb_path: str = ""
    embed_model: str = "BAAI/bge-m3"
    reference_collections: str = "k9_textbook,curriculum_std"
    exercise_collections: str = "k9_exercises"
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        validation_alias="TEACHER_RAG_TOPK",
    )
    minimum_similarity: float = Field(
        default=0.44,
        ge=0.0,
        le=1.0,
        validation_alias="TEACHER_RAG_MIN_SIMILARITY",
    )
    subject_filter: str = "数学"
    device: str = ""
    cpu_threads: int = Field(default=4, ge=1)

    model_config = SettingsConfigDict(
        env_prefix="TEACHER_RAG_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    @staticmethod
    def _split_collections(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def reference_collection_names(self) -> list[str]:
        return self._split_collections(self.reference_collections)

    @property
    def exercise_collection_names(self) -> list[str]:
        return self._split_collections(self.exercise_collections)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_retrieval_query(problem_context: str, current_message: str = "") -> str:
    """Build a focused query without deleting equations, LaTeX, or symbols."""

    problem = _normalize_whitespace(problem_context)
    message = _normalize_whitespace(current_message)
    adds_retrieval_intent = any(
        marker in message
        for marker in ("为什么", "为啥", "解释", "概念", "举例", "例子", "例题", "练习")
    )
    if message and adds_retrieval_intent and message not in problem:
        return f"{problem} {message}".strip()
    return problem


def _default_client_factory(settings: SemanticRagSettings) -> Any:
    import chromadb

    if settings.chroma_server:
        raw = settings.chroma_server.strip()
        parsed = urlsplit(raw if "://" in raw else f"http://{raw}")
        if not parsed.hostname:
            raise ValueError("invalid Chroma server")
        return chromadb.HttpClient(host=parsed.hostname, port=parsed.port or 8000)
    if settings.vectordb_path:
        return chromadb.PersistentClient(path=settings.vectordb_path)
    raise ValueError("Chroma location is not configured")


def _default_embedding_factory(model_name: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _sanitize_metadata(metadata: Any) -> dict[str, str | int | float | bool]:
    if not isinstance(metadata, dict):
        return {}
    return {
        str(key): value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


def _first_row(result: dict[str, Any], key: str) -> list[Any]:
    rows = result.get(key) or [[]]
    if not isinstance(rows, list) or not rows:
        return []
    first = rows[0]
    return first if isinstance(first, list) else []


class ChromaSemanticRetrievalProvider:
    name = "chroma_vector"

    def __init__(
        self,
        settings: SemanticRagSettings | None = None,
        *,
        client_factory: Callable[[SemanticRagSettings], Any] | None = None,
        embedding_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.settings = settings or SemanticRagSettings()
        self._client_factory = client_factory or _default_client_factory
        self._embedding_factory = embedding_factory or _default_embedding_factory
        self._client: Any | None = None
        self._embedder: Any | None = None

    def _error(
        self,
        query: str,
        error_code: str,
        started_at: float,
    ) -> SemanticRetrievalResult:
        return SemanticRetrievalResult(
            status=RetrievalStatus.ERROR,
            provider=self.name,
            query=query,
            elapsed_seconds=max(0.0, perf_counter() - started_at),
            error_code=error_code,
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(self.settings)
        return self._client

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            configure_cpu_threads(str(self.settings.cpu_threads))
            device = resolve_rag_device(self.settings.device or None)
            self._embedder = self._embedding_factory(self.settings.embed_model, device)
        return self._embedder

    @staticmethod
    def _collection_plan(
        settings: SemanticRagSettings,
        request: SemanticRetrievalRequest,
    ) -> list[tuple[str, RetrievalContentKind]]:
        plan = [
            (name, RetrievalContentKind.REFERENCE)
            for name in settings.reference_collection_names
        ]
        if RetrievalContentKind.EXAMPLE in request.allowed_content_kinds:
            plan.extend(
                (name, RetrievalContentKind.EXAMPLE)
                for name in settings.exercise_collection_names
            )
        return plan

    @staticmethod
    def _dedupe_key(
        collection: str,
        item_id: str,
        text: str,
        metadata: dict[str, str | int | float | bool],
    ) -> str:
        source = str(metadata.get("source_file") or metadata.get("book_title") or "")
        chunk = str(
            metadata.get("chunk_index")
            or metadata.get("question_id")
            or metadata.get("id")
            or ""
        )
        if source:
            return f"{source}|{chunk}|{_normalize_whitespace(text)}"
        return f"{collection}|{item_id}"

    def retrieve(
        self,
        request: SemanticRetrievalRequest,
    ) -> SemanticRetrievalResult:
        query = build_retrieval_query(request.problem_context, request.current_message)
        if not self.settings.enabled:
            return SemanticRetrievalResult(
                status=RetrievalStatus.DISABLED,
                provider=self.name,
                query=query,
            )
        started_at = perf_counter()
        try:
            client = self._ensure_client()
            embedder = self._ensure_embedder()
        except ImportError:
            return self._error(query, "dependency_unavailable", started_at)
        except Exception:  # noqa: BLE001 - initialization is an optional boundary
            return self._error(query, "backend_unavailable", started_at)

        try:
            vector = embedder.encode(query, normalize_embeddings=True)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            query_embedding = list(vector)
        except Exception:  # noqa: BLE001 - embedding failure must not block teaching
            return self._error(query, "embedding_failed", started_at)

        candidates: dict[str, RetrievalHit] = {}
        effective_top_k = min(request.top_k, self.settings.top_k)
        effective_minimum_similarity = max(
            request.minimum_similarity,
            self.settings.minimum_similarity,
        )
        try:
            for collection_name, content_kind in self._collection_plan(
                self.settings,
                request,
            ):
                collection = client.get_collection(name=collection_name)
                query_args: dict[str, Any] = {
                    "query_embeddings": [query_embedding],
                    "n_results": effective_top_k,
                }
                if self.settings.subject_filter:
                    query_args["where"] = {"subject": self.settings.subject_filter}
                raw = collection.query(**query_args)
                ids = _first_row(raw, "ids")
                documents = _first_row(raw, "documents")
                distances = _first_row(raw, "distances")
                metadatas = _first_row(raw, "metadatas")
                for index, item_id in enumerate(ids):
                    text = documents[index] if index < len(documents) else ""
                    if not isinstance(text, str) or not text.strip():
                        continue
                    try:
                        distance = float(distances[index])
                    except (IndexError, TypeError, ValueError):
                        continue
                    similarity = max(0.0, min(1.0, 1.0 - distance))
                    if similarity < effective_minimum_similarity:
                        continue
                    metadata = _sanitize_metadata(
                        metadatas[index] if index < len(metadatas) else {}
                    )
                    source_label = str(
                        metadata.get("book_title")
                        or metadata.get("source_file")
                        or collection_name
                    )
                    hit = RetrievalHit(
                        reference_id=f"{collection_name}:{item_id}",
                        source_label=source_label,
                        collection=collection_name,
                        content_kind=content_kind,
                        text=text.strip(),
                        similarity=similarity,
                        metadata=metadata,
                    )
                    key = self._dedupe_key(
                        collection_name,
                        str(item_id),
                        hit.text,
                        metadata,
                    )
                    if key not in candidates or candidates[key].similarity < similarity:
                        candidates[key] = hit
        except ImportError:
            return self._error(query, "dependency_unavailable", started_at)
        except Exception:  # noqa: BLE001 - backend failures are a soft degradation
            return self._error(query, "backend_unavailable", started_at)

        hits = sorted(
            candidates.values(),
            key=lambda item: item.similarity,
            reverse=True,
        )[:effective_top_k]
        return SemanticRetrievalResult(
            status=(RetrievalStatus.SUCCESS if hits else RetrievalStatus.EMPTY),
            provider=self.name,
            query=query,
            hits=hits,
            elapsed_seconds=max(0.0, perf_counter() - started_at),
        )


def get_semantic_retrieval_provider() -> SemanticRetrievalProvider:
    settings = SemanticRagSettings()
    if not settings.enabled:
        return DisabledSemanticRetrievalProvider()
    return ChromaSemanticRetrievalProvider(settings)
