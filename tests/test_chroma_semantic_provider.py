from __future__ import annotations

import pytest

from app.retrieval.chroma_semantic_provider import (
    ChromaSemanticRetrievalProvider,
    SemanticRagSettings,
    build_retrieval_query,
    get_semantic_retrieval_provider,
)
from app.retrieval.semantic_provider import (
    RetrievalContentKind,
    RetrievalStatus,
    SemanticRetrievalRequest,
)


class _FakeEmbedder:
    def __init__(self, vector: list[float] | Exception) -> None:
        self.vector = vector
        self.queries: list[str] = []

    def encode(self, query: str, normalize_embeddings: bool = True):
        self.queries.append(query)
        if isinstance(self.vector, Exception):
            raise self.vector
        return self.vector


class _FakeCollection:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _FakeClient:
    def __init__(self, collections: dict[str, _FakeCollection]) -> None:
        self.collections = collections
        self.requested_collections: list[str] = []

    def get_collection(self, name: str):
        self.requested_collections.append(name)
        return self.collections[name]


def _settings(**updates) -> SemanticRagSettings:
    values = {
        "enabled": True,
        "chroma_server": "localhost:8002",
        "embed_model": "BAAI/bge-m3",
        "reference_collections": "k9_textbook,curriculum_std",
        "exercise_collections": "k9_exercises",
        "top_k": 5,
        "minimum_similarity": 0.44,
        "subject_filter": "数学",
    }
    values.update(updates)
    return SemanticRagSettings(**values)


def _result(
    *,
    ids: list[str],
    documents: list[str],
    distances: list[float],
    metadatas: list[dict] | None = None,
) -> dict:
    return {
        "ids": [ids],
        "documents": [documents],
        "distances": [distances],
        "metadatas": [metadatas or [{} for _ in ids]],
    }


def test_build_retrieval_query_preserves_math_notation() -> None:
    problem = r"  已知 $x+1=2$，计算 \frac{1}{2}x + \sqrt{3}。  "

    query = build_retrieval_query(problem, "请解释这个概念")

    assert query == r"已知 $x+1=2$，计算 \frac{1}{2}x + \sqrt{3}。 请解释这个概念"


def test_public_environment_variable_names_are_loaded(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_RAG_ENABLED", "true")
    monkeypatch.setenv("TEACHER_RAG_TOPK", "3")
    monkeypatch.setenv("TEACHER_RAG_MIN_SIMILARITY", "0.72")
    monkeypatch.setenv("CHROMA_SERVER", "chroma.internal:8002")

    settings = SemanticRagSettings(_env_file=None)

    assert settings.enabled is True
    assert settings.top_k == 3
    assert settings.minimum_similarity == 0.72
    assert settings.chroma_server == "chroma.internal:8002"


def test_reference_only_request_does_not_query_exercise_collection() -> None:
    collections = {
        "k9_textbook": _FakeCollection(
            _result(
                ids=["t1"],
                documents=["等式两边同时进行相同运算，等式仍成立。"],
                distances=[0.1],
                metadatas=[{"source_file": "七年级数学", "chunk_index": 2}],
            )
        ),
        "curriculum_std": _FakeCollection(_result(ids=[], documents=[], distances=[])),
        "k9_exercises": _FakeCollection(
            _result(ids=["e1"], documents=["例题：解2x+3=7"], distances=[0.05])
        ),
    }
    client = _FakeClient(collections)
    embedder = _FakeEmbedder([0.1, 0.2])
    provider = ChromaSemanticRetrievalProvider(
        _settings(),
        client_factory=lambda settings: client,
        embedding_factory=lambda model, device: embedder,
    )

    result = provider.retrieve(
        SemanticRetrievalRequest(
            problem_context="2x+3=7",
            allowed_content_kinds={RetrievalContentKind.REFERENCE},
            top_k=3,
            minimum_similarity=0.6,
        )
    )

    assert client.requested_collections == ["k9_textbook", "curriculum_std"]
    assert result.status is RetrievalStatus.SUCCESS
    assert [hit.reference_id for hit in result.hits] == ["k9_textbook:t1"]
    assert result.hits[0].content_kind is RetrievalContentKind.REFERENCE
    assert result.hits[0].similarity == 0.9


def test_example_enabled_request_queries_exercise_collection() -> None:
    collections = {
        "k9_textbook": _FakeCollection(_result(ids=[], documents=[], distances=[])),
        "curriculum_std": _FakeCollection(_result(ids=[], documents=[], distances=[])),
        "k9_exercises": _FakeCollection(
            _result(
                ids=["e1"],
                documents=["例题：解2x+3=7"],
                distances=[0.08],
                metadatas=[{"source_file": "题库", "question_id": 7}],
            )
        ),
    }
    client = _FakeClient(collections)
    provider = ChromaSemanticRetrievalProvider(
        _settings(),
        client_factory=lambda settings: client,
        embedding_factory=lambda model, device: _FakeEmbedder([0.1, 0.2]),
    )

    result = provider.retrieve(
        SemanticRetrievalRequest(
            problem_context="2x+3=7",
            allowed_content_kinds={
                RetrievalContentKind.REFERENCE,
                RetrievalContentKind.EXAMPLE,
            },
        )
    )

    assert client.requested_collections == [
        "k9_textbook",
        "curriculum_std",
        "k9_exercises",
    ]
    assert collections["k9_textbook"].calls[0]["where"] == {"subject": "数学"}
    assert collections["k9_exercises"].calls[0]["where"] == {"subject": "Math"}
    assert result.hits[0].content_kind is RetrievalContentKind.EXAMPLE


def test_hits_are_filtered_deduplicated_sorted_and_capped() -> None:
    textbook = _FakeCollection(
        _result(
            ids=["low", "same", "malformed"],
            documents=["低分片段", "重复片段", ""],
            distances=[0.7, 0.2, 0.01],
            metadatas=[
                {"source_file": "book-a", "chunk_index": 1},
                {"source_file": "book-b", "chunk_index": 2},
                {"source_file": "book-c"},
            ],
        )
    )
    curriculum = _FakeCollection(
        _result(
            ids=["same", "top"],
            documents=["重复片段", "最高分片段"],
            distances=[0.1, 0.05],
            metadatas=[
                {"source_file": "book-b", "chunk_index": 2},
                {"source_file": "standard", "chunk_index": 1},
            ],
        )
    )
    client = _FakeClient(
        {"k9_textbook": textbook, "curriculum_std": curriculum}
    )
    provider = ChromaSemanticRetrievalProvider(
        _settings(),
        client_factory=lambda settings: client,
        embedding_factory=lambda model, device: _FakeEmbedder([0.3]),
    )

    result = provider.retrieve(
        SemanticRetrievalRequest(
            problem_context="等式性质",
            top_k=2,
            minimum_similarity=0.5,
        )
    )

    assert [hit.text for hit in result.hits] == ["最高分片段", "重复片段"]
    assert [hit.similarity for hit in result.hits] == [0.95, 0.9]


def test_environment_limits_bound_default_request() -> None:
    collection = _FakeCollection(
        _result(
            ids=["below", "accepted"],
            documents=["低于配置阈值", "达到配置阈值"],
            distances=[0.15, 0.05],
        )
    )
    client = _FakeClient({"k9_textbook": collection})
    provider = ChromaSemanticRetrievalProvider(
        _settings(
            reference_collections="k9_textbook",
            top_k=1,
            minimum_similarity=0.9,
        ),
        client_factory=lambda settings: client,
        embedding_factory=lambda model, device: _FakeEmbedder([0.3]),
    )

    result = provider.retrieve(
        SemanticRetrievalRequest(problem_context="等式性质")
    )

    assert collection.calls[0]["n_results"] == 1
    assert [hit.text for hit in result.hits] == ["达到配置阈值"]


def test_missing_optional_collection_keeps_successful_hits() -> None:
    client = _FakeClient(
        {
            "k9_textbook": _FakeCollection(
                _result(
                    ids=["t1"],
                    documents=["有效教材片段"],
                    distances=[0.1],
                )
            )
        }
    )
    provider = ChromaSemanticRetrievalProvider(
        _settings(reference_collections="k9_textbook,missing_collection"),
        client_factory=lambda settings: client,
        embedding_factory=lambda model, device: _FakeEmbedder([0.3]),
    )

    result = provider.retrieve(
        SemanticRetrievalRequest(problem_context="等式性质")
    )

    assert result.status is RetrievalStatus.SUCCESS
    assert [hit.text for hit in result.hits] == ["有效教材片段"]


def test_absolute_source_path_is_reduced_to_safe_filename() -> None:
    client = _FakeClient(
        {
            "k9_textbook": _FakeCollection(
                _result(
                    ids=["/home/private/chunks/t1"],
                    documents=["教材片段"],
                    distances=[0.1],
                    metadatas=[
                        {
                            "source_file": "/home/private/textbooks/math.pdf",
                            "book_title": "C:\\private\\titles\\math-book.pdf",
                            "local_path": "/srv/private/page-1.png",
                            "chunk_index": 1,
                        }
                    ],
                )
            )
        }
    )
    provider = ChromaSemanticRetrievalProvider(
        _settings(reference_collections="k9_textbook"),
        client_factory=lambda settings: client,
        embedding_factory=lambda model, device: _FakeEmbedder([0.3]),
    )

    result = provider.retrieve(
        SemanticRetrievalRequest(problem_context="等式性质")
    )

    assert result.hits[0].reference_id == "k9_textbook:t1"
    assert result.hits[0].source_label == "math-book.pdf"
    assert result.hits[0].metadata["source_file"] == "math.pdf"
    assert result.hits[0].metadata["local_path"] == "page-1.png"
    assert "/home/private" not in result.model_dump_json()
    assert "C:\\private" not in result.model_dump_json()
    assert "/srv/private" not in result.model_dump_json()


@pytest.mark.parametrize(
    ("client_factory", "embedding_factory", "error_code"),
    [
        (
            lambda settings: (_ for _ in ()).throw(ImportError("chromadb")),
            lambda model, device: _FakeEmbedder([0.1]),
            "dependency_unavailable",
        ),
        (
            lambda settings: (_ for _ in ()).throw(ConnectionError("offline")),
            lambda model, device: _FakeEmbedder([0.1]),
            "backend_unavailable",
        ),
        (
            lambda settings: _FakeClient({}),
            lambda model, device: _FakeEmbedder(RuntimeError("model failed")),
            "embedding_failed",
        ),
    ],
)
def test_provider_failures_are_sanitized_and_non_throwing(
    client_factory,
    embedding_factory,
    error_code: str,
) -> None:
    provider = ChromaSemanticRetrievalProvider(
        _settings(),
        client_factory=client_factory,
        embedding_factory=embedding_factory,
    )

    result = provider.retrieve(SemanticRetrievalRequest(problem_context="x+1=2"))

    assert result.status is RetrievalStatus.ERROR
    assert result.error_code == error_code
    assert "offline" not in result.model_dump_json()
    assert "model failed" not in result.model_dump_json()


def test_embedding_initialization_failure_is_classified_correctly() -> None:
    provider = ChromaSemanticRetrievalProvider(
        _settings(),
        client_factory=lambda settings: _FakeClient({}),
        embedding_factory=lambda model, device: (_ for _ in ()).throw(
            RuntimeError("private model initialization detail")
        ),
    )

    result = provider.retrieve(
        SemanticRetrievalRequest(problem_context="等式性质")
    )

    assert result.status is RetrievalStatus.ERROR
    assert result.error_code == "embedding_failed"
    assert "private model initialization detail" not in result.model_dump_json()


def test_disabled_provider_does_not_initialize_runtime() -> None:
    calls: list[str] = []
    provider = ChromaSemanticRetrievalProvider(
        _settings(enabled=False),
        client_factory=lambda settings: calls.append("client"),
        embedding_factory=lambda model, device: calls.append("embedding"),
    )

    result = provider.retrieve(SemanticRetrievalRequest(problem_context="x+1=2"))

    assert result.status is RetrievalStatus.DISABLED
    assert calls == []


def test_default_provider_is_reused_within_process(monkeypatch) -> None:
    get_semantic_retrieval_provider.cache_clear()
    monkeypatch.setenv("TEACHER_RAG_ENABLED", "true")
    try:
        first = get_semantic_retrieval_provider()
        second = get_semantic_retrieval_provider()
    finally:
        get_semantic_retrieval_provider.cache_clear()

    assert first is second
