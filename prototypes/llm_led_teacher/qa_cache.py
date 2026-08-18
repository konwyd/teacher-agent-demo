"""Curated high-quality Q&A reference library backed by the existing Chroma server.

历史优质回答只作为教师生成时的有界参考，绝不直接返回给学生。在线回合只读，
只有夜间增量任务或显式 seed 命令可以写入，避免模型回答自我回流和错误放大。

检索结果（包括“未找到”）按 ``问题 + 优质问答库版本`` 缓存。夜间发布新版本
后旧缓存自然失效，但没有被路由到优质问答库的问题不会因此触发向量检索。

Chroma 的 cosine 空间返回 distance = 1 - cosine，故 similarity = 1 - distance。
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from prototypes.llm_led_teacher.rag_hook import CHROMA_HOST, CHROMA_PORT, _get_embedder

QA_CACHE_COLLECTION = os.environ.get(
    "TEACHER_QUALITY_QA_COLLECTION",
    os.environ.get("TEACHER_QA_CACHE_COLLECTION", "quality_qa"),
)
LOOKUP_ENABLED = os.environ.get(
    "TEACHER_QUALITY_QA_ENABLE",
    os.environ.get("TEACHER_QA_CACHE_ENABLE", "0"),
) == "1"
SIMILARITY_THRESHOLD = float(
    os.environ.get(
        "TEACHER_QUALITY_QA_THRESHOLD",
        os.environ.get("TEACHER_QA_CACHE_THRESHOLD", "0.78"),
    )
)
DUPLICATE_THRESHOLD = float(
    os.environ.get("TEACHER_QUALITY_QA_DUPLICATE_THRESHOLD", "0.94")
)
TOP_K = int(
    os.environ.get(
        "TEACHER_QUALITY_QA_TOPK",
        os.environ.get("TEACHER_QA_CACHE_TOPK", "3"),
    )
)
STATE_DB_PATH = os.environ.get(
    "TEACHER_QUALITY_QA_STATE_DB",
    "data/runtime/quality_qa_state.sqlite3",
)
RETRIEVAL_CACHE_TTL_SECONDS = int(
    os.environ.get("TEACHER_QUALITY_QA_CACHE_TTL_SECONDS", "3600")
)
RETRIEVAL_CACHE_MAX_ENTRIES = int(
    os.environ.get("TEACHER_QUALITY_QA_CACHE_MAX_ENTRIES", "2048")
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_STATE_LOCK = threading.RLock()
_CACHE_LOCK = threading.RLock()
_RETRIEVAL_CACHE: OrderedDict[
    tuple[str, int, str, str, int, float],
    tuple[float, tuple["QaCacheHit", ...]],
] = OrderedDict()


@dataclass(frozen=True)
class QaCacheHit:
    """A single high-similarity match returned by :func:`lookup`."""

    id: str
    question: str
    answer: str
    similarity: float
    subject: str
    stage: str
    hit_count: int
    quality_score: float = 0.0
    category: str = ""
    source_conversation_id: str = ""
    content_hash: str = ""


@dataclass(frozen=True)
class QualityQaLookup:
    """Version-aware retrieval result, including cached empty results."""

    hits: tuple[QaCacheHit, ...]
    version: int
    retrieval_cache_hit: bool

    @property
    def text(self) -> str:
        chunks: list[str] = []
        used = 0
        for index, hit in enumerate(self.hits, 1):
            chunk = (
                f"[优质问答:{index}] 仅作历史经验参考，不代表标准答案\n"
                f"历史问题：{hit.question}\n"
                f"历史回答：{hit.answer}"
            )
            if used + len(chunk) > 3000:
                break
            chunks.append(chunk)
            used += len(chunk)
        return "\n\n".join(chunks)

    @property
    def refs(self) -> list[str]:
        return [f"[quality_qa:{hit.id}]" for hit in self.hits]


@dataclass(frozen=True)
class ReplaceSourceResult:
    changed: bool
    stored: int
    duplicates: int
    removed: int


def lookup_enabled() -> bool:
    return LOOKUP_ENABLED


def write_enabled() -> bool:
    """Legacy compatibility: online auto-write is intentionally disabled."""

    return False


def _embed(text: str) -> list[float]:
    return _get_embedder().encode(text, normalize_embeddings=True).tolist()


def _chroma_client():
    import chromadb

    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def _collection():
    """Get or create the dedicated Q&A collection in cosine space."""
    client = _chroma_client()
    try:
        return client.get_collection(name=QA_CACHE_COLLECTION)
    except Exception:
        return client.create_collection(
            name=QA_CACHE_COLLECTION,
            metadata={"hnsw:space": "cosine", "kind": "curated_quality_qa"},
        )


def _connect_state() -> sqlite3.Connection:
    path = Path(STATE_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS quality_qa_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    return connection


def current_version() -> int:
    """Return the currently published quality-Q&A library version."""
    try:
        with _STATE_LOCK, _connect_state() as connection:
            row = connection.execute(
                "SELECT value FROM quality_qa_metadata WHERE key = 'version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO quality_qa_metadata (key, value, updated_at) "
                    "VALUES ('version', '1', ?)",
                    (time.time(),),
                )
                return 1
            return max(1, int(row[0]))
    except (OSError, sqlite3.Error, ValueError):
        return 1


def publish_new_version() -> int:
    """Increment the version only after a batch has changed the library."""
    with _STATE_LOCK, _connect_state() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT value FROM quality_qa_metadata WHERE key = 'version'"
        ).fetchone()
        version = max(1, int(row[0])) + 1 if row is not None else 2
        connection.execute(
            """
            INSERT INTO quality_qa_metadata (key, value, updated_at)
            VALUES ('version', ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (str(version), time.time()),
        )
    clear_local_cache()
    return version


def clear_local_cache() -> None:
    with _CACHE_LOCK:
        _RETRIEVAL_CACHE.clear()


def _digit_set(text: str) -> frozenset[str]:
    return frozenset(_NUMBER_RE.findall(text))


def _numbers_compatible(incoming: str, cached: str) -> bool:
    """措辞相似但数字不同 ⇒ 不是同一道题，不能复用答案。

    任一侧没有数字（概念/定义类问题）时交给纯语义相似度判断。
    """
    incoming_digits = _digit_set(incoming)
    cached_digits = _digit_set(cached)
    if not incoming_digits or not cached_digits:
        return True
    return incoming_digits == cached_digits


def _query_hits(
    message: str,
    *,
    subject: str,
    stage: str,
    top_k: int,
) -> tuple[QaCacheHit, ...]:
    query: dict = {
        "query_embeddings": [_embed(message)],
        "n_results": max(TOP_K, top_k),
        "include": ["documents", "metadatas", "distances"],
    }
    filters: list[dict[str, str]] = []
    if subject:
        filters.append({"subject": subject})
    if stage:
        filters.append({"stage": stage})
    if len(filters) == 1:
        query["where"] = filters[0]
    elif filters:
        query["where"] = {"$and": filters}
    results = _collection().query(**query)
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    hits: list[QaCacheHit] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        similarity = 1.0 - float(distance)
        cached_question = str(metadata.get("question") or document or "")
        answer = str(metadata.get("answer") or "")
        if (
            similarity < SIMILARITY_THRESHOLD
            or not _numbers_compatible(message, cached_question)
            or not cached_question
            or not answer
        ):
            continue
        hits.append(
            QaCacheHit(
                id=str(metadata.get("id") or ""),
                question=cached_question,
                answer=answer,
                similarity=round(similarity, 4),
                subject=str(metadata.get("subject") or ""),
                stage=str(metadata.get("stage") or ""),
                hit_count=int(metadata.get("hit_count") or 0),
                quality_score=float(metadata.get("quality_score") or 0.0),
                category=str(metadata.get("category") or ""),
                source_conversation_id=str(
                    metadata.get("source_conversation_id") or ""
                ),
                content_hash=str(metadata.get("content_hash") or ""),
            )
        )
        if len(hits) >= top_k:
            break
    return tuple(hits)


def retrieve(
    message: str,
    *,
    subject: str = "",
    stage: str = "",
    top_k: int | None = None,
) -> QualityQaLookup:
    """Retrieve bounded references with positive/negative versioned caching."""
    normalized = " ".join(message.split()).strip()
    version = current_version()
    if not normalized:
        return QualityQaLookup((), version, False)
    effective_top_k = max(1, top_k or TOP_K)
    key = (
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        version,
        subject,
        stage,
        effective_top_k,
        float(SIMILARITY_THRESHOLD),
    )
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _RETRIEVAL_CACHE.get(key)
        if cached is not None and cached[0] > now:
            _RETRIEVAL_CACHE.move_to_end(key)
            return QualityQaLookup(cached[1], version, True)
        _RETRIEVAL_CACHE.pop(key, None)
    try:
        hits = _query_hits(
            normalized,
            subject=subject,
            stage=stage,
            top_k=effective_top_k,
        )
    except Exception:
        # 优质问答库故障不能阻断教材检索或正常生成。
        hits = ()
    with _CACHE_LOCK:
        _RETRIEVAL_CACHE[key] = (
            now + max(RETRIEVAL_CACHE_TTL_SECONDS, 1),
            hits,
        )
        _RETRIEVAL_CACHE.move_to_end(key)
        while len(_RETRIEVAL_CACHE) > max(RETRIEVAL_CACHE_MAX_ENTRIES, 1):
            _RETRIEVAL_CACHE.popitem(last=False)
    return QualityQaLookup(hits, version, False)


def lookup(message: str) -> QaCacheHit | None:
    """Compatibility wrapper returning the best reference, never a direct reply."""
    result = retrieve(message, top_k=1)
    return result.hits[0] if result.hits else None


def content_fingerprint(question: str, answer: str) -> str:
    canonical = f'{" ".join(question.split()).strip()}\n{" ".join(answer.split()).strip()}'
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def store(
    question: str,
    answer: str,
    *,
    subject: str = "",
    stage: str = "",
    quality_score: float = 1.0,
    category: str = "",
    source_conversation_id: str = "manual",
    source_updated_at: str = "",
    item_key: str = "",
) -> bool:
    """Explicitly store one reviewed Q&A pair; online turns never call this."""
    question = " ".join(question.split()).strip()
    answer = " ".join(answer.split()).strip()
    if not question or not answer:
        return False
    try:
        embedding = _embed(question)
        fingerprint = content_fingerprint(question, answer)
        stable_key = f"{source_conversation_id}:{item_key or fingerprint}"
        qid = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:24]
        _collection().upsert(
            ids=[qid],
            embeddings=[embedding],
            documents=[question[:4000]],
            metadatas=[
                {
                    "id": qid,
                    "question": question[:4000],
                    "answer": answer[:16000],
                    "subject": subject or "",
                    "stage": stage or "",
                    "quality_score": float(quality_score),
                    "category": category[:120],
                    "source_conversation_id": source_conversation_id,
                    "source_updated_at": source_updated_at,
                    "content_hash": fingerprint,
                    "hit_count": 0,
                    "created_at": time.time(),
                }
            ],
        )
        clear_local_cache()
        return True
    except Exception:
        return False


def _source_records(
    source_conversation_id: str,
) -> tuple[list[str], list[dict]]:
    result = _collection().get(
        where={"source_conversation_id": source_conversation_id},
        include=["metadatas"],
    )
    return (
        list(result.get("ids") or []),
        list(result.get("metadatas") or []),
    )


def _is_semantic_duplicate(
    question: str,
    answer: str,
    *,
    source_conversation_id: str,
) -> bool:
    fingerprint = content_fingerprint(question, answer)
    try:
        result = _collection().query(
            query_embeddings=[_embed(question)],
            n_results=max(TOP_K, 5),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return False
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        if (
            str(metadata.get("source_conversation_id") or "")
            == source_conversation_id
        ):
            continue
        if str(metadata.get("content_hash") or "") == fingerprint:
            return True
        cached_question = str(metadata.get("question") or document or "")
        if (
            1.0 - float(distance) >= DUPLICATE_THRESHOLD
            and _numbers_compatible(question, cached_question)
        ):
            return True
    return False


def replace_source(
    source_conversation_id: str,
    source_updated_at: str,
    entries: list[Mapping[str, object]],
) -> ReplaceSourceResult:
    """Replace one conversation's extracted entries and remove stale ones."""
    source_conversation_id = source_conversation_id.strip()
    if not source_conversation_id:
        raise ValueError("source_conversation_id is required")
    existing_ids, existing_metadata = _source_records(source_conversation_id)
    existing_hashes = {
        str(metadata.get("content_hash") or "")
        for metadata in existing_metadata
        if metadata
    }
    prepared: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    for index, raw in enumerate(entries):
        question = " ".join(str(raw.get("question") or "").split()).strip()
        answer = " ".join(str(raw.get("answer") or "").split()).strip()
        quality_score = float(raw.get("quality_score") or 0.0)
        if not question or not answer or quality_score <= 0:
            continue
        fingerprint = content_fingerprint(question, answer)
        if fingerprint in seen_hashes:
            continue
        seen_hashes.add(fingerprint)
        prepared.append(
            {
                **raw,
                "question": question,
                "answer": answer,
                "quality_score": quality_score,
                "fingerprint": fingerprint,
                "item_key": str(raw.get("item_key") or index),
            }
        )
    candidate_hashes = {
        str(item["fingerprint"]) for item in prepared
    }
    if existing_hashes == candidate_hashes:
        return ReplaceSourceResult(False, len(prepared), 0, 0)

    accepted: list[dict[str, object]] = []
    duplicates = 0
    for item in prepared:
        if _is_semantic_duplicate(
            str(item["question"]),
            str(item["answer"]),
            source_conversation_id=source_conversation_id,
        ):
            duplicates += 1
            continue
        accepted.append(item)

    desired_ids: set[str] = set()
    for item in accepted:
        stable_key = (
            f"{source_conversation_id}:{item['item_key']}"
        )
        qid = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:24]
        desired_ids.add(qid)
        if not store(
            str(item["question"]),
            str(item["answer"]),
            subject=str(item.get("subject") or ""),
            stage=str(item.get("stage") or ""),
            quality_score=float(item["quality_score"]),
            category=str(item.get("category") or ""),
            source_conversation_id=source_conversation_id,
            source_updated_at=source_updated_at,
            item_key=str(item["item_key"]),
        ):
            raise RuntimeError("failed to store curated quality-Q&A entry")

    stale_ids = [qid for qid in existing_ids if qid not in desired_ids]
    if stale_ids:
        _collection().delete(ids=stale_ids)
        clear_local_cache()
    return ReplaceSourceResult(
        changed=bool(accepted or stale_ids),
        stored=len(accepted),
        duplicates=duplicates,
        removed=len(stale_ids),
    )


def bump_hit(qid: str) -> None:
    """Fire-and-forget hit-count increment；失败不影响主流程。"""
    if not qid:
        return
    try:
        collection = _collection()
        existing = collection.get(ids=[qid], include=["metadatas"])
        metas = (existing.get("metadatas") or [None])[0]
        if not metas:
            return
        collection.update(
            ids=[qid],
            metadatas=[{**metas, "hit_count": int(metas.get("hit_count") or 0) + 1}],
        )
    except Exception:
        pass
