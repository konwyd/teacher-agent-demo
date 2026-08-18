"""Tests for the semantic Q&A cache (prototypes.llm_led_teacher.qa_cache).

Uses an in-memory Chroma client plus a deterministic fake embedder so the suite
runs without a live Chroma server or the bge-m3 model.  Chroma is imported via
pytest.importorskip; the whole file is skipped when chromadb is missing.
"""

from __future__ import annotations

import uuid

import pytest

from prototypes.llm_led_teacher import qa_cache


def fake_embed(text: str) -> list[float]:
    """Deterministic character-bag embedding: near-duplicates get high cosine.

    "什么是方程？" 与 "什么是方程呢？" 的余弦约为 0.93，接近真实语义
    模型的量级；无关问题会显著更低。
    """
    vector = [0.0] * 64
    for char in text:
        vector[hash(char) % 64] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm:
        vector = [value / norm for value in vector]
    return vector


@pytest.fixture()
def chroma_ready(monkeypatch, tmp_path):
    pytest.importorskip("chromadb")
    import chromadb

    client = chromadb.EphemeralClient()
    monkeypatch.setattr(qa_cache, "_chroma_client", lambda: client)
    monkeypatch.setattr(qa_cache, "QA_CACHE_COLLECTION", f"qa_test_{uuid.uuid4().hex[:8]}")
    monkeypatch.setattr(
        qa_cache,
        "STATE_DB_PATH",
        str(tmp_path / "quality_qa_state.sqlite3"),
    )
    monkeypatch.setattr(qa_cache, "_embed", fake_embed)
    qa_cache.clear_local_cache()
    return client


# --- 纯逻辑：数字守卫 -------------------------------------------------------


def test_numbers_compatible():
    assert qa_cache._numbers_compatible(
        "小明有12个苹果，吃了3个，还剩几个？",
        "小明有12个苹果，吃了3个，还剩几个？",
    )
    assert not qa_cache._numbers_compatible(
        "小明有12个苹果，吃了3个，还剩几个？",
        "小红有5个橘子，吃了2个，还剩几个？",
    )
    # 概念/定义类问题不含数字，交给语义相似度判断
    assert qa_cache._numbers_compatible("什么是方程？", "小明有12个苹果，吃了3个")
    assert qa_cache._numbers_compatible("小明有12个苹果，吃了3个", "什么是方程？")


# --- 集成：命中 / 未命中 / 阈值 / 数字拦截 ---------------------------------


def test_lookup_returns_stored_answer(chroma_ready):
    assert qa_cache.store(
        "什么是方程？",
        "方程就是含有未知数的等式。",
        subject="数学",
        stage="初中",
    )
    hit = qa_cache.lookup("什么是方程？")
    assert hit is not None
    assert "方程就是含有未知数的等式" in hit.answer
    assert hit.similarity >= qa_cache.SIMILARITY_THRESHOLD
    assert hit.subject == "数学"
    assert hit.stage == "初中"


def test_lookup_misses_on_unrelated(chroma_ready):
    qa_cache.store("什么是方程？", "方程就是含有未知数的等式。")
    assert qa_cache.lookup("推荐一台笔记本电脑") is None


def test_digit_guard_blocks_different_numbers(chroma_ready, monkeypatch):
    qa_cache.store("小明有12个苹果，吃了3个，还剩几个？", "还剩9个。")
    # 放低阈值，让"措辞相似"能通过语义相似度这一关，专门检验数字守卫
    monkeypatch.setattr(qa_cache, "SIMILARITY_THRESHOLD", 0.5)
    # 数字相同 → 命中
    assert qa_cache.lookup("小明有12个苹果 吃了3个 还剩几个？") is not None
    # 措辞相似但数字不同 → 即使语义相似也拦截
    assert qa_cache.lookup("小红有5个橘子，吃了2个，还剩几个？") is None


def test_threshold_blocks_low_similarity(chroma_ready, monkeypatch):
    qa_cache.store("什么是方程？", "方程就是含有未知数的等式。")
    # 放低阈值 → 相近措辞命中
    monkeypatch.setattr(qa_cache, "SIMILARITY_THRESHOLD", 0.8)
    assert qa_cache.lookup("什么是方程呢？") is not None
    # 拉高阈值 → 同一句话被拦截
    monkeypatch.setattr(qa_cache, "SIMILARITY_THRESHOLD", 0.99)
    assert qa_cache.lookup("什么是方程呢？") is None


def test_store_rejects_blank_input(chroma_ready):
    assert not qa_cache.store("", "答案")
    assert not qa_cache.store("问题", "   ")
    assert not qa_cache.store("", "  ")


def test_published_version_invalidates_positive_and_negative_cache(
    chroma_ready,
    monkeypatch,
):
    calls = 0

    def fake_query(message: str, *, subject: str, stage: str, top_k: int):
        nonlocal calls
        calls += 1
        return ()

    monkeypatch.setattr(qa_cache, "_query_hits", fake_query)
    first = qa_cache.retrieve("还没有入库的问题")
    second = qa_cache.retrieve("还没有入库的问题")

    assert calls == 1
    assert first.retrieval_cache_hit is False
    assert second.retrieval_cache_hit is True
    assert second.version == first.version

    published = qa_cache.publish_new_version()
    third = qa_cache.retrieve("还没有入库的问题")

    assert published == first.version + 1
    assert third.version == published
    assert third.retrieval_cache_hit is False
    assert calls == 2


def test_replace_source_is_idempotent_and_removes_stale_entries(chroma_ready):
    entry = {
        "question": "移项为什么要变号？",
        "answer": "本质是等式两边同时进行相反运算。",
        "quality_score": 0.95,
        "subject": "数学",
        "stage": "初中",
        "item_key": "turn-1",
    }

    first = qa_cache.replace_source("conv_a", "v1", [entry])
    second = qa_cache.replace_source("conv_a", "v1", [entry])
    removed = qa_cache.replace_source("conv_a", "v2", [])

    assert first.changed is True
    assert first.stored == 1
    assert second.changed is False
    assert removed.changed is True
    assert removed.removed == 1
    assert qa_cache.lookup("移项为什么要变号？") is None


def test_semantic_duplicate_from_another_source_does_not_publish_change(
    chroma_ready,
):
    entry = {
        "question": "移项为什么要变号？",
        "answer": "本质是等式两边同时进行相反运算。",
        "quality_score": 0.95,
        "subject": "数学",
        "stage": "初中",
        "item_key": "turn-1",
    }
    qa_cache.replace_source("conv_a", "v1", [entry])

    duplicate = qa_cache.replace_source("conv_b", "v1", [entry])

    assert duplicate.changed is False
    assert duplicate.stored == 0
    assert duplicate.duplicates == 1
