"""Textbook RAG: 向量检索 K9 教材知识库（ChromaDB + bge-m3）。

本模块在项目原有的 Graph RAG（概念图 mock）之上，补充基于向量检索的
教材原文召回，让 Teacher Agent 的讲解能 grounded 在真实教材内容上。

设计要点：
- 懒加载 chromadb / sentence_transformers，缺失时优雅降级（返回空结果），
  不影响 agent 主流程；
- 进程级单例（lru_cache），bge-m3 模型只加载一次；
- 路径 / 模型 / 集合名 / topk / 学科过滤均可通过环境变量配置；
- 检索接口返回带元数据的片段，并提供 format_context 直接注入 prompt。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from app.retrieval.chroma_compat import ensure_chroma_compat, get_chroma_client

ensure_chroma_compat()  # 在 import chromadb 之前：切 SegmentAPI + 补 hnswlib 扩展

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 配置

DEFAULT_VECTORDB_PATH = "/data/education/data/china_textbook_k9_vectordb"
DEFAULT_EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"
DEFAULT_COLLECTION = "k9_textbook"
DEFAULT_TOPK = 5
DEFAULT_SUBJECT_FILTER = "数学"  # agent 是数学老师，默认只召回数学教材
DEFAULT_MAX_CONTEXT_CHARS = 2400


def _rag_enabled() -> bool:
    flag = os.getenv("TEACHER_RAG_ENABLED", "1").lower()
    return flag not in ("0", "false", "no", "off")


def _vectordb_path() -> str:
    return os.getenv("TEACHER_RAG_VECTORDB_PATH", DEFAULT_VECTORDB_PATH)


def _embed_model() -> str:
    return os.getenv("TEACHER_RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def _collection_name() -> str:
    return os.getenv("TEACHER_RAG_COLLECTION", DEFAULT_COLLECTION)


def _topk() -> int:
    try:
        return max(1, int(os.getenv("TEACHER_RAG_TOPK", str(DEFAULT_TOPK))))
    except ValueError:
        return DEFAULT_TOPK


def _subject_filter() -> str:
    """返回学科过滤值，空串表示不过滤。"""
    return os.getenv("TEACHER_RAG_SUBJECT_FILTER", DEFAULT_SUBJECT_FILTER)


# ---------------------------------------------------------------- 可用性探测


def is_rag_available() -> bool:
    """RAG 是否可用（依赖已装 + 向量库存在 + 未被 env 关闭）。"""
    if not _rag_enabled():
        return False
    if not Path(_vectordb_path()).exists():
        return False
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _rag_deps_available() -> bool:
    """检查依赖和向量库是否存在（不看环境变量开关，供强制模式使用）。"""
    if not Path(_vectordb_path()).exists():
        return False
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------- 检索器


class TextbookRAG:
    """基于 ChromaDB 的教材向量检索器。

    与 wyd-workspace/scripts/rag_retriever.py 同源：相同的集合名、嵌入模型
    （BAAI/bge-m3, 1024 维, cosine）与元数据结构，因此可直接复用已建好的
    向量库，无需重新入库。
    """

    def __init__(self) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        db_path = _vectordb_path()
        model = _embed_model()
        collection = _collection_name()

        logger.info("初始化 TextbookRAG: db=%s collection=%s model=%s", db_path, collection, model)
        self._client = get_chroma_client(db_path)
        self._emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model,
            normalize_embeddings=True,
        )
        self._coll = self._client.get_collection(name=collection, embedding_function=self._emb_fn)
        logger.info("TextbookRAG 就绪，集合 %s 共 %d 条", collection, self._coll.count())

    def retrieve(
        self,
        query: str,
        topk: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """检索教材片段，返回 [{"text","meta","distance"}, ...]。"""
        k = topk or _topk()
        filt = where
        if filt is None:
            subj = _subject_filter()
            if subj:
                filt = {"subject": subj}
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": k}
        if filt:
            kwargs["where"] = filt
        try:
            res = self._coll.query(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TextbookRAG 检索失败: %s", exc)
            return []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [
            {"text": doc, "meta": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def format_context(self, hits: list[dict[str, Any]], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> str:
        """把命中片段格式化为可注入 prompt 的教材上下文。"""
        parts: list[str] = []
        used = 0
        for i, h in enumerate(hits, 1):
            m = h.get("meta", {}) or {}
            header = (
                f"{m.get('stage', '')}{m.get('subject', '')} "
                f"{m.get('grade', '')}{m.get('semester', '')} "
                f"《{m.get('chapter', '')}》"
            ).strip()
            block = f"[教材参考{i}] {header}\n{h.get('text', '')}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def format_refs(self, hits: list[dict[str, Any]], limit: int = 3) -> list[str]:
        """简短引用（供 UI / 日志展示，不含正文）。"""
        refs: list[str] = []
        for h in hits[:limit]:
            m = h.get("meta", {}) or {}
            ref = (
                f"{m.get('subject', '')}·{m.get('grade', '')}{m.get('semester', '')} "
                f"《{m.get('chapter', '')}》(相似度{1 - h.get('distance', 0):.2f})"
            ).strip()
            refs.append(ref)
        return refs


@lru_cache
def get_textbook_rag() -> TextbookRAG | None:
    """进程级单例；不可用时返回 None（agent 主流程继续，仅缺教材召回）。"""
    if not is_rag_available():
        logger.info("TextbookRAG 不可用（依赖缺失/向量库不存在/已被 env 关闭），跳过教材检索")
        return None
    try:
        return TextbookRAG()
    except Exception as exc:  # noqa: BLE001
        logger.warning("TextbookRAG 初始化失败，本轮起降级为无教材检索: %s", exc)
        return None


# 强制模式的模块级缓存（use_rag=True 时绕过 env 检查）
_forced_rag: TextbookRAG | None = None
_forced_rag_tried: bool = False


def get_textbook_rag_forced() -> TextbookRAG | None:
    """强制获取 RAG 实例（跳过 TEACHER_RAG_ENABLED 检查，但仍需依赖+向量库）。

    用于 use_rag=True 场景：即使环境变量关闭，也尝试初始化。
    已初始化时优先复用常规单例；否则用独立的模块级缓存避免重复加载模型。
    """
    global _forced_rag, _forced_rag_tried
    # 优先复用已初始化的常规实例
    existing = get_textbook_rag()
    if existing is not None:
        return existing
    # 常规实例不可用（env 关闭），走强制路径
    if _forced_rag_tried:
        return _forced_rag
    _forced_rag_tried = True
    if not _rag_deps_available():
        logger.info("TextbookRAG 强制初始化失败：依赖缺失或向量库不存在")
        return None
    try:
        _forced_rag = TextbookRAG()
        logger.info("TextbookRAG 强制初始化成功（绕过 env 开关）")
        return _forced_rag
    except Exception as exc:  # noqa: BLE001
        logger.warning("TextbookRAG 强制初始化失败: %s", exc)
        return None


def reset_forced_rag() -> None:
    """重置强制模式缓存（测试用）。"""
    global _forced_rag, _forced_rag_tried
    _forced_rag = None
    _forced_rag_tried = False


def retrieve_textbook(query: str, topk: int | None = None) -> list[dict[str, Any]]:
    """便捷调用：不可用时返回空列表。"""
    rag = get_textbook_rag()
    if rag is None:
        return []
    return rag.retrieve(query, topk=topk)
