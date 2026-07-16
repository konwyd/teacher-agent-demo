"""多集合知识库管理器（Multi-Collection KB Manager）—— B 方案。

在**同一个 ChromaDB 向量库目录**下管理多个 collection，按知识类型分库：

    china_textbook_k9_vectordb/          ← 一个向量库目录（一个 sqlite3）
    ├── collection: k9_textbook          （已有，由 textbook_rag.py 管理）
    ├── collection: k9_exercises         （新增：初中题库 JSONL + 中考真题）
    ├── collection: curriculum_std       （新增：课程标准/新课标，待补数据）
    ├── collection: teacher_cert         （新增：教师资格证考试，待补数据）
    └── collection: teacher_guide        （新增：教师教学用书/教参，待补数据）

「多集合」≠「多向量库」：所有 collection 共存于一个 ChromaDB PersistentClient
（一个目录），共享同一套 bge-m3 嵌入模型，区别仅在 collection 名与元数据。

设计要点：
- 复用 textbook_rag 已加载的 bge-m3 emb_fn，避免重复加载 ~2.2GB 模型；
- 复用同一个 ChromaDB 路径（多 collection 共存于一个目录）；
- lazy 加载：collection 不存在或无数据时跳过，不影响其他集合与 agent 主流程；
- use_rag 开关语义与 textbook_rag 一致（None=跟随env / True=强制 / False=跳过）；
- 检索接口与 textbook_rag 对齐：retrieve / format_context / format_refs。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from app.retrieval.chroma_compat import ensure_chroma_compat, get_chroma_client

ensure_chroma_compat()  # 在 import chromadb 之前：切 SegmentAPI + 补 hnswlib 扩展

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 配置
# 与 textbook_rag 共享路径/模型，保证 collection 共存于同一向量库目录

DEFAULT_VECTORDB_PATH = "/data/education/data/china_textbook_k9_vectordb"
DEFAULT_EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"
DEFAULT_MAX_CONTEXT_CHARS = 2000


def _vectordb_path() -> str:
    return os.getenv("TEACHER_RAG_VECTORDB_PATH", DEFAULT_VECTORDB_PATH)


def _embed_model() -> str:
    return os.getenv("TEACHER_RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def _rag_enabled() -> bool:
    flag = os.getenv("TEACHER_RAG_ENABLED", "1").lower()
    return flag not in ("0", "false", "no", "off")


def _deps_available() -> bool:
    """依赖与向量库目录是否存在（不看 env 开关，供强制模式使用）。"""
    if not Path(_vectordb_path()).exists():
        return False
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------- Collection 注册表


@dataclass(frozen=True)
class CollectionSpec:
    """单个知识库集合的规格定义。"""

    name: str  # ChromaDB collection 名
    doc_type: str  # 文档类型标识（写入 metadata.doc_type）
    label: str  # 中文标签（格式化 header / 日志用）
    default_topk: int


# 除 k9_textbook（由 textbook_rag.py 独立管理）外的所有知识库集合
COLLECTIONS: dict[str, CollectionSpec] = {
    "k9_exercises": CollectionSpec("k9_exercises", "exercise", "题库", 3),
    "curriculum_std": CollectionSpec("curriculum_std", "curriculum_std", "课程标准", 3),
    "teacher_cert": CollectionSpec("teacher_cert", "teacher_cert", "教资", 3),
    "teacher_guide": CollectionSpec("teacher_guide", "teacher_guide", "教参", 3),
}


# ---------------------------------------------------------------- 共享嵌入函数

_emb_fn_instance: Any = None
_emb_fn_tried: bool = False


def _get_shared_emb_fn() -> Any:
    """复用 textbook_rag 已加载的 bge-m3 emb_fn，避免重复加载 ~2.2GB 模型。

    若 textbook_rag 不可用（env 关闭或依赖缺失），则自行加载。
    """
    global _emb_fn_instance, _emb_fn_tried
    if _emb_fn_instance is not None:
        return _emb_fn_instance
    if _emb_fn_tried:
        return _emb_fn_instance  # 之前尝试过（可能为 None）
    _emb_fn_tried = True
    # 优先复用 textbook_rag 已加载的 emb_fn（retrieve_node 中教材检索先于题库检索）
    try:
        from app.retrieval.textbook_rag import get_textbook_rag, get_textbook_rag_forced

        rag = get_textbook_rag()
        if rag is None:
            rag = get_textbook_rag_forced()
        if rag is not None and hasattr(rag, "_emb_fn"):
            _emb_fn_instance = rag._emb_fn
            logger.info("KBManager: 复用 textbook_rag 的 emb_fn（省去重复加载 bge-m3）")
            return _emb_fn_instance
    except Exception:  # noqa: BLE001
        pass
    # 自行加载
    try:
        from chromadb.utils import embedding_functions

        _emb_fn_instance = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=_embed_model(),
            normalize_embeddings=True,
        )
        logger.info("KBManager: 自行加载 emb_fn (%s)", _embed_model())
    except Exception as exc:  # noqa: BLE001
        logger.warning("KBManager: emb_fn 加载失败: %s", exc)
    return _emb_fn_instance


# ---------------------------------------------------------------- 检索器


class KBManager:
    """多集合知识库管理器：一个 ChromaDB client + 多个 collection。

    与 textbook_rag.TextbookRAG 共享同一个向量库目录与嵌入模型，
    在此之上管理 k9_exercises / curriculum_std / teacher_cert / teacher_guide
    等额外集合。每个集合 lazy 加载，不存在时优雅跳过。
    """

    def __init__(self) -> None:
        import chromadb

        db_path = _vectordb_path()
        logger.info("初始化 KBManager: db=%s", db_path)
        self._client = get_chroma_client(db_path)
        self._emb_fn = _get_shared_emb_fn()
        self._coll_cache: dict[str, Any] = {}  # name -> Collection | None

    def _get_collection(self, key: str) -> Any:
        """lazy 获取 collection；不存在则缓存 None 并返回 None。"""
        if key in self._coll_cache:
            return self._coll_cache[key]
        spec = COLLECTIONS.get(key)
        if spec is None:
            self._coll_cache[key] = None
            return None
        try:
            coll = self._client.get_collection(name=spec.name, embedding_function=self._emb_fn)
            logger.info("KBManager: 集合 %s 就绪，共 %d 条", spec.name, coll.count())
            self._coll_cache[key] = coll
            return coll
        except Exception as exc:  # noqa: BLE001
            logger.info("KBManager: 集合 %s 不可用（%s），跳过", spec.name, exc)
            self._coll_cache[key] = None
            return None

    def retrieve(
        self,
        query: str,
        collection: str,
        topk: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """从指定集合检索，返回 [{"text","meta","distance","collection"}, ...]。"""
        spec = COLLECTIONS.get(collection)
        if spec is None:
            logger.warning("KBManager: 未知集合 %s", collection)
            return []
        coll = self._get_collection(collection)
        if coll is None:
            return []
        k = topk or spec.default_topk
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": k}
        if where:
            kwargs["where"] = where
        try:
            res = coll.query(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("KBManager 检索 %s 失败: %s", collection, exc)
            return []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [
            {"text": doc, "meta": meta, "distance": dist, "collection": collection}
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    # ---- 题库便捷检索 ----

    def retrieve_exercises(
        self,
        query: str,
        topk: int | None = None,
        subject: str = "数学",
        difficulty: str | None = None,
    ) -> list[dict[str, Any]]:
        """检索题库，默认按学科=数学过滤（agent 是数学老师）。"""
        where: dict[str, Any] = {"subject": subject} if subject else {}
        if difficulty:
            where["difficulty"] = difficulty
        return self.retrieve(query, "k9_exercises", topk=topk, where=where or None)

    # ---- 格式化 ----

    def format_context(
        self,
        hits: list[dict[str, Any]],
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> str:
        """把命中片段格式化为可注入 prompt 的知识库上下文。

        按 collection 类型生成不同 header：
        - exercise:  题库·数学·单选题·较易《知识点》
        - 其他:       {label}·{subject}{grade}《{chapter}》
        """
        parts: list[str] = []
        used = 0
        for i, h in enumerate(hits, 1):
            m = h.get("meta", {}) or {}
            coll = h.get("collection", "")
            spec = COLLECTIONS.get(coll)
            label = spec.label if spec else "参考"
            header = self._build_header(label, m)
            block = f"[{label}{i}] {header}\n{h.get('text', '')}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    @staticmethod
    def _build_header(label: str, m: dict[str, Any]) -> str:
        doc_type = m.get("doc_type", "")
        if doc_type == "exercise":
            kvs = [
                m.get("subject", ""),
                m.get("ques_type", ""),
                m.get("difficulty", ""),
            ]
            head = "·".join(x for x in kvs if x)
            klg = m.get("knowledges", "")
            if klg:
                head += f"《{klg}》"
            return head
        # 通用 header：label·学段学科 年级学期《章节》
        stage_subj = f"{m.get('stage', '')}{m.get('subject', '')}".strip()
        grade_sem = f"{m.get('grade', '')}{m.get('semester', '')}".strip()
        chapter = m.get("chapter", "")
        parts = [p for p in (stage_subj, grade_sem, f"《{chapter}》" if chapter else "") if p]
        return " ".join(parts)

    def format_refs(self, hits: list[dict[str, Any]], limit: int = 3) -> list[str]:
        """简短引用（供 UI / 日志展示，不含正文）。"""
        refs: list[str] = []
        for h in hits[:limit]:
            m = h.get("meta", {}) or {}
            coll = h.get("collection", "")
            spec = COLLECTIONS.get(coll)
            label = spec.label if spec else "参考"
            header = self._build_header(label, m)
            sim = 1 - h.get("distance", 0)
            refs.append(f"{label}·{header}(相似度{sim:.2f})".strip("·"))
        return refs


# ---------------------------------------------------------------- 单例管理（与 textbook_rag 同构）


_kb_manager: KBManager | None = None
_kb_tried: bool = False
_forced_kb: KBManager | None = None
_forced_kb_tried: bool = False


def get_kb_manager() -> KBManager | None:
    """进程级单例（跟随 env 开关）；不可用时返回 None。"""
    global _kb_manager, _kb_tried
    if not _rag_enabled():
        return None
    if _kb_tried:
        return _kb_manager
    _kb_tried = True
    if not _deps_available():
        logger.info("KBManager 不可用（依赖缺失或向量库不存在），跳过多集合检索")
        return None
    try:
        _kb_manager = KBManager()
    except Exception as exc:  # noqa: BLE001
        logger.warning("KBManager 初始化失败，降级为无多集合检索: %s", exc)
        return None
    return _kb_manager


def get_kb_manager_forced() -> KBManager | None:
    """强制获取（绕过 env 开关，供 use_rag=True）；仍需依赖+向量库。"""
    global _forced_kb, _forced_kb_tried
    existing = get_kb_manager()
    if existing is not None:
        return existing
    if _forced_kb_tried:
        return _forced_kb
    _forced_kb_tried = True
    if not _deps_available():
        logger.info("KBManager 强制初始化失败：依赖缺失或向量库不存在")
        return None
    try:
        _forced_kb = KBManager()
        logger.info("KBManager 强制初始化成功（绕过 env 开关）")
    except Exception as exc:  # noqa: BLE001
        logger.warning("KBManager 强制初始化失败: %s", exc)
        return None
    return _forced_kb


def reset_kb_manager() -> None:
    """重置所有缓存（测试用）。"""
    global _kb_manager, _kb_tried, _forced_kb, _forced_kb_tried, _emb_fn_instance, _emb_fn_tried
    _kb_manager = None
    _kb_tried = False
    _forced_kb = None
    _forced_kb_tried = False
    _emb_fn_instance = None
    _emb_fn_tried = False
