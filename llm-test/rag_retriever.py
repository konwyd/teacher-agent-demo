"""
RAG 检索器：对接 Chroma Server + bge-m3 向量库，支持多集合检索
支持 BM25（关键词）+ bge-m3（语义）双路混合检索 + RRF 融合排序
严格按照 RAG_DEPLOYMENT_GUIDE.md 规范实现
"""

from __future__ import annotations

import logging
import re
import os
from functools import lru_cache
from typing import Any

import numpy as np
import jieba

from rag_device import resolve_rag_device

logger = logging.getLogger(__name__)

# ============================================================
# 默认路径（与 teacher-agent-demo 共用同一份向量库）
# ============================================================
DEFAULT_VECTORDB_PATH = "/data/education/data/china_textbook_k9_vectordb"
DEFAULT_EMBED_MODEL = "/home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3"
DEFAULT_COLLECTION = "k9_textbook"
DEFAULT_EXERCISE_COLLECTION = "k9_exercises"
DEFAULT_TOPK = 5
FINAL_TOPK = 10         # 最大注入候选数
EXERCISE_TOPK = 5        # 题库检索数
MIN_SIM_THRESHOLD = 0.50 # 本地/联网统一注入阈值

# Chroma Server 地址（从配置读取，默认 127.0.0.1:8002）
_CHROMA_SERVER = os.environ.get("CHROMA_SERVER", "127.0.0.1:8002").strip()


def _strip_latex(text: str) -> str:
    """去除 LaTeX 公式，保留纯文本用于语义检索"""
    text = re.sub(r'\\frac\{[^}]*\}\{[^}]*\}', '', text)
    text = re.sub(r'\\sqrt\{[^}]*\}', '', text)
    text = re.sub(r'\\left\|?', '', text)
    text = re.sub(r'\\right\|?', '', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$]*\$', '', text)
    text = text.replace('{', '').replace('}', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_keywords(text: str, max_len: int = 200) -> str:
    """从题目中提取关键语义信息用于检索 query"""
    clean = _strip_latex(text)
    # 跳过"注意事项"等模板化前缀，直接从题目内容开始提取
    m = re.search(r'(一[、，.]\s*(?:选择题|单项选择|积累与运用|基础知识)|第[ⅠI一二]卷)', clean)
    if m:
        clean = clean[m.start():]
    # 去掉纯数字/符号噪音，保留中文词和≥2字的英文词
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{2,}', clean)
    return ' '.join(tokens)[:max_len]


def _is_near_duplicate(question: str, candidate: str, threshold: float = 0.72) -> bool:
    """检测题库候选是否复述当前题目；只用于 clean profile 排除近重复题。"""
    normalize = lambda s: re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9]', '', _strip_latex(s)).lower()
    q = normalize(question)
    c = normalize(candidate)
    if len(q) < 16 or len(c) < 16:
        return False
    if q in c:
        return True
    width = 4
    qgrams = {q[i:i + width] for i in range(len(q) - width + 1)}
    cgrams = {c[i:i + width] for i in range(len(c) - width + 1)}
    if not qgrams:
        return False
    # 使用题目覆盖率；候选可能是包含该题的长试卷，不能用对称 Jaccard。
    return len(qgrams & cgrams) / len(qgrams) >= threshold


# ============================================================
# 混合检索：BM25（关键词）+ bge-m3（语义）+ RRF 融合
# ============================================================

def _tokenize(text: str) -> list[str]:
    """jieba 分词，用于 BM25"""
    return [w.strip() for w in jieba.cut(text) if w.strip()]


def _rrf_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    """RRF (Reciprocal Rank Fusion) 融合双路检索结果。

    不依赖绝对分值，只看排名。对 BM25 和向量相似度的量纲差异不敏感。

    Args:
        dense_results: bge-m3 语义检索结果（含 _rank 和 sim）
        sparse_results: BM25 关键词检索结果（含 _rank 和 bm25_score）
        k: RRF 平滑参数，默认 60
        alpha: dense 权重，0.5 表示两路等权

    Returns:
        融合后按 rrf_score 降序排列的结果
    """
    rrf_scores: dict[str, float] = {}

    for r in dense_results:
        doc_id = r.get("_doc_id", r.get("text", "")[:80])
        rank = r.get("_rank", 100) + 1
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + alpha / (k + rank)

    for r in sparse_results:
        doc_id = r.get("_doc_id", r.get("text", "")[:80])
        rank = r.get("_rank", 100) + 1
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - alpha) / (k + rank)

    # 按 RRF 分数排序
    all_results = {r.get("_doc_id", r.get("text", "")[:80]): r for r in dense_results + sparse_results}
    for r in all_results.values():
        doc_id = r.get("_doc_id", r.get("text", "")[:80])
        r["rrf_score"] = rrf_scores.get(doc_id, 0)

    merged = sorted(all_results.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
    return merged


def _build_bm25_index(documents: list[str]) -> Any:
    """构建 BM25 索引"""
    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize(doc) for doc in documents]
    return BM25Okapi(tokenized)


def _bm25_search(query: str, bm25_index: Any, documents: list[dict[str, Any]], topk: int = 30) -> list[dict[str, Any]]:
    """BM25 关键词检索"""
    tokenized_query = _tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    ranked = np.argsort(scores)[::-1][:topk]
    results = []
    for rank, idx in enumerate(ranked):
        if scores[idx] <= 0:
            continue
        r = dict(documents[idx])
        r["bm25_score"] = float(scores[idx])
        r["_rank"] = rank
        r["_doc_id"] = r.get("text", "")[:80]
        results.append(r)
    return results


def hybrid_retrieve(
    query: str,
    coll: Any,
    emb_fn: Any,
    topk: int = 5,
    where: dict | None = None,
    recall_topk: int = 30,
) -> list[dict[str, Any]]:
    """双路混合检索：BM25 + bge-m3，RRF 融合。

    流程：
      1. 从 ChromaDB 拉取候选集（用 metadata 过滤后的全部文档，或向量粗筛 top-recall_topk）
      2. BM25 关键词检索 → top-recall_topk
      3. bge-m3 语义检索 → top-recall_topk
      4. RRF 融合 → top-k

    Args:
        query: 检索 query
        coll: ChromaDB collection
        emb_fn: embedding function
        topk: 最终返回条数
        where: metadata 过滤条件
        recall_topk: 粗筛召回数（BM25 和向量各取这么多）

    Returns:
        [{"text","meta","distance","source","sim","rrf_score"}, ...]
    """
    # Step 1: 向量粗筛 → 拉取候选文档集（不拉全量，避免 SQLite too many SQL variables）
    candidate_docs: list[str] = []
    candidate_metas: list[dict] = []
    try:
        vec_kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(recall_topk * 5, 500),
            "include": ["documents", "metadatas"],
        }
        if where:
            vec_kwargs["where"] = where
        vec_res = coll.query(**vec_kwargs)
        candidate_docs = vec_res.get("documents", [[]])[0]
        candidate_metas = vec_res.get("metadatas", [[]])[0]
    except Exception:
        logger.warning("hybrid_retrieve: 向量粗筛失败", exc_info=True)
        return []

    if not candidate_docs:
        return []

    logger.info("hybrid_retrieve: 向量粗筛 → %d 条 (where=%s)", len(candidate_docs), where)

    # Step 2: 构建 BM25 索引 + 检索
    candidate_list = [
        {"text": doc, "meta": meta}
        for doc, meta in zip(candidate_docs, candidate_metas)
    ]
    bm25_index = _build_bm25_index([d["text"] for d in candidate_list])
    sparse_results = _bm25_search(query, bm25_index, candidate_list, topk=recall_topk)
    logger.info("BM25 检索: %d 条命中", len(sparse_results))

    # Step 3: bge-m3 向量检索
    dense_results: list[dict[str, Any]] = []
    try:
        vec_kwargs: dict[str, Any] = {"query_texts": [query], "n_results": min(recall_topk, len(candidate_docs))}
        if where:
            vec_kwargs["where"] = where
        vec_res = coll.query(**vec_kwargs)
        vec_docs = vec_res.get("documents", [[]])[0]
        vec_metas = vec_res.get("metadatas", [[]])[0]
        vec_dists = vec_res.get("distances", [[]])[0]
        for rank, (doc, meta, dist) in enumerate(zip(vec_docs, vec_metas, vec_dists)):
            sim = 1.0 - dist if dist < 2 else 0.0
            dense_results.append({
                "text": doc, "meta": meta, "distance": dist, "source": "hybrid",
                "sim": sim, "_rank": rank, "_doc_id": doc[:80],
            })
    except Exception:
        logger.warning("bge-m3 向量检索失败", exc_info=True)
    logger.info("bge-m3 检索: %d 条命中", len(dense_results))

    # Step 4: RRF 融合
    merged = _rrf_fusion(dense_results, sparse_results)

    # 取 top-k，统一字段
    final = []
    for r in merged[:topk]:
        rrf = r.get("rrf_score", 0)
        vec_sim = r.get("sim", 0)
        bm25 = r.get("bm25_score", 0)
        # 两路权重 alpha + (1-alpha) = 1，因此理论最大值是 1/(k+1)。
        # 旧代码误用了 2/(k+1)，会把所有 RRF 综合分数压低一半。
        rrf_max = 1.0 / (60 + 1)
        rrf_normalized = min(rrf / rrf_max, 1.0) if rrf_max > 0 else 0.0
        # RRF 只表示相对名次，单独使用时第一名即使不相关也会接近 1。
        # 融合分保留语义相似度的绝对信息，才可以用于 0.65 这类触发阈值。
        fusion_sim = max(0.0, min(1.0, 0.7 * vec_sim + 0.3 * rrf_normalized))
        final.append({
            "text": r["text"],
            "meta": r.get("meta", {}),
            "distance": 1.0 - fusion_sim,
            "source": r.get("source", "hybrid"),
            "sim": fusion_sim,
            "rrf_score": rrf,
            "rrf_normalized": rrf_normalized,
            "vec_sim": vec_sim,
            "bm25_score": bm25,
        })

    return final

def get_chroma_client() -> Any:
    """获取 chromadb client：优先 HttpClient（Server 模式），fallback PersistentClient。

    通过 CHROMA_SERVER=host:port 指定 server 地址。
    文档要求使用 Server 模式（不推荐嵌入式，有数据丢失风险）。
    """
    import chromadb

    server = _CHROMA_SERVER
    if server:
        host, _, port = server.partition(":")
        logger.info("ChromaDB 使用 Server 模式: %s:%s", host, port)
        return chromadb.HttpClient(host=host or "127.0.0.1", port=int(port) if port else 8000)
    logger.warning("CHROMA_SERVER 未配置，使用 PersistentClient（嵌入式，不推荐）")
    return chromadb.PersistentClient(path=DEFAULT_VECTORDB_PATH)


def ensure_chroma_compat() -> None:
    """chromadb 兼容层：patch 旧版 pickle 格式（在 PersistentClient 回退时有用）

    注意：HttpClient（Server 模式）不需要此 patch，Server 端已处理兼容性。
    """
    import pickle
    import sqlite3

    try:
        from chromadb.segment.impl.vector.local_persistent_hnsw import PersistentData

        @staticmethod
        def _patched_load_from_file(filename: str) -> Any:
            with open(filename, "rb") as f:
                ret = pickle.load(f)
            if not isinstance(ret, dict):
                return ret
            dim = ret.get("dimensionality")
            if dim is None:
                try:
                    seg_dir = os.path.dirname(filename)
                    seg_uuid = os.path.basename(seg_dir)
                    db_root = os.path.dirname(seg_dir)
                    db_file = os.path.join(db_root, "chroma.sqlite3")
                    con = sqlite3.connect(db_file)
                    row = con.execute(
                        "SELECT c.dimension FROM segments s "
                        "JOIN collections c ON c.id = s.collection WHERE s.id = ?",
                        (seg_uuid,),
                    ).fetchone()
                    con.close()
                    dim = row[0] if row else None
                except Exception:
                    dim = None
            pd = PersistentData(
                dimensionality=dim,
                total_elements_added=ret.get("total_elements_added", 0),
                id_to_label=ret.get("id_to_label", {}),
                label_to_id=ret.get("label_to_id", {}),
                id_to_seq_id=ret.get("id_to_seq_id", {}),
            )
            pd.max_seq_id = ret.get("max_seq_id")
            return pd

        PersistentData.load_from_file = _patched_load_from_file  # type: ignore[assignment]
    except Exception:
        pass


def _create_embedding_function(embed_model: str):
    from chromadb.utils import embedding_functions

    device = resolve_rag_device()
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model,
            device=device,
            normalize_embeddings=True,
        )
    except Exception:
        if device == "cpu":
            raise
        logger.warning("RAG embedding init failed on %s; falling back to CPU", device)
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model,
            device="cpu",
            normalize_embeddings=True,
        )


# ============================================================
# 教材检索器（k9_textbook 集合）
# ============================================================

class TextbookRAG:
    """教材向量检索器，对接 Chroma Server 的 k9_textbook 集合"""

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        import chromadb

        ensure_chroma_compat()
        logger.info("初始化 TextbookRAG: collection=%s model=%s", collection_name, embed_model)

        self._client = get_chroma_client()
        self._emb_fn = _create_embedding_function(embed_model)
        self._coll = self._client.get_collection(
            name=collection_name, embedding_function=self._emb_fn
        )
        logger.info("TextbookRAG 就绪，%s 共 %d 条", collection_name, self._coll.count())

    def retrieve(
        self,
        query: str,
        topk: int = DEFAULT_TOPK,
        subject: str | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """检索教材片段（双路混合检索：BM25 + bge-m3 + RRF 融合）

        支持多维度元数据过滤：subject（学科）、stage（学段：小学/初中）
        """
        # 构造 where 条件
        conditions: list[dict[str, str]] = []
        if subject:
            conditions.append({"subject": subject})
        if stage:
            conditions.append({"stage": stage})

        where: dict | None = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        results = hybrid_retrieve(query, self._coll, self._emb_fn, topk=topk, where=where)
        for r in results:
            r["source"] = "textbook"
        return results

    def format_header(self, meta: dict[str, Any], i: int = 1) -> str:
        """格式化单条结果的标题行"""
        m = meta or {}
        header = (
            f"{m.get('stage', '')}{m.get('subject', '')} "
            f"{m.get('grade', '')}{m.get('semester', '')} "
            f"《{m.get('chapter', '')}》"
        ).strip()
        return f"[教材参考{i}] {header}"


# ============================================================
# 题库检索器（k9_exercises 集合）
# ============================================================

class ExerciseRAG:
    """题库向量检索器，对接 Chroma Server 的 k9_exercises 集合"""

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        collection_name: str = DEFAULT_EXERCISE_COLLECTION,
    ) -> None:
        import chromadb

        logger.info("初始化 ExerciseRAG: collection=%s", collection_name)

        self._client = get_chroma_client()
        self._emb_fn = _create_embedding_function(embed_model)
        self._coll = self._client.get_collection(
            name=collection_name, embedding_function=self._emb_fn
        )
        logger.info("ExerciseRAG 就绪，%s 共 %d 条", collection_name, self._coll.count())

    def retrieve(
        self,
        query: str,
        topk: int = 3,
        subject: str | None = None,
        exclude_source_file: str | None = None,
    ) -> list[dict[str, Any]]:
        """检索相关练习题（双路混合检索：BM25 + bge-m3 + RRF 融合）"""
        where: dict | None = None
        conditions: list[dict[str, Any]] = []
        if subject:
            conditions.append({"subject": subject})
        if exclude_source_file:
            conditions.append({"source_file": {"$ne": exclude_source_file}})
        if len(conditions) == 1:
            where = conditions[0]
        elif conditions:
            where = {"$and": conditions}

        results = hybrid_retrieve(query, self._coll, self._emb_fn, topk=topk, where=where)
        for r in results:
            r["source"] = "exercise"
        return results

    def format_header(self, meta: dict[str, Any], i: int = 1) -> str:
        """格式化单条结果的标题行"""
        m = meta or {}
        header = (
            f"{m.get('subject', '')}·{m.get('ques_type', '')}·"
            f"{m.get('difficulty', '')} "
            f"《{m.get('knowledges', '')}》"
        ).strip()
        return f"[题库参考{i}] {header}"


# ============================================================
# 自适应截断：长文档不截断，短文档正常截断
# ============================================================

LONG_CHUNK_THRESHOLD = 2000  # 超过此字符数的 chunk 视为长文档（如完整试卷）


# ============================================================
# 格式化工具
# ============================================================

def format_context(
    hits: list[dict[str, Any]],
    textbook_rag: TextbookRAG | None = None,
    exercise_rag: ExerciseRAG | None = None,
) -> str:
    """把命中片段格式化为可注入 prompt 的上下文。

    保持融合排序顺序输出，不因文档长度改变相关性排名。
    """
    ordered = hits

    parts: list[str] = []
    idx = 0

    def _format_header(h: dict[str, Any], i: int) -> str:
        m = h.get("meta", {}) or {}
        source = h.get("source", "textbook")
        if source == "online":
            src_name = m.get("source", "在线来源")
            url = m.get("url", "")
            ci = m.get("chunk_index")
            ct = m.get("chunk_total")
            tag = f" 片段{ci+1}/{ct}" if ci is not None and ct is not None else ""
            hdr = f"[网络参考{i}] {src_name}{tag}"
            if url:
                hdr += f" ({url})"
            return hdr
        elif source == "exercise" and exercise_rag is not None:
            return exercise_rag.format_header(m, i)
        elif textbook_rag is not None:
            return textbook_rag.format_header(m, i)
        else:
            return f"[参考{i}] {m.get('chapter', m.get('knowledges', ''))}"

    for h in ordered:
        idx += 1
        header = _format_header(h, idx)
        text = h.get("text", "")
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def format_refs(hits: list[dict[str, Any]], max_refs: int = 5) -> list[str]:
    """生成引用列表"""
    refs: list[str] = []
    for h in hits[:max_refs]:
        m = h.get("meta", {}) or {}
        source = h.get("source", "textbook")

        # 统一相似度：合并后的结果有 "sim" 字段，原始本地结果用 distance 换算
        if "sim" in h:
            sim = h["sim"]
        else:
            dist = h.get("distance", 0)
            sim = 1.0 - dist if dist < 2 else 0.0

        if source == "online":
            src_name = m.get("source", "在线来源")
            url = m.get("url", "")
            ci = m.get("chunk_index")
            ct = m.get("chunk_total")
            chunk_tag = f" [片段{ci+1}/{ct}]" if ci is not None and ct is not None else ""
            ref = f"🌐 {src_name}{chunk_tag} (相似度 {sim:.2f})"
            if url:
                ref += f" ({url})"
        elif source == "exercise":
            ref = (
                f"{m.get('subject', '')}·{m.get('ques_type', '')}·"
                f"{m.get('difficulty', '')} "
                f"《{m.get('knowledges', '')}》(相似度{sim:.2f})"
            ).strip()
        else:
            ref = (
                f"{m.get('subject', '')}·{m.get('grade', '')}{m.get('semester', '')} "
                f"《{m.get('chapter', '')}》(相似度{sim:.2f})"
            ).strip()
        refs.append(ref)
    return refs


# ============================================================
# 进程级单例
# ============================================================

@lru_cache
def _get_textbook_rag() -> TextbookRAG | None:
    """进程级单例，仅加载一次 bge-m3 模型"""
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        logger.warning("chromadb 或 sentence-transformers 未安装，RAG 不可用")
        return None
    try:
        return TextbookRAG()
    except Exception:
        logger.warning("TextbookRAG 初始化失败", exc_info=True)
        return None


@lru_cache
def _get_exercise_rag() -> ExerciseRAG | None:
    """进程级单例，题库检索器"""
    try:
        return ExerciseRAG()
    except Exception:
        logger.warning("ExerciseRAG 初始化失败（题库不可用，不影响教材检索）", exc_info=True)
        return None


# ============================================================
# 主检索入口
# ============================================================

def retrieve_for_exam(
    paper_content: str,
    subject: str,
    topk: int = DEFAULT_TOPK,
    online_mode: str | None = None,
    verbose: bool = False,
    exclude_source_file: str | None = None,
    exclude_near_duplicates: bool = False,
    textbook_only: bool = False,
    min_sim_threshold: float | None = None,
) -> tuple[str, list[str]]:
    """为题目检索教材+题库上下文，支持阈值触发联网。

    检索策略：
      1. 先用关键词提取+去LaTeX 构造语义 query（而非 800 字原文）
      2. 同时检索 k9_textbook（教材）+ k9_exercises（题库）
      3. 按学科+学段过滤，top-1 相似度不够时触发联网

    online_mode:
      - None / "off": 只用本地 RAG
      - "auto": 本地 top-1 相似度 < 阈值时自动联网
      - "online": 跳过本地，直接联网搜索

    verbose: 打印完整检索过程（query、每条命中、相似度、是否触发联网等）

    返回 (格式化的上下文, 引用列表)
    """
    from config import ONLINE_RAG, RAG_CONFIG

    mode = online_mode if online_mode is not None else "off"
    threshold = ONLINE_RAG.get("local_threshold", 0.5)

    # 归一化学科名（"初中数学" → "数学"，向量库 metadata 存的是纯学科名）
    _subject_clean = subject.removeprefix("初中").removeprefix("小学").strip()

    # 用关键词提取构造语义聚焦的 query（而非 800 字原文）
    keywords = _extract_keywords(paper_content, max_len=300)
    query = f"{_subject_clean} 中考 {keywords}" if keywords else paper_content[:200]

    if verbose:
        print(f"━━━━━━ RAG 检索 ━━━━━━")
        print(f"📝 题目预览: {paper_content[:120].replace(chr(10), ' ')}")
        print(f"🔍 检索 Query: {query[:200]}")
        print(f"⚙️  模式: {mode} | 阈值: {threshold}")
        print()

    # ---- 本地教材检索 ----
    textbook_rag = _get_textbook_rag()
    textbook_hits: list = []
    local_top_score = 0.0

    if textbook_rag is not None and mode != "online":
        # 按学科+学段过滤（初中数学）
        textbook_hits = textbook_rag.retrieve(
            query, topk=topk, subject=_subject_clean, stage="初中"
        )
        if textbook_hits:
            local_top_score = 1.0 - min(h.get("distance", 999) for h in textbook_hits)
            logger.info("本地教材 top-1 相似度: %.3f (阈值=%.3f)", local_top_score, threshold)
            if verbose:
                print(f"📚 本地教材检索 ({len(textbook_hits)} 条命中):")
                for i, h in enumerate(textbook_hits[:topk], 1):
                    m = h.get("meta", {})
                    sim = 1.0 - h.get("distance", 0)
                    header = textbook_rag.format_header(m)
                    print(f"    [{i}] sim={sim:.4f} │ {header}")
                    print(f"        内容: {h.get('text', '')[:80].replace(chr(10), ' ')}...")
                print()
        elif verbose:
            print(f"📚 本地教材检索: 无匹配结果\n")

    # ---- 本地题库检索 ----
    exercise_rag = None if textbook_only else _get_exercise_rag()
    exercise_hits: list = []
    if exercise_rag is not None and mode != "online":
        # 学科名映射：中文→英文（向量库中不同数据源的 subject 格式不统一）
        _subject_en = {
            "数学": "Math", "物理": "Physics", "化学": "Chemistry",
            "英语": "English", "语文": "Chinese", "历史": "History",
            "道法": "Politics", "生物": "Biology", "地理": "Geography",
        }.get(_subject_clean, _subject_clean)
        # 统一走混合检索（BM25 + bge-m3 + RRF），按学科过滤
        exercise_hits = exercise_rag.retrieve(
            query, topk=EXERCISE_TOPK, subject=_subject_en,
            exclude_source_file=exclude_source_file,
        )
        if exclude_near_duplicates:
            exercise_hits = [
                h for h in exercise_hits
                if not _is_near_duplicate(paper_content, h.get("text", ""))
            ]
        if exercise_hits:
            ex_top_score = 1.0 - min(h.get("distance", 999) for h in exercise_hits)
            logger.info("本地题库 top-1 相似度: %.3f", ex_top_score)
            local_top_score = max(local_top_score, ex_top_score)
            if verbose:
                print(f"📝 题库检索 ({len(exercise_hits)} 条命中):")
                for i, h in enumerate(exercise_hits, 1):
                    m = h.get("meta", {})
                    sim = 1.0 - h.get("distance", 0)
                    header = exercise_rag.format_header(m)
                    print(f"    [{i}] sim={sim:.4f} │ {header}")
                    print(f"        内容: {h.get('text', '')[:80].replace(chr(10), ' ')}...")
                print()
        elif verbose:
            print(f"📝 题库检索: 无匹配结果\n")

    # ---- 判断题库是否命中同题 ----
    is_exact_match = False
    if exercise_hits:
        ex_top_sim = 1.0 - min(h.get("distance", 999) for h in exercise_hits)
        EXACT_MATCH_THRESHOLD = 0.7
        is_exact_match = ex_top_sim >= EXACT_MATCH_THRESHOLD
        if verbose and is_exact_match:
            print(f"🎯 题库 top-1 相似度 {ex_top_sim:.3f} >= {EXACT_MATCH_THRESHOLD}，判定为同题命中\n")

    # ---- 判断是否需要联网（auto 模式：相似度 ≥ threshold 的本地命中数 < topk）----
    local_all: list[dict[str, Any]] = textbook_hits + exercise_hits
    # 给每条本地结果加上 sim 字段
    for h in local_all:
        h["sim"] = 1.0 - h.get("distance", 999)

    local_good_count = sum(1 for h in local_all if h["sim"] >= threshold)

    need_online = False
    online_reason = ""
    if mode == "online":
        need_online = True
        online_reason = "强制联网模式"
    elif mode == "auto":
        if textbook_rag is None and exercise_rag is None:
            need_online = True
            online_reason = "本地 RAG 不可用"
            logger.info("本地 RAG 不可用，触发联网搜索")
        elif local_good_count < topk:
            need_online = True
            online_reason = f"本地满足阈值(≥{threshold})的命中 {local_good_count} 条 < topk({topk})"
            logger.info("本地满足阈值 %.3f 的命中 %d < topk(%d)，触发联网搜索",
                        threshold, local_good_count, topk)
        else:
            logger.info("本地满足阈值 %.3f 的命中 %d ≥ topk(%d)，使用本地结果",
                        threshold, local_good_count, topk)

    if verbose:
        if need_online:
            print(f"🌐 触发联网搜索: {online_reason}")
            print(f"   ──→ 调用 Tavily API...")
            print()
        else:
            print(f"📊 本地满足阈值({threshold})的命中数: {local_good_count} ≥ topk({topk}) → 仅本地")
            print()

    # ---- 联网搜索（auto/online 模式）----
    online_hits: list[dict[str, Any]] = []
    if need_online:
        from online_rag import get_online_rag, _strip_latex as _strip_tex_online
        online = get_online_rag()
        if online:
            clean = _strip_tex_online(paper_content)
            prefix = f"{subject} 中考 知识点 考点 "
            max_query = 380 - len(prefix)
            online_query = prefix + clean[:max_query]
            if verbose:
                print(f"   🔍 联网Query: {online_query[:200]}")
            online_hits = online.search_with_rerank(online_query, max_results=topk)
            if online_hits and verbose:
                print(f"   ✅ 联网命中 {len(online_hits)} 条（已重排序）:")
                for i, h in enumerate(online_hits, 1):
                    m = h.get("meta", {}) or {}
                    ci = m.get("chunk_index")
                    ct = m.get("chunk_total")
                    tag = f" [片段{ci+1}/{ct}]" if ci is not None and ct is not None else ""
                    print(f"      [{i}] score={h.get('score', 0):.4f} │ {m.get('source', '?')}{tag}")
            elif verbose:
                print(f"   ⚠️  联网搜索无结果")
        elif verbose:
            print(f"   ❌ 联网RAG不可用")

    # ---- 合并本地 + 联网，按相似度统一排序，取 top-k ----
    all_hits: list[dict[str, Any]] = list(local_all)  # 本地（已有 sim）

    for h in online_hits:
        hc = dict(h)
        hc["source"] = "online"
        hc["sim"] = hc.get("score", 0)
        all_hits.append(hc)

    if not all_hits:
        if verbose:
            print(f"❌ 无任何命中结果\n")
        return "", [], False

    # 本地 RRF 分数和联网 bge-m3 重排分数均归一化到 0~1，统一按 sim
    # 排序并应用同一注入阈值；来源不会改变优先级。
    all_hits.sort(key=lambda h: h.get("sim", 0), reverse=True)
    effective_min_sim = MIN_SIM_THRESHOLD if min_sim_threshold is None else min_sim_threshold
    all_hits = [h for h in all_hits if h.get("sim", 0) >= effective_min_sim][:FINAL_TOPK]

    if verbose and online_hits and local_all:
        local_in_final = sum(1 for h in all_hits if h.get("source") not in ("online",))
        online_in_final = sum(1 for h in all_hits if h.get("source") == "online")
        print(f"\n📊 合并排序后 top-{len(all_hits)} (sim≥{effective_min_sim}): 本地 {local_in_final} 条 + 联网 {online_in_final} 条")
        for i, h in enumerate(all_hits, 1):
            src_tag = "🌐" if h.get("source") == "online" else "📚"
            print(f"   [{i}] {src_tag} sim={h.get('sim', 0):.4f}")

    # 不截断，全量注入，让 LLM 自己判断相关性（对标 GPT-4/Gemini/Claude）
    context = format_context(all_hits, textbook_rag=textbook_rag, exercise_rag=exercise_rag)
    refs = format_refs(all_hits, max_refs=10)

    if verbose:
        print(f"\n📤 最终注入上下文 ({len(context)} 字符，全量不截断):")
        for r in refs:
            print(f"   {r}")
        print(f"━━━━━━ RAG 检索结束 ━━━━━━\n")

    return context, refs, is_exact_match


def is_rag_available() -> bool:
    """检查 RAG 是否可用"""
    rag = _get_textbook_rag()
    return rag is not None


def _split_exam_questions(paper_content: str) -> list[str]:
    """按行首题号切分整卷，(1)/(2) 小问不会被单独切出。"""
    lines = paper_content.splitlines()
    section_starts = [i for i, line in enumerate(lines)
                      if re.match(r"^\s*(?:#{0,3}\s*)?[一二三四五六七八九十]+[、.]\s*\S", line)]
    if section_starts:
        lines = lines[min(section_starts):]
    starts = [i for i, line in enumerate(lines)
              if re.match(r"^\s*\d{1,3}\s*[\.、]\s*\S", line)]
    questions: list[str] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        if text:
            questions.append(text)
    return questions


def _split_numbered_answer_hints(answer_content: str) -> dict[int, str]:
    """从答案文件提取题号片段，仅用于 Oracle 的检索 query，不会注入模型。"""
    marker = re.compile(r'(?<![\d.])(1[0-9]|2[01]|[1-9])\s*[.、]\s*')
    matches = list(marker.finditer(answer_content))
    hints: dict[int, str] = {}
    for pos, match in enumerate(matches):
        number = int(match.group(1))
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(answer_content)
        text = answer_content[match.end():end].strip()
        if number not in hints and text:
            hints[number] = text[:600]
    return hints


def retrieve_for_exam_questionwise(
    paper_content: str,
    subject: str,
    topk: int = DEFAULT_TOPK,
    online_mode: str | None = None,
    verbose: bool = False,
    profile: str = "standard",
    target_source_file: str | None = None,
    answer_content: str = "",
) -> tuple[str, list[str], bool]:
    """逐题检索并去重；auto 模式下每题按本地阈值决定是否联网。"""
    questions = _split_exam_questions(paper_content)
    if not questions:
        return retrieve_for_exam(paper_content, subject, topk=topk,
                                  online_mode=online_mode, verbose=verbose)

    contexts: list[str] = []
    refs: list[str] = []
    seen_contexts: set[str] = set()
    seen_refs: set[str] = set()
    mode = online_mode if online_mode is not None else "off"
    used_online = False
    answer_hints = _split_numbered_answer_hints(answer_content) if profile == "oracle" else {}

    for i, question in enumerate(questions, 1):
        query = question
        if profile == "oracle" and answer_hints.get(i):
            query = f"{question}\n用于定位教材知识点的正确结论：{answer_hints[i]}"
        context, question_refs, _ = retrieve_for_exam(
            query, subject, topk=1,
            # web profile 在逐题本地检索后统一做一次不受限联网，避免一张试卷触发 21 次 Tavily。
            online_mode="off" if profile in ("oracle", "web") else mode,
            verbose=verbose,
            exclude_source_file=target_source_file if profile in ("local_clean", "web") else None,
            exclude_near_duplicates=profile in ("local_clean", "web"),
            textbook_only=profile == "oracle",
            min_sim_threshold=0.0 if profile == "oracle" else None,
        )
        if context and context not in seen_contexts:
            contexts.append(context)
            seen_contexts.add(context)
        for ref in question_refs:
            used_online = used_online or "🌐" in ref
            if ref not in seen_refs:
                refs.append(ref)
                seen_refs.add(ref)
        if verbose:
            print(f"逐题 RAG [{i}/{len(questions)}] 命中 {len(question_refs)} 条")

    if profile == "web":
        # 不受限联网明确表达“找原卷答案”的意图；这是开卷搜索能力，不是教材增强。
        compact_query = (
            f"2026年河南省中考{subject}试卷 含答案及解析 参考答案\n"
            + "\n".join(questions[: min(len(questions), 5)])
        )
        online_context, online_refs, _ = retrieve_for_exam(
            compact_query, subject, topk=DEFAULT_TOPK, online_mode="online", verbose=verbose,
        )
        if online_context and online_context not in seen_contexts:
            contexts.append(online_context)
            seen_contexts.add(online_context)
        for ref in online_refs:
            used_online = used_online or "🌐" in ref
            if ref not in seen_refs:
                refs.append(ref)
                seen_refs.add(ref)

    return "\n\n".join(contexts)[:16000], refs[:10], used_online
