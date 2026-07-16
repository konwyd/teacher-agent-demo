"""
RAG 检索器：对接 Chroma Server + bge-m3 向量库，支持多集合检索
严格按照 RAG_DEPLOYMENT_GUIDE.md 规范实现
"""

from __future__ import annotations

import logging
import re
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# 默认路径（与 teacher-agent-demo 共用同一份向量库）
# ============================================================
DEFAULT_VECTORDB_PATH = "/data/education/data/china_textbook_k9_vectordb"
DEFAULT_EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"
DEFAULT_COLLECTION = "k9_textbook"
DEFAULT_EXERCISE_COLLECTION = "k9_exercises"
DEFAULT_TOPK = 5
DEFAULT_MAX_CONTEXT_CHARS = 2400

# Chroma Server 地址（从配置读取，默认 127.0.0.1:8002）
_CHROMA_SERVER = os.environ.get("CHROMA_SERVER", "").strip()


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
    # 去掉纯数字/符号噪音，保留中文词和≥2字的英文词
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{2,}', clean)
    return ' '.join(tokens)[:max_len]


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
        from chromadb.utils import embedding_functions

        ensure_chroma_compat()
        logger.info("初始化 TextbookRAG: collection=%s model=%s", collection_name, embed_model)

        self._client = get_chroma_client()
        self._emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model,
            normalize_embeddings=True,
        )
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
        """检索教材片段

        支持多维度元数据过滤：subject（学科）、stage（学段：小学/初中）
        """
        # 多条件过滤需要用 $and
        conditions: list[dict[str, str]] = []
        if subject:
            conditions.append({"subject": subject})
        if stage:
            conditions.append({"stage": stage})

        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": topk}
        if len(conditions) == 1:
            kwargs["where"] = conditions[0]
        elif len(conditions) > 1:
            kwargs["where"] = {"$and": conditions}

        try:
            res = self._coll.query(**kwargs)
        except Exception:
            logger.warning("TextbookRAG 检索失败", exc_info=True)
            return []

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [
            {"text": doc, "meta": meta, "distance": dist, "source": "textbook"}
            for doc, meta, dist in zip(docs, metas, dists)
        ]

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
        from chromadb.utils import embedding_functions

        logger.info("初始化 ExerciseRAG: collection=%s", collection_name)

        self._client = get_chroma_client()
        self._emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model,
            normalize_embeddings=True,
        )
        self._coll = self._client.get_collection(
            name=collection_name, embedding_function=self._emb_fn
        )
        logger.info("ExerciseRAG 就绪，%s 共 %d 条", collection_name, self._coll.count())

    def retrieve(
        self,
        query: str,
        topk: int = 3,
        subject: str | None = None,
    ) -> list[dict[str, Any]]:
        """检索相关练习题"""
        filt: dict[str, str] | None = None
        if subject:
            filt = {"subject": subject}

        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": topk}
        if filt:
            kwargs["where"] = filt

        try:
            res = self._coll.query(**kwargs)
        except Exception:
            logger.warning("ExerciseRAG 检索失败", exc_info=True)
            return []

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [
            {"text": doc, "meta": meta, "distance": dist, "source": "exercise"}
            for doc, meta, dist in zip(docs, metas, dists)
        ]

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
# 格式化工具
# ============================================================

def format_context(
    hits: list[dict[str, Any]],
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    textbook_rag: TextbookRAG | None = None,
    exercise_rag: ExerciseRAG | None = None,
) -> str:
    """把命中片段格式化为可注入 prompt 的上下文，支持截断（不跳过后续结果）"""
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        m = h.get("meta", {}) or {}
        source = h.get("source", "textbook")

        # 用对应的 formatter
        if source == "online":
            src_name = m.get("source", "在线来源")
            url = m.get("url", "")
            ci = m.get("chunk_index")
            ct = m.get("chunk_total")
            if ci is not None and ct is not None:
                header = f"[网络参考{i}] {src_name} 片段{ci+1}/{ct}"
            else:
                header = f"[网络参考{i}] {src_name}"
            if url:
                header += f" ({url})"
        elif source == "exercise" and exercise_rag is not None:
            header = exercise_rag.format_header(m, i)
        elif textbook_rag is not None:
            header = textbook_rag.format_header(m, i)
        else:
            header = f"[参考{i}] {m.get('chapter', m.get('knowledges', ''))}"

        text = h.get("text", "")

        remaining = max_chars - used
        header_len = len(header) + 2
        if remaining < header_len + 20:
            break

        max_text_len = remaining - header_len
        if len(text) > max_text_len:
            text = text[: max_text_len - 3] + "..."

        block = f"{header}\n{text}"
        parts.append(block)
        used += len(block)

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
    exercise_rag = _get_exercise_rag()
    exercise_hits: list = []
    if exercise_rag is not None and mode != "online":
        exercise_hits = exercise_rag.retrieve(query, topk=3, subject=_subject_clean)
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

    # 按 sim 降序排序，取 topk
    all_hits.sort(key=lambda h: h.get("sim", 0), reverse=True)
    all_hits = all_hits[:topk]

    if verbose and online_hits and local_all:
        local_in_final = sum(1 for h in all_hits if h.get("source") not in ("online",))
        online_in_final = sum(1 for h in all_hits if h.get("source") == "online")
        print(f"\n📊 合并排序后 top-{topk}: 本地 {local_in_final} 条 + 联网 {online_in_final} 条")
        for i, h in enumerate(all_hits, 1):
            src_tag = "🌐" if h.get("source") == "online" else "📚"
            print(f"   [{i}] {src_tag} sim={h.get('sim', 0):.4f}")

    max_chars = RAG_CONFIG.get("max_context_chars", DEFAULT_MAX_CONTEXT_CHARS)
    context = format_context(all_hits, max_chars=max_chars,
                             textbook_rag=textbook_rag, exercise_rag=exercise_rag)
    refs = format_refs(all_hits, max_refs=5)

    if verbose:
        print(f"\n📤 最终注入上下文 ({len(context)} 字符):")
        for r in refs:
            print(f"   {r}")
        print(f"━━━━━━ RAG 检索结束 ━━━━━━\n")

    return context, refs, is_exact_match


def is_rag_available() -> bool:
    """检查 RAG 是否可用"""
    rag = _get_textbook_rag()
    return rag is not None
