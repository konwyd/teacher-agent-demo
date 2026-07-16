"""
联网 RAG 检索器：Tavily 搜索 → 全文抓取 → 分块 → 向量化重排序
当本地教材向量库检索结果相关度不足时，自动触发联网搜索。

完整流程：
  用户提问 → Tavily 搜索 → 抓取网页全文 → 切割成文本块
  → bge-m3 向量化 → 余弦相似度重排序 → 取最相关 3 片段 → 注入上下文
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 共享 bge-m3 嵌入模型（进程级单例，与本地 RAG 复用）
# ============================================================

_embed_model: Any = None
DEFAULT_EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"


def _get_embed_model() -> Any:
    """延迟加载 bge-m3，一次加载多处复用"""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer(DEFAULT_EMBED_MODEL, device="cpu")
    _embed_model.encode(["warmup"], normalize_embeddings=True)
    logger.info("bge-m3 嵌入模型已加载（联网重排序）")
    return _embed_model


# ============================================================
# 文本分块
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
    """将长文本按段落/句子切分为重叠块

    策略：优先按空行(\n\n)切段，段内按句号/问号/感叹号切句。
    块间有 overlap 保证语义连续性。
    单句超长时按固定长度硬切。
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    # 第一步：按段落分割
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= chunk_size:
            chunks.append(para)
            continue

        # 长段落按句子切分（中文句号、问号、感叹号 + 英文标点）
        sentences = re.split(r'(?<=[。！？.!?])\s*', para)

        current = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            if len(current) + len(sent) <= chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current)
                # 如果单个句子超过 chunk_size，硬切
                if len(sent) > chunk_size:
                    for i in range(0, len(sent), chunk_size - chunk_overlap):
                        chunks.append(sent[i:i + chunk_size])
                    current = ""
                else:
                    current = sent

        if current:
            chunks.append(current)

    # 过滤过短碎片（纯标点/空白）
    chunks = [c for c in chunks if len(c.strip()) >= 20]

    if not chunks:
        return [text[:chunk_size]]

    return chunks


# ============================================================
# 向量重排序
# ============================================================

def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    topk: int = 3,
) -> list[dict[str, Any]]:
    """用 bge-m3 对 chunks 做向量化重排序，返回 top-k 最相关片段

    每个 chunk 保留原始 meta（来源URL、标题等）。
    """
    if not chunks:
        return []

    model = _get_embed_model()
    chunk_texts = [c["text"] for c in chunks]

    # query + 所有 chunks 一起编码（一次推理）
    all_texts = [query] + chunk_texts
    embeddings = model.encode(all_texts, normalize_embeddings=True)

    query_emb = embeddings[0]
    chunk_embs = embeddings[1:]

    # cosine 相似度 = 归一化向量的点积
    scores = np.dot(chunk_embs, query_emb)

    # 排序取 top-k
    ranked_indices = np.argsort(scores)[::-1][:topk]

    results: list[dict[str, Any]] = []
    for idx in ranked_indices:
        c = dict(chunks[idx])
        c["score"] = float(scores[idx])
        results.append(c)

    best = results[0]["score"] if results else 0
    logger.info("联网重排序: %d chunks → top-%d, 最高分=%.3f",
                len(chunks), len(results), best)
    return results


# ============================================================
# 工具函数
# ============================================================

def _contains_chinese(text: str) -> bool:
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def _strip_latex(text: str) -> str:
    """去除 LaTeX 公式，保留纯文本"""
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


# ============================================================
# OnlineRAG
# ============================================================

class OnlineRAG:
    """基于 Tavily 的联网检索器，支持分块 + 向量重排序"""

    def __init__(self, api_key: str, max_results: int = 5):
        if not api_key:
            raise ValueError("Tavily API key 未配置")

        from tavily import TavilyClient
        self._client = TavilyClient(api_key=api_key)
        self._max_results = max_results
        logger.info("OnlineRAG 就绪 (Tavily, max_results=%d)", max_results)

    # 排除无法提取正文的域名
    _EXCLUDE_DOMAINS = [
        "scribd.com", "doc88.com", "docin.com",
        "max.book118.com", "wenku.baidu.com", "slideplayer.com",
    ]

    # 超过此长度的 raw_content 才做分块（短的直接用）
    _CHUNK_MIN_LEN = 500
    # 分块参数
    _CHUNK_SIZE = 300
    _CHUNK_OVERLAP = 50
    # 重排序后保留的 chunk 数
    _RERANK_TOPK = 3

    def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """废弃：请使用 search_with_rerank()"""
        raise DeprecationWarning("请使用 search_with_rerank(query, max_results)")

    def search_with_rerank(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """联网搜索 → 全文抓取 → 分块 → 向量重排序

        返回: [{"text","meta","score"}, ...]  已按相关度排序
        """
        max_r = max_results or self._max_results

        # ---- 第一步：Tavily advanced 搜索（获取 raw_content）----
        raw_results = self._do_search(query, max_r, "advanced")
        logger.info("Tavily 返回 %d 条原始结果", len(raw_results))

        if not raw_results:
            return []

        # ---- 第二步：分离 AI 摘要 和 网页结果 ----
        ai_summaries: list[dict[str, Any]] = []
        web_results: list[dict[str, Any]] = []

        for r in raw_results:
            if r["meta"].get("source") == "Tavily AI 摘要":
                ai_summaries.append(r)
            else:
                web_results.append(r)

        # ---- 第三步：对网页全文做分块 ----
        all_chunks: list[dict[str, Any]] = []

        # AI 摘要不切分，作为独立 chunk
        for r in ai_summaries:
            if r.get("text") and len(r["text"]) >= 20:
                all_chunks.append(r)

        # 网页结果：长的切分，短的直接保留
        total_raw = 0
        total_chunked = 0
        for r in web_results:
            text = r.get("text", "")
            if not text:
                continue
            meta = r["meta"]
            base_score = r.get("score", 0.5)

            if len(text) > self._CHUNK_MIN_LEN:
                # 长文本 → 分块
                sub_chunks = chunk_text(text, self._CHUNK_SIZE, self._CHUNK_OVERLAP)
                total_raw += 1
                total_chunked += len(sub_chunks) if sub_chunks else 0
                for i, sc in enumerate(sub_chunks):
                    chunk_meta = dict(meta)
                    chunk_meta["chunk_index"] = i
                    chunk_meta["chunk_total"] = len(sub_chunks)
                    all_chunks.append({
                        "text": sc,
                        "meta": chunk_meta,
                        "score": base_score,
                    })
            else:
                # 短文本（摘要等）→ 不分块，直接保留
                all_chunks.append(r)

        if total_raw > 0:
            logger.info("分块: %d 篇网页 → %d 个 chunks (size=%d, overlap=%d)",
                        total_raw, total_chunked, self._CHUNK_SIZE, self._CHUNK_OVERLAP)

        # ---- 第四步：向量化重排序 ----
        if not all_chunks:
            return []

        reranked = rerank_chunks(query, all_chunks, topk=self._RERANK_TOPK)
        return reranked

    def _do_search(
        self,
        query: str,
        max_results: int,
        search_depth: str,
    ) -> list[dict[str, Any]]:
        """执行单次 Tavily 搜索"""
        is_advanced = (search_depth == "advanced")
        try:
            resp = self._client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=True,
                include_raw_content=is_advanced,
                exclude_domains=self._EXCLUDE_DOMAINS,
            )
        except Exception:
            logger.warning("Tavily 搜索失败 (depth=%s)", search_depth, exc_info=True)
            return []

        results: list[dict[str, Any]] = []

        # Tavily AI 摘要
        answer = (resp.get("answer") or "").strip()
        if answer:
            results.append({
                "text": answer,
                "meta": {"source": "Tavily AI 摘要", "url": ""},
                "score": 1.0,
            })

        # 各条搜索结果
        for item in resp.get("results", [])[:max_results]:
            raw = (item.get("raw_content") or "").strip()
            summary = (item.get("content") or "").strip()

            # 过滤垃圾 raw_content
            raw_is_junk = (
                raw.startswith("data:image")
                or raw.count("base64,") > 3
                or (
                    len(raw) >= 150
                    and not _contains_chinese(raw)
                    and _contains_chinese(summary)
                )
            )

            if is_advanced and raw and len(raw) >= 150 and not raw_is_junk:
                text = raw
            elif summary and len(summary) >= 30:
                text = summary
            else:
                continue

            results.append({
                "text": text,
                "meta": {
                    "source": item.get("title", ""),
                    "url": item.get("url", ""),
                },
                "score": item.get("score", 0.5),
            })

        return results

    def format_context(
        self,
        hits: list[dict[str, Any]],
        max_chars: int = 2400,
    ) -> str:
        """把重排序后的命中片段格式化为 prompt 上下文（支持截断，不跳过后续）"""
        parts: list[str] = []
        used = 0
        for i, h in enumerate(hits, 1):
            m = h.get("meta", {}) or {}
            source = m.get("source", "未知来源")
            url = m.get("url", "")
            url_hint = f" ({url})" if url else ""

            # 分块标注
            ci = m.get("chunk_index")
            ct = m.get("chunk_total")
            if ci is not None and ct is not None:
                header = f"[网络参考{i}] {source} 片段{ci+1}/{ct}{url_hint}"
            else:
                header = f"[网络参考{i}] {source}{url_hint}"

            text = h.get("text", "")

            remaining = max_chars - used
            header_len = len(header) + 2
            if remaining < header_len + 20:
                break

            max_text_len = remaining - header_len
            if len(text) > max_text_len:
                text = text[:max_text_len - 3] + "..."

            block = f"{header}\n{text}"
            parts.append(block)
            used += len(block)

        return "\n\n".join(parts)


# ============================================================
# 单例 & 对外接口
# ============================================================

_online_rag_instance: OnlineRAG | None = None


def get_online_rag() -> OnlineRAG | None:
    global _online_rag_instance
    if _online_rag_instance is not None:
        return _online_rag_instance

    from config import ONLINE_RAG
    api_key = ONLINE_RAG.get("tavily_api_key", "")
    if not api_key:
        logger.warning("Tavily API key 未配置，联网搜索不可用")
        return None

    try:
        _online_rag_instance = OnlineRAG(
            api_key=api_key,
            max_results=ONLINE_RAG.get("search_topk", 5),
        )
        return _online_rag_instance
    except Exception:
        logger.warning("OnlineRAG 初始化失败", exc_info=True)
        return None


def search_online(
    paper_content: str,
    subject: str,
    topk: int = 5,
    verbose: bool = False,
) -> tuple[str, list[str]]:
    """联网搜索（含分块+重排序），返回 (上下文, 引用列表)

    与本地 RAG 的 retrieve_for_exam 输出格式完全相同，可直接替换。
    """
    online = get_online_rag()
    if online is None:
        if verbose:
            print("   ❌ 联网RAG不可用")
        return "", []

    # 构造搜索 query
    clean = _strip_latex(paper_content)
    prefix = f"{subject} 中考 知识点 考点 "
    max_content_len = 380 - len(prefix)
    query = prefix + clean[:max_content_len]

    if verbose:
        print(f"   🔍 联网Query: {query[:200]}")
        print(f"   ⏳ 搜索+分块+重排序中...")

    # 核心流程：搜索 → 分块 → 重排序
    hits = online.search_with_rerank(query, max_results=topk)

    if not hits:
        if verbose:
            print(f"   ❌ 联网搜索无结果")
        return "", []

    if verbose:
        print(f"   ✅ 联网搜索命中 {len(hits)} 条（已重排序）:")
        for i, h in enumerate(hits, 1):
            m = h.get("meta", {}) or {}
            ci = m.get("chunk_index")
            ct = m.get("chunk_total")
            chunk_tag = f" [片段{ci+1}/{ct}]" if ci is not None and ct is not None else ""
            print(f"      [{i}] score={h.get('score', 0):.4f} │ {m.get('source', '?')}{chunk_tag}")
            print(f"          内容: {h.get('text', '')[:80].replace(chr(10), ' ')}...")

    context = online.format_context(hits)
    refs = []
    for h in hits[:3]:
        m = h.get("meta", {}) or {}
        source = m.get("source", "未知来源")
        url = m.get("url", "")
        # 标注分块信息
        ci = m.get("chunk_index")
        ct = m.get("chunk_total")
        chunk_tag = f" [片段{ci+1}/{ct}]" if ci is not None and ct is not None else ""
        ref = f"🌐 {source}{chunk_tag} (相似度 {h.get('score', 0):.2f})"
        if url:
            ref += f" ({url})"
        refs.append(ref)

    return context, refs


def is_online_rag_available() -> bool:
    return get_online_rag() is not None
