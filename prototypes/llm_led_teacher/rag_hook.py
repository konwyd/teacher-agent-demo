"""RAG hook: inject local ChromaDB + online Tavily context into the prototype.

策略：
- 本地 ChromaDB 检索（用 bge-m3 客户端编码 query，再 query_embeddings）
- 联网 Tavily 搜索（auto 模式自动触发）
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

# 把 llm-test/ 加入 sys.path
_LLM_TEST_DIR = Path(__file__).resolve().parents[2] / "llm-test"
if str(_LLM_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_TEST_DIR))

# 复用原项目的完整检索策略：BM25 + bge-m3 + RRF、学科/学段过滤、
# 0.65 联网回退和 0.44 最终过滤均由 llm-test/rag_retriever 统一负责。
from rag_device import configure_cpu_threads, resolve_rag_device
from rag_retriever import retrieve_for_exam as _original_retrieve_for_exam
from rag_retriever import retrieve_local_hits as _retrieve_local_hits

from prototypes.llm_led_teacher.orchestration_models import SchoolStage, Subject

logger = logging.getLogger(__name__)

CHROMA_HOST = os.environ.get("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8002"))
TEXTBOOK_COLLECTION = os.environ.get("TEACHER_RAG_COLLECTION", "k9_textbook")
EXERCISE_COLLECTION = os.environ.get("TEACHER_EXERCISE_COLLECTION", "k9_exercises")
TOPK = int(os.environ.get("TEACHER_RAG_TOPK", "5"))
LOCAL_SIMILARITY_THRESHOLD = float(os.environ.get("TEACHER_RAG_LOCAL_THRESHOLD", "0.65"))
BGE_MODEL_PATH = os.environ.get(
    "TEACHER_RAG_EMBED_MODEL",
    "/home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3",
)

_EXERCISE_SUBJECTS = {
    Subject.MATH: "Math",
    Subject.CHINESE: "Chinese",
    Subject.ENGLISH: "English",
    Subject.PHYSICS: "Physics",
    Subject.CHEMISTRY: "Chemistry",
    Subject.BIOLOGY: "Biology",
    Subject.HISTORY: "History",
    Subject.GEOGRAPHY: "Geography",
    Subject.POLITICS: "Politics",
}


@lru_cache(maxsize=1)
def warmup_rag() -> dict[str, object]:
    """Warm local RAG models in the Streamlit process without network calls."""
    started = time.perf_counter()
    device = resolve_rag_device()
    cpu_threads = configure_cpu_threads()
    try:
        _original_retrieve_for_exam(
            "一元一次方程",
            "初中数学",
            topk=1,
            online_mode="off",
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - warmup must not block teaching
        elapsed = round(time.perf_counter() - started, 3)
        logger.warning("RAG warmup failed on %s after %.3fs: %s", device, elapsed, type(exc).__name__)
        return {
            "ok": False,
            "device": device,
            "cpu_threads": cpu_threads,
            "elapsed_seconds": elapsed,
            "error": type(exc).__name__,
        }
    elapsed = round(time.perf_counter() - started, 3)
    logger.info("RAG warmup complete: device=%s elapsed=%.3fs", device, elapsed)
    return {
        "ok": True,
        "device": device,
        "cpu_threads": cpu_threads,
        "elapsed_seconds": elapsed,
        "error": "",
    }


class RagContext:
    def __init__(
        self,
        text: str = "",
        refs: list[str] | None = None,
        is_exact_match: bool = False,
        source: str = "none",
        hits: list[dict] | None = None,
        online_triggered: bool = False,
        online_status: str = "not_triggered",
        online_reason: str = "",
        local_threshold: float = LOCAL_SIMILARITY_THRESHOLD,
        local_top_score: float = 0.0,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        self.text = text
        self.refs: list[str] = refs or []
        self.is_exact_match = is_exact_match
        self.source = source
        self.hits: list[dict] = hits or []
        self.online_triggered = online_triggered
        self.online_status = online_status
        self.online_reason = online_reason
        self.local_threshold = local_threshold
        self.local_top_score = local_top_score
        self.diagnostics = diagnostics or {}


def subject_metadata_filters(
    subject: Subject,
    stage: SchoolStage,
) -> dict[str, dict[str, str]]:
    """Translate orchestration enums to collection metadata without routing."""
    return {
        "textbook": {"subject": subject.value, "stage": stage.value},
        "exercise": {"subject": _EXERCISE_SUBJECTS[subject]},
    }


def retrieve_local_context(
    problem: str,
    *,
    subject: Subject,
    stage: SchoolStage,
    topk: int = TOPK,
    sources: tuple[str, ...] = ("textbook", "exercise"),
) -> RagContext:
    """Retrieve only local evidence using caller-selected subject and stage."""
    if not problem.strip():
        return RagContext(source="empty_problem")
    try:
        result = _retrieve_local_hits(
            problem,
            subject.value,
            stage.value,
            topk,
            sources,
        )
    except Exception as exc:  # noqa: BLE001 - tool errors are recoverable
        return RagContext(
            source=f"retrieval_error: {type(exc).__name__}",
            diagnostics={"error": str(exc), "error_type": type(exc).__name__},
        )

    raw_hits = list(result.get("hits", []))
    text_parts: list[str] = []
    refs: list[str] = []
    hits: list[dict] = []
    used = 0
    for index, raw in enumerate(raw_hits, 1):
        document = str(raw.get("text", ""))
        if not document:
            continue
        excerpt = document[:600] + ("…" if len(document) > 600 else "")
        metadata = raw.get("metadata", {}) or {}
        source = str(raw.get("source", "local"))
        title = metadata.get("chapter") or metadata.get("knowledges") or metadata.get("source_file", "")
        reference = (
            f"[{source}:{index}] {metadata.get('subject', subject.value)} · "
            f"{metadata.get('stage', stage.value)} · {str(title)[:80]}"
        )
        chunk = f"{reference}\n{excerpt}"
        if used + len(chunk) > 2400:
            break
        refs.append(reference)
        text_parts.append(chunk)
        hits.append(
            {
                **raw,
                "document": excerpt,
                "similarity": float(raw.get("sim", 0.0)),
                "reference": reference,
            }
        )
        used += len(chunk)

    diagnostics = {
        "query": result.get("query", problem),
        "filters": result.get("filters", subject_metadata_filters(subject, stage)),
        "hit_count": len(hits),
        "elapsed_seconds": float(result.get("elapsed_seconds", 0.0)),
    }
    return RagContext(
        text="\n\n".join(text_parts),
        refs=refs,
        source="local" if hits else "local_empty",
        hits=hits,
        online_triggered=False,
        online_status="not_triggered",
        diagnostics=diagnostics,
        local_top_score=max((float(hit.get("sim", 0.0)) for hit in hits), default=0.0),
    )


def retrieve_web_context(
    query: str,
    *,
    subject: Subject,
    stage: SchoolStage,
    evidence_gap: str,
    topk: int = 3,
) -> RagContext:
    """Run web-only retrieval after the tool registry has enforced its gate."""
    results = _tavily_search(f"{stage.value}{subject.value} {query}", max_results=topk)
    text, refs, hits = _format_results(results)
    return RagContext(
        text=text,
        refs=refs,
        source="online" if hits else "online_empty",
        hits=hits,
        online_triggered=True,
        online_status="used" if hits else "no_results",
        online_reason=evidence_gap,
        diagnostics={"query": query, "hit_count": len(hits), "evidence_gap": evidence_gap},
    )


@lru_cache(maxsize=1)
def _get_embedder():
    """懒加载 bge-m3 嵌入模型（单例）。"""
    warnings.filterwarnings("ignore")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(BGE_MODEL_PATH, device="cpu")


def _strip_latex(text: str) -> str:
    text = re.sub(r"\\frac\{[^}]*\}\{[^}]*\}", "", text)
    text = re.sub(r"\\sqrt\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]*\$", "", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _query_chroma(
    collection_name: str,
    query_embedding: list[float],
    topk: int,
    where: dict | None = None,
) -> list[dict]:
    """用预计算的 embedding 调用 Chroma Server 检索。"""
    try:
        import chromadb
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        coll = client.get_collection(name=collection_name)
        kw: dict = {
            "query_embeddings": [query_embedding],
            "n_results": topk,
        }
        if where:
            kw["where"] = where
        results = coll.query(**kw)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        out: list[dict] = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 1.0
            sim = max(0.0, 1.0 - float(dist))
            out.append({
                "document": doc,
                "metadata": meta,
                "sim": sim,
                "source": collection_name,
            })
        return out
    except Exception as exc:  # noqa: BLE001
        return [{"_error": f"{type(exc).__name__}: {exc}"}]


def _format_results(results: list[dict], max_chars: int = 2400) -> tuple[str, list[str], list[dict]]:
    pieces: list[str] = []
    refs: list[str] = []
    hits: list[dict] = []
    used = 0
    for r in results:
        doc = r.get("document", "")
        meta = r.get("metadata", {})
        if not doc:
            continue
        src = r.get("source", "")
        ref = (
            f"[{src}] {meta.get('subject', '')} · {meta.get('stage', '')} · "
            f"{meta.get('book_title', meta.get('source_file', ''))[:60]}"
        )
        if ref not in refs:
            refs.append(ref)
        if len(doc) > 600:
            doc = doc[:600] + "…"
        chunk = f"{ref}\n{doc}\n"
        if used + len(chunk) > max_chars:
            break
        pieces.append(chunk)
        hits.append({
            "source": src,
            "reference": ref,
            "similarity": round(float(r.get("sim", 0.0)), 3),
            "document": doc,
        })
        used += len(chunk)
    return "\n".join(pieces), refs, hits


def _tavily_search(query: str, max_results: int = 3) -> list[dict]:
    """调用原 llm-test 的 Tavily 单例入口，未配置时非阻断降级。"""
    try:
        from online_rag import get_online_rag
        rag = get_online_rag()
        if rag is None:
            return []
        results = rag.search_with_rerank(query, max_results=max_results)
        out: list[dict] = []
        for r in results:
            text = r.get("text", "")
            if not text:
                continue
            out.append({
                "document": text,
                "metadata": {"source_file": r.get("meta", {}).get("url", "online")},
                "sim": r.get("score", 0.5),
                "source": "online_tavily",
            })
        return out
    except Exception:
        return []


def retrieve_context(
    problem: str,
    *,
    online_mode: str = "auto",
    topk: int = TOPK,
) -> RagContext:
    """复用 llm-test 的统一 RAG 策略，适配 Prototype 的上下文结构。"""
    if not problem.strip():
        return RagContext(source="empty_problem")
    diagnostics: dict[str, object] = {}
    try:
        context, refs, exact_match = _original_retrieve_for_exam(
            problem,
            "初中数学",
            topk=topk,
            online_mode=online_mode,
            verbose=False,
            diagnostics=diagnostics,
        )
    except Exception as exc:  # noqa: BLE001 - RAG 失败不阻断教学
        return RagContext(
            source=f"retrieval_error: {type(exc).__name__}",
            online_triggered=bool(diagnostics.get("online_triggered", False)),
            online_status=str(diagnostics.get("online_status", "error")),
            online_reason=str(diagnostics.get("online_reason", "")),
            local_threshold=float(diagnostics.get("local_threshold", LOCAL_SIMILARITY_THRESHOLD)),
            local_top_score=float(diagnostics.get("local_top_score", 0.0)),
        )
    if not context.strip() or not refs:
        return RagContext(
            source="retrieval_error",
            online_triggered=bool(diagnostics.get("online_triggered", False)),
            online_status=str(diagnostics.get("online_status", "no_results")),
            online_reason=str(diagnostics.get("online_reason", "")),
            local_threshold=float(diagnostics.get("local_threshold", LOCAL_SIMILARITY_THRESHOLD)),
            local_top_score=float(diagnostics.get("local_top_score", 0.0)),
        )

    source = "hybrid" if any("🌐" in ref for ref in refs) else "local"
    hits: list[dict] = []
    for ref in refs:
        match = re.search(r"相似度\s*([0-9.]+)", ref)
        similarity = float(match.group(1)) if match else 0.0
        hits.append({
            "source": "online_tavily" if "🌐" in ref else "local",
            "reference": ref,
            "similarity": similarity,
            "document": "",
        })
    return RagContext(
        text=context,
        refs=refs,
        is_exact_match=exact_match,
        source=source,
        hits=hits,
        online_triggered=bool(diagnostics.get("online_triggered", False)),
        online_status=str(diagnostics.get("online_status", "not_triggered")),
        online_reason=str(diagnostics.get("online_reason", "")),
        local_threshold=float(diagnostics.get("local_threshold", LOCAL_SIMILARITY_THRESHOLD)),
        local_top_score=float(diagnostics.get("local_top_score", 0.0)),
    )
