"""Textbook RAG（K9 教材向量检索）测试。

覆盖：
- RAG 关闭时的优雅降级（get_textbook_rag 返回 None、retrieve_textbook 返回空）
- format_context / format_refs 格式化辅助函数（纯函数，不需加载模型）
- 真实向量库可用时的端到端检索（集成测试，向量库不存在则跳过）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.retrieval import textbook_rag as _rag_module
from app.retrieval.textbook_rag import (
    TextbookRAG,
    get_textbook_rag,
    is_rag_available,
    retrieve_textbook,
)

_VECTORDB_PATH = "/data/wyd-workspace/education/data/china_textbook_k9_vectordb"


# ---------------------------------------------------------------- 降级


def _clear_rag_cache() -> None:
    """清除 lru_cache 单例和强制模式缓存，使 env 变更生效。"""
    get_textbook_rag.cache_clear()
    _rag_module.reset_forced_rag()


@pytest.fixture(autouse=True)
def _reset_rag_cache():
    _clear_rag_cache()
    yield
    _clear_rag_cache()


def test_is_rag_available_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """TEACHER_RAG_ENABLED=0 时 is_rag_available 返回 False。"""
    monkeypatch.setenv("TEACHER_RAG_ENABLED", "0")
    assert is_rag_available() is False


def test_get_textbook_rag_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG 关闭时单例返回 None，不影响主流程。"""
    monkeypatch.setenv("TEACHER_RAG_ENABLED", "0")
    assert get_textbook_rag() is None


def test_retrieve_textbook_empty_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG 关闭时便捷函数返回空列表。"""
    monkeypatch.setenv("TEACHER_RAG_ENABLED", "0")
    assert retrieve_textbook("解方程 2x+3=7") == []


# ---------------------------------------------------------------- 格式化


_SYNTH_HITS = [
    {
        "text": "解一元一次方程的一般步骤包括：去分母、去括号、移项、合并同类项、系数化为1。",
        "meta": {
            "stage": "初中",
            "subject": "数学",
            "grade": "七年级",
            "semester": "上册",
            "chapter": "归纳",
        },
        "distance": 0.25,
    },
    {
        "text": "移项就是把等式一边的某项变号后移到另一边。",
        "meta": {
            "stage": "初中",
            "subject": "数学",
            "grade": "七年级",
            "semester": "上册",
            "chapter": "思考",
        },
        "distance": 0.30,
    },
]


def test_format_context_basic() -> None:
    """format_context 生成带编号的教材参考块。"""
    ctx = TextbookRAG.format_context(None, _SYNTH_HITS)  # type: ignore[arg-type]
    assert "[教材参考1]" in ctx
    assert "七年级" in ctx
    assert "归纳" in ctx
    assert "去分母" in ctx
    assert "[教材参考2]" in ctx


def test_format_context_max_chars() -> None:
    """max_chars 限制生效，超出部分截断。"""
    ctx = TextbookRAG.format_context(None, _SYNTH_HITS, max_chars=70)  # type: ignore[arg-type]
    # 第一个片段放得下，第二个放不下
    assert "[教材参考1]" in ctx
    assert "[教材参考2]" not in ctx


def test_format_refs() -> None:
    """format_refs 返回简短引用列表。"""
    refs = TextbookRAG.format_refs(None, _SYNTH_HITS)  # type: ignore[arg-type]
    assert len(refs) == 2
    assert "七年级上册" in refs[0]
    assert "相似度" in refs[0]


def test_format_refs_limit() -> None:
    """limit 参数截断引用数量。"""
    refs = TextbookRAG.format_refs(None, _SYNTH_HITS, limit=1)  # type: ignore[arg-type]
    assert len(refs) == 1


def test_format_context_empty_hits() -> None:
    """空命中列表返回空串。"""
    assert TextbookRAG.format_context(None, []) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------- 集成测试（需要真实向量库）


def _vectordb_exists() -> bool:
    return Path(_VECTORDB_PATH).exists() and (Path(_VECTORDB_PATH) / "chroma.sqlite3").exists()


@pytest.mark.skipif(not _vectordb_exists(), reason="K9 向量库不存在")
def test_real_retrieve_returns_math_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实向量库可用时，检索解方程相关 query 应返回数学教材片段。"""
    monkeypatch.setenv("TEACHER_RAG_ENABLED", "1")
    monkeypatch.setenv("TEACHER_RAG_VECTORDB_PATH", _VECTORDB_PATH)
    _clear_rag_cache()

    rag = get_textbook_rag()
    if rag is None:
        pytest.skip("chromadb / sentence-transformers 未安装或模型加载失败")

    hits = rag.retrieve("解一元一次方程 移项 变号")
    assert len(hits) > 0
    # 默认学科过滤为数学
    for h in hits:
        assert h["meta"].get("subject") == "数学"
    # 语义相关性：top-1 应包含方程/移项相关内容
    assert any("移项" in h["text"] or "方程" in h["text"] for h in hits)


@pytest.mark.skipif(not _vectordb_exists(), reason="K9 向量库不存在")
def test_real_retrieve_in_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """retrieve_node 集成：教材上下文注入 RetrievalResult。"""
    from app.agent.state import TeacherState
    from app.agent.workflow import retrieve_node

    monkeypatch.setenv("TEACHER_RAG_ENABLED", "1")
    monkeypatch.setenv("TEACHER_RAG_VECTORDB_PATH", _VECTORDB_PATH)
    _clear_rag_cache()

    state = TeacherState(
        student_id="rag_test",
        message="老师 3x+4=7 怎么解？",
        original_problem="3x+4=7",
        target_equation="3x+4=7",
        detected_concept_id="c_linear",
        detected_concept_name="一元一次方程",
        use_rag=True,
    )
    result = retrieve_node(state)
    retrieval = result["retrieval"]
    if get_textbook_rag() is not None:
        assert retrieval.textbook_context != ""
        assert len(retrieval.textbook_refs) > 0


# ---------------------------------------------------------------- 开关控制


@pytest.mark.skipif(not _vectordb_exists(), reason="K9 向量库不存在")
def test_use_rag_false_skips_textbook(monkeypatch: pytest.MonkeyPatch) -> None:
    """use_rag=False 时完全跳过教材 RAG，textbook 字段为空。"""
    from app.agent.state import TeacherState
    from app.agent.workflow import retrieve_node

    monkeypatch.setenv("TEACHER_RAG_ENABLED", "1")
    _clear_rag_cache()

    state = TeacherState(
        student_id="switch_off",
        message="老师 3x+4=7 怎么解？",
        original_problem="3x+4=7",
        target_equation="3x+4=7",
        detected_concept_id="c_linear",
        detected_concept_name="一元一次方程",
        use_rag=False,
    )
    result = retrieve_node(state)
    retrieval = result["retrieval"]
    assert retrieval.textbook_context == ""
    assert retrieval.textbook_refs == []


@pytest.mark.skipif(not _vectordb_exists(), reason="K9 向量库不存在")
def test_use_rag_true_forces_textbook(monkeypatch: pytest.MonkeyPatch) -> None:
    """use_rag=True 即使环境变量关闭也强制检索教材。"""
    from app.agent.state import TeacherState
    from app.agent.workflow import retrieve_node

    monkeypatch.setenv("TEACHER_RAG_ENABLED", "0")  # 环境变量关闭
    monkeypatch.setenv("TEACHER_RAG_VECTORDB_PATH", _VECTORDB_PATH)
    _clear_rag_cache()

    state = TeacherState(
        student_id="switch_on",
        message="老师 3x+4=7 怎么解？",
        original_problem="3x+4=7",
        target_equation="3x+4=7",
        detected_concept_id="c_linear",
        detected_concept_name="一元一次方程",
        use_rag=True,  # 但请求级强制开启
    )
    result = retrieve_node(state)
    retrieval = result["retrieval"]
    if get_textbook_rag() is not None:
        assert retrieval.textbook_context != ""
        assert len(retrieval.textbook_refs) > 0
