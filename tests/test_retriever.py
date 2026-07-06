"""JSON mock Graph RAG 检索测试。"""

from app.retrieval.graph_retriever import JsonGraphRetriever, get_retriever


def test_get_concept_by_name() -> None:
    r = get_retriever()
    concept = r.get_concept("一元一次方程")
    assert concept is not None
    assert concept.id == "c_linear"
    assert concept.grade == "七年级"


def test_get_concept_by_id_and_fuzzy_message() -> None:
    r = get_retriever()
    assert r.get_concept("c_frac") is not None
    # 名称出现在完整消息里也能匹配
    concept = r.get_concept("我不会合并同类项怎么办")
    assert concept is not None
    assert concept.id == "c_combine"


def test_get_prerequisites() -> None:
    r = get_retriever()
    prereq_ids = {p.id for p in r.get_prerequisites("c_linear")}
    assert prereq_ids == {"c_eq_prop", "c_move", "c_combine"}


def test_get_prerequisites_depth_2() -> None:
    r = get_retriever()
    # c_frac -> c_linear/c_bracket -> (depth 2) c_eq_prop/c_move/c_combine
    ids = {p.id for p in r.get_prerequisites("c_frac", depth=2)}
    assert {"c_linear", "c_bracket", "c_eq_prop"}.issubset(ids)


def test_get_related_exercises_sorted_by_difficulty() -> None:
    r = get_retriever()
    exercises = r.get_related_exercises("c_linear")
    assert len(exercises) >= 2
    difficulties = [e.difficulty for e in exercises]
    assert difficulties == sorted(difficulties)


def test_get_misconceptions() -> None:
    r = get_retriever()
    ms = r.get_misconceptions("c_move")
    assert any("变号" in m.description for m in ms)


def test_retriever_singleton() -> None:
    assert get_retriever() is get_retriever()
    assert isinstance(get_retriever(), JsonGraphRetriever)
