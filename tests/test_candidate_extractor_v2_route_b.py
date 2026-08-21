import pytest

from app.memory.candidate_extractor_v2_route_b import (
    BgeM3Embedder,
    RouteBEmbeddingError,
    SemanticFamily,
    SemanticFamilyRanker,
)
from app.memory.candidate_extractor_v2_route_b.qualifiers import (
    analyze_clause,
    resolve_temporal_scope,
    segment_clauses,
)
from app.memory.contracts import TemporalScope


def test_qualifier_parser_preserves_negation_condition_and_scope():
    negative = analyze_clause("我不太喜欢分步骤说明")
    conditional = analyze_clause("只在化学课讲得慢一些")
    assert (negative.negated, negative.polarity) == (True, "negative")
    assert (conditional.condition, conditional.scope) == ("化学课", "chemistry")


def test_double_negation_and_temporal_precedence_are_deterministic():
    positive = analyze_clause("我并不是不想看例子")
    bounded = analyze_clause("本次给个例子，之后恢复默认")
    assert (positive.negated, positive.polarity) == (False, "positive")
    assert bounded.temporal_expression == "本次；之后恢复默认"
    assert resolve_temporal_scope(bounded, TemporalScope.PERSISTENT) is TemporalScope.TURN_ONLY


def test_correction_uncertainty_and_clause_segmentation_are_preserved():
    correction = analyze_clause("改一下，我其实已经掌握比例了")
    uncertain = analyze_clause("可能以后说得慢一些")
    assert correction.explicit_correction is True
    assert correction.correction_of == "prior claim"
    assert uncertain.uncertainty == "hedged"
    assert resolve_temporal_scope(uncertain, TemporalScope.PERSISTENT) is TemporalScope.UNKNOWN
    assert segment_clauses("叫我阿泽，并且讲得慢些") == ("叫我阿泽", "讲得慢些")


def test_explicit_turn_bound_wins_over_hedging():
    analysis = analyze_clause("可能本次给个例子")
    assert analysis.uncertainty == "hedged"
    assert analysis.temporal_hint is TemporalScope.TURN_ONLY
    assert resolve_temporal_scope(analysis, TemporalScope.PERSISTENT) is TemporalScope.TURN_ONLY


def test_clause_segmentation_binds_corrections_and_temporal_boundaries():
    assert segment_clauses("改一下，我其实已经掌握比例了") == ("改一下,我其实已经掌握比例了",)
    assert segment_clauses("本次给个例子，之后恢复默认") == ("本次给个例子,之后恢复默认",)
    assert segment_clauses("叫我阿泽，并且讲得慢些") == ("叫我阿泽", "讲得慢些")


def test_hard_negative_guards_are_explicit():
    assert analyze_clause("书中的人物叫阿泽").hard_negative_reason == "third_person"
    assert analyze_clause("他喜欢分步骤说明").hard_negative_reason == "third_person"
    assert analyze_clause("注意：他喜欢分步骤说明").hard_negative_reason == "third_person"
    assert analyze_clause("我在其他科目也想讲得慢一些").hard_negative_reason == ""
    assert analyze_clause("‘分步骤’这个词是什么含义").hard_negative_reason == "meta_quoted"
    assert analyze_clause("不要保存我的姓名").hard_negative_reason == "privacy_prohibition"


class FakeEmbeddingBackend:
    def __init__(self, vectors: dict[str, tuple[float, ...]]):
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts):
        batch = tuple(texts)
        self.calls.append(batch)
        return [self.vectors[text] for text in batch]


def test_ranker_batches_clauses_and_returns_multiple_semantic_families():
    backend = FakeEmbeddingBackend(
        {
            "学生希望老师采用较慢节奏": (1.0, 0.0),
            "学生陈述自己的姓名或称呼": (0.0, 1.0),
            "请放慢说明速度": (0.99, 0.01),
            "称呼我为阿泽": (0.01, 0.99),
        }
    )
    ranker = SemanticFamilyRanker(
        backend,
        prototypes={
            SemanticFamily.PREFERENCE: ("学生希望老师采用较慢节奏",),
            SemanticFamily.PROFILE: ("学生陈述自己的姓名或称呼",),
        },
        thresholds={family: 0.8 for family in SemanticFamily},
    )

    rows = ranker.rank(("请放慢说明速度", "称呼我为阿泽"))

    assert rows[0][0].family is SemanticFamily.PREFERENCE
    assert rows[1][0].family is SemanticFamily.PROFILE
    assert len(backend.calls) == 2  # one prototype batch, one clause batch


def test_bge_loader_rejects_missing_path_without_download(tmp_path):
    missing_path = tmp_path / "missing"

    with pytest.raises(RouteBEmbeddingError, match="local BGE-M3 model"):
        BgeM3Embedder(missing_path)

    assert not missing_path.exists()


def test_ranker_rejects_non_finite_or_ragged_vectors():
    backend = FakeEmbeddingBackend(
        {"prototype": (1.0, 0.0), "query": (float("nan"), 0.0)}
    )
    ranker = SemanticFamilyRanker(
        backend,
        prototypes={SemanticFamily.PREFERENCE: ("prototype",)},
        thresholds={family: 0.8 for family in SemanticFamily},
    )

    with pytest.raises(RouteBEmbeddingError, match="finite"):
        ranker.rank(("query",))


def test_ranker_rejects_ragged_vectors():
    backend = FakeEmbeddingBackend(
        {"prototype": (1.0, 0.0), "query": (1.0, 0.0, 0.0)}
    )
    ranker = SemanticFamilyRanker(
        backend,
        prototypes={SemanticFamily.PREFERENCE: ("prototype",)},
        thresholds={family: 0.8 for family in SemanticFamily},
    )

    with pytest.raises(RouteBEmbeddingError, match="dimension"):
        ranker.rank(("query",))


def test_ranker_caches_prototypes_and_tracks_encode_work():
    backend = FakeEmbeddingBackend(
        {
            "generic preference intent": (1.0, 0.0),
            "first request": (1.0, 0.0),
            "second request": (1.0, 0.0),
        }
    )
    ranker = SemanticFamilyRanker(
        backend,
        prototypes={SemanticFamily.PREFERENCE: ("generic preference intent",)},
        thresholds={family: 0.8 for family in SemanticFamily},
    )

    ranker.rank(("first request",))
    ranker.rank(("second request",))

    assert backend.calls == [
        ("generic preference intent",),
        ("first request",),
        ("second request",),
    ]
    assert ranker.encode_call_count == 3
    assert ranker.encoded_text_count == 3


def test_ranker_stably_orders_equal_similarity_matches():
    backend = FakeEmbeddingBackend(
        {
            "profile prototype": (1.0, 0.0),
            "preference prototype": (1.0, 0.0),
            "query": (1.0, 0.0),
        }
    )
    ranker = SemanticFamilyRanker(
        backend,
        prototypes={
            SemanticFamily.PREFERENCE: ("preference prototype",),
            SemanticFamily.PROFILE: ("profile prototype",),
        },
        thresholds={family: 0.8 for family in SemanticFamily},
    )

    matches = ranker.rank(("query",))[0]

    assert tuple(match.family for match in matches) == (
        SemanticFamily.PROFILE,
        SemanticFamily.PREFERENCE,
    )
