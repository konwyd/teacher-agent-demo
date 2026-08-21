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
