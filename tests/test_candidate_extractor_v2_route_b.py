from pathlib import Path
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
import unicodedata

import pytest

from app.evaluation.candidate_extractor.models import BenchmarkCase, ExpectedCandidate
from app.memory.candidate_extractor_v2_route_b.benchmark import run_route_b_benchmark
from scripts.run_candidate_extractor_route_b import build_parser

from app.memory.candidate_extractor_v2_route_b import (
    EMBEDDING_DIMENSION,
    MODEL_ID,
    MODEL_LICENSE,
    BgeM3Embedder,
    CandidateExtractorRouteB,
    FamilyMatch,
    RouteBConfig,
    RouteBEmbeddingError,
    SemanticFamily,
    SemanticFamilyRanker,
)
from app.memory.candidate_extractor_v2_route_b.qualifiers import (
    analyze_clause,
    resolve_temporal_scope,
    segment_clauses,
)
from app.memory.contracts import (
    MemorySourceKind,
    MemoryTargetClass,
    MemoryType,
    TemporalScope,
)


def test_route_b_dependency_and_model_metadata_are_explicit():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["optional-dependencies"]["candidate-extractor-route-b"] == [
        "sentence-transformers==5.6.0"
    ]
    assert MODEL_ID == "BAAI/bge-m3"
    assert MODEL_LICENSE == "MIT"
    assert EMBEDDING_DIMENSION == 1024


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
    assert (
        resolve_temporal_scope(bounded, TemporalScope.PERSISTENT)
        is TemporalScope.TURN_ONLY
    )


def test_correction_uncertainty_and_clause_segmentation_are_preserved():
    correction = analyze_clause("改一下，我其实已经掌握比例了")
    uncertain = analyze_clause("可能以后说得慢一些")
    assert correction.explicit_correction is True
    assert correction.correction_of == "prior claim"
    assert uncertain.uncertainty == "hedged"
    assert (
        resolve_temporal_scope(uncertain, TemporalScope.PERSISTENT)
        is TemporalScope.UNKNOWN
    )
    assert segment_clauses("叫我阿泽，并且讲得慢些") == ("叫我阿泽", "讲得慢些")


def test_explicit_turn_bound_wins_over_hedging():
    analysis = analyze_clause("可能本次给个例子")
    assert analysis.uncertainty == "hedged"
    assert analysis.temporal_hint is TemporalScope.TURN_ONLY
    assert (
        resolve_temporal_scope(analysis, TemporalScope.PERSISTENT)
        is TemporalScope.TURN_ONLY
    )


def test_clause_segmentation_binds_corrections_and_temporal_boundaries():
    assert segment_clauses("改一下，我其实已经掌握比例了") == (
        "改一下,我其实已经掌握比例了",
    )
    assert segment_clauses("本次给个例子，之后恢复默认") == (
        "本次给个例子,之后恢复默认",
    )
    assert segment_clauses("我读六年级，请称呼我Leo") == (
        "我读六年级",
        "请称呼我Leo",
    )
    assert segment_clauses("我读六年级。请称呼我Leo") == (
        "我读六年级",
        "请称呼我Leo",
    )
    assert segment_clauses("我读六年级，请称呼我Leo，然后讲慢些") == (
        "我读六年级",
        "请称呼我Leo",
        "讲慢些",
    )
    assert segment_clauses("叫我阿泽，并且讲得慢些") == ("叫我阿泽", "讲得慢些")


def test_hard_negative_guards_are_explicit():
    assert analyze_clause("书中的人物叫阿泽").hard_negative_reason == "third_person"
    assert analyze_clause("他喜欢分步骤说明").hard_negative_reason == "third_person"
    assert (
        analyze_clause("注意：他喜欢分步骤说明").hard_negative_reason == "third_person"
    )
    assert analyze_clause("我在其他科目也想讲得慢一些").hard_negative_reason == ""
    assert (
        analyze_clause("‘分步骤’这个词是什么含义").hard_negative_reason == "meta_quoted"
    )
    assert (
        analyze_clause("不要保存我的姓名").hard_negative_reason == "privacy_prohibition"
    )


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
    backend = FakeEmbeddingBackend({"prototype": (1.0, 0.0), "query": (1.0, 0.0, 0.0)})
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


class StubFamilyRanker:
    def __init__(
        self,
        families_by_text: dict[str, SemanticFamily],
        *,
        similarity: float = 1.0,
    ):
        self.families_by_text = {
            unicodedata.normalize("NFKC", text): family
            for text, family in families_by_text.items()
        }
        self.similarity = similarity
        self.calls: list[tuple[str, ...]] = []
        self.encode_call_count = 0
        self.encoded_text_count = 0

    def rank(self, texts):
        texts = tuple(texts)
        self.calls.append(texts)
        if texts:
            self.encode_call_count += 1
            self.encoded_text_count += len(texts)
        return tuple(
            (
                FamilyMatch(
                    family=self.families_by_text[text],
                    similarity=self.similarity,
                ),
            )
            if text in self.families_by_text
            else ()
            for text in texts
        )


@pytest.fixture
def route_b_factory():
    def make(families_by_text: dict[str, SemanticFamily]) -> CandidateExtractorRouteB:
        return CandidateExtractorRouteB(StubFamilyRanker(families_by_text))

    return make


def test_embedding_selects_preference_but_parser_owns_qualifiers(route_b_factory):
    extractor = route_b_factory({"我不想看示范": SemanticFamily.PREFERENCE})
    [candidate] = extractor.extract(
        "我不想看示范",
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-test",
    )
    assert candidate.memory_type is MemoryType.PREFERENCE
    assert candidate.payload == {
        "teaching_preferences": ["示例讲解"],
        "negated": True,
        "polarity": "negative",
    }
    assert candidate.temporal_scope is TemporalScope.PERSISTENT


def test_temporary_expression_cannot_inflate_to_persistent(route_b_factory):
    extractor = route_b_factory({"这一会儿请慢一些": SemanticFamily.PREFERENCE})
    [candidate] = extractor.extract(
        "这一会儿请慢一些",
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-test",
    )
    assert candidate.payload["temporal_expression"] == "这一会儿"
    assert candidate.temporal_scope is TemporalScope.SHORT_LIVED


def test_profile_multi_field_and_difficulty_payloads_are_bounded(route_b_factory):
    extractor = route_b_factory(
        {
            "我读六年级": SemanticFamily.PROFILE,
            "请称呼我Leo": SemanticFamily.PROFILE,
            "我经常把正负号搞反": SemanticFamily.DIFFICULTY,
        }
    )
    [profile] = extractor.extract(
        "我读六年级，请称呼我Ｌｅｏ",
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-profile",
    )
    assert extractor.ranker.calls == [("我读六年级", "请称呼我Leo")]
    [difficulty] = extractor.extract(
        "我经常把正负号搞反",
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-difficulty",
    )
    assert profile.payload == {"grade": "六年级", "preferred_name": "leo"}
    assert difficulty.payload == {
        "subject": "数学",
        "topic": "正负号",
        "statement": "搞反正负号",
    }


@pytest.mark.parametrize(
    ("utterance", "families_by_text"),
    (
        (
            "我不是六年级，请称呼我Leo",
            {
                "我不是六年级": SemanticFamily.PROFILE,
                "请称呼我Leo": SemanticFamily.PROFILE,
            },
        ),
        (
            "请称呼我Leo，我妹妹读六年级",
            {
                "请称呼我Leo": SemanticFamily.PROFILE,
                "我妹妹读六年级": SemanticFamily.PROFILE,
            },
        ),
        (
            "请称呼我Leo，张三是六年级",
            {
                "请称呼我Leo": SemanticFamily.PROFILE,
                "张三是六年级": SemanticFamily.PROFILE,
            },
        ),
    ),
)
def test_profile_qualifiers_and_referents_do_not_cross_comma_clauses(
    route_b_factory, utterance, families_by_text
):
    extractor = route_b_factory(families_by_text)

    [candidate] = extractor.extract(utterance, {}, conversation_id="route-b-profile")

    assert candidate.payload == {"preferred_name": "leo"}


def test_duplicate_compatible_candidates_are_merged_once(route_b_factory):
    extractor = route_b_factory({"我读六年级": SemanticFamily.PROFILE})

    candidates = extractor.extract(
        "我读六年级，并且我读六年级",
        {},
        conversation_id="route-b-profile-duplicate",
    )

    assert len(candidates) == 1
    assert candidates[0].payload == {"grade": "六年级"}
    assert len({candidate.candidate_id for candidate in candidates}) == 1


@pytest.mark.parametrize(
    ("utterance", "family"),
    (
        ("小明不会二次函数", SemanticFamily.DIFFICULTY),
        ("我弟弟不会二次函数", SemanticFamily.DIFFICULTY),
        ("小明是六年级", SemanticFamily.PROFILE),
        ("小明经常把正负号搞反", SemanticFamily.DIFFICULTY),
        ("小明连续三次都算对了方程", SemanticFamily.BEHAVIOR_EVIDENCE),
        ("张三是六年级", SemanticFamily.PROFILE),
        ("张三经常把正负号搞反", SemanticFamily.DIFFICULTY),
        ("张三连续三次都算对了方程", SemanticFamily.BEHAVIOR_EVIDENCE),
        ("张三更喜欢图示讲解", SemanticFamily.PREFERENCE),
        ("昨天张三练习解出了方程", SemanticFamily.LEARNING_EPISODE),
        ("本次张三更喜欢图示讲解", SemanticFamily.PREFERENCE),
        ("我希望张三更喜欢图示讲解", SemanticFamily.PREFERENCE),
        ("本次讲慢点张三更喜欢", SemanticFamily.PREFERENCE),
        ("本次张三讲慢点", SemanticFamily.PREFERENCE),
        ("今天老师逐步说明", SemanticFamily.PREFERENCE),
        ("逐步说明", SemanticFamily.PREFERENCE),
        ("麻烦张三逐步说明", SemanticFamily.PREFERENCE),
        ("如果是张三请示范", SemanticFamily.PREFERENCE),
        ("本次请给张三一个例子", SemanticFamily.PREFERENCE),
        ("以后都给 张三一个例子", SemanticFamily.PREFERENCE),
        ("以后都给张三非常耐心地讲一个例子", SemanticFamily.PREFERENCE),
        ("只在数学课给张三讲慢一点", SemanticFamily.PREFERENCE),
        ("只在数学课给张三非常耐心地讲慢一点", SemanticFamily.PREFERENCE),
        ("只在数学课讲慢点张三更喜欢", SemanticFamily.PREFERENCE),
        ("如果张三需要请讲慢一些", SemanticFamily.PREFERENCE),
        ("以后都用图示张三更喜欢", SemanticFamily.PREFERENCE),
        ("昨天练习时张三解出了方程", SemanticFamily.LEARNING_EPISODE),
        ("昨天课堂张三回答了问题", SemanticFamily.LEARNING_EPISODE),
        ("昨天练习时张三讲了方法", SemanticFamily.LEARNING_EPISODE),
    ),
)
def test_third_person_fact_is_a_hard_negative(route_b_factory, utterance, family):
    extractor = route_b_factory({utterance: family})

    assert (
        extractor.extract(
            utterance,
            {"classification": {"subject": "数学"}},
            conversation_id="route-b-third-person",
        )
        == []
    )


def test_incompatible_temporal_profile_candidates_remain_separate(route_b_factory):
    extractor = route_b_factory(
        {
            "我读六年级": SemanticFamily.PROFILE,
            "本次请称呼我Leo": SemanticFamily.PROFILE,
        }
    )

    candidates = extractor.extract(
        "我读六年级，并且本次请称呼我Leo",
        {},
        conversation_id="route-b-profile-scope",
    )

    assert len(candidates) == 2
    assert {candidate.temporal_scope for candidate in candidates} == {
        TemporalScope.PERSISTENT,
        TemporalScope.TURN_ONLY,
    }


def test_implicit_first_person_difficulty_is_not_suppressed(route_b_factory):
    utterance = "还是不会二次函数"
    extractor = route_b_factory({utterance: SemanticFamily.DIFFICULTY})

    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-first-person",
    )

    assert candidate.payload["topic"] == "二次函数"


def test_implicit_first_person_progress_is_not_suppressed(route_b_factory):
    utterance = "已经掌握一元一次方程"
    extractor = route_b_factory({utterance: SemanticFamily.TOPIC_PROGRESS})

    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-first-person-progress",
    )

    assert candidate.payload["topic"] == "一元一次方程"


@pytest.mark.parametrize(
    ("utterance", "family", "memory_type"),
    (
        ("比较喜欢图示讲解", SemanticFamily.PREFERENCE, MemoryType.PREFERENCE),
        ("有点不懂二次函数", SemanticFamily.DIFFICULTY, MemoryType.DIFFICULTY),
        ("本人读六年级", SemanticFamily.PROFILE, MemoryType.PROFILE_FACT),
        ("老师讲慢点我更喜欢", SemanticFamily.PREFERENCE, MemoryType.PREFERENCE),
        ("小例子我更喜欢", SemanticFamily.PREFERENCE, MemoryType.PREFERENCE),
        ("阿拉伯数字我更喜欢用图示", SemanticFamily.PREFERENCE, MemoryType.PREFERENCE),
        ("特别喜欢图示讲解", SemanticFamily.PREFERENCE, MemoryType.PREFERENCE),
        ("完全不懂二次函数", SemanticFamily.DIFFICULTY, MemoryType.DIFFICULTY),
        ("我现在读六年级", SemanticFamily.PROFILE, MemoryType.PROFILE_FACT),
        ("我今年读六年级", SemanticFamily.PROFILE, MemoryType.PROFILE_FACT),
        (
            "我真的特别喜欢图示讲解",
            SemanticFamily.PREFERENCE,
            MemoryType.PREFERENCE,
        ),
        (
            "我还是更喜欢图示讲解",
            SemanticFamily.PREFERENCE,
            MemoryType.PREFERENCE,
        ),
        (
            "我真的完全不懂二次函数",
            SemanticFamily.DIFFICULTY,
            MemoryType.DIFFICULTY,
        ),
    ),
)
def test_common_first_person_claim_is_not_suppressed(
    route_b_factory, utterance, family, memory_type
):
    extractor = route_b_factory({utterance: family})

    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-first-person-common",
    )

    assert candidate.memory_type is memory_type


@pytest.mark.parametrize(
    "utterance",
    (
        "请给我一个例子",
        "这道题能画图吗",
        "我希望这道题能画图",
        "我希望你给我一个例子",
        "我希望你讲个例题",
        "我希望这题能画图",
        "我希望你演示一道例题",
        "我希望你做一道示范",
        "我希望张三演示一道例题",
        "我希望张三给我一个例子",
        "我希望老师给我一个例子",
    ),
)
def test_current_instructional_request_is_not_a_persistent_preference(
    route_b_factory, utterance
):
    extractor = route_b_factory({utterance: SemanticFamily.PREFERENCE})

    assert extractor.extract(utterance, {}, conversation_id="route-b-request") == []


@pytest.mark.parametrize(
    ("utterance", "expected_scope"),
    (
        ("本次请给我一个例子", TemporalScope.TURN_ONLY),
        ("以后都请给我一个例子", TemporalScope.PERSISTENT),
    ),
)
def test_explicitly_bounded_instructional_preference_is_extractable(
    route_b_factory, utterance, expected_scope
):
    extractor = route_b_factory({utterance: SemanticFamily.PREFERENCE})

    [candidate] = extractor.extract(utterance, {}, conversation_id="route-b-bounded")

    assert candidate.temporal_scope is expected_scope


def test_conditional_instructional_preference_is_extractable(route_b_factory):
    utterance = "如果需要请讲慢一些"
    extractor = route_b_factory({utterance: SemanticFamily.PREFERENCE})

    [candidate] = extractor.extract(utterance, {}, conversation_id="route-b-condition")

    assert candidate.payload["teaching_preferences"] == ["慢节奏讲解"]
    assert candidate.payload["condition"] == "需要请讲慢一些"
    assert candidate.temporal_scope is TemporalScope.PERSISTENT


def test_stated_durable_preference_remains_extractable(route_b_factory):
    extractor = route_b_factory({"我喜欢看示范": SemanticFamily.PREFERENCE})

    [candidate] = extractor.extract(
        "我喜欢看示范", {}, conversation_id="route-b-preference"
    )

    assert candidate.payload == {"teaching_preferences": ["示例讲解"]}
    assert candidate.temporal_scope is TemporalScope.PERSISTENT


def test_learning_goal_parser_requires_subject_and_preserves_deadline(route_b_factory):
    utterance = "我想在期末前掌握平面几何"
    extractor = route_b_factory({utterance: SemanticFamily.LEARNING_GOAL})

    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-goal",
    )

    assert candidate.memory_type is MemoryType.LEARNING_GOAL
    assert candidate.payload == {
        "subject": "数学",
        "goal": "掌握平面几何",
        "deadline_expression": "期末前",
    }
    assert candidate.temporal_scope is TemporalScope.PERSISTENT


def test_topic_progress_parser_emits_bounded_status(route_b_factory):
    utterance = "我已经掌握一元一次方程了"
    extractor = route_b_factory({utterance: SemanticFamily.TOPIC_PROGRESS})

    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-progress",
    )

    assert candidate.memory_type is MemoryType.TOPIC_PROGRESS
    assert candidate.payload == {
        "subject": "数学",
        "topic": "一元一次方程",
        "status": "mastered",
    }
    assert candidate.temporal_scope is TemporalScope.SHORT_LIVED


def test_task_state_parser_emits_current_requested_target(route_b_factory):
    utterance = "我现在要完成第十二题"
    extractor = route_b_factory({utterance: SemanticFamily.TASK_STATE})

    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-task",
    )

    assert candidate.memory_type is MemoryType.TASK_STATE
    assert candidate.payload == {"subject": "数学", "requested_target": "完成第十二题"}
    assert candidate.temporal_scope is TemporalScope.SESSION


def test_learning_episode_parser_requires_a_specific_episode_marker(route_b_factory):
    utterance = "上次练习我独立解出了方程题"
    extractor = route_b_factory({utterance: SemanticFamily.LEARNING_EPISODE})

    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-episode",
    )

    assert candidate.memory_type is MemoryType.LEARNING_EPISODE
    assert candidate.payload == {
        "subject": "数学",
        "summary": "上次练习我独立解出了方程题",
    }
    assert candidate.temporal_scope is TemporalScope.SHORT_LIVED


def test_behavior_evidence_parser_requires_an_observed_pattern(route_b_factory):
    utterance = "我连续三次都独立算对了方程"
    extractor = route_b_factory({utterance: SemanticFamily.BEHAVIOR_EVIDENCE})

    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-behavior",
    )

    assert candidate.memory_type is MemoryType.BEHAVIOR_EVIDENCE
    assert candidate.payload == {
        "subject": "数学",
        "evidence_pattern": "连续三次都独立算对了方程",
    }
    assert candidate.temporal_scope is TemporalScope.SHORT_LIVED


@pytest.mark.parametrize(
    ("utterance", "family"),
    (
        ("这是我的个人情况", SemanticFamily.PROFILE),
        ("我有一种讲课偏好", SemanticFamily.PREFERENCE),
        ("我有一个目标", SemanticFamily.LEARNING_GOAL),
        ("我遇到学习麻烦", SemanticFamily.DIFFICULTY),
        ("我有一些进展", SemanticFamily.TOPIC_PROGRESS),
        ("我有当前任务", SemanticFamily.TASK_STATE),
        ("我有一次学习经历", SemanticFamily.LEARNING_EPISODE),
        ("我有学习表现", SemanticFamily.BEHAVIOR_EVIDENCE),
    ),
)
def test_high_family_similarity_cannot_bypass_mandatory_fields(
    route_b_factory, utterance, family
):
    extractor = route_b_factory({utterance: family})

    assert (
        extractor.extract(
            utterance,
            {"classification": {"subject": "数学"}},
            conversation_id="route-b-guard",
        )
        == []
    )


def test_user_stated_candidate_has_deterministic_id_and_bounded_provenance(
    route_b_factory,
):
    extractor = route_b_factory({"请叫我Mika": SemanticFamily.PROFILE})

    [first] = extractor.extract(
        "请叫我Mika", {}, conversation_id="route-b-provenance", turn_id=7
    )
    [second] = extractor.extract(
        "请叫我Mika", {}, conversation_id="route-b-provenance", turn_id=7
    )

    assert first.candidate_id == second.candidate_id
    assert first.candidate_id.startswith("route_b_")
    assert first.target_memory_class is MemoryTargetClass.STUDENT
    assert first.source_kind is MemorySourceKind.USER_STATED
    assert first.confidence == 1.0
    assert len(first.provenance) == 1
    provenance = first.provenance[0]
    assert (
        provenance.conversation_id,
        provenance.turn_id,
        provenance.event_reference,
    ) == ("route-b-provenance", 7, "turn:7")


def test_route_config_threshold_rejects_a_weaker_family_match():
    extractor = CandidateExtractorRouteB(
        StubFamilyRanker({"请叫我Nia": SemanticFamily.PROFILE}, similarity=0.8),
        RouteBConfig(family_thresholds={family: 0.9 for family in SemanticFamily}),
    )

    assert extractor.extract("请叫我Nia", {}, conversation_id="route-b-threshold") == []


def test_difficulty_parser_handles_student_stated_noncomprehension(route_b_factory):
    utterance = "我还是不懂二次函数"
    extractor = route_b_factory({utterance: SemanticFamily.DIFFICULTY})

    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-difficulty-paraphrase",
    )

    assert candidate.payload == {
        "subject": "数学",
        "topic": "二次函数",
        "statement": "不懂二次函数",
        "negated": True,
        "polarity": "negative",
    }
    assert candidate.temporal_scope is TemporalScope.SHORT_LIVED


def test_multiple_clauses_emit_multiple_candidates_without_qualifier_leak(
    route_b_factory,
):
    extractor = route_b_factory(
        {
            "称呼我阿泽": SemanticFamily.PROFILE,
            "今天给我一些例子": SemanticFamily.PREFERENCE,
        }
    )
    actual = extractor.extract(
        "称呼我阿泽，并且今天给我一些例子",
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-multi",
    )
    assert [item.memory_type for item in actual] == [
        MemoryType.PROFILE_FACT,
        MemoryType.PREFERENCE,
    ]
    assert "temporal_expression" not in actual[0].payload
    assert actual[1].payload["temporal_expression"] == "今天"
    assert actual[1].temporal_scope is TemporalScope.SESSION


def test_verification_and_model_inference_keep_declared_source(route_b_factory):
    extractor = route_b_factory({})
    actual = extractor.extract(
        "",
        {
            "classification": {"subject": "英语"},
            "verification": {
                "status": "verified_correct",
                "summary": "checked",
                "verifier_name": "math-checker",
            },
            "model_inferences": ({"memory_type": "difficulty", "topic": "past tense"},),
        },
        conversation_id="route-b-upstream",
    )
    assert {item.source_kind for item in actual} == {
        MemorySourceKind.VERIFIED_RESULT,
        MemorySourceKind.MODEL_INFERENCE,
    }
    inferred = next(
        item for item in actual if item.source_kind is MemorySourceKind.MODEL_INFERENCE
    )
    assert inferred.temporal_scope is TemporalScope.UNKNOWN
    verified = next(
        item for item in actual if item.source_kind is MemorySourceKind.VERIFIED_RESULT
    )
    assert verified.provenance[0].source_kind is MemorySourceKind.VERIFIED_RESULT
    assert verified.provenance[0].verifier_reference == "math-checker"


def test_observations_preserve_source_and_exact_payload_deduplication(route_b_factory):
    extractor = route_b_factory({})
    observation = {"evidence_pattern": "连续三次算对方程"}
    actual = extractor.extract(
        "",
        {
            "classification": {"subject": "数学"},
            "observations": (observation, observation),
        },
        conversation_id="route-b-observation",
    )
    assert len(actual) == 1
    assert actual[0].source_kind is MemorySourceKind.OBSERVED_BEHAVIOR
    assert actual[0].payload == {
        "subject": "数学",
        "evidence_pattern": "连续三次算对方程",
    }
    assert actual[0].provenance[0].source_kind is MemorySourceKind.OBSERVED_BEHAVIOR


def test_malformed_optional_upstream_items_fail_safe_and_are_counted(
    route_b_factory,
):
    extractor = route_b_factory({})
    actual = extractor.extract(
        "",
        {
            "verification": "not-a-mapping",
            "observations": (
                None,
                {"missing": "evidence_pattern"},
                {"evidence_pattern": ""},
            ),
            "model_inferences": (
                "not-a-mapping",
                {"memory_type": "difficulty", "topic": ""},
            ),
        },
        conversation_id="route-b-malformed",
    )
    assert actual == []
    assert extractor.diagnostics.malformed_upstream_items == 6


def test_hard_negatives_emit_no_persistent_candidate(route_b_factory):
    extractor = route_b_factory(
        {
            "课本里的学生叫阿泽": SemanticFamily.PROFILE,
            "请解释‘慢慢说明’是什么意思": SemanticFamily.PREFERENCE,
            "别保存我的姓名": SemanticFamily.PROFILE,
        }
    )
    for text in (
        "课本里的学生叫阿泽",
        "请解释‘慢慢说明’是什么意思",
        "别保存我的姓名",
    ):
        assert extractor.extract(text, {}, conversation_id="route-b-hard") == []


def test_frozen_topic_status_observation_uses_verification_source(route_b_factory):
    extractor = route_b_factory({})

    actual = extractor.extract(
        "我会独立解方程了",
        {
            "classification": {"subject": "数学"},
            "verification": {"status": "verified_correct", "summary": "verified"},
            "observations": ({"topic": "方程", "status": "independent"},),
        },
        conversation_id="route-b-frozen-topic-status",
    )

    assert len(actual) == 1
    assert actual[0].memory_type is MemoryType.TOPIC_PROGRESS
    assert actual[0].payload == {
        "subject": "数学",
        "topic": "方程",
        "status": "independent",
    }
    assert actual[0].source_kind is MemorySourceKind.VERIFIED_RESULT
    assert actual[0].provenance[0].source_kind is MemorySourceKind.VERIFIED_RESULT


def test_frozen_episode_observation_keeps_observed_source(route_b_factory):
    extractor = route_b_factory({})

    [candidate] = extractor.extract(
        "画图后我理解了",
        {
            "classification": {"subject": "数学"},
            "observations": ({"episode": "diagram enabled understanding"},),
        },
        conversation_id="route-b-frozen-episode",
    )

    assert candidate.memory_type is MemoryType.LEARNING_EPISODE
    assert candidate.payload == {
        "subject": "数学",
        "episode": "diagram enabled understanding",
    }
    assert candidate.source_kind is MemorySourceKind.OBSERVED_BEHAVIOR
    assert candidate.provenance[0].source_kind is MemorySourceKind.OBSERVED_BEHAVIOR


def test_frozen_pattern_count_observation_maps_to_evidence_pattern(route_b_factory):
    extractor = route_b_factory({})

    [candidate] = extractor.extract(
        "连续三题先列式",
        {
            "classification": {"subject": "数学"},
            "observations": ({"pattern": "equation first", "count": 3},),
        },
        conversation_id="route-b-frozen-pattern",
    )

    assert candidate.memory_type is MemoryType.BEHAVIOR_EVIDENCE
    assert candidate.payload == {
        "subject": "数学",
        "evidence_pattern": "equation first",
    }
    assert candidate.source_kind is MemorySourceKind.OBSERVED_BEHAVIOR


def test_frozen_claim_inference_maps_to_difficulty(route_b_factory):
    extractor = route_b_factory({})

    [candidate] = extractor.extract(
        "",
        {
            "classification": {"subject": "英语"},
            "model_inferences": ({"claim": "difficulty", "topic": "past tense"},),
        },
        conversation_id="route-b-frozen-inference",
    )

    assert candidate.memory_type is MemoryType.DIFFICULTY
    assert candidate.payload == {
        "subject": "英语",
        "topic": "past tense",
        "uncertainty": "model_inference",
    }
    assert candidate.source_kind is MemorySourceKind.MODEL_INFERENCE
    assert candidate.provenance[0].source_kind is MemorySourceKind.MODEL_INFERENCE


def test_candidate_ids_distinguish_identical_payloads_from_different_sources(
    route_b_factory,
):
    utterance = "我连续三次都独立算对了方程"
    evidence_pattern = "连续三次都独立算对了方程"
    extractor = route_b_factory({utterance: SemanticFamily.BEHAVIOR_EVIDENCE})

    actual = extractor.extract(
        utterance,
        {
            "classification": {"subject": "数学"},
            "observations": ({"evidence_pattern": evidence_pattern},),
        },
        conversation_id="route-b-source-ids",
    )

    assert len(actual) == 2
    assert {item.source_kind for item in actual} == {
        MemorySourceKind.USER_STATED,
        MemorySourceKind.OBSERVED_BEHAVIOR,
    }
    assert len({item.candidate_id for item in actual}) == 2


def test_whitespace_required_upstream_strings_are_rejected_and_counted(
    route_b_factory,
):
    extractor = route_b_factory({})

    actual = extractor.extract(
        "",
        {
            "verification": {"status": "   "},
            "observations": (
                {"episode": "   "},
                {"pattern": "   ", "count": 3},
                {"evidence_pattern": "   "},
                {"topic": "方程", "status": "   "},
            ),
            "model_inferences": (
                {"claim": "difficulty", "topic": "   "},
                {"memory_type": "difficulty", "topic": "   "},
            ),
        },
        conversation_id="route-b-whitespace",
    )

    assert actual == []
    assert extractor.diagnostics.malformed_upstream_items == 7


def test_falsey_wrong_upstream_containers_are_counted_as_malformed(route_b_factory):
    extractor = route_b_factory({})

    assert (
        extractor.extract(
            "",
            {
                "verification": [],
                "observations": {},
                "model_inferences": "",
            },
            conversation_id="route-b-falsey-containers",
        )
        == []
    )
    assert extractor.diagnostics.malformed_upstream_items == 3


def test_non_string_required_upstream_values_are_rejected_and_counted(
    route_b_factory,
):
    extractor = route_b_factory({})

    actual = extractor.extract(
        "",
        {
            "verification": {"status": ["verified_correct"]},
            "observations": (
                {"episode": {"text": "diagram enabled understanding"}},
                {"pattern": 3},
                {"evidence_pattern": True},
                {"topic": ["方程"], "status": "independent"},
                {"topic": "方程", "status": {"value": "independent"}},
            ),
            "model_inferences": (
                {"claim": ["difficulty"], "topic": "past tense"},
                {"memory_type": "difficulty", "topic": {"name": "past tense"}},
            ),
        },
        conversation_id="route-b-non-string-required",
    )

    assert actual == []
    assert extractor.diagnostics.malformed_upstream_items == 8


def test_conflicting_model_inference_discriminators_fail_safe_and_are_counted(
    route_b_factory,
):
    extractor = route_b_factory({})

    actual = extractor.extract(
        "",
        {
            "classification": {"subject": "英语"},
            "model_inferences": (
                {
                    "claim": "mastery",
                    "memory_type": "difficulty",
                    "topic": "past tense",
                },
                {
                    "claim": "difficulty",
                    "memory_type": "mastery",
                    "topic": "past tense",
                },
            ),
        },
        conversation_id="route-b-conflicting-discriminator",
    )

    assert actual == []
    assert extractor.diagnostics.malformed_upstream_items == 2


def test_route_b_package_import_does_not_eagerly_load_evaluation_runtime():
    script = (
        "import sys; "
        "import app.memory.candidate_extractor_v2_route_b; "
        "assert 'app.evaluation.candidate_extractor.baseline' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _synthetic_profile_case(utterance: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="synthetic-profile",
        split="development",
        utterance=utterance,
        labels=("PROFILE",),
        rationale="route-local adapter test",
        expected_candidates=(
            ExpectedCandidate(
                memory_type="profile_fact",
                payload={"preferred_name": "阿泽"},
                source_kind="user_stated",
                temporal_scope="persistent",
            ),
        ),
    )


def test_route_runner_exposes_only_development_and_layer_b():
    parser = build_parser()
    split_action = next(action for action in parser._actions if action.dest == "splits")

    assert set(split_action.choices) == {"development", "layer_b"}


def test_route_benchmark_rejects_an_unsupported_split(route_b_factory):
    extractor = route_b_factory({})

    with pytest.raises(ValueError, match="development and layer_b only"):
        run_route_b_benchmark(extractor, splits=("unsupported",))


def test_route_benchmark_rejects_case_with_mismatched_declared_split(
    route_b_factory,
):
    extractor = route_b_factory({})
    case = _synthetic_profile_case("称呼我阿泽").model_copy(
        update={"split": "unsupported"}
    )

    with pytest.raises(ValueError, match="case split must match selected split"):
        run_route_b_benchmark(
            extractor,
            splits=("development",),
            cases_by_split={"development": [case]},
        )


def test_benchmark_adapter_preserves_frozen_metrics_and_reports_slices(
    route_b_factory,
):
    utterance = "称呼我阿泽"
    extractor = route_b_factory({utterance: SemanticFamily.PROFILE})

    report = run_route_b_benchmark(
        extractor,
        splits=("development",),
        cases_by_split={"development": [_synthetic_profile_case(utterance)]},
        provisional=True,
    )

    split = report["reports"]["development"]
    assert split["exact_candidate_set_accuracy"]["value"] == 1.0
    assert split["operational"]["additional_model_calls"] == 0
    assert split["operational"]["deterministic_replay"] is True
    assert report["operational"]["remote_llm_calls"] == 0
    assert report["operational"]["provisional"] is True
    assert report["operational"]["deterministic_replay"] is True
    assert report["slice_metrics"]["development"]["multi_candidate_exact_match"] == {
        "value": None,
        "support": 0,
        "numerator": 0,
    }
    operational = report["operational"]
    assert operational["cold_load_case_id"] == "synthetic-profile"
    assert operational["primary_encode_calls"] == 1
    assert operational["replay_encode_calls"] == 1
    assert operational["encode_calls"] == 2
    assert operational["primary_encoded_text_count"] == 1
    assert operational["replay_encoded_text_count"] == 1
    assert operational["encoded_text_count"] == 2


def test_cold_load_is_first_embedding_consuming_case(route_b_factory):
    hard_negative = BenchmarkCase(
        case_id="synthetic-hard-negative",
        split="development",
        utterance="书中的人物叫阿泽",
        labels=("HARD_NEGATIVE",),
        rationale="no embedding work before model load",
        expected_candidates=(),
    )
    profile = _synthetic_profile_case("称呼我阿泽")
    extractor = route_b_factory({"称呼我阿泽": SemanticFamily.PROFILE})

    report = run_route_b_benchmark(
        extractor,
        splits=("development",),
        cases_by_split={"development": [hard_negative, profile]},
    )

    assert report["operational"]["cold_load_case_id"] == "synthetic-profile"
    assert report["operational"]["primary_encode_calls"] == 1


def test_tuned_default_threshold_admits_a_moderate_generic_family_match():
    extractor = CandidateExtractorRouteB(
        StubFamilyRanker({"请称呼我小禾": SemanticFamily.PROFILE}, similarity=0.5)
    )
    [candidate] = extractor.extract(
        "请称呼我小禾", {}, conversation_id="route-b-tuned-threshold"
    )
    assert candidate.memory_type is MemoryType.PROFILE_FACT


@pytest.mark.parametrize(
    ("utterance", "expected_payload", "expected_scope"),
    (
        (
            "麻烦逐步说明",
            {"teaching_preferences": ["分步骤讲解"]},
            TemporalScope.PERSISTENT,
        ),
        (
            "麻烦尽量逐步说明",
            {"teaching_preferences": ["分步骤讲解"]},
            TemporalScope.PERSISTENT,
        ),
        (
            "如果是代数题请示范",
            {"teaching_preferences": ["示例讲解"], "condition": "代数题"},
            TemporalScope.PERSISTENT,
        ),
        (
            "或许以后请讲慢一些",
            {
                "teaching_preferences": ["慢节奏讲解"],
                "temporal_expression": "以后",
                "uncertainty": "hedged",
            },
            TemporalScope.UNKNOWN,
        ),
        (
            "平时请举例，这道题不要",
            {
                "teaching_preferences": ["示例讲解"],
                "condition": "except current problem",
                "scope": "general",
            },
            TemporalScope.PERSISTENT,
        ),
    ),
)
def test_generalized_preference_requests_preserve_bounds(
    route_b_factory, utterance, expected_payload, expected_scope
):
    extractor = route_b_factory({utterance: SemanticFamily.PREFERENCE})
    [candidate] = extractor.extract(utterance, {}, conversation_id="route-b-tuning")
    assert candidate.payload == expected_payload
    assert candidate.temporal_scope is expected_scope


def test_generalized_deadline_first_goal_preserves_temporal_expression(route_b_factory):
    utterance = "月考前掌握百分数"
    extractor = route_b_factory({utterance: SemanticFamily.LEARNING_GOAL})
    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-tuning-goal",
    )
    assert candidate.payload == {
        "subject": "数学",
        "goal": "掌握百分数",
        "temporal_expression": "月考前",
    }
    assert candidate.temporal_scope is TemporalScope.SHORT_LIVED


def test_generalized_stable_confusion_maps_to_a_topic(route_b_factory):
    utterance = "我总弄混被除数和除数"
    extractor = route_b_factory({utterance: SemanticFamily.DIFFICULTY})
    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-tuning-difficulty",
    )
    assert candidate.payload == {
        "subject": "数学",
        "topic": "除法",
        "statement": "弄混被除数和除数",
    }
    assert candidate.temporal_scope is TemporalScope.PERSISTENT


def test_generalized_continuation_is_turn_bounded(route_b_factory):
    utterance = "接着上一题"
    extractor = route_b_factory({utterance: SemanticFamily.TASK_STATE})
    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-tuning-task",
    )
    assert candidate.payload == {"subject": "数学", "requested_target": "上一题"}
    assert candidate.temporal_scope is TemporalScope.TURN_ONLY


def test_generalized_resolution_transition_is_topic_progress(route_b_factory):
    utterance = "我以前怕小数，现在会了"
    extractor = route_b_factory({utterance: SemanticFamily.TOPIC_PROGRESS})
    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-tuning-progress",
    )
    assert candidate.payload == {
        "subject": "数学",
        "topic": "小数",
        "status": "resolved",
        "explicit_correction": True,
        "correction_of": "difficulty",
    }
    assert candidate.temporal_scope is TemporalScope.SHORT_LIVED


def test_profile_hedge_uses_contract_uncertainty_value(route_b_factory):
    utterance = "我可能叫小禾"
    extractor = route_b_factory({utterance: SemanticFamily.PROFILE})
    [candidate] = extractor.extract(
        utterance, {}, conversation_id="route-b-tuning-name"
    )
    assert candidate.payload == {
        "preferred_name": "小禾",
        "uncertainty": "uncertain",
    }
    assert candidate.temporal_scope is TemporalScope.UNKNOWN


def test_preference_correction_has_typed_correction_target(route_b_factory):
    utterance = "纠正一下，我其实偏好图解"
    extractor = route_b_factory({utterance: SemanticFamily.PREFERENCE})
    [candidate] = extractor.extract(
        utterance, {}, conversation_id="route-b-tuning-correction"
    )
    assert candidate.payload["correction_of"] == "prior preference"


@pytest.mark.parametrize(
    ("utterance", "expression", "scope"),
    (
        ("今天先逐步说明，明天恢复", "今天;明天恢复", TemporalScope.SESSION),
        ("本轮请示范，之后不用", "本轮;之后不用", TemporalScope.TURN_ONLY),
    ),
)
def test_generalized_restore_bound_stays_clause_local(
    route_b_factory, utterance, expression, scope
):
    extractor = route_b_factory({utterance: SemanticFamily.PREFERENCE})
    [candidate] = extractor.extract(
        utterance, {}, conversation_id="route-b-tuning-restore"
    )
    assert candidate.payload["temporal_expression"] == expression
    assert "negated" not in candidate.payload
    assert "polarity" not in candidate.payload
    assert candidate.temporal_scope is scope


def test_explicit_self_double_negation_keeps_preference_owner(route_b_factory):
    utterance = "我不是不偏好图解"
    extractor = route_b_factory({utterance: SemanticFamily.PREFERENCE})
    [candidate] = extractor.extract(
        utterance, {}, conversation_id="route-b-tuning-negation"
    )
    assert candidate.payload["negated"] is False


def test_grammatical_and_lexical_bu_do_not_cancel_each_other(route_b_factory):
    difficulty_text = "我不懂不等式"
    preference_text = "我不喜欢用不等式图示讲解"
    extractor = route_b_factory(
        {
            difficulty_text: SemanticFamily.DIFFICULTY,
            preference_text: SemanticFamily.PREFERENCE,
        }
    )

    [difficulty] = extractor.extract(
        difficulty_text,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-mixed-negation-difficulty",
    )
    [preference] = extractor.extract(
        preference_text, {}, conversation_id="route-b-mixed-negation-preference"
    )

    assert difficulty.payload["negated"] is True
    assert difficulty.payload["polarity"] == "negative"
    assert preference.payload["negated"] is True
    assert preference.payload["polarity"] == "negative"


def test_lexical_bu_inside_profile_and_task_payload_is_not_negation(route_b_factory):
    profile_text = "我可能叫不凡"
    task_text = "接着做不等式"
    extractor = route_b_factory(
        {
            profile_text: SemanticFamily.PROFILE,
            task_text: SemanticFamily.TASK_STATE,
        }
    )

    [profile] = extractor.extract(
        profile_text, {}, conversation_id="route-b-lexical-profile"
    )
    [task] = extractor.extract(
        task_text,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-lexical-task",
    )

    assert profile.payload == {"preferred_name": "不凡", "uncertainty": "uncertain"}
    assert task.payload == {"subject": "数学", "requested_target": "不等式"}


@pytest.mark.parametrize(
    ("utterance", "family"),
    (
        ("我以前怕不等式，现在会了", SemanticFamily.TOPIC_PROGRESS),
        ("我总弄混不等式和等式", SemanticFamily.DIFFICULTY),
    ),
)
def test_lexical_bu_does_not_leak_into_qualifiers(route_b_factory, utterance, family):
    extractor = route_b_factory({utterance: family})
    [candidate] = extractor.extract(
        utterance,
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-lexical-negation",
    )
    assert "negated" not in candidate.payload
    assert "polarity" not in candidate.payload
