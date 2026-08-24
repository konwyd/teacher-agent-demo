import unicodedata

import pytest

from app.memory.candidate_extractor_v2_route_b import (
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

    def rank(self, texts):
        self.calls.append(tuple(texts))
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
