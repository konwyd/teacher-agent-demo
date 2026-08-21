# CandidateExtractorV2 Route B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the isolated Route B extractor in which BGE-M3 proposes semantic families and deterministic code preserves every qualifier and temporal boundary.

**Architecture:** A route-local Memory package separates normalized dense family ranking, deterministic qualifier analysis, candidate construction, and benchmark adaptation. The package consumes frozen upstream mappings and emits the existing `MemoryCandidate` contract without wiring into the authoritative runtime or changing evaluation semantics.

**Tech Stack:** Python 3.10+, Pydantic v2, `sentence-transformers==5.6.0`, local `BAAI/bge-m3`, NumPy, pytest, frozen CandidateExtractorV2 evaluator.

**Spec:** `docs/superpowers/specs/2026-08-21-candidate-extractor-v2-route-b-design.md`

## Global Constraints

- Work only in `.worktrees/candidate-extractor-v2-embedding-hybrid` on `spike/candidate-extractor-v2-embedding-hybrid`, starting from required SHA `e072586cfda38f88ee3d86dda615bee7da4c4cb9`.
- Do not modify `app/evaluation/candidate_extractor/**`, any development or sealed-holdout label, existing Layer B labels, AdmissionPolicyV1, matcher, ledger, baseline writer, Memory V2 authority, Agent Runtime, Agent-to-Memory contracts, or causal identity.
- Use only `development.jsonl` for iterative tuning; do not inspect sealed-holdout labels and run Route B against sealed holdout zero times.
- Embeddings may propose semantic families only. Deterministic logic owns negation, temporal scope, conditions, corrections, polarity, uncertainty, and contextual scope.
- The model is local `BAAI/bge-m3`, MIT license, 1024 dimensions, approximately 4.3 GB snapshot, loaded through `sentence-transformers==5.6.0` on CPU by default.
- Never download weights implicitly, call DeepSeek, use a remote extraction API, use an LLM judge, or read Route A/Route C implementations.
- Do not encode development case IDs, utterances, or expected outputs as one-off rules.
- Keep all generated reports and model/runtime artifacts under `/tmp`, not Git.
- Every production behavior follows RED → verify RED → GREEN → verify GREEN before refactoring.

## File Map

- `app/memory/candidate_extractor_v2_route_b/qualifiers.py`: typed deterministic qualifier analysis, segmentation, hard-negative gating, and temporal resolution.
- `app/memory/candidate_extractor_v2_route_b/embedding.py`: semantic-family enum, embedding protocol, local BGE-M3 backend, prototypes, thresholds, and ranking.
- `app/memory/candidate_extractor_v2_route_b/extractor.py`: bounded family parsers, frozen-upstream adapters, candidate construction, and deduplication.
- `app/memory/candidate_extractor_v2_route_b/benchmark.py`: frozen evaluator adapter and route-local operational/slice reporting.
- `app/memory/candidate_extractor_v2_route_b/__init__.py`: narrow public Route B exports.
- `tests/test_candidate_extractor_v2_route_b.py`: deterministic unit and adapter coverage using a fake embedder.
- `scripts/run_candidate_extractor_route_b.py`: development/Layer B-only local BGE-M3 runner.
- `docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md`: model card, procedure, metrics, limitations, and exact verification evidence.
- `pyproject.toml`: route-specific optional dependency group only.
- `PROGRESS.md`: only the Memory System section.

---

### Task 1: Deterministic Qualifier Parser

**Files:**
- Create: `app/memory/candidate_extractor_v2_route_b/qualifiers.py`
- Create: `app/memory/candidate_extractor_v2_route_b/__init__.py`
- Create: `tests/test_candidate_extractor_v2_route_b.py`

**Interfaces:**
- Consumes: `app.memory.contracts.TemporalScope`.
- Produces: `QualifierAnalysis`, `segment_clauses(text: str) -> tuple[str, ...]`, `analyze_clause(text: str) -> QualifierAnalysis`, and `resolve_temporal_scope(analysis: QualifierAnalysis, default: TemporalScope) -> TemporalScope`.

- [ ] **Step 1: Write focused failing qualifier tests**

```python
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


def test_hard_negative_guards_are_explicit():
    assert analyze_clause("书中的人物叫阿泽").hard_negative_reason == "third_person"
    assert analyze_clause("‘分步骤’这个词是什么含义").hard_negative_reason == "meta_quoted"
    assert analyze_clause("不要保存我的姓名").hard_negative_reason == "privacy_prohibition"
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py -q`

Expected: collection fails with `ModuleNotFoundError: app.memory.candidate_extractor_v2_route_b`.

- [ ] **Step 3: Implement the typed parser with explicit precedence**

```python
@dataclass(frozen=True)
class QualifierAnalysis:
    text: str
    hard_negative_reason: str = ""
    negated: bool | None = None
    polarity: str = ""
    condition: str = ""
    scope: str = ""
    temporal_expression: str = ""
    temporal_hint: TemporalScope | None = None
    uncertainty: str = ""
    explicit_correction: bool = False
    correction_of: str = ""


def resolve_temporal_scope(
    analysis: QualifierAnalysis, default: TemporalScope
) -> TemporalScope:
    if analysis.temporal_hint is not None:
        return analysis.temporal_hint
    if analysis.uncertainty:
        return TemporalScope.UNKNOWN
    return default
```

Implement Unicode normalization, bounded conjunction splitting, deterministic
hard-negative patterns, negation parity, condition/scope mapping, temporal
precedence, correction markers, and hedges in named constants. Keep the parser
independent of similarity values and semantic-family thresholds.

- [ ] **Step 4: Run qualifier tests and verify GREEN**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the parser checkpoint**

```bash
git add app/memory/candidate_extractor_v2_route_b tests/test_candidate_extractor_v2_route_b.py
git commit -m "feat: add deterministic route B qualifier parser"
```

### Task 2: Offline Embedding Family Ranker

**Files:**
- Create: `app/memory/candidate_extractor_v2_route_b/embedding.py`
- Modify: `app/memory/candidate_extractor_v2_route_b/__init__.py`
- Modify: `tests/test_candidate_extractor_v2_route_b.py`

**Interfaces:**
- Consumes: normalized clause strings from Task 1.
- Produces: `SemanticFamily`, `EmbeddingBackend.encode(texts: Sequence[str]) -> Sequence[Sequence[float]]`, `BgeM3Embedder`, `FamilyMatch`, `SemanticFamilyRanker.rank(texts: Sequence[str]) -> tuple[tuple[FamilyMatch, ...], ...]`, and `DEFAULT_MODEL_PATH`.

- [ ] **Step 1: Add failing ranker and local-only loader tests**

```python
class FakeEmbeddingBackend:
    def __init__(self, vectors: dict[str, tuple[float, ...]]):
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts):
        batch = tuple(texts)
        self.calls.append(batch)
        return [self.vectors[text] for text in batch]


def test_ranker_batches_clauses_and_returns_multiple_semantic_families():
    backend = FakeEmbeddingBackend({
        "学生希望老师采用较慢节奏": (1.0, 0.0),
        "学生陈述自己的姓名或称呼": (0.0, 1.0),
        "请放慢说明速度": (0.99, 0.01),
        "称呼我为阿泽": (0.01, 0.99),
    })
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
    with pytest.raises(RouteBEmbeddingError, match="local BGE-M3 model"):
        BgeM3Embedder(tmp_path / "missing")


def test_ranker_rejects_non_finite_or_ragged_vectors():
    backend = FakeEmbeddingBackend({"prototype": (1.0, 0.0), "query": (float("nan"), 0.0)})
    ranker = SemanticFamilyRanker(
        backend,
        prototypes={SemanticFamily.PREFERENCE: ("prototype",)},
        thresholds={family: 0.8 for family in SemanticFamily},
    )
    with pytest.raises(RouteBEmbeddingError, match="finite"):
        ranker.rank(("query",))
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py -q`

Expected: import errors for the missing embedding interfaces.

- [ ] **Step 3: Implement local BGE-M3 loading and cosine ranking**

```python
class SemanticFamily(str, Enum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    LEARNING_GOAL = "learning_goal"
    DIFFICULTY = "difficulty"
    TOPIC_PROGRESS = "topic_progress"
    TASK_STATE = "task_state"
    LEARNING_EPISODE = "learning_episode"
    BEHAVIOR_EVIDENCE = "behavior_evidence"


class EmbeddingBackend(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise NotImplementedError


@dataclass(frozen=True)
class FamilyMatch:
    family: SemanticFamily
    similarity: float


class BgeM3Embedder:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, device: str = "cpu"):
        path = Path(model_path)
        if not path.is_dir():
            raise RouteBEmbeddingError(f"local BGE-M3 model directory not found: {path}")
        self.model_path = path
        self.device = device
        self._model = None

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(str(self.model_path), device=self.device)
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()
```

Implement prototype caching, dimension/finite validation, normalized dot-product
ranking, stable tie ordering, configurable per-family thresholds, and encode
call/text counters. Keep prototype text generic and independent of benchmark
utterances.

- [ ] **Step 4: Run the route tests and verify GREEN**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py -q`

Expected: all Task 1–2 tests pass without loading BGE-M3.

- [ ] **Step 5: Commit the embedding checkpoint**

```bash
git add app/memory/candidate_extractor_v2_route_b tests/test_candidate_extractor_v2_route_b.py
git commit -m "feat: add offline route B semantic ranker"
```

### Task 3: Utterance Candidate Extraction

**Files:**
- Create: `app/memory/candidate_extractor_v2_route_b/extractor.py`
- Modify: `app/memory/candidate_extractor_v2_route_b/__init__.py`
- Modify: `tests/test_candidate_extractor_v2_route_b.py`

**Interfaces:**
- Consumes: `SemanticFamilyRanker.rank`, `analyze_clause`, `resolve_temporal_scope`, and canonical Memory v2 contract enums/models.
- Produces: `RouteBConfig`, `CandidateExtractorRouteB.extract(utterance: str, upstream: Mapping[str, Any], *, conversation_id: str, turn_id: int = 1) -> list[MemoryCandidate]`.

- [ ] **Step 1: Add failing tests proving embeddings cannot override qualifiers**

```python
class StubFamilyRanker:
    def __init__(self, families_by_text: dict[str, SemanticFamily]):
        self.families_by_text = families_by_text

    def rank(self, texts):
        return tuple(
            (FamilyMatch(family=self.families_by_text[text], similarity=1.0),)
            if text in self.families_by_text else ()
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
        "我不想看示范", {"classification": {"subject": "数学"}},
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
        "这一会儿请慢一些", {"classification": {"subject": "数学"}},
        conversation_id="route-b-test",
    )
    assert candidate.payload["temporal_expression"] == "这一会儿"
    assert candidate.temporal_scope is TemporalScope.SHORT_LIVED


def test_profile_multi_field_and_difficulty_payloads_are_bounded(route_b_factory):
    extractor = route_b_factory({
        "我读六年级，请称呼我Leo": SemanticFamily.PROFILE,
        "我经常把正负号搞反": SemanticFamily.DIFFICULTY,
    })
    [profile] = extractor.extract(
        "我读六年级，请称呼我Leo", {"classification": {"subject": "数学"}},
        conversation_id="route-b-profile",
    )
    [difficulty] = extractor.extract(
        "我经常把正负号搞反", {"classification": {"subject": "数学"}},
        conversation_id="route-b-difficulty",
    )
    assert profile.payload == {"grade": "六年级", "preferred_name": "leo"}
    assert difficulty.payload == {
        "subject": "数学", "topic": "正负号", "statement": "搞反正负号"
    }
```

- [ ] **Step 2: Run extractor tests and verify RED**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py -q`

Expected: import failure for `CandidateExtractorRouteB` or missing extraction behavior.

- [ ] **Step 3: Implement bounded family parsers and candidate factory**

```python
@dataclass(frozen=True)
class RouteBConfig:
    family_thresholds: Mapping[SemanticFamily, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )


class CandidateExtractorRouteB:
    def __init__(self, ranker: SemanticFamilyRanker, config: RouteBConfig | None = None):
        self.ranker = ranker
        self.config = config or RouteBConfig()

    def extract(
        self,
        utterance: str,
        upstream: Mapping[str, Any],
        *,
        conversation_id: str,
        turn_id: int = 1,
    ) -> list[MemoryCandidate]:
        clauses = segment_clauses(utterance)
        analyses = tuple(analyze_clause(clause) for clause in clauses)
        eligible = tuple(item.text for item in analyses if not item.hard_negative_reason)
        rankings = self.ranker.rank(eligible) if eligible else ()
        return self._build_and_merge(analyses, rankings, upstream, conversation_id, turn_id)
def _make_candidate(
    *,
    conversation_id: str,
    turn_id: int,
    memory_type: MemoryType,
    payload: dict[str, Any],
    source_kind: MemorySourceKind,
    confidence: float,
    temporal_scope: TemporalScope,
    verifier_reference: str = "",
) -> MemoryCandidate:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(
        f"{conversation_id}\\n{turn_id}\\n{memory_type.value}\\n{canonical}".encode("utf-8")
    ).hexdigest()[:24]
    provenance = ProvenanceReference(
        conversation_id=conversation_id,
        turn_id=turn_id,
        source_kind=source_kind,
        event_reference=f"turn:{turn_id}",
        verifier_reference=verifier_reference,
    )
    return MemoryCandidate(
        candidate_id=f"route_b_{digest}",
        target_memory_class=MemoryTargetClass.STUDENT,
        memory_type=memory_type,
        payload=payload,
        source_kind=source_kind,
        provenance=[provenance],
        confidence=confidence,
        temporal_scope=temporal_scope,
    )

```

Implement deterministic IDs, provenance, fixed source-appropriate confidence,
mandatory-field guards, canonical preference values, profile parsing, goal,
difficulty, progress, task-state, episode, and observed-pattern parsing. Attach
only non-empty qualifier fields and resolve temporal scope after parsing.

- [ ] **Step 4: Run route tests and existing contract tests**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py tests/test_memory_v2_contracts.py -q`

Expected: all tests pass and every emitted candidate validates against the
unchanged `MemoryCandidate` model.

- [ ] **Step 5: Commit utterance extraction**

```bash
git add app/memory/candidate_extractor_v2_route_b tests/test_candidate_extractor_v2_route_b.py
git commit -m "feat: extract route B utterance candidates"
```

### Task 4: Multi-Candidate and Frozen-Upstream Evidence

**Files:**
- Modify: `app/memory/candidate_extractor_v2_route_b/extractor.py`
- Modify: `tests/test_candidate_extractor_v2_route_b.py`

**Interfaces:**
- Consumes: `CandidateExtractorRouteB.extract` and frozen upstream mapping keys `classification`, `teaching_contract`, `verification`, `observations`, and `model_inferences`.
- Produces: order-stable multi-candidate output and source-kind-safe upstream candidates without changing any upstream schema.

- [ ] **Step 1: Add failing multi-source tests**

```python
def test_multiple_clauses_emit_multiple_candidates_without_qualifier_leak(route_b_factory):
    extractor = route_b_factory({
        "称呼我阿泽": SemanticFamily.PROFILE,
        "今天给我一些例子": SemanticFamily.PREFERENCE,
    })
    actual = extractor.extract(
        "称呼我阿泽，并且今天给我一些例子",
        {"classification": {"subject": "数学"}},
        conversation_id="route-b-multi",
    )
    assert [item.memory_type for item in actual] == [MemoryType.PROFILE_FACT, MemoryType.PREFERENCE]
    assert "temporal_expression" not in actual[0].payload
    assert actual[1].payload["temporal_expression"] == "今天"
    assert actual[1].temporal_scope is TemporalScope.SESSION


def test_verification_and_model_inference_keep_declared_source(route_b_factory):
    extractor = route_b_factory({})
    actual = extractor.extract(
        "",
        {
            "classification": {"subject": "英语"},
            "verification": {"status": "verified_correct", "summary": "checked"},
            "model_inferences": ({"memory_type": "difficulty", "topic": "past tense"},),
        },
        conversation_id="route-b-upstream",
    )
    assert {item.source_kind for item in actual} == {
        MemorySourceKind.VERIFIED_RESULT,
        MemorySourceKind.MODEL_INFERENCE,
    }
    inferred = next(item for item in actual if item.source_kind is MemorySourceKind.MODEL_INFERENCE)
    assert inferred.temporal_scope is TemporalScope.UNKNOWN


def test_hard_negatives_emit_no_persistent_candidate(route_b_factory):
    extractor = route_b_factory({
        "课本里的学生叫阿泽": SemanticFamily.PROFILE,
        "请解释‘慢慢说明’是什么意思": SemanticFamily.PREFERENCE,
        "别保存我的姓名": SemanticFamily.PROFILE,
    })
    for text in (
        "课本里的学生叫阿泽",
        "请解释‘慢慢说明’是什么意思",
        "别保存我的姓名",
    ):
        assert extractor.extract(text, {}, conversation_id="route-b-hard") == []
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py -q`

Expected: failures show missing multi-clause merge or upstream candidate behavior.

- [ ] **Step 3: Implement source-safe upstream adapters and stable merging**

Add private functions with exact return boundaries:

```python
def _verification_candidates(
    upstream: Mapping[str, Any], *, conversation_id: str, turn_id: int
) -> list[MemoryCandidate]:
    verification = dict(upstream.get("verification") or {})
    status = str(verification.get("status") or "").strip()
    if status not in {"verified_correct", "verified_incorrect"}:
        return []
    subject = str(dict(upstream.get("classification") or {}).get("subject") or "")[:20]
    return [_make_candidate(
        conversation_id=conversation_id,
        turn_id=turn_id,
        memory_type=MemoryType.BEHAVIOR_EVIDENCE,
        payload={
            "subject": subject,
            "verification_status": status,
            "summary": str(verification.get("summary") or "")[:180],
        },
        source_kind=MemorySourceKind.VERIFIED_RESULT,
        confidence=0.95,
        temporal_scope=TemporalScope.SHORT_LIVED,
        verifier_reference=str(verification.get("verifier_name") or "")[:160],
    )]


def _observation_candidates(
    upstream: Mapping[str, Any], *, conversation_id: str, turn_id: int
) -> list[MemoryCandidate]:
    subject = str(dict(upstream.get("classification") or {}).get("subject") or "")[:20]
    candidates = []
    for observation in tuple(upstream.get("observations") or ()):
        if not isinstance(observation, Mapping) or not observation.get("evidence_pattern"):
            continue
        candidates.append(_make_candidate(
            conversation_id=conversation_id,
            turn_id=turn_id,
            memory_type=MemoryType.BEHAVIOR_EVIDENCE,
            payload={
                "subject": subject,
                "evidence_pattern": str(observation["evidence_pattern"])[:180],
            },
            source_kind=MemorySourceKind.OBSERVED_BEHAVIOR,
            confidence=0.75,
            temporal_scope=TemporalScope.SHORT_LIVED,
        ))
    return candidates


def _model_inference_candidates(
    upstream: Mapping[str, Any], *, conversation_id: str, turn_id: int
) -> list[MemoryCandidate]:
    subject = str(dict(upstream.get("classification") or {}).get("subject") or "")[:20]
    candidates = []
    for inference in tuple(upstream.get("model_inferences") or ()):
        if not isinstance(inference, Mapping) or inference.get("memory_type") != "difficulty":
            continue
        candidates.append(_make_candidate(
            conversation_id=conversation_id,
            turn_id=turn_id,
            memory_type=MemoryType.DIFFICULTY,
            payload={
                "subject": subject,
                "topic": str(inference.get("topic") or "")[:160],
                "uncertainty": "model_inference",
            },
            source_kind=MemorySourceKind.MODEL_INFERENCE,
            confidence=0.4,
            temporal_scope=TemporalScope.UNKNOWN,
        ))
    return candidates


def _merge_candidates(candidates: Iterable[MemoryCandidate]) -> list[MemoryCandidate]:
    unique: dict[tuple[str, str, str], MemoryCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.memory_type.value,
            candidate.source_kind.value,
            json.dumps(candidate.payload, ensure_ascii=False, sort_keys=True),
        )
        unique.setdefault(key, candidate)
    return list(unique.values())
```

Skip malformed optional upstream items, count them on the extractor's bounded
diagnostics object, and never relabel `model_inference` or `observed_behavior` as
`user_stated`/`verified_result`. Deduplicate only exact semantic payloads while
preserving first-seen order.

- [ ] **Step 4: Run Route B and Memory contract tests**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py tests/test_memory_v2_contracts.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit multi-candidate/upstream behavior**

```bash
git add app/memory/candidate_extractor_v2_route_b/extractor.py tests/test_candidate_extractor_v2_route_b.py
git commit -m "feat: support route B contextual candidates"
```

### Task 5: Development and Layer B Benchmark Runner

**Files:**
- Create: `app/memory/candidate_extractor_v2_route_b/benchmark.py`
- Create: `scripts/run_candidate_extractor_route_b.py`
- Modify: `app/memory/candidate_extractor_v2_route_b/__init__.py`
- Modify: `tests/test_candidate_extractor_v2_route_b.py`

**Interfaces:**
- Consumes: frozen `load_cases`, `load_layer_b`, and `evaluate_cases` read-only; `CandidateExtractorRouteB.extract`.
- Produces: `run_route_b_benchmark(extractor, splits=("development", "layer_b"), *, provisional=True) -> dict[str, Any]` and a CLI that cannot select sealed holdout.

- [ ] **Step 1: Add failing adapter, split-safety, and metrics tests**

```python
def synthetic_profile_case(utterance: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="synthetic-profile",
        split="development",
        utterance=utterance,
        labels=("PROFILE",),
        rationale="route-local adapter test",
        expected_candidates=(ExpectedCandidate(
            memory_type="profile_fact",
            payload={"preferred_name": "阿泽"},
            source_kind="user_stated",
            temporal_scope="persistent",
        ),),
    )



def test_route_runner_exposes_only_development_and_layer_b():
    parser = build_parser()
    split_action = next(action for action in parser._actions if action.dest == "splits")
    assert set(split_action.choices) == {"development", "layer_b"}


def test_benchmark_adapter_preserves_frozen_evaluator_and_slice_metrics(route_b_factory):
    extractor = route_b_factory({"称呼我阿泽": SemanticFamily.PROFILE})
    report = run_route_b_benchmark(
        extractor,
        splits=("development",),
        cases_by_split={"development": [synthetic_profile_case("称呼我阿泽")]},
        provisional=True,
    )
    split = report["reports"]["development"]
    assert split["exact_candidate_set_accuracy"]["value"] == 1.0
    assert report["operational"]["remote_llm_calls"] == 0
    assert report["operational"]["provisional"] is True
    assert report["slice_metrics"]["development"]["multi_candidate_exact_match"]["support"] == 0
```

- [ ] **Step 2: Run runner tests and verify RED**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py -q`

Expected: import failures for `benchmark.py` and the route runner parser.

- [ ] **Step 3: Implement the read-only benchmark adapter and CLI**

```python
ALLOWED_SPLITS = ("development", "layer_b")


def run_route_b_benchmark(
    extractor: CandidateExtractorRouteB,
    splits: tuple[str, ...] = ALLOWED_SPLITS,
    *,
    cases_by_split: Mapping[str, Sequence[BenchmarkCase]] | None = None,
    provisional: bool = True,
) -> dict[str, Any]:
    if not splits or any(split not in ALLOWED_SPLITS for split in splits):
        raise ValueError("Route B runner supports development and layer_b only")
    # Load frozen cases, call evaluate_cases unchanged, then derive label slices
    # from case_results without changing frozen metric values.
```

Measure cold load, wall throughput, `resource.getrusage().ru_maxrss`, CUDA peak
memory only when an explicitly selected CUDA device is used, embed encode calls,
encoded text count, dependency versions, deterministic replay, and implementation
line count. Write JSON only to an explicit path whose default is under `/tmp`.

- [ ] **Step 4: Run runner and manifest regression tests**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py tests/test_candidate_extractor_benchmark.py -q`

Expected: all tests pass; `verify_manifest()` remains empty.

- [ ] **Step 5: Commit the benchmark adapter**

```bash
git add app/memory/candidate_extractor_v2_route_b scripts/run_candidate_extractor_route_b.py tests/test_candidate_extractor_v2_route_b.py
git commit -m "feat: add route B benchmark runner"
```

### Task 6: Dependency, Route Documentation, and Progress

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md`
- Modify: `PROGRESS.md:41-143`
- Modify: `tests/test_candidate_extractor_v2_route_b.py`

**Interfaces:**
- Consumes: model/config/operational constants from Tasks 2 and 5.
- Produces: optional dependency group `candidate-extractor-route-b`, reproducible local invocation documentation, and an updated Memory System status only.

- [ ] **Step 1: Add a failing dependency and model-metadata test**

```python
def test_route_b_dependency_and_model_metadata_are_explicit():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["optional-dependencies"]["candidate-extractor-route-b"] == [
        "sentence-transformers==5.6.0"
    ]
    assert MODEL_ID == "BAAI/bge-m3"
    assert MODEL_LICENSE == "MIT"
    assert EMBEDDING_DIMENSION == 1024
```

- [ ] **Step 2: Run the metadata test and verify RED**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py::test_route_b_dependency_and_model_metadata_are_explicit -q`

Expected: failure because the optional dependency group is absent.

- [ ] **Step 3: Add the optional dependency and route report structure**

Add exactly this entry inside the existing `[project.optional-dependencies]` table:

```toml
candidate-extractor-route-b = ["sentence-transformers==5.6.0"]
```

Preserve the existing `ui`, `dev`, and `chem-structure` groups. Document model
identity/version/license, 4.3 GB snapshot, 2.27 GB PyTorch weights, CPU default,
dependencies, offline-only path, exact development/Layer B commands, frozen
metrics table, qualifier safety table, contamination table, provisional resource
table, repeatability, implementation complexity, limitations, and zero Route B
sealed-holdout runs. Update only `PROGRESS.md` lines in the Memory System section
with phase, completed work, current findings, blockers, next action, branch, and
date.

- [ ] **Step 4: Run metadata tests and inspect scoped documentation diff**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py -q`

Run: `git diff -- pyproject.toml docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md PROGRESS.md`

Expected: tests pass; `PROGRESS.md` changes occur only between the Memory System
and Agent Runtime headings.

- [ ] **Step 5: Commit dependencies and documentation**

```bash
git add pyproject.toml docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md PROGRESS.md tests/test_candidate_extractor_v2_route_b.py
git commit -m "docs: record route B model and procedure"
```

### Task 7: Development-Only BGE-M3 Tuning

**Files:**
- Modify: `app/memory/candidate_extractor_v2_route_b/embedding.py`
- Modify: `app/memory/candidate_extractor_v2_route_b/qualifiers.py`
- Modify: `app/memory/candidate_extractor_v2_route_b/extractor.py`
- Modify: `tests/test_candidate_extractor_v2_route_b.py`
- Modify: `docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md`

**Interfaces:**
- Consumes: the real local BGE-M3 backend and development/Layer B-only runner.
- Produces: development-tuned generic prototypes/thresholds and generalized parser rules with recorded frozen metrics.

- [ ] **Step 1: Run the first real development and Layer B benchmark**

Run:

```bash
python scripts/run_candidate_extractor_route_b.py \
  --model-path /home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3 \
  --device cpu \
  --splits development layer_b \
  --output /tmp/candidate_extractor_route_b_initial.json
```

Expected: the runner loads only development and Layer B, produces all frozen
metrics, labels performance provisional, and performs zero remote/LLM calls.

- [ ] **Step 2: Turn each justified development failure into a generalized RED test**

For every prototype/threshold/parser adjustment, first add a new synthetic
paraphrase that is not copied from `development.jsonl`. Use the narrow form:

```python
@pytest.mark.parametrize(
    ("utterance", "family"),
    [
        ("说明时多给一个具体示范", SemanticFamily.PREFERENCE),
        ("我准备在月考前掌握百分数", SemanticFamily.LEARNING_GOAL),
    ],
)
def test_local_bge_generic_paraphrases_reach_the_intended_family(utterance, family):
    if not DEFAULT_MODEL_PATH.is_dir():
        pytest.skip("route B local model is not installed")
    ranker = SemanticFamilyRanker(BgeM3Embedder(DEFAULT_MODEL_PATH))
    matches = ranker.rank((utterance,))
    assert matches[0]
    assert matches[0][0].family is family
```

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py -q`

Expected: each newly added test fails for the intended missing generalization,
not because of an import, typo, or fixture error.

- [ ] **Step 3: Make the minimal generic catalog/threshold/parser change**

Change only named generic prototypes, family thresholds, or reusable qualifier
patterns. Do not add case IDs, exact development utterances, expected payload
tables, or branches that identify benchmark cases.

- [ ] **Step 4: Verify unit GREEN and rerun development/Layer B**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py -q`

Run:

```bash
python scripts/run_candidate_extractor_route_b.py \
  --model-path /home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3 \
  --device cpu \
  --splits development layer_b \
  --output /tmp/candidate_extractor_route_b_final_dev.json
```

Expected: unit tests pass; report includes development and Layer B only. Copy
the final metrics and provisional resource values into the route documentation.

- [ ] **Step 5: Audit for benchmark memorization and commit tuning**

Run: `rg -n "dev-[0-9]+|holdout|sealed" app/memory/candidate_extractor_v2_route_b tests/test_candidate_extractor_v2_route_b.py`

Expected: no case-ID branches and no sealed-holdout access; documentation-only
mentions of the prohibition are outside searched implementation paths.

```bash
git add app/memory/candidate_extractor_v2_route_b tests/test_candidate_extractor_v2_route_b.py docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md
git commit -m "perf: tune route B on development split"
```

### Task 8: Full Verification, Protected-Scope Audit, and Final Route Commit

**Files:**
- Modify: `docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md` only if fresh results differ from the recorded evidence.

**Interfaces:**
- Consumes: all Route B implementation and repository verification harnesses.
- Produces: fresh completion evidence, clean assigned worktree, and final handoff commit.

- [ ] **Step 1: Run targeted Route B and frozen benchmark tests**

Run: `python -m pytest tests/test_candidate_extractor_v2_route_b.py tests/test_candidate_extractor_benchmark.py -q`

Expected: all tests pass and frozen manifest verification remains green.

- [ ] **Step 2: Run the final development and existing Layer B benchmark**

Run:

```bash
python scripts/run_candidate_extractor_route_b.py \
  --model-path /home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3 \
  --device cpu \
  --splits development layer_b \
  --output /tmp/candidate_extractor_route_b_verified.json
```

Expected: both splits complete, all required metrics are populated where the
frozen evaluator has support, performance is labeled provisional, and sealed
holdout is absent.

- [ ] **Step 3: Run full repository pytest and offline evaluation**

Run: `python -m pytest`

Expected: exit 0 with no failures.

Run: `python scripts/run_eval.py`

Expected: exit 0 with `38/38` offline mock cases passing. The ignored `logs/`
artifact is not staged.

- [ ] **Step 4: Run Memory admission robustness and manifest verification**

Run:

```bash
python scripts/run_memory_admission_robustness.py \
  --output /tmp/route_b_memory_admission_robustness.json \
  --markdown-output /tmp/route_b_memory_admission_robustness.md \
  --existing-full-test-results PASS \
  --existing-evaluation-results 38/38
```

Expected: exit 0; existing Layer B and PolicyV1 robustness semantics remain
unchanged.

Run: `python scripts/run_candidate_extractor_benchmark.py --verify-manifest --output /tmp/route_b_current_baseline.json`

Expected: exit 0 with no `manifest drift`. This invokes only the unchanged
baseline adapter; it is not a Route B sealed-holdout run.

- [ ] **Step 5: Audit protected files, source scope, and secret/runtime hygiene**

Run: `git diff e072586cfda38f88ee3d86dda615bee7da4c4cb9 --name-only`

Expected changed paths: the Route B package, Route B test/runner/docs/plans/spec,
`pyproject.toml`, and the Memory section of `PROGRESS.md` only.

Run: `git diff e072586cfda38f88ee3d86dda615bee7da4c4cb9 -- app/evaluation/candidate_extractor app/evaluation/memory_admission app/memory/admission_policy.py app/memory/admission_matcher.py app/memory/admission_ledger.py app/memory/contracts.py`

Expected: empty.

Run: `git diff --check`

Expected: exit 0.

Run: `git status --short`

Expected: no secrets, model weights, `/tmp` reports, `logs/`, or unrelated files.

- [ ] **Step 6: Update fresh evidence and create the final scoped commit if needed**

If verification numbers differ, update only
`docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md`, then run its targeted tests again.

```bash
git add docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md
git commit -m "docs: finalize route B verification evidence"
```

If no documentation changed, do not create an empty commit. Confirm final HEAD,
branch, upstream, intended push target `origin/spike/candidate-extractor-v2-embedding-hybrid`,
and clean worktree before the handoff. Push only the assigned branch; never merge,
rebase, cherry-pick, or update the remote default branch.
