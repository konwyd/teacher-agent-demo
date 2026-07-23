# Physics Paper RAG Foundation Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independently hybrid, measurable local text retriever whose Recall@5 is at least 80% on 30–50 manually labeled physics questions before re-running the paper-score ablation.

**Architecture:** Keep `llm-test/rag_retriever.py` as the public compatibility layer, but move query preparation, corpus caching, scoring, and context allocation into focused helpers in `llm-test/rag_foundation.py`. Dense retrieval and a cached full-corpus BM25 retrieval run independently; their union is rescored with BGE-M3 cosine similarity, while RRF remains ranking evidence only. A separate offline benchmark module evaluates stable document IDs without invoking an answer-generating LLM.

**Tech Stack:** Python 3.10+, ChromaDB HTTP client, BGE-M3 embedding function, `rank_bm25`, `jieba`, NumPy, pytest.

## Global Constraints

- First-phase scope is local physics textbook and exercise text only; do not change web search, Oracle, image understanding, or the production Teacher Agent workflow.
- Answers may be used by a human to identify relevance, but must never enter a retrieval query or model context.
- Recall@5 on 30–50 manually reviewed physics questions must be at least 80% before the paper-score ablation is treated as valid.
- RRF is a rank signal only; injection thresholds must use BGE-M3 `relevance_score`.
- Preserve the return contract of `retrieve_for_exam_questionwise()`.
- Do not add a cross-encoder model in this phase.
- Run `python -m pytest` and `python scripts/run_eval.py` before completion.

---

## File Map

- Create `llm-test/rag_foundation.py`: pure query preparation, document identity, quality filtering, cached corpus loading, independent BM25 recall, union rescoring, and fair context allocation.
- Modify `llm-test/rag_retriever.py`: delegate hybrid retrieval and question-wise allocation to the new helpers while retaining current public classes and function signatures.
- Create `tests/test_paper_rag_foundation.py`: offline unit tests with fake Chroma and fake embedding functions.
- Create `llm-test/rag_benchmark.py`: offline/Chroma-backed Recall@K and MRR runner.
- Create `llm-test/benchmarks/physics_retrieval.jsonl`: 30–50 reviewed physics question-to-document labels.
- Create `tests/test_rag_benchmark.py`: benchmark metric and schema tests.

### Task 1: Query Preparation and Conservative Quality Filtering

**Files:**
- Create: `llm-test/rag_foundation.py`
- Create: `tests/test_paper_rag_foundation.py`

**Interfaces:**
- Produces: `prepare_queries(question: str, subject: str) -> tuple[str, str]`
- Produces: `stable_doc_id(doc_id: str | None, text: str, meta: dict[str, Any]) -> str`
- Produces: `quality_rejection_reason(text: str) -> str | None`

- [ ] **Step 1: Write failing tests for formula preservation, stable IDs, and OCR filtering**

```python
# tests/test_paper_rag_foundation.py
import importlib
import sys
from pathlib import Path

LLM_TEST_DIR = Path(__file__).resolve().parents[1] / "llm-test"
if str(LLM_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_TEST_DIR))


def foundation():
    return importlib.import_module("rag_foundation")


def test_prepare_queries_preserves_formula_number_and_unit():
    dense, sparse = foundation().prepare_queries(
        "12. 质量为 $2kg$ 的物体，速度由 $3m/s$ 增至 $5m/s$，求动能变化。",
        "物理",
    )
    assert "2kg" in dense and "3m/s" in dense and "5m/s" in dense
    assert "动能" in sparse and "2" in sparse and "kg" in sparse


def test_stable_doc_id_prefers_chroma_id_and_has_content_fallback():
    f = foundation()
    assert f.stable_doc_id("doc-7", "正文", {}) == "doc-7"
    first = f.stable_doc_id(None, "同一正文", {"source_file": "教材.md", "chunk_index": 3})
    second = f.stable_doc_id(None, "同一正文", {"source_file": "教材.md", "chunk_index": 3})
    assert first == second
    assert first.startswith("derived:")


def test_quality_filter_rejects_garbage_but_keeps_formula_dense_physics():
    f = foundation()
    assert f.quality_rejection_reason("Ү੣ҰΞЖ" * 20) == "ocr_garbage"
    assert f.quality_rejection_reason("U=IR，P=UI=I²R；当 I=2A、R=5Ω 时计算功率。") is None
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run: `python -m pytest tests/test_paper_rag_foundation.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rag_foundation'`.

- [ ] **Step 3: Implement the pure helpers**

```python
# llm-test/rag_foundation.py
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def prepare_queries(question: str, subject: str) -> tuple[str, str]:
    text = re.sub(r"!\[[^]]*]\([^)]*\)", " 图片 ", question)
    text = text.replace("$$", " ").replace("$", " ")
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = re.sub(r"[#*_`]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    dense = f"{subject.strip()} {text}".strip()
    tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+(?:\.\d+)?|[Ω℃°/%²³]+", text)
    return dense[:600], " ".join(tokens)[:600]


def stable_doc_id(doc_id: str | None, text: str, meta: dict[str, Any]) -> str:
    if doc_id:
        return str(doc_id)
    identity = {
        "source_file": meta.get("source_file", ""),
        "chunk_index": meta.get("chunk_index", ""),
        "text": text,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"derived:{digest}"


def quality_rejection_reason(text: str) -> str | None:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 12:
        return "too_short"
    readable = re.findall(r"[\u4e00-\u9fffA-Za-z0-9，。；：、,.!?=+\-*/%Ω℃°²³()（）]", compact)
    if len(readable) / len(compact) < 0.55:
        return "ocr_garbage"
    return None
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_paper_rag_foundation.py -v`

Expected: 3 tests PASS.

- [ ] **Step 5: Commit the helpers and tests**

```bash
git add llm-test/rag_foundation.py tests/test_paper_rag_foundation.py
git commit -m "test: define paper RAG query and quality contracts"
```

### Task 2: Paginated Corpus Cache and Truly Independent BM25 Recall

**Files:**
- Modify: `llm-test/rag_foundation.py`
- Modify: `tests/test_paper_rag_foundation.py`

**Interfaces:**
- Consumes: `stable_doc_id(...)`, `quality_rejection_reason(...)`
- Produces: `CorpusSnapshot` dataclass
- Produces: `CorpusCache.get(coll: Any, where: dict[str, Any] | None) -> CorpusSnapshot`
- Produces: `sparse_retrieve(query: str, snapshot: CorpusSnapshot, topk: int) -> list[dict[str, Any]]`

- [ ] **Step 1: Add a fake paginated collection and failing independence/cache tests**

```python
class FakeCollection:
    name = "physics"

    def __init__(self, rows):
        self.rows = rows
        self.get_calls = []

    def count(self):
        return len(self.rows)

    def get(self, *, limit, offset, include, where=None):
        self.get_calls.append((limit, offset, where))
        page = self.rows[offset:offset + limit]
        return {
            "ids": [row[0] for row in page],
            "documents": [row[1] for row in page],
            "metadatas": [row[2] for row in page],
        }


def test_sparse_recall_reads_full_filtered_corpus_and_caches_it():
    f = foundation()
    rows = [
        ("dense-only", "机械运动和速度", {"subject": "物理"}),
        ("lexical-hit", "托里拆利实验测量大气压", {"subject": "物理"}),
    ]
    coll = FakeCollection(rows)
    cache = f.CorpusCache(page_size=1)
    snapshot = cache.get(coll, {"subject": "物理"})
    hits = f.sparse_retrieve("托里拆利 大气压", snapshot, topk=1)
    assert hits[0]["_doc_id"] == "lexical-hit"
    assert len(coll.get_calls) == 2
    cache.get(coll, {"subject": "物理"})
    assert len(coll.get_calls) == 2


def test_corpus_cache_invalidates_when_collection_count_changes():
    f = foundation()
    coll = FakeCollection([("a", "速度的变化", {"subject": "物理"})])
    cache = f.CorpusCache(page_size=10)
    cache.get(coll, None)
    coll.rows.append(("b", "密度的测量方法", {"subject": "物理"}))
    snapshot = cache.get(coll, None)
    assert len(snapshot.documents) == 2
    assert len(coll.get_calls) == 2
```

- [ ] **Step 2: Run the new tests and verify missing cache types fail**

Run: `python -m pytest tests/test_paper_rag_foundation.py -v`

Expected: FAIL because `CorpusCache` is not defined.

- [ ] **Step 3: Implement paginated loading, cache invalidation, and BM25**

Add immutable `CorpusSnapshot(documents, bm25)` data, canonicalize `where` with sorted JSON, page using `coll.get(limit=..., offset=..., include=["documents", "metadatas"], where=where)`, filter only rejected text, and cache by `(collection name, canonical where, coll.count())`. Each document dict must contain `text`, `meta`, `_doc_id`, and `quality_flags`. Build `BM25Okapi` over the sparse tokens produced by `prepare_queries(text, "")[1]`, and return copied results with `_rank` and `bm25_score`.

- [ ] **Step 4: Run the cache and sparse tests**

Run: `python -m pytest tests/test_paper_rag_foundation.py -v`

Expected: 5 tests PASS.

- [ ] **Step 5: Commit independent sparse recall**

```bash
git add llm-test/rag_foundation.py tests/test_paper_rag_foundation.py
git commit -m "feat: add independent cached BM25 recall"
```

### Task 3: Candidate Union, BGE-M3 Rescoring, and Rank-Only RRF

**Files:**
- Modify: `llm-test/rag_foundation.py`
- Modify: `llm-test/rag_retriever.py`
- Modify: `tests/test_paper_rag_foundation.py`

**Interfaces:**
- Consumes: `CorpusCache.get(...)`, `sparse_retrieve(...)`, `prepare_queries(...)`
- Produces: `rescore_union(dense_results, sparse_results, dense_query, emb_fn) -> list[dict[str, Any]]`
- Modifies: `hybrid_retrieve(..., corpus_cache: CorpusCache | None = None)` while preserving existing callers
- Result fields: `_doc_id`, `text`, `meta`, `vec_sim`, `bm25_score`, `rank_score`, `relevance_score`, `sim`, `distance`

- [ ] **Step 1: Add failing tests proving sparse-only candidates survive and RRF cannot pass the gate**

```python
import numpy as np


class FakeEmbedding:
    vectors = {
        "物理 大气压强": [1.0, 0.0],
        "无关章节": [0.0, 1.0],
        "托里拆利实验说明大气压强": [0.98, 0.02],
    }

    def __call__(self, texts):
        return [self.vectors[text] for text in texts]


def test_rescore_union_keeps_sparse_only_hit_and_uses_semantic_gate():
    f = foundation()
    dense = [{"_doc_id": "wrong", "text": "无关章节", "meta": {}, "_rank": 0, "sim": 0.1}]
    sparse = [{
        "_doc_id": "right", "text": "托里拆利实验说明大气压强", "meta": {},
        "_rank": 0, "bm25_score": 8.0,
    }]
    hits = f.rescore_union(dense, sparse, "物理 大气压强", FakeEmbedding())
    assert hits[0]["_doc_id"] == "right"
    assert hits[0]["relevance_score"] > 0.9
    wrong = next(hit for hit in hits if hit["_doc_id"] == "wrong")
    assert wrong["rank_score"] > 0
    assert wrong["relevance_score"] < 0.5
```

- [ ] **Step 2: Run the focused test and confirm rescoring is absent**

Run: `python -m pytest tests/test_paper_rag_foundation.py::test_rescore_union_keeps_sparse_only_hit_and_uses_semantic_gate -v`

Expected: FAIL because `rescore_union` is not defined.

- [ ] **Step 3: Implement normalized batch cosine rescoring**

Implement `rescore_union` by deduplicating on `_doc_id`, calculating weighted RRF with `k=60`, calling `emb_fn([dense_query] + candidate_texts)` once, L2-normalizing the NumPy matrix, and assigning query-to-document cosine similarity as `relevance_score`. Sort by `(relevance_score, rank_score)` descending. Set compatibility fields `sim=relevance_score` and `distance=1.0-relevance_score`; never mix `rank_score` into either field.

- [ ] **Step 4: Replace `hybrid_retrieve` internals without changing its external signature**

Use one Chroma `query(..., n_results=recall_topk)` for dense recall and `CorpusCache.get(...)` plus `sparse_retrieve(..., topk=recall_topk)` for independent sparse recall. Preserve Chroma IDs by requesting/reading the returned `ids`. If corpus loading or BM25 fails, log the exception and rescore dense candidates only. If Chroma query fails, continue with sparse candidates; return an empty list only if both routes fail.

- [ ] **Step 5: Run all paper-RAG foundation tests**

Run: `python -m pytest tests/test_paper_rag_foundation.py -v`

Expected: all tests PASS, including a sparse-only top result with a high `relevance_score`.

- [ ] **Step 6: Commit hybrid retrieval repair**

```bash
git add llm-test/rag_foundation.py llm-test/rag_retriever.py tests/test_paper_rag_foundation.py
git commit -m "fix: make paper RAG hybrid recall independent"
```

### Task 4: Fair Per-Question Context Allocation

**Files:**
- Modify: `llm-test/rag_foundation.py`
- Modify: `llm-test/rag_retriever.py`
- Modify: `tests/test_paper_rag_foundation.py`

**Interfaces:**
- Produces: `allocate_question_contexts(groups: list[QuestionContext], total_budget: int = 16000, per_question_budget: int = 1200) -> str`
- Produces: `QuestionContext(question_number: int, hits: list[dict[str, Any]], formatted_hits: list[str])`
- Modifies: `retrieve_for_exam_questionwise()` to retrieve Top5 and inject at most Top2 above the configured relevance threshold

- [ ] **Step 1: Add a failing first/last-question fairness test**

```python
def test_context_allocator_keeps_first_and_last_questions_within_budget():
    f = foundation()
    groups = [
        f.QuestionContext(i, [{"relevance_score": 0.9}], [f"核心片段-{i}-" + "知" * 300])
        for i in range(1, 22)
    ]
    context = f.allocate_question_contexts(groups, total_budget=5000, per_question_budget=400)
    assert "第1题参考" in context
    assert "第21题参考" in context
    assert "核心片段-1" in context and "核心片段-21" in context
    assert len(context) <= 5000
```

- [ ] **Step 2: Run the allocator test and verify it fails**

Run: `python -m pytest tests/test_paper_rag_foundation.py::test_context_allocator_keeps_first_and_last_questions_within_budget -v`

Expected: FAIL because `QuestionContext` is not defined.

- [ ] **Step 3: Implement two-pass fair allocation**

Create a dataclass for each question. In pass one, reserve a proportional core budget for the first formatted hit of every nonempty question and prefix it with `[第N题参考]`. In pass two, add second hits in descending `relevance_score` order while respecting both the per-question and total budgets. Trim at paragraph boundaries when possible, then hard-trim only the individual fragment. Assert the final join never exceeds `total_budget`.

- [ ] **Step 4: Wire the allocator into question-wise retrieval**

Change local question retrieval from `topk=1` to `topk=5`. Retain at most two hits whose `sim` equals the calibrated `relevance_score` and exceeds `min_sim_threshold`. Build `QuestionContext` groups rather than appending opaque context strings, then call `allocate_question_contexts`; remove the final `[:16000]` tail truncation. Keep the existing `(context, refs, used_online)` return tuple and leave the `web` profile branch unchanged.

- [ ] **Step 5: Run the focused suite**

Run: `python -m pytest tests/test_paper_rag_foundation.py -v`

Expected: all tests PASS and the 21-question allocation contains both endpoints under budget.

- [ ] **Step 6: Commit question-wise budgeting**

```bash
git add llm-test/rag_foundation.py llm-test/rag_retriever.py tests/test_paper_rag_foundation.py
git commit -m "fix: allocate paper RAG context fairly per question"
```

### Task 5: Retrieval Benchmark Harness and Manual Physics Labels

**Files:**
- Create: `llm-test/rag_benchmark.py`
- Create: `llm-test/benchmarks/physics_retrieval.jsonl`
- Create: `tests/test_rag_benchmark.py`

**Interfaces:**
- Produces: `BenchmarkCase(question_id, question, subject, relevant_docs, relevance_note)`
- Produces: `compute_metrics(cases: list[BenchmarkCase], ranked_ids: dict[str, list[str]], k: int = 5) -> dict[str, Any]`
- CLI: `python llm-test/rag_benchmark.py --benchmark llm-test/benchmarks/physics_retrieval.jsonl --topk 5 --output-dir llm-test/results/retrieval`

- [ ] **Step 1: Write failing metric tests**

```python
# tests/test_rag_benchmark.py
import importlib
import sys
from pathlib import Path

LLM_TEST_DIR = Path(__file__).resolve().parents[1] / "llm-test"
if str(LLM_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_TEST_DIR))


def test_compute_metrics_reports_recall_mrr_and_misses():
    mod = importlib.import_module("rag_benchmark")
    cases = [
        mod.BenchmarkCase("q1", "速度题", "物理", ["d1"], "速度定义"),
        mod.BenchmarkCase("q2", "密度题", "物理", ["d2"], "密度测量"),
    ]
    metrics = mod.compute_metrics(cases, {"q1": ["x", "d1"], "q2": ["x", "y"]}, k=5)
    assert metrics["recall_at_5"] == 0.5
    assert metrics["mrr"] == 0.25
    assert metrics["misses"] == ["q2"]
```

- [ ] **Step 2: Run the metric test and confirm the module is missing**

Run: `python -m pytest tests/test_rag_benchmark.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rag_benchmark'`.

- [ ] **Step 3: Implement strict JSONL loading, metrics, and report output**

Use a frozen dataclass, reject empty `relevant_docs`, compute Recall@1, Recall@5, and reciprocal rank against any accepted relevant ID, and include each miss's returned top-five IDs/text previews in JSON and Markdown. The CLI must instantiate `TextbookRAG`, retrieve exactly once per case, and never load an answer file.

- [ ] **Step 4: Export and manually review 30–50 physics labels**

Split `llm-test/testmd/paper/2026物理试卷.md` with `_split_exam_questions`. For each selected question, run the repaired retriever at Top20, inspect the original textbook chunks, and record one or more genuinely relevant stable IDs plus a Chinese `relevance_note`. Cover at least five mechanics, five electricity, five optics, five thermal/sound, and five experiment/calculation samples. Reject any label sourced from the same exam paper or an answer-bearing source. Validate the JSONL by loading it through `BenchmarkCase`; the file must contain between 30 and 50 nonempty records.

- [ ] **Step 5: Run benchmark unit tests**

Run: `python -m pytest tests/test_rag_benchmark.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit the harness and reviewed labels**

```bash
git add llm-test/rag_benchmark.py llm-test/benchmarks/physics_retrieval.jsonl tests/test_rag_benchmark.py
git commit -m "test: add labeled physics RAG benchmark"
```

### Task 6: Calibrate, Diagnose Misses, and Verify the Foundation

**Files:**
- Modify: `llm-test/config.py`
- Modify: `llm-test/rag_retriever.py`
- Modify if evidence requires it: `llm-test/ingest_zhongkao_v4.py`
- Test: `tests/test_paper_rag_foundation.py`

**Interfaces:**
- Consumes: benchmark CLI and `relevance_score`
- Produces: a named local injection threshold in `RAG_CONFIG`
- Produces: timestamped JSON and Markdown benchmark reports

- [ ] **Step 1: Run the labeled benchmark before calibration**

Run: `python llm-test/rag_benchmark.py --benchmark llm-test/benchmarks/physics_retrieval.jsonl --topk 5 --output-dir llm-test/results/retrieval`

Expected: a report containing Recall@1, Recall@5, MRR, every miss, candidate scores, and filter reasons.

- [ ] **Step 2: Sweep relevance thresholds without changing ranking**

Evaluate thresholds from 0.30 through 0.75 in increments of 0.05. Select the highest threshold that retains Recall@5 ≥ 0.80 and record both recall and average injected chunks per question. Add the selected value as `RAG_CONFIG["local_relevance_threshold"]`; update local injection code to read this value. Add a test asserting that a candidate below the value is not injected even when its `rank_score` is first.

- [ ] **Step 3: Classify every remaining miss before changing chunks**

For each miss, record exactly one category in the report: `coverage_missing`, `chunk_split`, `ocr_corrupt`, or `ranking_error`. Do not reindex for `coverage_missing` or `ranking_error`. If `chunk_split` or `ocr_corrupt` is material, make the smallest ingestion change for only the affected source pattern: inherit the nearest nonempty heading, merge undersized adjacent paragraphs, and retain a 120-character overlap. Add a pure unit test for every ingestion rule changed before rebuilding affected documents.

- [ ] **Step 4: Re-run the benchmark to the acceptance gate**

Run: `python llm-test/rag_benchmark.py --benchmark llm-test/benchmarks/physics_retrieval.jsonl --topk 5 --output-dir llm-test/results/retrieval`

Expected: `recall_at_5 >= 0.80`; otherwise stop and report the miss categories rather than claiming completion.

- [ ] **Step 5: Run all offline regressions**

Run: `python -m pytest`

Expected: all tests PASS.

Run: `python scripts/run_eval.py`

Expected: all golden evaluation cases PASS.

- [ ] **Step 6: Run the controlled local-RAG paper comparison**

Run the same local quantized model, seed, context size, prompt, and physics paper once without RAG and once with `--rag-profile local_clean`. Save both outputs and a retrieval audit listing per-question candidates, injected IDs, relevance scores, and empty-context questions. Treat the score difference as observational; the acceptance claim remains tied to Recall@5.

- [ ] **Step 7: Commit calibration and any evidence-backed ingestion repair**

```bash
git add llm-test/config.py llm-test/rag_retriever.py llm-test/ingest_zhongkao_v4.py tests/test_paper_rag_foundation.py
git commit -m "fix: calibrate local paper RAG relevance gate"
```

