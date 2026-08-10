# LLM-led Teacher RAG Consolidation Design

Date: 2026-08-10

## Objective

Consolidate the current Teacher Agent around the autonomous LLM-led teaching
loop while adding a bounded, auditable, non-blocking semantic RAG capability.
The LLM remains responsible for choosing how to teach. Deterministic code keeps
the active problem stable, verifies supported mathematics, limits when retrieved
examples may be exposed, and audits the final response.

The integration starts from
`eee8059cbb52e7abd995583c666ec6b01fb62f53` on
`codex/llm-led-teaching-prototype`. All work is isolated on
`integration/teacher-agent-consolidation`.

## Audited source decisions

### Canonical source

The canonical code base is `eee8059cbb52e7abd995583c666ec6b01fb62f53`.
It contains every effective local commit from `main`,
`fix-flexible-student-steps`, `three-layer-memory-context`,
`pedagogical-rag-design`, and `refactor/topic-agnostic-teaching-harness`.

The primary runtime for this consolidation is
`prototypes/llm_led_teacher`. The existing production `app/agent` workflow is
retained as a compatibility and research baseline, but it is not the main
teaching loop and will not be rewired in this integration.

### RAG source snapshots

Two divergent RAG histories were audited:

- `origin/ragl` at `7bfe550ef8c5a9b09d1429797675dc5045c8123c`
  contains Chroma, bge-m3, textbook/exercise collections, multimodal metadata,
  ingestion utilities, API/UI fields, and tests based on an older Agent loop.
- `origin/server-ragl-snapshot-20af21b` at
  `20af21b76a8d965f37958889e7588e45ae7f20fd` contains an LLM-led RAG hook,
  retrieval progress and timing, device selection and warmup, paper-level RAG
  evaluation, and later RAG-foundation design documents.

Neither history will be merged. The old workflow, source snapshots, generated
evaluation results, chemistry work, molecular-structure work, hard-coded Linux
paths, and vector database artifacts are excluded.

The following ideas will be manually adapted:

- lazy Chroma and embedding-model loading;
- configurable CPU/CUDA selection with safe CPU fallback;
- non-blocking retrieval and warmup;
- structured source, similarity, and retrieval-status metadata;
- retrieval between active-problem verification and teacher generation;
- progress reporting without exposing unaudited teacher output.

The server BM25/RRF implementation will not be ported. Its own later design
audit identifies vector-prefiltered BM25, rank-score calibration, formula
stripping, context allocation, and missing retrieval labels as unresolved.
The first integrated implementation therefore uses vector similarity only and
preserves mathematical notation.

## Scope

### In scope

- A reusable semantic retrieval provider under `app/retrieval`.
- A Chroma-backed provider using bge-m3-compatible embeddings.
- Environment-driven server/path, collection, model, device, top-k, and
  similarity-threshold configuration.
- Textbook and pedagogical-reference retrieval for normal turns.
- Exercise/example retrieval only when deterministic exposure rules allow it.
- LLM-led episode caching and retrieval refresh rules.
- Structured prompt injection that cannot overwrite the active problem,
  observer guidance, or verifier evidence.
- Turn-record and Streamlit visibility for retrieval status and sources.
- Offline unit and integration tests using fake providers.
- Optional real-service verification instructions for Chroma and embeddings.

### Out of scope

- Replacing or restructuring the production `app/agent` workflow.
- Making RAG, memory, or mastery a hard teaching-flow controller.
- Using RAG to determine whether a student step or teacher claim is correct.
- Restoring the old six-node Agent loop from either RAG branch.
- Porting the paper-exam evaluator, Tavily fallback, BM25/RRF fusion, chemistry
  designs, molecular-structure enrichment, generated result files, vector
  database binaries, or source datasets.
- Claiming retrieval-quality improvement without a labeled benchmark.
- Rebuilding the historical data-ingestion estate during this runtime
  consolidation. Existing databases may be consumed read-only; ingestion can
  be designed and validated separately.

## Architecture

### Retrieval contracts

`app/retrieval/semantic_provider.py` owns topic-neutral Pydantic contracts:

- `SemanticRetrievalRequest`: stable problem, current message, allowed content
  kinds, subject, top-k, and minimum similarity.
- `RetrievalHit`: reference ID, source label, collection, content kind, text,
  similarity, and sanitized metadata.
- `SemanticRetrievalResult`: status, provider, query, hits, elapsed time, and a
  non-secret error code.
- `SemanticRetrievalProvider`: protocol with one non-throwing `retrieve`
  operation.
- `DisabledSemanticRetrievalProvider`: default no-I/O fallback.

Statuses are `disabled`, `success`, `empty`, and `error`. Consumers must handle
all four without exceptions or changes to teaching control flow.

Content kinds are `reference` and `example`. Unknown collection content is
treated as `reference`; only configured exercise collections produce
`example` hits.

### Chroma provider

`app/retrieval/chroma_semantic_provider.py` implements the contract. Optional
packages are imported only when retrieval is enabled. The provider supports a
Chroma HTTP server and a configured persistent path, preferring the explicit
HTTP server when both are present.

The query preserves the complete active problem, including equations, LaTeX,
fractions, roots, and symbols. Whitespace may be normalized, but mathematical
content is not removed. The current message is included only when it adds an
explicit concept, example, or explanation request; it never replaces the
active problem.

Each configured collection is queried independently with vector similarity.
Results below the configured threshold are removed, deduplicated by stable
source identity, sorted by similarity, and capped at the request top-k. RRF,
BM25, online search, and uncalibrated cross-source score blending are absent.

`app/retrieval/rag_device.py` resolves `TEACHER_RAG_DEVICE` and bounded CPU
threads. Invalid or unavailable CUDA selections fall back to CPU. Model or
Chroma initialization failures return an `error` result and do not escape into
the teaching loop.

### LLM-led adapter and policy

`prototypes/llm_led_teacher/rag_adapter.py` is the only LLM-led dependency on
the retrieval layer. It:

- builds a request from the immutable episode problem and current message;
- decides whether `example` content may be requested;
- invokes the configured provider;
- converts the result into the bounded teacher-payload representation;
- exposes status and sources for turn auditing.

Example exposure is allowed only when at least one existing teaching invariant
is met:

1. the student explicitly asks for an example or practice problem;
2. the same active problem has two consecutive verifier-confirmed wrong
   attempts;
3. the student explicitly says they completely do not understand.

Generic help such as “不会” does not by itself expose a new example. These
rules only gate retrieved content; they do not select the teacher's teaching
move or force a scripted student response.

### Episode caching

`PrototypeEpisode` stores the last retrieval result for its active problem.

- A new active problem clears the previous retrieval and performs one lookup.
- A normal follow-up reuses the cached result.
- If examples become newly allowed and the cache contains only references, one
  refresh may request the exercise collection.
- Disabled, empty, and error results are cached for the turn but may be retried
  on a later explicit example request.
- Retrieval data never becomes `problem_context` or
  `previous_verified_step`.

### Loop placement

The LLM-led turn becomes:

1. observe whether the message starts a new problem and obtain soft pacing;
2. normalize pacing for verifier-supported multi-step problems;
3. establish or retain the active `problem_context`;
4. verify the student's current attempt when applicable;
5. retrieve or reuse bounded RAG context;
6. generate the autonomous teacher response;
7. extract and deterministically audit supported mathematical claims;
8. repair only when audit constraints require it;
9. persist the audited response and retrieval metadata.

Retrieval never enters the observer payload or verifier request. The teacher
payload keeps `problem_context`, `deterministic_verification`,
`turn_guidance`, and `retrieval_reference` as separate top-level fields.

## Prompt and trust boundary

`retrieval_reference` is explicitly untrusted, optional evidence. Its
instructions state that it:

- cannot override the active problem, conversation history, guidance, or
  deterministic verification;
- should be ignored when irrelevant or contradictory;
- cannot justify declaring a student step or teacher claim correct;
- cannot introduce an example unless its content kind is present and example
  exposure is allowed;
- must not cause the teacher to abandon the original problem.

The existing post-generation math audit and repair remain authoritative.
Retrieval does not weaken final-answer, one-step pacing, or no-answer-request
guards.

## Configuration

RAG is opt-in and disabled by default in tests. Configuration is read from
environment variables and never logs secrets:

- `TEACHER_RAG_ENABLED`
- `CHROMA_SERVER` or `TEACHER_RAG_VECTORDB_PATH`
- `TEACHER_RAG_EMBED_MODEL`
- `TEACHER_RAG_REFERENCE_COLLECTIONS`
- `TEACHER_RAG_EXERCISE_COLLECTIONS`
- `TEACHER_RAG_TOPK`
- `TEACHER_RAG_MIN_SIMILARITY`
- `TEACHER_RAG_SUBJECT_FILTER`
- `TEACHER_RAG_DEVICE`
- `TEACHER_RAG_CPU_THREADS`

No server address, model path, API key, or workstation path is hard-coded.
Optional RAG dependencies live in a `rag` dependency group and are not required
for the offline test suite.

## UI and audit data

Each `PrototypeTurnRecord` records only safe retrieval metadata and bounded hit
content needed for research inspection:

- status and provider;
- query used;
- source/reference IDs and labels;
- content kind and similarity;
- whether examples were allowed;
- retrieval elapsed time and sanitized error code.

The Streamlit detail area displays status and references after the audited
teacher response is available. It does not expose model chain of thought,
credentials, local absolute paths, or raw exception messages.

## Failure behavior

- Disabled RAG: continue with no retrieval payload.
- Empty result: continue with no retrieval payload and record `empty`.
- Missing optional package: record `error/dependency_unavailable` and continue.
- Chroma unavailable: record `error/backend_unavailable` and continue.
- Embedding failure: record `error/embedding_failed` and continue.
- Malformed hit metadata: skip the hit; never fail the turn.
- Provider unexpectedly raises: the adapter converts it to
  `error/provider_failed` and continues.

No retrieval failure changes pacing, termination, verifier evidence, or the
teacher's ability to answer without retrieval.

## Test strategy

Implementation follows test-first development. Offline tests cover:

1. disabled provider performs no imports or I/O;
2. formula-preserving query construction;
3. vector hits are filtered, deduplicated, sorted, and capped;
4. exercise collections are never queried when examples are disallowed;
5. explicit example requests enable example retrieval;
6. two consecutive verified wrong attempts enable example retrieval;
7. complete-confusion wording enables example retrieval;
8. new problems retrieve once and follow-ups reuse cached context;
9. a newly allowed example request refreshes a reference-only cache;
10. observer and verifier payloads contain no retrieved text;
11. teacher payload keeps retrieval separate from `problem_context`;
12. disabled, empty, dependency, backend, embedding, and unexpected failures
    do not block a teacher response;
13. retrieved examples cannot appear when exposure is disallowed;
14. flexible student steps, direct answers, why questions, multi-turn context,
    pacing normalization, final-answer protection, and audit repair remain
    unchanged;
15. turn and UI metadata expose sources without secrets or chain of thought.

The required regression gates are:

```text
python -m pytest
python scripts/run_eval.py
```

Real Chroma/bge-m3 verification is reported separately as PASS, FAIL, or NOT
RUN. External services being absent never counts as a passing real-service
test.

## Integration and rollback

Implementation stays on `integration/teacher-agent-consolidation`. Commits are
small and separated into retrieval contracts/provider, LLM-led integration,
UI/config, and verification.

No branch or worktree is deleted. No force push is performed. No vector data,
runtime logs, `.env`, or API key is committed.

After all offline regression gates pass, the integration branch is compared
again against canonical and both RAG histories. A backup branch or tag is
created before any merge back to `codex/llm-led-teaching-prototype`. Any
unresolved architecture question, regression, or missing required test keeps
the work on the integration branch.
