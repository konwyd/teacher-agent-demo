# LLM-led Teacher RAG Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the autonomous LLM-led teaching loop the primary consolidated Teacher Agent and add bounded, auditable, non-blocking Chroma vector RAG without restoring the older planner-controlled Agent architecture.

**Architecture:** Add a topic-neutral retrieval contract and lazy Chroma provider under `app/retrieval`, then connect it to `prototypes/llm_led_teacher` through a small policy adapter. Retrieval runs after active-problem anchoring and student verification but before teacher generation; it never enters the observer or verifier and never controls mathematical correctness. Existing math-checker audits remain authoritative, and retrieved examples are withheld unless one of the three established example triggers is present.

**Tech Stack:** Python 3.10+, Pydantic 2, pytest, ChromaDB 1.5.9, sentence-transformers/bge-m3, Streamlit, existing DeepSeek client and deterministic math checker.

## Global Constraints

- Base all work on `eee8059cbb52e7abd995583c666ec6b01fb62f53` through `integration/teacher-agent-consolidation`.
- The main runtime is `prototypes/llm_led_teacher`; do not rewire the production `app/agent` workflow.
- RAG, memory, and mastery remain soft evidence and never override `math_checker`, the active problem, or verifier output.
- Retrieved examples are visible only for an explicit example/practice request, two consecutive verifier-confirmed wrong attempts on the active problem, or an explicit complete-confusion statement.
- Do not port the server BM25/RRF implementation, formula-stripping query cleanup, Tavily fallback, paper evaluator, chemistry code, generated result files, data sets, or vector database files.
- RAG is optional. Missing packages, disabled configuration, empty results, Chroma failure, and embedding failure must not block teaching.
- Do not commit `.env`, API keys, logs, runtime data, absolute workstation/server paths, or vector database artifacts.
- Every production behavior change follows red-green-refactor and is committed only after its focused tests pass.
- Final offline gates are exactly `python -m pytest` and `python scripts/run_eval.py`.

---

## File map

- Create `app/retrieval/semantic_provider.py`: retrieval enums, request/result/hit models, provider protocol, disabled provider, and safe provider-call boundary.
- Create `app/retrieval/rag_device.py`: CPU/CUDA device and bounded-thread selection adapted from server commit `1bfef256`.
- Create `app/retrieval/chroma_semantic_provider.py`: environment settings, formula-preserving query construction, lazy Chroma/embedding initialization, collection querying, filtering, deduplication, and factory.
- Create `prototypes/llm_led_teacher/rag_adapter.py`: example-exposure policy, episode-cache decision, provider invocation, and bounded teacher-payload projection.
- Modify `prototypes/llm_led_teacher/logic.py`: server-only pacing safeguards, retrieval state, retrieval placement, progress/timing, prompt boundary, and audit record fields.
- Create `prototypes/llm_led_teacher/live_progress.py`: safe user-visible stage trace derived from server commit `1bfef256`.
- Modify `prototypes/llm_led_teacher/streamlit_app.py`: primary-product wording, retrieval/progress display, safe source audit, and session schema bump.
- Create `.env.example`: documented DeepSeek and optional RAG configuration without secrets or machine-specific paths.
- Modify `pyproject.toml`: add optional `rag` dependencies without changing the default offline environment.
- Modify `prototypes/llm_led_teacher/README.md`: document the consolidated autonomous loop and optional RAG setup.
- Add focused tests under `tests/` and `tests/prototypes/`; preserve all existing tests unchanged unless a schema-version assertion must be updated.

---

### Task 1: Preserve server-only LLM-led pacing and audit safeguards

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py`

**Interfaces:**
- Produces: `_normalize_guidance(episode: PrototypeEpisode, message: str, guidance: TurnGuidance) -> TurnGuidance`
- Produces: `PrototypeTurnRecord.observer_raw_guidance: TurnGuidance`
- Preserves: `run_turn(episode, message, *, client=...) -> PrototypeTurnRecord`

- [ ] **Step 1: Add a failing regression for incorrect observer pacing**

  Add an observed client that returns `complete_response` for `2x+9=14，这道题怎么解`, then returns a draft containing both `2x=5` and `x=5/2`, followed by a repaired one-step response. Assert:

  ```python
  assert turn.observer_raw_guidance.pacing is PacingMode.COMPLETE_RESPONSE
  assert turn.turn_guidance.pacing is PacingMode.SINGLE_MOVE
  assert client.calls == 2
  assert "2x=5" in turn.teacher_output.response
  assert "x=5/2" not in turn.teacher_output.response
  ```

- [ ] **Step 2: Run the focused test and verify RED**

  Run:

  ```text
  python -m pytest tests/prototypes/test_llm_led_turn_contract.py::test_wrong_observer_complete_response_is_normalized_and_final_answer_repaired -q
  ```

  Expected: FAIL because `observer_raw_guidance` and `_normalize_guidance` do not exist and the observer's `complete_response` is currently trusted.

- [ ] **Step 3: Add deterministic guidance normalization**

  Implement `_is_supported_transform_problem` with `LinearEquationVerifier` and `math_checker`, then implement `_normalize_guidance`. It changes only pacing to `single_move` when the active problem is a supported non-final equation and the student did not request a full solution or ask why. Retain the observer's original object in `observer_raw_guidance`.

- [ ] **Step 4: Add a failing regression for post-audit final-answer protection**

  Force effective guidance to remain `complete_response` for a first-turn transform problem and return a verified final form. Assert that `_collect_warnings` triggers repair even when guidance is wrong.

- [ ] **Step 5: Run the post-audit test and verify RED**

  Expected: FAIL because the current warning condition depends on `guidance.pacing is SINGLE_MOVE`.

- [ ] **Step 6: Make the post-audit invariant independent of observer pacing**

  Change only the multi-step disclosure warning condition: for a new supported transform problem, a verified final-form claim requires repair unless the student explicitly requests a full solution. Keep existing no-answer and verifier-conflict checks unchanged.

- [ ] **Step 7: Run the complete prototype contract file and verify GREEN**

  Run:

  ```text
  python -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
  ```

  Expected: all tests PASS.

- [ ] **Step 8: Commit the safeguard port**

  ```text
  git add prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_turn_contract.py
  git commit -m "fix: harden llm-led pacing audits"
  ```

---

### Task 2: Add retrieval contracts and safe device selection

**Files:**
- Create: `app/retrieval/semantic_provider.py`
- Create: `app/retrieval/rag_device.py`
- Create: `tests/test_semantic_retrieval_contract.py`
- Create: `tests/test_rag_device.py`

**Interfaces:**
- Produces: `RetrievalStatus`, `RetrievalContentKind`, `SemanticRetrievalRequest`, `RetrievalHit`, `SemanticRetrievalResult`
- Produces: `SemanticRetrievalProvider.retrieve(request) -> SemanticRetrievalResult`
- Produces: `safe_retrieve(provider, request) -> SemanticRetrievalResult`
- Produces: `resolve_rag_device(requested: str | None = None, *, torch_module=None) -> str`
- Produces: `configure_cpu_threads(raw: str | None = None, *, torch_module=None) -> int`

- [ ] **Step 1: Write failing contract tests**

  Test Pydantic validation, a disabled provider returning `disabled`, and an exploding provider converted by `safe_retrieve` to `error/provider_failed` without including the raw exception message.

  ```python
  result = safe_retrieve(ExplodingProvider(), SemanticRetrievalRequest(problem_context="x+1=2"))
  assert result.status is RetrievalStatus.ERROR
  assert result.error_code == "provider_failed"
  assert "secret backend detail" not in result.model_dump_json()
  ```

- [ ] **Step 2: Verify contract tests RED**

  Run `python -m pytest tests/test_semantic_retrieval_contract.py -q`.
  Expected: collection error because `app.retrieval.semantic_provider` does not exist.

- [ ] **Step 3: Implement minimal retrieval contracts**

  Use string enums with values `disabled`, `success`, `empty`, `error` and `reference`, `example`. Validate similarity in `[0, 1]`, default lists with factories, and ensure `safe_retrieve` returns a sanitized error result for unexpected exceptions.

- [ ] **Step 4: Verify retrieval contracts GREEN**

  Run `python -m pytest tests/test_semantic_retrieval_contract.py -q`.

- [ ] **Step 5: Write failing device tests**

  Cover explicit CPU, default `cuda:0`, unavailable CUDA, out-of-range CUDA, invalid values, environment CPU thread count, and invalid thread-count fallback to `4`.

- [ ] **Step 6: Verify device tests RED**

  Run `python -m pytest tests/test_rag_device.py -q`.
  Expected: collection error because `app.retrieval.rag_device` does not exist.

- [ ] **Step 7: Implement lazy device helpers**

  Import `torch` inside functions only when no test/runtime object is injected, so importing the retrieval contracts never requires optional RAG packages. If Torch is unavailable, device resolution returns `cpu` and thread configuration returns the validated count without calling Torch. Use `TEACHER_RAG_DEVICE` and `TEACHER_RAG_CPU_THREADS`; invalid or unavailable CUDA returns `cpu`.

- [ ] **Step 8: Verify Task 2 GREEN and commit**

  Run:

  ```text
  python -m pytest tests/test_semantic_retrieval_contract.py tests/test_rag_device.py -q
  ```

  Then commit:

  ```text
  git add app/retrieval/semantic_provider.py app/retrieval/rag_device.py tests/test_semantic_retrieval_contract.py tests/test_rag_device.py
  git commit -m "feat: add semantic retrieval contracts"
  ```

---

### Task 3: Implement the formula-preserving Chroma provider

**Files:**
- Create: `app/retrieval/chroma_semantic_provider.py`
- Create: `tests/test_chroma_semantic_provider.py`

**Interfaces:**
- Consumes: Task 2 retrieval contracts and device helpers
- Produces: `ChromaSemanticRetrievalProvider(settings: SemanticRagSettings | None = None, *, client_factory=None, embedding_factory=None)`
- Produces: `build_retrieval_query(problem_context: str, current_message: str) -> str`
- Produces: `get_semantic_retrieval_provider() -> SemanticRetrievalProvider`
- Produces: `SemanticRagSettings` with environment-backed opt-in configuration

- [ ] **Step 1: Write failing formula/query tests**

  Assert that the query retains `\\frac{1}{2}`, `\\sqrt{3}`, `$x+1=2$`, braces, operators, and Chinese text while only collapsing repeated whitespace.

- [ ] **Step 2: Write failing collection-policy tests**

  With fake client, collections, and embedding factory, assert:

  ```python
  request = SemanticRetrievalRequest(
      problem_context="2x+3=7",
      allowed_content_kinds={RetrievalContentKind.REFERENCE},
      top_k=3,
      minimum_similarity=0.6,
  )
  result = provider.retrieve(request)
  assert queried_collections == ["k9_textbook", "curriculum_std"]
  assert all(hit.content_kind is RetrievalContentKind.REFERENCE for hit in result.hits)
  ```

  A second request allowing examples must additionally query `k9_exercises`.

- [ ] **Step 3: Write failing ranking and degradation tests**

  Cover distance-to-similarity conversion, threshold filtering, stable-source deduplication, descending sort, top-k cap, malformed metadata skipping, missing dependency, backend failure, and embedding failure. Assert sanitized error codes rather than raw messages.

- [ ] **Step 4: Run provider tests and verify RED**

  Run `python -m pytest tests/test_chroma_semantic_provider.py -q`.
  Expected: collection error because the provider module does not exist.

- [ ] **Step 5: Implement settings and lazy initialization**

  `SemanticRagSettings` reads:

  ```text
  TEACHER_RAG_ENABLED
  CHROMA_SERVER
  TEACHER_RAG_VECTORDB_PATH
  TEACHER_RAG_EMBED_MODEL
  TEACHER_RAG_REFERENCE_COLLECTIONS
  TEACHER_RAG_EXERCISE_COLLECTIONS
  TEACHER_RAG_TOPK
  TEACHER_RAG_MIN_SIMILARITY
  TEACHER_RAG_SUBJECT_FILTER
  TEACHER_RAG_DEVICE
  TEACHER_RAG_CPU_THREADS
  ```

  Defaults: disabled; reference collections `k9_textbook,curriculum_std`; exercise collection `k9_exercises`; top-k `5`; minimum similarity `0.44`; subject `数学`. Do not instantiate Chroma or sentence-transformers until the first enabled request.

- [ ] **Step 6: Implement vector-only collection retrieval**

  Generate one embedding for the formula-preserving query. Query each allowed collection independently, convert cosine distance with `max(0.0, min(1.0, 1.0 - distance))`, sanitize metadata to string/number/boolean values, deduplicate with `collection + source_file/id + chunk index`, then filter/sort/cap.

- [ ] **Step 7: Verify provider GREEN**

  Run `python -m pytest tests/test_chroma_semantic_provider.py -q`.

- [ ] **Step 8: Commit the provider**

  ```text
  git add app/retrieval/chroma_semantic_provider.py tests/test_chroma_semantic_provider.py
  git commit -m "feat: add bounded chroma retrieval provider"
  ```

---

### Task 4: Add LLM-led example policy and retrieval cache adapter

**Files:**
- Create: `prototypes/llm_led_teacher/rag_adapter.py`
- Modify: `prototypes/llm_led_teacher/logic.py`
- Create: `tests/prototypes/test_llm_led_rag_adapter.py`

**Interfaces:**
- Consumes: Task 2 provider contract
- Produces: `example_exposure_allowed(episode, message, verification) -> bool`
- Produces: `retrieve_for_turn(episode, message, verification, *, starts_new_problem, provider) -> SemanticRetrievalResult`
- Produces: `retrieval_payload(result) -> dict[str, object] | None`
- Produces in `PrototypeEpisode`: `active_retrieval`, `active_retrieval_problem`, and `active_retrieval_allows_examples`

- [ ] **Step 1: Write failing example-policy tests**

  Cover all allowed triggers independently: explicit “举个例子/给我一道练习”, two consecutive `VERIFIED_INCORRECT` attempts for the same active problem, and “我完全不懂”. Assert that ordinary “我不会” and a single wrong attempt remain false.

- [ ] **Step 2: Write failing cache tests**

  Using a counting fake provider, assert new problem retrieval, normal follow-up reuse, active-problem change cache reset, and one refresh when examples become newly allowed after a reference-only lookup.

- [ ] **Step 3: Write failing payload-boundary tests**

  Assert `retrieval_payload` returns `None` for disabled/empty/error, includes only successful bounded hits, labels the data untrusted, and does not contain fields named `problem_context`, `turn_guidance`, or `deterministic_verification`.

- [ ] **Step 4: Run adapter tests and verify RED**

  Run `python -m pytest tests/prototypes/test_llm_led_rag_adapter.py -q`.
  Expected: collection error because `rag_adapter` does not exist.

- [ ] **Step 5: Implement deterministic exposure policy**

  Use explicit marker tuples and count consecutive verifier-confirmed wrong turn records from the end of the active episode. Do not infer wrong attempts from LLM text or mastery state.

- [ ] **Step 6: Add the minimal episode cache schema**

  Add typed retrieval-cache fields to `PrototypeEpisode` with safe empty defaults. Do not call retrieval or change teacher payload generation in this task.

- [ ] **Step 7: Implement cache and safe provider boundary**

  New problems always replace the cache. Follow-ups reuse a success cache unless examples become newly allowed and were not requested previously. Invoke providers only through `safe_retrieve`.

- [ ] **Step 8: Verify adapter GREEN and commit**

  Run `python -m pytest tests/prototypes/test_llm_led_rag_adapter.py -q`, then:

  ```text
  git add prototypes/llm_led_teacher/rag_adapter.py prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_rag_adapter.py
  git commit -m "feat: add llm-led retrieval policy adapter"
  ```

---

### Task 5: Integrate RAG into the autonomous turn loop

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Create: `prototypes/llm_led_teacher/live_progress.py`
- Create: `tests/prototypes/test_llm_led_rag_integration.py`
- Create: `tests/prototypes/test_llm_led_progress.py`

**Interfaces:**
- Consumes: `SemanticRetrievalProvider`, `get_semantic_retrieval_provider`, and Task 4 adapter
- Changes: `run_turn(..., retrieval_provider: SemanticRetrievalProvider | None = None, progress_callback: Callable[[str], None] | None = None, clock: Callable[[], float] = perf_counter)`
- Consumes: Task 4 episode retrieval-cache fields
- Adds to turn record: status, provider, query, sources/hits, example permission, sanitized error code, stage timings, backend total, and repair flag

- [ ] **Step 1: Write failing loop-placement tests**

  Capture observer JSON, provider requests, and teacher JSON. Assert:

  ```python
  assert "retrieval_reference" not in observer_payload
  assert provider_request.problem_context == episode.problem_context
  assert teacher_payload["problem_context"] == episode.problem_context
  assert teacher_payload["retrieval_reference"]["hits"][0]["text"] == "有效教材片段"
  ```

  Also assert retrieved content never reaches `_verify` as the student attempt or previous verified step.

- [ ] **Step 2: Write failing non-blocking tests**

  Parameterize disabled, empty, error-result, and raising providers. Each case must still return the scripted teacher response and record the correct sanitized retrieval status/error code.

- [ ] **Step 3: Write failing example-isolation test**

  Return one reference and one example from a deliberately non-compliant fake provider while examples are disallowed. Assert only the reference enters the teacher payload and the example text is absent.

- [ ] **Step 4: Write failing progress/timing tests**

  With a deterministic clock, assert stage order:

  ```text
  observing, retrieving, generating, auditing, [repairing], complete
  ```

  Assert callback failure does not block the turn and no chain-of-thought field exists.

- [ ] **Step 5: Run Task 5 tests and verify RED**

  Run:

  ```text
  python -m pytest tests/prototypes/test_llm_led_rag_integration.py tests/prototypes/test_llm_led_progress.py -q
  ```

  Expected: failures because `run_turn` lacks provider/progress parameters and state fields.

- [ ] **Step 6: Add state fields and prompt boundary**

  Store retrieval result on the episode, project safe metadata onto the turn, and include `retrieval_reference` only for successful non-empty hits. Update the system prompt to describe RAG as untrusted soft evidence that cannot replace guidance, active problem, or deterministic verification.

- [ ] **Step 7: Place retrieval after verification and before generation**

  Call Task 4's adapter after `problem_context` and student verification are fixed. Filter any provider-returned example again at the prompt boundary when exposure is false. Preserve the existing observer, teacher, audit, repair, and previous-verified-step semantics.

- [ ] **Step 8: Add safe progress and timing**

  Adapt the server progress callback pattern. `live_progress.py` maps only public stage names to user-facing status text. Callback exceptions are swallowed; teacher and retrieval exceptions retain their existing explicit behavior.

- [ ] **Step 9: Verify Task 5 GREEN plus prior prototype contracts**

  Run:

  ```text
  python -m pytest tests/prototypes/test_llm_led_rag_integration.py tests/prototypes/test_llm_led_progress.py tests/prototypes/test_llm_led_turn_contract.py -q
  ```

- [ ] **Step 10: Commit loop integration**

  ```text
  git add prototypes/llm_led_teacher/logic.py prototypes/llm_led_teacher/live_progress.py tests/prototypes/test_llm_led_rag_integration.py tests/prototypes/test_llm_led_progress.py
  git commit -m "feat: integrate bounded rag into llm-led loop"
  ```

---

### Task 6: Expose safe RAG configuration and UI audit data

**Files:**
- Create: `.env.example`
- Modify: `pyproject.toml`
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Modify: `prototypes/llm_led_teacher/README.md`
- Modify: `tests/prototypes/test_llm_led_streamlit_session.py`
- Create: `tests/prototypes/test_llm_led_rag_ui.py`

**Interfaces:**
- Consumes: Task 5 turn metadata and `LiveProgressTrace`
- Produces: optional dependency group `rag = ["chromadb==1.5.9", "sentence-transformers>=2.7"]`
- Changes: `_SESSION_SCHEMA_VERSION` from `4` to `5`

- [ ] **Step 1: Write failing UI/session tests**

  Update the session migration assertion to `5`. Using the existing fake Streamlit pattern, assert a successful RAG turn displays status, provider, source labels, similarity, and whether examples were allowed, while an error turn displays only its sanitized error code.

- [ ] **Step 2: Run UI tests and verify RED**

  Run:

  ```text
  python -m pytest tests/prototypes/test_llm_led_streamlit_session.py tests/prototypes/test_llm_led_rag_ui.py -q
  ```

  Expected: schema assertion and missing retrieval UI failures.

- [ ] **Step 3: Update Streamlit product wording and audit panel**

  Remove wording that says the primary UI deliberately bypasses all RAG. Describe it as autonomous teaching with optional soft RAG and deterministic audit. Render only safe turn metadata in the existing detail expander and use `LiveProgressTrace` during a new turn.

- [ ] **Step 4: Add configuration documentation**

  Create `.env.example` with empty `DEEPSEEK_API_KEY`, official DeepSeek defaults, `TEACHER_RAG_ENABLED=false`, local placeholders without absolute machine paths, and every variable from the design. Update the prototype README with disabled/no-service behavior and a Chroma HTTP example.

- [ ] **Step 5: Add optional dependencies**

  Add the `rag` optional dependency group without moving RAG packages into default dependencies. Do not add Torch separately because sentence-transformers owns the compatible Torch dependency.

- [ ] **Step 6: Verify UI/config GREEN and inspect secrets**

  Run the focused UI tests, then:

  ```text
  git diff --check
  rg -n "sk-[A-Za-z0-9]|/home/|dockerContainerSave|127\.0\.0\.1:8002" .env.example prototypes/llm_led_teacher app/retrieval
  ```

  Expected: tests PASS; the secret/path scan returns no hard-coded key or server-specific path. A documented localhost example may use separate host and port variables, not the banned combined literal.

- [ ] **Step 7: Commit UI and configuration**

  ```text
  git add .env.example pyproject.toml prototypes/llm_led_teacher/streamlit_app.py prototypes/llm_led_teacher/README.md tests/prototypes/test_llm_led_streamlit_session.py tests/prototypes/test_llm_led_rag_ui.py
  git commit -m "feat: expose llm-led rag configuration"
  ```

---

### Task 7: Full regression verification and consolidation audit

**Files:**
- Modify only if a genuine regression is found: the smallest responsible production file and a new focused regression test
- Record results in the final report; do not create generated test-output artifacts in Git

**Interfaces:**
- Verifies all prior task interfaces and repository invariants

- [ ] **Step 1: Run focused retrieval and prototype suites**

  ```text
  python -m pytest tests/test_semantic_retrieval_contract.py tests/test_rag_device.py tests/test_chroma_semantic_provider.py tests/prototypes -q
  ```

- [ ] **Step 2: Run the complete offline test suite**

  ```text
  python -m pytest
  ```

  Expected: all tests PASS with no external Chroma, embedding model, or DeepSeek request.

- [ ] **Step 3: Run the golden evaluation gate**

  ```text
  python scripts/run_eval.py
  ```

  Expected: all offline golden cases PASS.

- [ ] **Step 4: Run static repository checks**

  ```text
  git diff --check codex/llm-led-teaching-prototype...HEAD
  git status --short
  git ls-files .env logs data/runtime
  ```

  Expected: no whitespace errors; no unexpected worktree changes; no `.env`, log, runtime, database, or generated evaluation artifacts added.

- [ ] **Step 5: Verify branch ancestry and integration scope**

  Confirm the integration branch descends from `eee8059c`, list every integration commit, and compare file-level changes against `origin/ragl@7bfe550` and `origin/server-ragl-snapshot-20af21b`. Confirm no old Agent workflow, `llm-test`, chemistry, data, or vector artifacts entered the branch.

- [ ] **Step 6: Classify external-service verification**

  If a configured Chroma service and bge-m3 model are available, run one reference query, one example-enabled query, one empty query, and one backend-failure query without using a paid LLM. Otherwise report all four as `NOT RUN` with the exact missing dependency/service reason.

- [ ] **Step 7: Request code review and fix only evidence-backed findings**

  Review the diff for active-problem stability, example isolation, math-checker priority, failure degradation, configuration secrecy, and optional-dependency isolation. Any fix starts with a reproducing failing test.

- [ ] **Step 8: Decide merge readiness**

  If every offline gate passes and no architecture question remains, create a backup branch or tag at the then-current canonical SHA before merging. If any required gate fails or a real-service risk remains unresolved, leave the completed work on `integration/teacher-agent-consolidation` and do not modify canonical.
