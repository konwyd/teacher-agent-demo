# Teacher Agent Memory System Architecture Audit

> 审计日期：2026-08-20  
> 范围：当前 8502 LLM-led Teacher、保留的 8501/LangGraph baseline、离线 quality-QA 管道及共享 memory/retrieval/cache 模块。  
> 方法：只读代码/文档、调用链搜索和现有测试；未修改业务代码、配置行为、memory schema 或 agent loop。  
> 术语：本文把“记忆”按功能分为 active context/STM、student-specific persistent memory、agent experiential memory、curated knowledge、archive/log 和 performance cache；不会因为模块名含 `memory` 就把它们混为同一类。

## 1. Executive summary

当前项目不是一个统一 memory system，而是两代运行链路和若干旁路存储的集合：

- 当前 8502 主链路以 `PrototypeEpisode`、SQLite conversation archive、SQLite structured learner memory、rolling context summary、可选 KV snapshot 为核心；入口是 `prototypes/llm_led_teacher/logic.py::run_turn()` / `_run_turn_impl()`。
- 旧 8501 baseline 同时维护进程内 `ConversationState`、JSON `EpisodeState`、JSON mastery、JSONL learning events、聚合 profile 和 turn logs；入口由 `app/agent/workflow.py` 委托给 `app/agent/linear_equation_adapter.py`。
- 教材/题库 Chroma、local pedagogical JSON graph、Tavily 是 curated/external knowledge，不是 student memory。
- `quality_qa` 是从历史回答离线提炼的 agent experiential reference corpus。它有当前项目里最完整的 candidate → admission → dedup → replace/delete → versioned retrieval 生命周期，但默认关闭，且不属于学生档案。
- 8502 structured learner memory 的 read boundary 已相对集中在 `StudentMemoryService.resolve()`，write boundary 也集中在 `enqueue_turn()` → `_extract_and_write()` → `upsert_topic_memory()`；这是低侵入接入两个 controller 的最佳切点。
- 但当前 student-memory admission 过宽：凡是非 ambiguous 的 K9 回合都排入 outbox，普通、未验证回合也会以 `learning_turn`、`confidence=0.75` 写入持久主题记忆。没有 future utility、novelty、temporal validity、provenance type、contradiction 或 stability 判定。
- student topic memory 没有正式 delete/forget/consolidate API；`status != 'deleted'` 只出现在读过滤，代码没有把状态设为 deleted。merge 使用“新非空值覆盖 + list 去重 + confidence 取最大值”，因此错误置信度不会因后续冲突下降。
- 8502 memory access 不是每轮固定 retrieve：global profile 每轮加载；topic memory 只有承接词或明确主题才候选；教材 RAG 由 fast-plan/agent 决定；quality-QA 还受独立 route 和 feature flag 控制。已有 NOOP、歧义澄清、去重调用签名和每轮一次预算，但没有 stuck-triggered re-retrieve、阶段化 memory policy、自动 student-memory consolidation/forget。
- 上下文管理已区分完整 archive 与 active working context：SQLite 保留完整 `episode_json`，工作上下文达到估算窗口 80% 时把旧完整轮次语义压缩，目标 55%，默认至少保留最近 12 轮；摘要失败使用最多 32 条 continuity notes 的确定性降级。
- 项目有可靠的离线 evaluation signals 和若干在线观测值，但没有把结果回传给 memory policy 的可靠在线 reward。

> No reliable online reward signal currently exists.

结论：后端可以保留并在外层增加 controller，但 Phase 2 前必须先定义统一的 dynamic write taxonomy/provenance、shadow decision record、冲突与删除语义、以及不依赖模拟器的 feedback contract。详见第 14、15 节。

## 2. Current architecture diagram

### 2.1 当前 8502 主路径

```text
student message
  │
  ├─ UI pending message ───────────────────────────────┐
  │  prototypes/.../streamlit_app.py                   │
  ▼                                                    ▼
PrototypeEpisode (active STM)                 LocalConversationStore
  logic.py::PrototypeEpisode                  conversations.sqlite3
  │                                           complete episode_json
  ├─ every turn: StudentMemoryService.load_profile()
  ├─ classify request
  ├─ StudentMemoryService.resolve()
  │    ├─ student_topic_memories (SQLite)
  │    └─ conversation archive candidates (SQLite)
  │         ├─ NONE → no topic injection
  │         ├─ UNIQUE → bounded topic payload / optional restore
  │         └─ AMBIGUOUS → clarification, no write-back
  ├─ fast plan or bounded agent loop
  │    ├─ textbook/exercise Chroma (curated knowledge)
  │    ├─ quality_qa Chroma (agent experience; optional)
  │    └─ Tavily (external current knowledge; gated)
  ├─ teacher prompt
  │    ├─ rolling semantic summary
  │    ├─ non-compacted recent turns
  │    ├─ retrieval_reference
  │    └─ student_memory_context
  ├─ verifier / generation / audit / repair
  ├─ append PrototypeTurnRecord
  ├─ save complete episode_json
  └─ enqueue StudentMemoryService outbox
        └─ deterministic extraction → upsert topic/global profile

Optional performance plane:
llama-server GPU/RAM KV ↔ app/llm/kv_cache.py SSD snapshots (not semantic memory)
```

### 2.2 Legacy 8501 baseline

```text
TeacherState
  → context_resolver_node
  → diagnose_node
  → retrieve_node (JSON graph; normally every non-pending turn)
  → teaching_policy_node
  → teach_node
  → memory_update_node
       └─ student_profiles.json mastery/misconceptions
  → log_node
       ├─ learning_events.jsonl
       ├─ student_profiles_v2.json aggregation
       ├─ episode_states.json
       ├─ in-process ConversationState
       └─ logs/turn_logs.jsonl
```

`app/agent/workflow.py::build_workflow()` exposes this retained graph; it is not the 8502 main path.

## 3. Memory type inventory

| Memory type | Storage representation | Persistence | Owner module | Write entry point | Read entry point | Consumers |
|---|---|---|---|---|---|---|
| 8502 active episode / conversation STM | `logic.py::PrototypeEpisode`: problem, verified step, subject/stage, turns, pending state, compaction state | Persisted as complete JSON blob in SQLite by UI; also live object | `prototypes/llm_led_teacher/logic.py`, `app/memory/conversation_archive.py` | `logic.py::_run_turn_impl()` appends `PrototypeTurnRecord`; UI calls `LocalConversationStore.create/save()` | `LocalConversationStore.load()`; `_history_payload()` | classifier/orchestrator, teacher generator, verifier, UI, learner-memory extraction |
| 8502 rolling semantic summary | `ContextCompactionState` + `RollingContextSummary` typed lists | Inside persisted `PrototypeEpisode` | `prototypes/llm_led_teacher/context_compaction.py` | `maybe_compact_context()` via `_merge_semantically()` or `_fallback_summary()` | `working_history_payload()`, `logic.py::_context_summary_payload()` | observer/teacher prompts and token budgeting |
| 8502 global student profile | `student_profiles.global_json` → `GlobalStudentProfile(grade, preferred_name, teaching_preferences)` | SQLite; by default same DB file as conversations | `app/memory/learner_memory.py::LearnerMemoryStore` | `_extract_and_write()` → `update_global_profile()` | `StudentMemoryService.load_profile()` | `StudentMemoryResolution.prompt_payload()` → teacher personalization |
| 8502 topic/student LTM | `student_topic_memories` row + `TopicMemory` JSON, evidence, confidence, status, version | SQLite | `app/memory/learner_memory.py` | `enqueue_turn()` → outbox → `_extract_and_write()` → `upsert_topic_memory()` | `list_topic_memories()` → `StudentMemoryService.resolve()` | optional context restore; `student_memory_context.topic_memory` |
| 8502 memory pending state | `student_memory_outbox` rows (`pending/processing/completed/failed`) and `PendingMemoryClarification` in episode | SQLite outbox + episode JSON | `learner_memory.py`, `logic.py` | `LearnerMemoryStore.enqueue()`; `logic.py` records ambiguity candidates | `claim_next_job()`; `resolve_pending_choice()` | async writer; next-turn clarification resolution |
| Cross-session archive-derived candidate | Synthetic `TopicMemory(memory_id='archive_<id>')` from prior `episode_json` | Source archive persists; candidate is transient | `StudentMemoryService._archive_candidates()`; older `cross_session.py::recall_learning_context()` | Conversation archive save, not a separate memory write | list/load up to 24 prior conversations | structured resolver; older compatibility recall |
| Full conversation archive | SQLite `conversations(conversation_id,user_id,title,episode_json,...)` | Persistent | `app/memory/conversation_archive.py::LocalConversationStore` | `create()`, `save()` | `load()`, `list_for_user()`, `list_updated_between()` | UI history, cross-session recall, learner memory fallback, quality-QA batch |
| Agent trajectory / audit trace | fields in `PrototypeTurnRecord`: classification, contract, observations, trace, audits, timings, retrieval metadata | In conversation `episode_json` | `logic.py::PrototypeTurnRecord` | `_run_turn_impl()` record construction | UI and offline diagnostics | audit/debug/evaluation; not retrieved as procedural memory directly |
| Agent experiential memory / quality Q&A | Dedicated Chroma collection `quality_qa`; metadata includes source conversation, quality, subject/stage, hash, hit count | Persistent Chroma + SQLite version/pipeline state | `app/data_pipeline/quality_qa_pipeline.py`, `prototypes/.../qa_cache.py` | nightly `QualityQaPipeline.run()` → `replace_source()` → `store()`; explicit seed/manual `store()` | `qa_cache.retrieve()` via `search_quality_qa` | teacher `retrieval_reference`; never direct answer |
| Curated textbook/exercise knowledge | Separate Chroma collections; bge-m3 + BM25 + RRF results | Persistent external index | `rag_hook.py`, `llm-test/rag_retriever.py` | ingestion outside turn path | `retrieve_local_context()` / `search_local_knowledge` | teacher generation reference |
| Pedagogical GraphRAG knowledge | Local JSON nodes/edges in `data/sample/pedagogical_graph.json` plus simpler JSON graph files | Repository-persistent, read-only at runtime | `app/retrieval/graph_retriever.py`, `context_provider.py` | no runtime write | `retrieve_evidence()`, `get_prerequisites()`, etc. | legacy diagnosis/policy/teach path when enabled |
| External web knowledge | Tavily result snippets | Not persisted by this project in dynamic memory | `rag_hook.py::retrieve_web_context()` | none | guarded `search_web` | current turn only |
| Legacy in-process conversation state | `ConversationState` dictionary keyed by synthetic student/conversation key | Process lifetime only | `app/memory/conversation_state.py` | `InMemoryConversationStore.save/reset()` | `get()` | legacy diagnose/log nodes |
| Legacy episode state | `EpisodeState` JSON keyed `(student_id, conversation_id)`, last 8 recent turns | JSON file | `app/memory/episode_state.py` | `JsonEpisodeStateStore.save/reset()` | `get/find/list_conversations()` | legacy context resolver/diagnosis/UI/API |
| Legacy mastery/misconception profile | `student_profiles.json` → `ConceptMastery` per concept | JSON file | `app/memory/student_state.py` | `update_mastery_after_turn()`, `record_misconception()` | `get_concept_mastery()`, `recommend_review_concepts()` | legacy retrieval/policy/prompt/API |
| Legacy learning event history | `LearningEvent` JSONL | Append-only persistent file | `app/memory/learning_history.py` | `append_learning_event()` | `get_*`, `summarize_learning_history()` | aggregation/research; not live retrieval in 8502 |
| Legacy aggregated profile | `student_profiles_v2.json` → concept/skill mastery, weaknesses | JSON file | `app/memory/profile_updater.py` | `update_profile_from_event()` | `summarize_profile()` | legacy context resolver and API |
| Research turn log | `TurnLog` JSONL with state/retrieval/evidence | Persistent log | `linear_equation_adapter.py::log_node()` | append `logs/turn_logs.jsonl` | evaluator/analysis tools | research analysis, not live memory |
| KV performance cache | llama-server GPU/RAM KV and optional SSD `kv_*.bin` snapshots | Volatile GPU/RAM; optional SSD | `app/llm/kv_cache.py` | `KvCacheLease.save()` | `restore()` | latency/context reuse only; carries opaque model KV, not an auditable fact store |
| Retrieval result cache | in-process `OrderedDict`, positive and negative quality-QA results keyed by query hash + library version + filters | Process lifetime, TTL 3600s default | `qa_cache.py` | `retrieve()` | `retrieve()` | quality-QA lookup optimization |

### 3.1 Three source classes

**A. Curated educational knowledge.** Textbook/exercise Chroma and `data/sample/pedagogical_graph.json` are curated knowledge. Tavily is external knowledge. They are not written by student-memory code. The legacy graph mixes curriculum concepts with generic misconception/strategy nodes, but these are authored corpus nodes, not observations about a particular student.

**B. Student-specific memory.** 8502 `GlobalStudentProfile`/`TopicMemory`, conversation archive, and legacy mastery/events/episodes are user- or student-scoped. The code separates them from textbook Chroma, but two different student-memory generations coexist without a shared taxonomy or migration contract.

**C. Agent experiential memory.** `quality_qa` is explicitly historical answer experience (`qa_cache.py` module docstring and `QualityQaCandidate`). It is logically isolated in a dedicated Chroma collection. It is not a reusable tool sequence, reasoning trajectory, teaching plan, or success/failure policy; no procedural-memory representation exists.

The principal semantic ambiguity is naming: quality-QA is called a “knowledge base/cache,” while it is actually curated agent experience derived from student conversations. It must not be admitted through the future student-memory Admission Controller. GraphRAG likewise stays outside that controller unless a future dynamic graph write path is intentionally added.

## 4. Current memory lifecycle

Legend: **implemented**, **partial**, **absent**, **LLM prompt**, **hard-coded**, **threshold**, **scheduled**, **manual config**.

### 4.1 8502 topic/student LTM

```text
raw successful K9 turn
  → candidate extraction: _extract_and_write() [implemented; hard-coded regex/heuristics]
  → admission: enqueue_turn() scope/ambiguity/turn-id gate only [partial]
  → store: upsert_topic_memory() [implemented]
  → index: SQL composite index + unique key [implemented; no vector index]
  → retrieve: resolve() list 24 + archive 24, score [implemented; threshold]
  → context injection: prompt_payload() [implemented; bounded]
  → update: merge by subject/domain/normalized_topic [implemented]
  → forget/delete: [absent]
  → consolidate/summarize: field replacement/list dedup only [partial; no semantic consolidation]
```

Candidate fields are not raw utterances but deterministic synthesized topic records. Global facts come directly from regex/pattern matches in the student message. Topic progress may be verifier-backed, explicit difficulty, or a generic inference. Outbox retry is hard-coded to three attempts (`fail_job()`), then terminal `failed`.

### 4.2 Conversation archive and STM

```text
raw student + accepted teacher response + metadata
  → admission: every successful UI turn [no semantic admission]
  → store/index: one SQLite episode JSON per user/conversation
  → retrieve/inject: load selected conversation; active working slice only
  → summarize: token-threshold-triggered LLM JSON summary; deterministic fallback
  → update: overwrite complete episode JSON
  → delete/retention: no conversation-delete API in LocalConversationStore
```

The archive is the source of truth and is intentionally not pruned by context compaction.

### 4.3 Quality-QA agent experience

```text
persisted conversation archive
  → candidate extraction: DeepSeekQualityQaExtractor [implemented; LLM prompt]
  → admission: reusable + no sensitive data + score>=0.85 + injection filters
               + exclude answers that used quality_qa [implemented; hybrid]
  → store/index: Chroma cosine embeddings [implemented]
  → dedup: content hash + semantic >=0.94 + number compatibility [implemented]
  → retrieve: route flag + similarity >=0.78 + subject/stage + top_k=3 [implemented]
  → context injection: <=3000 chars, “reference only” [implemented]
  → update/delete: replace by source; stale entries deleted [implemented]
  → forget: source replacement only; no age/utility decay [partial]
  → schedule: incremental watermark; launcher’s nightly UTC window [scheduled/manual flag]
```

### 4.4 Legacy mastery/events

```text
checker/context result
  → candidate/admission: memory_update_node maps every turn to a step_result
  → mastery update: fixed deltas, clamp [hard-coded]
  → misconception admission: only deterministic incorrect step; repeated twice → common
  → event generation: _generate_learning_events() branch heuristics
  → persistent JSON/JSONL + aggregate profile
  → read: every legacy turn for retrieval/policy/context
  → forget/delete: explicit reset APIs only; no decay/consolidation schedule
```

Notably, `practice_requested` is sent to `update_mastery_after_turn()` but has no delta; attempts still increment. Ordinary help is `help_requested=-0.05`, so a request for help becomes a persistent mastery penalty.

### 4.5 Cache lifecycle

- Quality-QA result cache: both hits and empty results, TTL 3600s, max 2048 entries, LRU eviction; library version change clears/invalidates it.
- KV SSD cache: feature flag, keep latest one version by default, high-water 80 GiB → low-water 70 GiB cleanup, hard budget 100 GiB; inactive oldest/incompatible snapshots are removed. This is the only automatic age/size-like pruning, and it is performance cache management—not semantic forgetting.

## 5. Admission audit

### 5.1 All persistent dynamic write paths

```text
8502 UI accepted turn
 ├─ LocalConversationStore.create/save
 │    └─ raw student/teacher + metadata → full episode_json
 ├─ StudentMemoryService.enqueue_turn
 │    ├─ reject non-K9
 │    ├─ reject ambiguous memory resolution
 │    └─ durable outbox
 │         └─ _extract_and_write
 │              ├─ regex global profile facts → update_global_profile
 │              └─ synthesized TopicMemory → upsert_topic_memory
 └─ optional KvCacheLease.save → opaque SSD KV snapshot

Nightly/archive path
 LocalConversationStore.list_updated_between
   → DeepSeekQualityQaExtractor.extract (LLM candidates)
   → QualityQaPipeline._accepted_entries (admission)
   → qa_cache.replace_source/store (Chroma agent experience)

Legacy 8501 turn
 ├─ memory_update_node → student_profiles.json
 └─ log_node
      ├─ append_learning_event → learning_events.jsonl
      ├─ update_profile_from_event → student_profiles_v2.json
      ├─ JsonEpisodeStateStore.save → episode_states.json
      └─ append TurnLog → logs/turn_logs.jsonl
```

### 5.2 8502 topic-memory admission answers

1. **Candidate generator:** `StudentMemoryService._extract_and_write()` after `enqueue_turn()`; not the agent loop or teacher LLM.
2. **Candidate form:** extracted/synthesized atomic-ish structured fields, not raw conversation. However `problem_context` and `requested_target` can carry model/classifier-derived or transient text.
3. **Save condition:** scope=`k9`, resolution not ambiguous, turn ID >=1, subject present, normalized topic non-empty. There is no utility/novelty/stability gate.
4. **Classification:** hard-coded heuristic with no explicit substantive admission; operationally “almost every eligible K9 turn stored.”
5. **Checks:**
   - future utility: no;
   - factual confidence: only assigns fixed confidence from verifier/difficulty/generic branch; no admission threshold;
   - novelty: only unique SQL key and merge; no candidate novelty rejection;
   - temporal validity: no;
   - memory type: coarse global vs topic only;
   - stability: no;
   - provenance: conversation ID/turn/kind retained, but not source class (user stated / external / inference / behavior) as a typed field.
6. **Agent inference entering memory:** yes indirectly. `classification.subject` and `TeachingContract.requested_target` can be model-produced on ambiguous routes and directly shape subject/topic/next step. Generic progress is code inference. The rejected draft answer itself is not written into `TopicMemory`.
7. **Retrieved knowledge feedback:** textbook/GraphRAG text is not included in the learner-memory payload, so it is not directly copied back. Quality-QA-derived answers are explicitly excluded from the quality-QA nightly extractor, but not specially marked in student topic memory; the latter stores only synthesized progress, so verbatim feedback is unlikely.
8. **Risks:** hallucination/inference persistence, duplicate near-topics, contradictions, and transient-to-permanent writes are all possible. Unique key prevents exact normalized-topic duplication only.
9. **Candidate/pending state:** durable outbox is a processing queue, not a reviewable admission candidate state. Ambiguity pending state belongs to read resolution, not write admission.
10. **Source distinction:** absent as a first-class enum. `evidence.kind` distinguishes verified/generic/difficulty but cannot represent explicit user fact vs verified external fact vs model inference vs observed behavior comprehensively.

### 5.3 Merge and contamination behavior

`LearnerMemoryStore.upsert_topic_memory()`:

- keeps only 20 evidence records and 12 source conversation IDs;
- de-duplicates evidence by `(conversation_id, turn_id, kind)`;
- de-duplicates difficulties by exact string;
- replaces non-empty problem/summary/progress/next_step with incoming values;
- sets confidence to `max(current, incoming)`;
- increments version.

It does not detect contradiction, close stale topics, lower confidence, or preserve validity intervals. A single high-confidence wrong classification can dominate future versions.

### 5.4 Quality-QA admission quality

This is the strongest current admission design: LLM candidate judgement plus deterministic quality threshold, privacy, prompt-injection, self-feedback exclusion, content hash, semantic duplicate threshold, source replacement, and library versioning. Remaining gaps are independent factual verification, explicit reviewer approval, temporal validity, and utility feedback. `quality_score` is model judgement, not an external correctness proof.

## 6. Retrieval/access audit

### 6.1 8502 student memory

- Global profile: loaded every turn when `memory_identity` exists (`logic.py::_run_turn_impl()` → `load_profile()`).
- Topic memory: `resolve()` executes after classification; it returns NONE immediately unless the message has a continuation marker or explicit topic.
- Candidate source: up to 24 profile memories plus up to 24 archive-derived memories, then deterministic dedup.
- Score: domain +0.45, continuation +0.25, explicit topic up to +0.5 or similarity contribution, confidence up to +0.1. Eligibility >=0.68. Unique if no runner-up, or top >=0.85 and margin >=0.15.
- Query/template: fixed domain/topic extraction and `SequenceMatcher`; not embedding retrieval.
- NOOP: implemented (`MemoryResolutionStatus.NONE`).
- Ambiguity: implemented deterministic clarification and next-turn `resolve_pending_choice()`.
- Re-retrieve when stuck: absent. Resolve runs once per turn; no outcome-triggered second access.
- Stage-aware strategy: `stage` and `is_new_problem` are accepted then discarded in `resolve()`.
- Repeated query: likely same candidates; recency tie-break uses `updated_at`, while `touch_memory()` updates only `last_accessed_at`, so access recency does not change ranking.
- Consolidate/forget: absent.

### 6.2 8502 knowledge and experiential retrieval

- Fast route retrieves textbook/exercise for new problems or concept requests with known subject/stage; continuation normally skips it (`run_fast_plan()`).
- Fast parameters are fixed: query=`student_message[:800]`, local top-k 5, sources textbook+exercise; quality-QA top-k 3.
- Bounded agent path can choose tool/query/parameters within schemas, but registry overwrites subject/stage with verified classification.
- Per-turn signatures prevent duplicate local/quality calls; local and quality search are each budgeted to one; web is at most one and requires successful local search plus a concrete gap.
- There is no general “stuck” detector that re-retrieves with another query/source. A provider failure may trigger one deterministic fallback local search; that is error recovery, not adaptive memory access.
- Graph hop in legacy `retrieve_evidence()` defaults to prerequisite depth 2; simple `get_prerequisites()` defaults depth 1.
- Retrieved results include references and similarity in turn metadata. Student topic memory includes confidence and source conversation IDs but not per-field provenance/confidence in injected payload.

### 6.3 Can a `MemoryAccessController` wrap existing backends?

**Yes, with adapters and without backend rewrite**, because raw backends already expose stable operations:

- student memory: `load_profile()`, `resolve()`, `list_topic_memories()`, `touch_memory()`;
- archive: `load/list_for_user()`;
- quality-QA: `qa_cache.retrieve()`;
- local knowledge: `retrieve_local_context()`;
- graph: `GraphContextProvider` / `retrieve_evidence()`.

The controller should sit before these calls and return an explicit access decision (NOOP/source/query/limits/retry/consolidate/forget proposal). It must not absorb storage/index implementation. The first implementation should wrap 8502 student memory and quality-QA/local retrieval separately; legacy graph/mastery can use compatibility adapters. GraphRAG remains curated knowledge and is not governed by student-memory admission.

## 7. STM/context audit

### 7.1 Persistent LTM versus active context

- Persistent LTM/archive: full `PrototypeEpisode.turns` in SQLite, global/topic learner memory, quality-QA Chroma.
- Active STM: problem/subject/stage/previous verified step, rolling summary, and only turns after `context_compaction.compacted_turns`.
- Performance state: KV cache. It may encode prior prompt state but must never be treated as auditable semantic memory.

### 7.2 Growth and compaction

`logic.py::_history_payload()` calls `working_history_payload()`; therefore the current code no longer always injects all turns after compaction, despite older docs stating full history payload.

`ContextCompactionSettings.from_env()` defaults:

- enabled;
- context window 524,288 tokens (backend-dependent override);
- trigger 80%; target 55%;
- retain at least 12 recent complete turns when possible;
- reserve 8,192 output tokens;
- summary output max 4,096 tokens.

Token size is an estimate of UTF-8 bytes / 2, not the backend tokenizer. At trigger, oldest eligible turns are summarized through LLM JSON into seven semantic categories. If summarization fails, `_fallback_summary()` records bounded excerpts for the latest 32 continuity notes. If still over the native window, additional recent turns are folded until fit or state becomes `hard_limit`.

### 7.3 Filtering/injection risks

- Query-aware: compaction preserves semantic categories but does not select old facts based on the current query.
- Provenance/confidence: each recent turn carries `evidence_level`; summary list items do not carry typed per-item provenance/confidence, only prompt instructions telling the summarizer to preserve certainty.
- Duplicate injection: a fact can appear in `problem_context`, rolling summary, topic memory, and recent turns. No cross-source dedup exists.
- Summary poisoning: archived text is marked untrusted in prompts, but summary generation is still LLM-based and no deterministic factual reconciliation checks the result.
- Archive and summary can diverge; no re-materialization validation compares summary claims with source turns.

## 8. Existing reward and feedback signals

### 8.1 Online/runtime observable signals

- deterministic student step status: `VerificationResult.status` / legacy `student_step_validity`;
- tool success/error/elapsed time: `ToolResult` and `AgentTraceEvent`;
- boundary pass/fail/repair/fallback and warnings: `PrototypeTurnRecord`;
- retrieval hit count/top similarity/source/cache hit;
- generation token/usage metadata and stage/backend latency;
- repeated verified errors and explicit difficulty in stored evidence;
- quality-QA hit count (usage only, not utility/outcome).

These are observations, not a single trusted reward, and none currently updates an access/admission policy.

### 8.2 Offline evaluation signals

- `app/evaluation/evaluator.py`: golden deterministic behavior checks and per-case pass/fail.
- `app/evaluation/hybrid_scorer.py::HybridConversationScore`: teacher quality pass rate, `task_success`, solution progress, misconception resolved, understanding gain, confidence change, teacher turns, hint exposure, termination reason.
- runtime boundary audit is compliance checking, not answer correctness.

The hybrid measures depend on a deterministic student simulator and offline traces. They are suitable for ablation evaluation, not direct online reward.

### 8.3 Future mapping

- `task_reward`: verified solve/step outcomes plus offline task success/solution progress; do not use LLM boundary pass alone.
- `efficiency_penalty`: teacher turns, tool calls/repeats, latency, prompt/output tokens, hint exposure, unnecessary retrieval.
- `memory_quality_feedback`: user correction (needs a new explicit typed signal), contradiction discoveries, retrieved-memory use followed by verified progress/regression, clarification frequency, and stale/duplicate detection.

## 9. Hard-coded heuristics inventory

| Location / symbol | Heuristic |
|---|---|
| `learner_memory.py::_CONTINUATION_MARKERS`, `_DOMAIN_CUES`, `_PREFERENCE_CUES` | Chinese lexical routing/extraction |
| `StudentMemoryService.resolve()` | list 24; eligible 0.68; unique top 0.85 and margin 0.15; max 5 candidates |
| `resolve_pending_choice()` | title similarity 0.72; winner margin 0.15; 80-char abandon heuristic |
| `_score_candidates()` | domain .45, continuation .25, topic up to .5, confidence max .1 |
| `_extract_and_write()` | confidence .95 verified, .85 explicit difficulty, .75 generic; all active |
| `upsert_topic_memory()` | evidence 20, sources 12, difficulties 8; max confidence |
| `LearnerMemoryStore.fail_job()` | three attempts |
| `context_compaction.py::ContextCompactionSettings` | 80% trigger, 55% target, 12 recent turns, 8192 reserve, 4096 summary |
| `_fallback_summary()` | latest 32 continuity notes; each side 120 chars |
| `fast_plan.py::run_fast_plan()` | local top-k 5, quality top-k 3, fixed query and source set |
| `orchestration_tools.py` schemas/runtime | local/quality/web at most once; local/quality 3s, web 12s |
| `rag_hook.py` | top-k 5; local threshold .65; max 2400 chars; excerpt 600 chars |
| `qa_cache.py` | lookup .78; duplicate .94; top-k 3; TTL 3600; max 2048; text 3000 chars |
| `QualityQaPipeline` | min quality .85; last 48 turns; input max 28,000 chars; at most 8 candidates |
| `student_state.py` | default mastery .5; deltas +.1/+.2/-.1/-.05; review threshold .6; keep 5 errors; one-day review |
| `record_misconception()` | promote after count >=2 |
| `profile_updater.py::update_profile_from_event()` | separate fixed concept/skill deltas, default .5 |
| `LocalPedagogicalGraphRetriever` | hop depth 2; mastery bands .3/.7; fixed evidence scores |
| `episode_state.py`, legacy `log_node()` | retain last 8 recent turns; list 20 episodes |
| `kv_cache.py::KvCacheSettings` | one version; 100/80/70 GiB budget watermarks |

## 10. Failure and contamination risks

1. **Generic-turn persistence:** unverified K9 turns become persistent `learning_turn` facts.
2. **Inference provenance loss:** model classification/contract fields become topic metadata without source typing.
3. **Confidence ratchet:** merge takes max confidence, so later uncertainty cannot correct earlier confidence.
4. **Contradictions:** no competing-fact representation, supersession, validity interval, or resolver.
5. **Near duplicates:** normalized topic unique key is too coarse for synonyms and too brittle for slightly different titles.
6. **Transient state permanence:** current requested target and active problem become next-step/topic memory without stability horizon.
7. **No semantic deletion:** student topic memory has a `deleted` read filter but no delete/forget writer; privacy/rectification is incomplete.
8. **Archive amplification:** archive-derived candidates can restore an older full problem even if no curated topic-memory row exists.
9. **Cross-source repeated injection:** archive summary, topic memory, active problem, recent turns, and quality-QA can repeat similar content.
10. **Summary distortion:** LLM compaction lacks source-linked claim validation.
11. **Quality-QA correctness:** admission score is model judgement; no independent verifier for general subjects.
12. **Legacy double accounting:** mastery is updated once in `student_state.py` and again through event aggregation in a separate profile model; consumers may disagree.
13. **Help-as-failure:** legacy `help_requested=-0.05` penalizes ordinary help-seeking.
14. **Reward leakage risk:** using offline simulator outcomes directly as online policy reward would optimize to the simulator.
15. **KV opacity:** persisted KV may retain prompt material outside semantic deletion workflows; operational privacy policy must treat it separately.

## 11. Proposed Memory v2 module boundaries

### 11.1 Recommended minimal-intrusion architecture

```text
Interaction / verified tool evidence / explicit user facts
  ↓
CandidateExtractor
  ├─ typed candidate payload
  ├─ source_kind: user_stated | externally_verified | model_inference | observed_behavior
  ├─ provenance refs + confidence + temporal scope
  └─ target_memory_class: student | agent_experience
  ↓
MemoryAdmissionController  (owns all dynamic persistent semantic writes)
  ├─ ADMIT | REJECT | PENDING | SUPERSEDE
  ├─ reason codes; shadow decision record
  └─ future utility / confidence / novelty / validity / stability / contradiction
  ↓
Existing backend adapters
  ├─ LearnerMemoryStore (student memory)
  ├─ LocalConversationStore (archive policy remains separate)
  └─ quality_qa repository (agent-experience-specific admission policy)
  ↓
MemoryAccessController  (owns use strategy, not storage)
  ├─ NOOP or source selection
  ├─ query, limits, thresholds, retry budget
  ├─ re-retrieve/consolidate/forget proposal
  └─ shadow decision + observed outcome
  ↓
Existing raw retrieval backends
  ↓
bounded, provenance-preserving Agent Context
```

The archive write is operational conversation persistence and should not be blocked by semantic admission. Semantic derivatives from it must pass admission. KV storage remains a cache plane with its own retention/privacy policy.

### 11.2 Alternatives considered

1. **Recommended: controller + adapters around existing service seams.** Lowest risk, preserves SQLite/Chroma/graph implementations and allows per-module ablation.
2. **Replace all memory with a unified vector/graph store.** Rejected for Phase 2: high migration risk, conflates source classes, violates minimal intrusion.
3. **Prompt-only controller inside the agent loop.** Rejected: cannot guarantee write ownership, deterministic NOOP, provenance, or shadow decisions.

### 11.3 GraphRAG boundary

Current GraphRAG is **curated educational knowledge**, not dynamic agent memory or student memory. Its authored misconception nodes are generic curricular teaching knowledge, not a particular learner’s misconception history. It remains a read backend under access strategy and does not pass through student-memory admission. Only a future feature that writes dynamic observations into the graph would introduce an admission boundary.

### 11.4 Ablation flags

Required flags:

```text
memory_controller.enabled
memory_controller.shadow_mode
memory_admission.enabled
memory_admission.shadow_mode
```

Semantics:

- disabled: exact current behavior;
- shadow: current behavior executes; controller logs decisions/differences only;
- enabled: controller decision is authoritative, with deterministic fallback to baseline on controller failure;
- admission and access flags are independent.

Required matrix:

| Run | Access controller | Admission controller |
|---|---:|---:|
| Baseline current memory | off | off |
| Baseline + MemCon | on | off |
| Baseline + Admission | off | on |
| Baseline + both | on | on |

Quality-QA and GraphRAG flags must be recorded separately so their existing on/off state does not confound these four cells.

## 12. Exact integration points

### 12.1 Admission

1. `prototypes/llm_led_teacher/logic.py::_run_turn_impl()` around the call to `StudentMemoryService.enqueue_turn()`: construct a typed candidate envelope and call the admission facade. This is the live 8502 semantic write boundary.
2. `app/memory/learner_memory.py::StudentMemoryService.enqueue_turn()`: keep as durable scheduling adapter; require an admitted decision/token when admission is enabled. Shadow mode may enqueue unchanged.
3. `StudentMemoryService._extract_and_write()`: split extraction from write in Phase 2, but preserve existing deterministic extraction as baseline implementation.
4. `LearnerMemoryStore.upsert_topic_memory()` / `update_global_profile()`: raw backend writes; do not put policy here beyond validating authorization/shape.
5. `app/data_pipeline/quality_qa_pipeline.py::QualityQaPipeline._accepted_entries()`: retain a distinct agent-experience admission adapter; do not route these entries to student memory.
6. `app/agent/linear_equation_adapter.py::memory_update_node()` and `log_node()`: legacy compatibility adapter for mastery/event candidates if the legacy path is included in experiments.

### 12.2 Access

1. `logic.py::_run_turn_impl()` before `StudentMemoryService.resolve()`: controller decides NOOP/profile-only/topic/archive strategy and parameters.
2. `app/memory/learner_memory.py::StudentMemoryService.resolve()`: baseline student-memory retriever adapter; expose current hard-coded values as an input policy object only after shadow parity is proven.
3. `prototypes/llm_led_teacher/fast_plan.py::run_fast_plan()`: controller seam for local/quality-QA source choice, query and top-k.
4. `prototypes/llm_led_teacher/orchestrator.py::MultisubjectAgentLoop.run()`: controller should constrain/authorize agent-selected retrieval, not be implemented as another unconstrained tool.
5. `prototypes/llm_led_teacher/orchestration_tools.py::OrchestrationToolRegistry.execute()`: enforcement point for budgets, verified filters and decision IDs.
6. `rag_hook.py::retrieve_local_context()`, `qa_cache.py::retrieve()`, and `GraphContextProvider`: unchanged raw backend adapters.
7. `logic.py::_teacher_payload()`: final injection boundary; attach source/provenance/confidence and deduplicate across selected memory blocks.

### 12.3 Context and outcome observation

- `context_compaction.py::maybe_compact_context()`: controller may propose consolidation timing, while this module keeps raw summarization mechanics.
- `PrototypeTurnRecord`: add future shadow/access/admission decision records and outcome linkage without storing private reasoning.
- `logic.py::_run_turn_impl()` after verifier/audit/record completion: outcome observer receives verified status, tool metrics, tokens and latency; it must not invent a scalar reward.

## 13. Recommended implementation order

1. Freeze baseline fixtures for all four information classes and record exact current decisions.
2. Define typed source/provenance/candidate/decision schemas and separate archive/cache/semantic-memory policies.
3. Add read-only shadow logging for Admission Controller at the 8502 enqueue seam; no behavior changes.
4. Add access shadow decisions around student-memory resolve; prove baseline parity when policy chooses baseline.
5. Add deterministic contradiction/duplicate/temporal tests and a reviewable PENDING state before authoritative admission.
6. Enable admission alone behind flags; evaluate Baseline vs Admission.
7. Add access strategies (NOOP, source/query/top-k, one re-retrieve) behind flags; evaluate Baseline vs MemCon.
8. Run the full 2×2 ablation matrix, including memory contamination and efficiency metrics.
9. Only after parity and rollback tests, consider consolidation/forget actions; keep backend migration out of scope.

This order is a recommendation only; this audit does not implement Phase 2.

## 14. Open technical risks and Phase 2 blockers

1. No canonical schema identifies source kind and target memory class across 8502 topic memory, quality-QA and legacy events.
2. No formal correction/supersession/delete semantics for student memory, including user-requested rectification and KV/cache retention.
3. No contradiction policy or confidence-decrease model.
4. No reviewable semantic candidate state; outbox status is operational, not admission status.
5. No reliable online reward or explicit user-correction event contract.
6. Quality-QA and student-memory experiments could be confounded unless all feature states and library versions are logged.
7. Legacy and current paths have different identity keys, schemas and mastery semantics; scope for Phase 2 must explicitly name which runtime is authoritative.
8. Summary items lack claim-level provenance, and repeated facts across context sources are not deduplicated.
9. Backends have inconsistent deletion capability: quality-QA source replacement can delete; learner memory cannot; conversation archive has no delete API.
10. Existing dynamic data may already contain contaminated/conflicting records; migration/quarantine policy is undefined.

## 15. Tests/evaluations required before refactor

### 15.1 Baseline characterization

- Golden call-chain tests for every live persistent write path, including skip reasons.
- Snapshot fixtures for `TopicMemory`, global profile, outbox, archive record, quality-QA metadata and legacy mastery/events.
- A source-separation test proving textbook/GraphRAG/web results cannot enter student memory verbatim.
- Current-decision parity tests for resolve NONE/UNIQUE/AMBIGUOUS and quality-QA routing.

### 15.2 Admission safety

- explicit user fact vs verified fact vs model inference vs observed behavior provenance tests;
- unverified generic K9 turn rejection/pending tests;
- duplicate, near-duplicate, contradiction, supersession, stale/transient and temporal-validity tests;
- hallucinated classifier/contract field contamination tests;
- quality-QA self-feedback, privacy, injection and independent correctness tests;
- idempotent retry tests proving outbox retries do not create duplicate semantic records;
- user correction/delete tests, including archive derivatives and caches.

### 15.3 Access/controller behavior

- deterministic NOOP tests;
- task-stage/source/query/top-k policy tests;
- repeated query and stuck-triggered re-retrieval budget tests;
- backend failure parity/fallback tests;
- injection provenance, confidence, token bound and cross-source dedup tests;
- shadow mode “no behavior change” tests and decision-diff audit tests.

### 15.4 Context and evaluation

- tokenizer-vs-estimator stress tests near the real backend limit;
- summary factual fidelity/source-link tests and prompt-injection tests;
- full archive preservation and hard-limit behavior tests;
- offline task-success/teacher-quality metrics plus efficiency (tokens, latency, calls, turns);
- memory-specific evaluation: precision of admitted items, contradiction rate, stale-memory rate, clarification rate, retrieval utility and contamination rate;
- four-cell ablation with fixed seeds/scenarios and recorded GraphRAG/quality-QA/library versions.

Existing relevant regression suites include `tests/behavior/test_structured_student_memory.py`, `tests/prototypes/test_context_compaction.py`, `tests/test_quality_qa_pipeline.py`, `tests/prototypes/test_qa_cache.py`, `tests/test_cross_session_memory.py`, `tests/test_three_layer_memory.py`, `tests/test_mastery.py`, `tests/test_pedagogical_rag.py`, `tests/test_kv_cache.py`, and the offline evaluation scripts. They cover mechanics but not the missing admission semantics above.

## Final verdict

```text
READY_FOR_PHASE_2 = NO
```

必须先解决：

1. 定义统一的 dynamic semantic-memory candidate/provenance/type/decision schema；
2. 明确 8502 为首要接入范围，并把 legacy/quality-QA/GraphRAG 作为独立 adapter/source 处理；
3. 定义 correction、contradiction、supersession、delete/forget 和 temporal validity 语义；
4. 增加 shadow-mode decision record 与 baseline parity/contamination 测试；
5. 定义可观测 feedback contract，并接受当前没有可靠 online reward 的事实，禁止临时拼接伪 reward。

完成这些设计与测试前，不应实现 MemCon-style policy 或让 Admission Controller 接管真实写入。
