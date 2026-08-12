# 8502 Multisubject Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace the fixed subject/RAG orchestration inside 8502 with a bounded agent loop that routes nine K9 subjects, uses local knowledge before optional web search, and preserves every existing login, conversation, context-meter, token, cache, verifier, and audit contract.

**Architecture:** Keep streamlit_app.py and its authentication/rendering shell intact. Add focused classification, tool, and loop modules under prototypes/llm_led_teacher; make logic.run_turn delegate only routing and retrieval decisions to the new loop, then reuse the existing teacher generation, deterministic audit, repair, persistence, and usage aggregation paths.

**Tech Stack:** Python 3.10, Pydantic v2, Streamlit, existing DeepSeekClient JSON mode, Chroma/bge-m3/BM25/RRF, pytest.

## Global Constraints

- 8502 remains the only user-facing entrypoint; do not redirect users to 8503.
- Supported subjects are 数学、语文、英语、物理、化学、生物、历史、地理、道法.
- Off-topic requests never call local RAG or Tavily; they receive a brief answer and a K9 redirect.
- A K9 web search is legal only after a successful local search in the same turn, with one web call maximum.
- Existing math verification remains mandatory where supported and cannot be overridden by model decisions.
- Existing account databases, conversation ownership checks, pending-message recovery, context meter, token display, and KV/prompt-cache metrics remain unchanged.
- All LLM calls continue through app/llm/deepseek_client.py.
- No new provider dependency, model weight, vector database, sub-agent, or parallel delegation is introduced.
- Use TDD for every behavior change and stage only the files named by each task when committing.

---

## File Structure

- Create prototypes/llm_led_teacher/orchestration_models.py: request classification, loop decisions, tool observations, and serializable trace contracts.
- Create prototypes/llm_led_teacher/orchestration_tools.py: guarded local/web/skill tools and subject metadata mapping.
- Create prototypes/llm_led_teacher/orchestrator.py: mandatory classification plus bounded action loop.
- Modify prototypes/llm_led_teacher/rag_hook.py: expose local-only and web-only adapters; retain warmup compatibility.
- Modify llm-test/rag_retriever.py: expose a public local-hit function with raw scoring diagnostics and no online side effects.
- Modify prototypes/llm_led_teacher/logic.py: persist active subject/stage, invoke the loop, reuse generation/audit/repair, and aggregate all LLM usage.
- Modify prototypes/llm_led_teacher/live_progress.py: format real agent events without exposing private reasoning.
- Modify prototypes/llm_led_teacher/streamlit_app.py: render agent trace while preserving the existing authentication and metrics functions.
- Modify app/memory/conversation_archive.py only if required for schema-version compatibility; do not change ownership queries or database paths.
- Create tests/prototypes/test_llm_led_orchestration_models.py.
- Create tests/prototypes/test_llm_led_orchestration_tools.py.
- Create tests/prototypes/test_llm_led_orchestrator.py.
- Modify tests/prototypes/test_llm_led_turn_contract.py.
- Modify tests/prototypes/test_llm_led_streaming.py.
- Modify tests/prototypes/test_llm_led_auth_ui.py.
- Modify tests/test_conversation_archive.py.

---

### Task 1: Multisubject Contracts and Backward-Compatible Episode State

**Files:**
- Create: prototypes/llm_led_teacher/orchestration_models.py
- Modify: prototypes/llm_led_teacher/logic.py:24-90
- Create: tests/prototypes/test_llm_led_orchestration_models.py
- Modify: tests/test_conversation_archive.py

**Interfaces:**
- Produces: Subject(str, Enum), SchoolStage(str, Enum), RequestScope(str, Enum), AnswerMode(str, Enum).
- Produces: RequestClassification(scope, subject, stage, is_new_problem, answer_mode, reason).
- Produces: AgentActionKind, AgentDecision, AgentTraceEvent, AgentOrchestrationResult.
- Extends PrototypeEpisode with active_subject: str = "" and active_stage: str = "".
- Extends PrototypeTurnRecord with classification, agent_trace, loaded_skills, and agent_termination_reason, all with safe defaults.

- [ ] **Step 1: Write failing enum and validation tests**

~~~python
def test_k9_classification_requires_supported_subject():
    with pytest.raises(ValidationError):
        RequestClassification(
            scope="k9",
            subject="计算机",
            stage="初中",
            is_new_problem=True,
            answer_mode="guided",
            reason="invalid subject",
        )


def test_off_topic_classification_forces_brief_redirect():
    item = RequestClassification(
        scope="off_topic",
        subject=None,
        stage="未知",
        is_new_problem=True,
        answer_mode="brief_redirect",
        reason="not K9",
    )
    assert item.subject is None
~~~

- [ ] **Step 2: Run the new contract tests and verify RED**

Run: python3 -m pytest tests/prototypes/test_llm_led_orchestration_models.py -q

Expected: collection/import failure because orchestration_models.py does not exist.

- [ ] **Step 3: Implement the contracts with Pydantic shape validation**

Use exact subject values and validators:

~~~python
class Subject(str, Enum):
    MATH = "数学"
    CHINESE = "语文"
    ENGLISH = "英语"
    PHYSICS = "物理"
    CHEMISTRY = "化学"
    BIOLOGY = "生物"
    HISTORY = "历史"
    GEOGRAPHY = "地理"
    POLITICS = "道法"


class RequestClassification(BaseModel):
    scope: RequestScope
    subject: Subject | None = None
    stage: SchoolStage = SchoolStage.UNKNOWN
    is_new_problem: bool
    answer_mode: AnswerMode
    reason: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_scope(self):
        if self.scope is RequestScope.K9 and self.subject is None:
            raise ValueError("k9 classification requires subject")
        if self.scope is RequestScope.OFF_TOPIC:
            if self.subject is not None or self.answer_mode is not AnswerMode.BRIEF_REDIRECT:
                raise ValueError("off_topic requires no subject and brief_redirect")
        return self
~~~

- [ ] **Step 4: Add episode/turn defaults and old-record compatibility tests**

Add a test that validates an old episode dictionary containing only problem_context and turns, then assert active_subject == "", active_stage == "", and each new turn field has its default.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: python3 -m pytest tests/prototypes/test_llm_led_orchestration_models.py tests/test_conversation_archive.py -q

Expected: all tests pass and existing user ownership tests remain unchanged.

- [ ] **Step 6: Commit only Task 1 files**

~~~bash
git add prototypes/llm_led_teacher/orchestration_models.py prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_orchestration_models.py tests/test_conversation_archive.py
git commit -m "feat: add multisubject orchestration contracts"
~~~

---

### Task 2: Local-Only Knowledge Retrieval with Transparent Scores

**Files:**
- Modify: llm-test/rag_retriever.py:601-888
- Modify: prototypes/llm_led_teacher/rag_hook.py:1-280
- Create: tests/prototypes/test_llm_led_orchestration_tools.py
- Modify: tests/prototypes/test_llm_led_turn_contract.py:450-490
- Modify: tests/prototypes/test_rag_gpu.py:168-210

**Interfaces:**
- Produces: retrieve_local_hits(problem: str, subject: str, stage: str, topk: int, sources: tuple[str, ...]) -> dict[str, Any].
- Produces: retrieve_local_context(problem: str, subject: Subject, stage: SchoolStage, topk: int = 5, sources: tuple[str, ...] = ("textbook", "exercise")) -> RagContext.
- Preserves: warmup_rag() continues to perform local-only warmup and remains cached.
- RagContext gains diagnostics containing filters, hit_count, elapsed_seconds, and per-hit raw vec_sim, bm25_score, rrf_normalized, and fusion sim.

- [ ] **Step 1: Write failing tests proving local retrieval never calls online**

~~~python
def test_local_adapter_uses_requested_subject_and_never_online(monkeypatch):
    calls = []

    def fake_local(problem, subject, stage, topk, sources):
        calls.append((problem, subject, stage, topk, sources))
        return {"hits": [], "query": problem, "filters": {"subject": subject, "stage": stage}}

    monkeypatch.setattr(rag_hook, "_retrieve_local_hits", fake_local)
    result = rag_hook.retrieve_local_context(
        "力为什么能改变运动状态",
        subject=Subject.PHYSICS,
        stage=SchoolStage.MIDDLE,
    )
    assert calls == [("力为什么能改变运动状态", "物理", "初中", 5, ("textbook", "exercise"))]
    assert result.online_triggered is False
~~~

- [ ] **Step 2: Write failing score-transparency and metadata-map tests**

Test all nine mappings. Assert textbook subject remains Chinese while exercise subject maps to Math, Chinese, English, Physics, Chemistry, Biology, History, Geography, or Politics. Assert returned hits retain vec_sim, bm25_score, rrf_normalized, sim, source, text, and metadata.

- [ ] **Step 3: Run focused tests and verify RED**

Run: python3 -m pytest tests/prototypes/test_llm_led_orchestration_tools.py tests/prototypes/test_rag_gpu.py -q

Expected: failures for missing retrieve_local_context/retrieve_local_hits and mappings.

- [ ] **Step 4: Extract the public local-hit function**

Move the existing local textbook/exercise portion of retrieve_for_exam into retrieve_local_hits. Do not duplicate embedding initialization. The function returns unfiltered local candidates plus their raw scoring fields; it does not import online_rag and does not apply ONLINE_RAG.local_threshold.

- [ ] **Step 5: Make legacy retrieve_for_exam call the extracted function**

Keep the paper-evaluation API compatible. Its auto mode may retain legacy behavior for llm-test callers, but 8502 must use retrieve_local_context and therefore cannot trigger web implicitly.

- [ ] **Step 6: Implement the RagContext adapter and warmup compatibility**

Format context/refs from local hits, keep empty local results as source="local_empty", and use source="retrieval_error: Type" only for actual exceptions.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: python3 -m pytest tests/prototypes/test_llm_led_orchestration_tools.py tests/prototypes/test_rag_gpu.py tests/prototypes/test_llm_led_turn_contract.py -q

Expected: all pass, including cached CPU warmup and old paper-RAG compatibility.

- [ ] **Step 8: Commit only Task 2 files**

~~~bash
git add llm-test/rag_retriever.py prototypes/llm_led_teacher/rag_hook.py tests/prototypes/test_llm_led_orchestration_tools.py tests/prototypes/test_rag_gpu.py tests/prototypes/test_llm_led_turn_contract.py
git commit -m "feat: expose local-only multisubject retrieval"
~~~

---

### Task 3: Guarded Tool Registry

**Files:**
- Create: prototypes/llm_led_teacher/orchestration_tools.py
- Modify: tests/prototypes/test_llm_led_orchestration_tools.py

**Interfaces:**
- Produces: LocalSearchArguments(query, subject, stage, top_k, sources).
- Produces: WebSearchArguments(query, subject, stage, evidence_gap).
- Produces: OrchestrationRuntime(classification, problem_context, student_message, local_search_succeeded, web_search_count, skill_catalog).
- Produces: OrchestrationToolRegistry.execute(name, arguments, runtime) -> ToolResult.
- Registers: search_local_knowledge, search_web, load_skill, inspect_student_context.

- [ ] **Step 1: Write failing policy tests**

~~~python
def test_web_requires_k9_and_successful_local_search(registry, runtime):
    result = registry.execute(
        "search_web",
        {"query": "最新教材变化", "subject": "物理", "stage": "初中", "evidence_gap": "本地无新课标"},
        runtime,
    )
    assert not result.ok
    assert result.error == "web search requires successful local search"


def test_off_topic_cannot_use_local_or_web(registry, off_topic_runtime):
    assert not registry.execute("search_local_knowledge", {}, off_topic_runtime).ok
    assert not registry.execute("search_web", {}, off_topic_runtime).ok
~~~

- [ ] **Step 2: Write failing call-budget tests**

After one successful web result, a second search_web call must return error="web search budget exhausted". A second local search is legal only when query or filters differ from the first call.

Add a blocking fake handler and an injected timeout runner. Assert a timed-out Tool returns
ok=False with error="tool timed out" and that the next loop decision can still execute; the
exception must not escape into Streamlit.

- [ ] **Step 3: Run tests and verify RED**

Run: python3 -m pytest tests/prototypes/test_llm_led_orchestration_tools.py -q

Expected: import failure for orchestration_tools.py.

- [ ] **Step 4: Implement the registry and guarded handlers**

search_local_knowledge delegates to retrieve_local_context. search_web delegates to a web-only adapter in rag_hook and never runs for off_topic. The runtime updates local_search_succeeded only for an ok local ToolResult, and increments web_search_count only after the web handler starts.

Execute local and web handlers through an injected bounded runner with a configurable
TEACHER_TOOL_TIMEOUT_SECONDS default of 45 seconds. Convert timeout and handler exceptions into
ToolResult errors; do not terminate the worker process or Streamlit server.

- [ ] **Step 5: Return bounded, serializable observations**

Limit each result excerpt to 600 characters and total tool context to the existing RAG context budget. Keep full citations, numeric diagnostics, and errors; never include secrets or hidden prompts.

- [ ] **Step 6: Run tests and verify GREEN**

Run: python3 -m pytest tests/prototypes/test_llm_led_orchestration_tools.py -q

Expected: all tool policy, mapping, budget, and serialization tests pass.

- [ ] **Step 7: Commit only Task 3 files**

~~~bash
git add prototypes/llm_led_teacher/orchestration_tools.py prototypes/llm_led_teacher/rag_hook.py tests/prototypes/test_llm_led_orchestration_tools.py
git commit -m "feat: add guarded knowledge tools"
~~~

---

### Task 4: Mandatory Classification and Bounded Agent Loop

**Files:**
- Create: prototypes/llm_led_teacher/orchestrator.py
- Modify: prototypes/llm_led_teacher/orchestration_models.py
- Create: tests/prototypes/test_llm_led_orchestrator.py

**Interfaces:**
- Produces: RequestClassifier.classify(message, problem_context, active_subject, active_stage) -> RequestClassification using DeepSeekClient.chat_json.
- Produces: MultisubjectAgentLoop.run(problem_context, student_message, active_subject, active_stage, progress_callback=None) -> AgentOrchestrationResult.
- AgentOrchestrationResult includes classification, rag_context, observations, trace, loaded_skills, termination_reason, and routing guidance.
- Consumes: OrchestrationToolRegistry from Task 3 and one shared DeepSeekClient so usage_history contains classification, decisions, and final generation calls.

- [ ] **Step 1: Write failing classification continuity tests**

Use scripted classifiers:

~~~python
def test_followup_inherits_math_subject_when_classifier_marks_same_problem():
    result = loop.run(
        problem_context="解方程 2x+3=7",
        student_message="为什么要减去3？",
        active_subject="数学",
        active_stage="初中",
    )
    assert result.classification.subject.value == "数学"
    assert result.classification.is_new_problem is False


def test_explicit_physics_problem_switches_subject():
    result = loop.run(
        problem_context="解方程 2x+3=7",
        student_message="物体为什么会做匀速直线运动？",
        active_subject="数学",
        active_stage="初中",
    )
    assert result.classification.subject.value == "物理"
    assert result.classification.is_new_problem is True
~~~

- [ ] **Step 2: Write failing loop-budget and off-topic tests**

Assert maximum six decisions, maximum two distinct local calls, maximum one web call, and no knowledge calls for off_topic. Assert malformed classification gets one repair attempt then brief_redirect fallback.

- [ ] **Step 3: Run tests and verify RED**

Run: python3 -m pytest tests/prototypes/test_llm_led_orchestrator.py -q

Expected: import failure for orchestrator.py.

- [ ] **Step 4: Implement RequestClassifier with a strict JSON prompt**

The system prompt lists the nine exact subjects and says that ordinary product/company/chat questions are off_topic. Feed only active problem, current message, and active subject/stage; do not feed RAG results into classification.

- [ ] **Step 5: Implement the action loop**

Each iteration validates AgentDecision, records a public rationale, executes at most one tool, and feeds ToolResult back to the next decision. A finish action returns the selected RagContext and routing guidance. Provider errors and exhausted iterations return a safe result rather than raising into Streamlit.

- [ ] **Step 6: Add deterministic tool-policy enforcement**

Reject search_web before a successful local call, duplicate web, identical duplicate local calls, unknown tools, and all knowledge calls under off_topic. Tool policy is enforced in code even if the model requests an illegal action.

- [ ] **Step 7: Run tests and verify GREEN**

Run: python3 -m pytest tests/prototypes/test_llm_led_orchestrator.py tests/prototypes/test_llm_led_orchestration_tools.py -q

Expected: all classification, continuity, error, and budget tests pass.

- [ ] **Step 8: Commit only Task 4 files**

~~~bash
git add prototypes/llm_led_teacher/orchestrator.py prototypes/llm_led_teacher/orchestration_models.py tests/prototypes/test_llm_led_orchestrator.py
git commit -m "feat: add bounded multisubject agent loop"
~~~

---

### Task 5: Integrate the Loop into the Existing Turn Contract

**Files:**
- Modify: prototypes/llm_led_teacher/logic.py:473-622
- Modify: tests/prototypes/test_llm_led_turn_contract.py

**Interfaces:**
- run_turn keeps its current signature and PrototypeTurnRecord return type.
- Add optional orchestrator dependency for tests without changing Streamlit callers.
- Replace fixed retrieve_context calls with orchestrator.run.
- Preserve _teacher_payload, _chat_teaching_turn, _audit_claims, _collect_warnings, _repair, and summarize_usage_metadata.

- [ ] **Step 1: Write failing integration tests**

Add tests that inject a fake orchestrator and assert:

- a new physics problem updates episode.active_subject and active_stage;
- a math follow-up preserves the active problem and subject;
- off_topic produces brief_redirect guidance with no RagContext text;
- the turn stores classification and agent trace;
- existing verification/audit/repair still run after orchestration.

- [ ] **Step 2: Add an explicit zero-regression usage test**

Extend the usage-history fake so classification, one decision, and teacher generation each append metadata. Assert model_generation_metadata still sums prompt_tokens, prompt_cache_hit_tokens, prompt_cache_miss_tokens, completion_tokens, reasoning_tokens, response_tokens, total_tokens, context_used_tokens, context_window_tokens, and cache_hit_rate.

- [ ] **Step 3: Run target tests and verify RED**

Run: python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q

Expected: failures because run_turn has no orchestrator dependency and turn models lack loop fields.

- [ ] **Step 4: Integrate orchestration before teacher generation**

Create the active client once, pass it to MultisubjectAgentLoop, apply classification to episode state, map routing guidance into the existing TurnGuidance, and place the selected RagContext into _teacher_payload. Keep starts_new_problem normalization and active problem protection.

- [ ] **Step 5: Preserve deterministic math and post-generation audits**

Do not remove _verify, claim audits, disclosure/pacing warnings, repair, or previous_verified_step updates. Ensure mandatory loop verification is not counted as permission to skip the existing final mathematical audits.

- [ ] **Step 6: Preserve pending-message failure semantics**

Let DeepSeekError and ValueError continue to reach the existing Streamlit handler. Convert tool failures to observations inside the loop; do not turn them into process-level exceptions.

- [ ] **Step 7: Run contract tests and verify GREEN**

Run: python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py tests/prototypes/test_llm_led_orchestrator.py -q

Expected: all tests pass, including original token/cache aggregation and math audit tests.

- [ ] **Step 8: Commit only Task 5 files**

~~~bash
git add prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_turn_contract.py
git commit -m "feat: route 8502 turns through agent loop"
~~~

---

### Task 6: Preserve Login, History, Context Meter, and Cache UI

**Files:**
- Modify: prototypes/llm_led_teacher/live_progress.py
- Modify: prototypes/llm_led_teacher/streamlit_app.py:178-625
- Modify: tests/prototypes/test_llm_led_streaming.py
- Modify: tests/prototypes/test_llm_led_auth_ui.py

**Interfaces:**
- LiveProgressTrace gains update_agent_event(event: AgentTraceEvent) -> None.
- render_context_usage and render_turn_token_usage signatures and call counts remain unchanged.
- Existing session keys authenticated_user, prototype_episode, prototype_conversation_id, prototype_user_id, conversation_selector, and pending student message remain unchanged.

- [ ] **Step 1: Strengthen the UI source contract before editing**

Add assertions that streamlit_app.py still contains:

~~~python
assert "render_login_or_registration()" in source
assert source.count("render_context_usage(episode)") == 1
assert source.count("render_turn_token_usage(turn)") == 2
assert "prompt_cache_hit_tokens" in source
assert "prompt_cache_miss_tokens" in source
assert "_save_active_conversation(episode)" in source
assert "_get_pending_student_message(episode)" in source
~~~

- [ ] **Step 2: Add AppTest login/history regression assertions**

Keep the existing logged-out, login restore, new conversation, and schema migration tests. Add a restored multisubject conversation containing active_subject="物理" and assert login restores that field without changing student ID or conversation ID.

- [ ] **Step 3: Run UI/auth tests before implementation**

Run: python3 -m pytest tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py -q

Expected: existing tests pass; the new agent-event rendering assertion fails.

- [ ] **Step 4: Render public agent events in the existing status box**

Map classification, skill, local retrieval, web retrieval, verifier, finish, fallback, and audit events to concise Chinese lines. Use reason summaries and counts only; never render raw prompts, full decision payloads, keys, or hidden reasoning.

- [ ] **Step 5: Keep authentication and metric blocks structurally untouched**

Do not move or rename render_login_or_registration, session initialization, render_context_usage, render_turn_token_usage, the st.bottom block, login/logout buttons, or conversation selector. Change only the progress callback and turn-detail sections needed for agent trace.

- [ ] **Step 6: Run UI/auth tests and verify GREEN**

Run: python3 -m pytest tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py tests/test_auth.py tests/test_conversation_archive.py -q

Expected: all tests pass and AppTest restores both legacy and multisubject sessions.

- [ ] **Step 7: Commit only Task 6 files**

~~~bash
git add prototypes/llm_led_teacher/live_progress.py prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py
git commit -m "feat: show agent trace without regressing 8502 ui"
~~~

---

### Task 7: Offline Full Regression and Documentation

**Files:**
- Modify: README.md
- Modify: docs/superpowers/specs/2026-08-12-8502-multisubject-agent-loop-design.md only if implementation reveals a factual correction

**Interfaces:**
- Documents 8502 as bounded multisubject agent loop.
- Documents local-first/web-optional policy and off-topic brief redirect.
- Does not claim the Git LFS cache size is the active Chroma database size.

- [ ] **Step 1: Run all focused prototype tests**

Run: python3 -m pytest tests/prototypes -q

Expected: all prototype tests pass.

- [ ] **Step 2: Run the complete suite**

Run: python3 -m pytest

Expected: all tests pass with zero failures.

- [ ] **Step 3: Run the required offline evaluation**

Run: python3 scripts/run_eval.py

Expected: 38 cases pass, zero fail, and logs/eval_results.jsonl is written.

- [ ] **Step 4: Update README architecture and RAG behavior**

Replace fixed observer-to-auto-RAG wording with the bounded loop, nine-subject routing, local-only tool, guarded web tool, mandatory math verifier, and preserved 8502 login/context/cache features.

- [ ] **Step 5: Check documentation and diff hygiene**

Run: git diff --check

Run: rg -n "硬编码.*初中数学|5 条.*0.65|固定.*RAG" README.md docs/superpowers/specs/2026-08-12-8502-multisubject-agent-loop-design.md

Expected: no stale statement describes 8502 as using hidden fixed-subject auto-web routing.

- [ ] **Step 6: Commit documentation**

~~~bash
git add README.md docs/superpowers/specs/2026-08-12-8502-multisubject-agent-loop-design.md
git commit -m "docs: document 8502 multisubject agent loop"
~~~

---

### Task 8: Real Local-Model and Chroma Acceptance

**Files:**
- No source edits unless a failing acceptance test returns the plan to the relevant earlier task.
- Runtime evidence: logs and terminal output only.

**Interfaces:**
- Uses scripts/run_llm_led_teacher.py with local DeepSeek on 8080, Chroma on 8002, and Streamlit on 8502.
- Leaves one healthy 8502 stack running for user testing.

- [ ] **Step 1: Start from one clean project stack**

Resolve only project-owned processes on 8080, 8002, and 8502. Start:

~~~bash
python3 scripts/run_llm_led_teacher.py
~~~

Expected: the model completes its inference probe, Chroma reports both collections, and 8502 health returns ok.

- [ ] **Step 2: Verify local mathematical coverage without web**

Submit “一元一次方程移项为什么要变号”. Expected trace: 数学 → local search → finish, with relevant textbook evidence and no search_web action.

- [ ] **Step 3: Verify a second K9 subject**

Submit a physics or chemistry question known to exist in the collection. Expected trace uses the matching Chinese textbook subject and matching English exercise subject; it must not query Math.

- [ ] **Step 4: Verify guarded web fallback**

Submit a K9 question whose requested current fact is absent locally. Expected trace: local search first, explicit evidence gap, one web search maximum, then finish.

- [ ] **Step 5: Verify off-topic behavior**

Submit “网易企业邮箱是什么”. Expected: brief answer plus K9 redirect, no local search, no web search, and no false “本地知识库不足” message.

- [ ] **Step 6: Verify conversation continuity and UI preservation**

Log in, switch from a math problem to physics, ask a follow-up, refresh, and reopen the conversation. Confirm active subject persists. Confirm context usage, per-turn token totals, cache hit/miss counts, login/logout, history selector, and pending-message retry remain visible and functional.

- [ ] **Step 7: Verify process health after all real turns**

Run:

~~~bash
curl -fsS --noproxy '*' http://127.0.0.1:8080/health
curl -fsS --noproxy '*' http://127.0.0.1:8002/api/v2/heartbeat
curl -fsS --noproxy '*' http://127.0.0.1:8502/_stcore/health
~~~

Expected: model status ok, Chroma heartbeat JSON, Streamlit ok. No CUDA OOM and no WebSocket termination.

- [ ] **Step 8: Final review**

Run git status --short and inspect every changed file. Confirm no account database, conversation database, Chroma data, generated cache, secret, or unrelated user change is staged.
