# Teacher Agent Fast Route Orchestration Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make routine K9 requests use one or two LLM calls while reserving the bounded Agent tool loop for explicit dynamic evidence requests.

**Architecture:** Keep `FastRoute`, `RequestClassification`, `TeachingContract`, verifier, memory, RAG, audit, trace, and UI contracts intact. Split routing into deterministic L0, classifier-assisted direct L1, and dynamic-agent L2; execute obvious skill/local/Quality-QA tools in Python, then apply risk-based semantic auditing after Teacher generation.

**Tech Stack:** Python 3.11, Pydantic, Streamlit, DeepSeek OpenAI-compatible client, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-fast-route-orchestration-optimization-design.md`

## Global Constraints

- Preserve all unrelated and overlapping uncommitted work already present in the dirty worktree.
- Do not commit during execution unless the user separately requests it; inspect scoped diffs after every task.
- Do not add the out-of-scope `2+2` short-answer/closed-computation policy.
- Do not remove or weaken TeachingContract, `LinearEquationVerifier`, claim audits, Memory, RAG, SemanticBoundaryAuditor, repair, trace, or existing UI diagnostics.
- All LLM calls continue through `app/llm/deepseek_client.py`; structured calls continue using Pydantic validation.
- Local search max 1, Quality QA max 1, Web max 1, tool timeouts, and local-before-web remain enforced.
- New persisted fields must have defaults so old `PrototypeTurnRecord`, `PrototypeEpisode`, and `ConversationRecord` payloads remain loadable.
- Finish with fresh `python -m pytest` and `python scripts/run_eval.py` evidence.

## File Structure

- `prototypes/llm_led_teacher/fast_routing.py`: high-confidence subject/stage routing, continuity, Quality QA benefit policy, dynamic-tool policy.
- `prototypes/llm_led_teacher/fast_plan.py`: deterministic contract/skill/retrieval preparation shared by direct and dynamic routes.
- `prototypes/llm_led_teacher/orchestrator.py`: L0/L1/L2 selection, deterministic pre-tools, bounded dynamic loop, route metadata.
- `prototypes/llm_led_teacher/orchestration_models.py`: backward-compatible orchestration iteration metadata only.
- `prototypes/llm_led_teacher/logic.py`: risk-based audit selection and persisted performance metadata.
- `prototypes/llm_led_teacher/streamlit_app.py`: route and performance metadata display.
- `tests/prototypes/test_fast_agent_routing.py`: deterministic cues, continuity, local-evidence routing.
- `tests/prototypes/test_quality_qa_routing.py`: narrower Quality QA policy.
- `tests/prototypes/test_llm_led_orchestrator.py`: classifier-direct and dynamic-agent behavior.
- `tests/prototypes/test_agent_call_budget.py`: semantic-audit and LLM-call budgets.
- `tests/prototypes/test_llm_led_turn_contract.py`: high-risk audit/verifier preservation and metadata compatibility.
- `tests/prototypes/test_llm_led_auth_ui.py`: UI label and hot-reload compatibility.

---

### Task 1: High-confidence routing and evidence-demand policies

**Files:**
- Modify: `tests/prototypes/test_fast_agent_routing.py`
- Modify: `tests/prototypes/test_quality_qa_routing.py`
- Modify: `prototypes/llm_led_teacher/fast_routing.py`

**Interfaces:**
- Produces: `needs_dynamic_tool_loop(message: str, classification: RequestClassification) -> bool`
- Produces: `needs_local_evidence(message: str) -> bool`
- Preserves: `choose_fast_route(message, problem_context, active_subject, active_stage) -> FastRoute | None`

- [ ] **Step 1: Add failing FastRoute cue tests**

Add parametrized tests asserting that each prompt-required math topic returns K9/Math without a classifier call, including these minimum cases:

```python
@pytest.mark.parametrize(
    "message",
    [
        "勾股定理是什么",
        "什么是平方根",
        "一次函数是什么",
        "一元二次方程怎么理解",
        "全等三角形是什么",
        "圆周角是什么",
    ],
)
def test_common_middle_school_math_topics_use_fast_route(message: str) -> None:
    route = choose_fast_route(message, "", "", "")

    assert route is not None
    assert route.classification.scope.value == "k9"
    assert route.classification.subject.value == "数学"
    assert route.classification.stage.value == "初中"
```

Add a broader parametrized assertion covering every mathematics cue listed in the spec. Add explicit high-confidence cases for English and Politics plus current Chinese/Physics/Chemistry/Biology/History/Geography cases. Add negative assertions showing that ambiguous single words such as `力`、`水`、`人`、`国家`、`文章` do not determine a subject by themselves.

- [ ] **Step 2: Add failing stage and continuity tests**

```python
def test_physics_cue_infers_middle_stage_without_positional_lookup() -> None:
    route = choose_fast_route("惯性是什么", "", "", "")
    assert route is not None
    assert route.classification.subject.value == "物理"
    assert route.classification.stage.value == "初中"


def test_followup_why_inherits_active_math_context() -> None:
    route = choose_fast_route(
        "那为什么呢？",
        "勾股定理是什么",
        "数学",
        "初中",
    )
    assert route is not None
    assert route.classification.subject.value == "数学"
    assert route.classification.stage.value == "初中"
    assert route.classification.is_new_problem is False
```

- [ ] **Step 3: Add failing Quality QA and evidence-policy tests**

```python
@pytest.mark.parametrize(
    "message",
    ["为什么勾股定理成立？", "勾股定理怎么理解？", "勾股定理的思路是什么？"],
)
def test_routine_explanation_does_not_enable_quality_qa(message: str) -> None:
    route = choose_fast_route(message, "", "", "")
    assert route is not None
    assert route.classification.use_quality_qa is False


@pytest.mark.parametrize(
    "message",
    ["怎么记勾股定理", "这类题总是错怎么办", "请批改这篇作文"],
)
def test_experiential_requests_can_enable_quality_qa(message: str) -> None:
    route = choose_fast_route(message, "", "数学", "初中")
    assert route is not None
    assert route.classification.use_quality_qa is True


def test_only_explicit_evidence_language_requests_local_search() -> None:
    assert needs_local_evidence("勾股定理是什么") is False
    assert needs_local_evidence("根据教材解释一下勾股定理") is True
    assert needs_local_evidence("课程标准原文在哪一页") is True
```

Add dynamic-tool assertions:

```python
classification = RequestClassification(
    scope="k9", subject="数学", stage="初中", is_new_problem=True,
    answer_mode="guided", reason="test",
)
assert needs_dynamic_tool_loop("请用三种方法证明勾股定理", classification) is False
assert needs_dynamic_tool_loop("联网查一下最新初中数学课程标准", classification) is True
```

- [ ] **Step 4: Run focused tests and verify RED**

Run:

```bash
python -m pytest tests/prototypes/test_fast_agent_routing.py tests/prototypes/test_quality_qa_routing.py -q
```

Expected: failures because the new math cues, generic follow-up marker, evidence policy, and dynamic-loop policy are missing, and because routine “为什么” still enables Quality QA.

- [ ] **Step 5: Implement the minimal routing policies**

In `fast_routing.py`:

```python
_SUBJECT_CUES: dict[Subject, tuple[str, ...]] = {
    Subject.MATH: _MATH_TOPIC_CUES,
    Subject.CHINESE: _CHINESE_TOPIC_CUES,
    Subject.ENGLISH: _ENGLISH_TOPIC_CUES,
    Subject.PHYSICS: _PHYSICS_TOPIC_CUES,
    Subject.CHEMISTRY: _CHEMISTRY_TOPIC_CUES,
    Subject.BIOLOGY: _BIOLOGY_TOPIC_CUES,
    Subject.HISTORY: _HISTORY_TOPIC_CUES,
    Subject.GEOGRAPHY: _GEOGRAPHY_TOPIC_CUES,
    Subject.POLITICS: _POLITICS_TOPIC_CUES,
}

_MIDDLE_ONLY_CUE_SUBJECTS = {Subject.PHYSICS, Subject.CHEMISTRY}
```

Use `_SUBJECT_CUES.get(subject, ())`; remove every positional lookup such as `_UNIQUE_SUBJECT_CUES[0][1]`. Infer middle stage for the prompt-listed middle-school math topics and high-confidence Physics/Chemistry cues, but not for arbitrary explicit subject labels.

Add `"那为什么"`, `"为什么呢"`, and equivalent bounded follow-up forms to continuity matching without adding a general short-message rule.

Implement policies using explicit marker tuples:

```python
def needs_local_evidence(message: str) -> bool:
    return any(marker in message for marker in _LOCAL_EVIDENCE_MARKERS)


def needs_dynamic_tool_loop(
    message: str,
    classification: RequestClassification,
) -> bool:
    return bool(
        classification.scope is RequestScope.K9
        and any(marker in message for marker in _DYNAMIC_TOOL_MARKERS)
    )
```

Do not include `证明`、`详细解释`、`为什么`、`多步骤` or message length as dynamic markers. Narrow `_QUALITY_QA_BENEFIT_MARKERS` to experiential requests.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the same focused pytest command. Expected: all tests pass with no model calls.

- [ ] **Step 7: Inspect scoped diff**

Run:

```bash
git diff --check
git diff -- prototypes/llm_led_teacher/fast_routing.py tests/prototypes/test_fast_agent_routing.py tests/prototypes/test_quality_qa_routing.py
```

Confirm only Task 1 behavior changed and existing unrelated work remains intact.

---

### Task 2: Deterministic plan preparation and on-demand RAG

**Files:**
- Modify: `tests/prototypes/test_fast_agent_routing.py`
- Modify: `tests/prototypes/test_quality_qa_routing.py`
- Modify: `prototypes/llm_led_teacher/fast_plan.py`

**Interfaces:**
- Consumes: `needs_local_evidence(message: str) -> bool`
- Extends: `run_fast_plan(..., router_type: str = "deterministic_fast_path", force_local_search: bool = False) -> AgentOrchestrationResult`

- [ ] **Step 1: Add failing ordinary-concept and explicit-evidence tests**

```python
def test_fast_concept_does_not_search_any_knowledge_source(monkeypatch) -> None:
    monkeypatch.setattr(qa_cache, "LOOKUP_ENABLED", True)
    registry = _FastRegistry()
    result = MultisubjectAgentLoop(
        client=_NoModelCalls(), registry=registry
    ).run("", "勾股定理是什么", "", "")

    assert [name for name, _ in registry.calls] == ["load_skill"]
    assert result.retrieval_decision.mode is RetrievalMode.NONE


def test_explicit_textbook_request_runs_one_local_search() -> None:
    registry = _FastRegistry()
    result = MultisubjectAgentLoop(
        client=_NoModelCalls(), registry=registry
    ).run("", "根据教材解释一下勾股定理", "", "")

    assert [name for name, _ in registry.calls] == [
        "load_skill", "search_local_knowledge"
    ]
    assert result.retrieval_decision.mode is RetrievalMode.LOCAL
```

Add a continuation case proving no repeated local retrieval unless the current message itself requests evidence.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python -m pytest tests/prototypes/test_fast_agent_routing.py tests/prototypes/test_quality_qa_routing.py -q
```

Expected: ordinary new concepts still call `search_local_knowledge` because current code uses `is_new_problem or is_concept_request`.

- [ ] **Step 3: Reuse the deterministic plan and change retrieval policy**

Extend `run_fast_plan()` so the same deterministic implementation can serve a
router hit, classifier-direct execution, or dynamic pre-tools. It must:

1. Build the safe TeachingContract.
2. Select/load exactly one deterministic skill when available.
3. Set `needs_local_search = force_local_search or needs_local_evidence(student_message)`; never use `classification.is_new_problem` or `is_concept_request` as sufficient conditions.
4. Execute Quality QA only when lookup is enabled and classification explicitly enables it.
5. Merge results with the existing `_merge_rag()` behavior.
6. Append public trace events without exposing private reasoning.

Return the result with consistent router metadata:

```python
return AgentOrchestrationResult(
    classification=classification,
    teaching_contract=contract,
    retrieval_decision=retrieval_decision,
    rag_context=rag_context,
    observations=observations,
    trace=trace,
    loaded_skills=loaded_skills,
    loaded_skill_guidance=loaded_skill_guidance,
    termination_reason=router_type,
    routing_guidance={
        "answer_mode": classification.answer_mode.value,
        "router": router_type,
    },
)
```

Use retrieval reasons that distinguish explicit evidence, Quality QA, and no retrieval; do not describe ordinary new concepts as requiring RAG.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 2 focused pytest command. Expected: ordinary concept uses only `load_skill`; textbook request performs one local search; continuation does not re-search.

- [ ] **Step 5: Inspect scoped diff**

Run `git diff --check` and inspect only Task 2 files, preserving the existing retrieval-decision work already present in `fast_plan.py`.

---

### Task 3: Classifier-direct L1 and bounded dynamic-agent L2

**Files:**
- Modify: `tests/prototypes/test_llm_led_orchestrator.py`
- Modify: `tests/prototypes/test_fast_agent_routing.py`
- Modify: `prototypes/llm_led_teacher/orchestration_models.py`
- Modify: `prototypes/llm_led_teacher/orchestrator.py`

**Interfaces:**
- Consumes: `needs_dynamic_tool_loop(...)`
- Consumes: `run_fast_plan(..., force_local_search=True)` for deterministic pre-tools
- Extends: `AgentOrchestrationResult.agent_iterations: int = 0`
- Preserves injected-test behavior when `enable_fast_routing=False`; production/default optimized routing uses L0/L1/L2.

- [ ] **Step 1: Add failing classifier-direct test**

Create counting fakes and assert:

```python
def test_fast_miss_k9_classification_uses_direct_plan_without_action_model() -> None:
    classifier = _CountingClassifier(_k9("数学", new=True))
    provider = _CountingActionProvider()
    result = MultisubjectAgentLoop(
        classifier=classifier,
        action_provider=provider,
        registry=_Registry(),
        enable_fast_routing=True,
    ).run("", "某冷门数学概念是什么意思", "", "")

    assert classifier.calls == 1
    assert provider.calls == 0
    assert result.termination_reason == "model_classifier_direct_path"
    assert result.routing_guidance["router"] == "model_classifier_direct_path"
    assert result.agent_iterations == 0
```

Ensure the test forces `choose_fast_route()` to return `None`, either with a message outside deterministic cues or a scoped monkeypatch.

- [ ] **Step 2: Add failing dynamic-path tests**

Use `联网查一下最新的初中数学课程标准` and fakes that first select Web and then finish. Assert:

```python
assert result.routing_guidance["router"] == "dynamic_agent_path"
assert result.termination_reason in {"model_finished", "iteration_budget_exhausted"}
assert provider.calls <= 2
assert result.agent_iterations == provider.calls
assert [name for name, _ in registry.calls].count("search_local_knowledge") == 1
assert [name for name, _ in registry.calls].count("search_web") <= 1
```

Also assert that local search appears before Web and that a failed local search still prevents Web.

- [ ] **Step 3: Run focused orchestrator tests and verify RED**

Run:

```bash
python -m pytest tests/prototypes/test_llm_led_orchestrator.py tests/prototypes/test_fast_agent_routing.py -q
```

Expected: Fast miss enters ActionProvider, the loop default is 4, and router/iteration metadata are missing.

- [ ] **Step 4: Implement L0/L1/L2 selection**

In `MultisubjectAgentLoop.__init__`, change the production default to `max_iterations=2`.

After classification and memory handling:

```python
dynamic_route = bool(
    self.enable_fast_routing
    and needs_dynamic_tool_loop(student_message, classification)
)
if (
    self.enable_fast_routing
    and fast_route is None
    and classification.scope is RequestScope.K9
    and not dynamic_route
):
    fast_route = FastRoute(
        classification=classification,
        reason="模型分类器已确定课程边界，后续使用确定性执行计划",
    )
    classification_router = "model_classifier_direct_path"
```

Pass `classification_router` to `run_fast_plan()` so the trace, termination reason, and routing guidance agree.

For dynamic requests:

- set classification trace router to `dynamic_agent_path`;
- call `run_fast_plan(..., force_local_search=True)` before the ActionProvider loop and reuse its result as the pre-tool state;
- initialize observations, skills, guidance, RAG, and TeachingContract from the preparation result;
- give ActionProvider only the remaining choice of Web or finish;
- increment `agent_iterations` once for each actual `decide()` invocation, including schema-repair retries if the client made another call;
- retain existing provider/tool fallbacks and Pydantic validation.

Do not alter `enable_fast_routing=False` injected-loop tests except where default max iteration expectations need an explicit value; this preserves existing unit coverage of low-level loop mechanics.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Task 3 command. Fix production code rather than weakening call-count assertions.

- [ ] **Step 6: Inspect scoped diff**

Confirm the existing retrieval-decision summary code in `orchestrator.py` remains intact and dynamic Web still obeys registry budgets.

---

### Task 4: Risk-based semantic audit and performance metadata

**Files:**
- Modify: `tests/prototypes/test_agent_call_budget.py`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py`
- Modify: `prototypes/llm_led_teacher/logic.py`

**Interfaces:**
- Produces: `_router_type(orchestration_result: AgentOrchestrationResult) -> str`
- Produces: `_is_low_risk_fast_generation(orchestration_result, verification, student_message) -> bool`
- Extends: `PrototypeTurnRecord.router_type: str = ""`
- Extends: `PrototypeTurnRecord.agent_iterations: int = 0`
- Extends: `PrototypeTurnRecord.semantic_audit_performed: bool = False`

- [ ] **Step 1: Replace the old concept budget expectation with a failing low-risk test**

Construct a new-problem K9 concept result whose router is `deterministic_fast_path` and whose Teacher draft contains no forbidden literal marker:

```python
def test_low_risk_fast_concept_uses_one_generation_and_no_semantic_audit() -> None:
    client = _BudgetClient(
        "勾股定理说明直角三角形三边之间的关系。",
        allow_audit=False,
    )
    result = _result("concept_explanation").model_copy(
        update={
            "termination_reason": "deterministic_fast_path",
            "routing_guidance": {"router": "deterministic_fast_path"},
        }
    )
    turn = run_turn(
        PrototypeEpisode(),
        "勾股定理是什么？",
        client=client,
        orchestrator=_ResultOrchestrator(result),
    )

    assert client.chat_limits == [1024]
    assert client.audit_limits == []
    assert turn.semantic_audit_performed is False
    assert turn.repair_performed is False
```

Add an equivalent classifier-direct budget test using a spy client that records one Classifier `chat_json()` plus one Teacher `chat()` and no ActionProvider/audit call.

- [ ] **Step 2: Add failing high-risk preservation tests**

Add cases for:

- active-problem `one_step` or `guided_solution` still calls SemanticBoundaryAuditor;
- explicit “不要答案” still calls SemanticBoundaryAuditor;
- verified linear-equation continuation still runs deterministic verifier and claim audits;
- low-risk hard-boundary literal violation triggers repair rather than silent return;
- length truncation remains conditional repair.

Representative assertion:

```python
assert turn.semantic_audit_performed is True
assert client.audit_limits == [320]
assert turn.student_verification.status.value == "verified_correct"
```

- [ ] **Step 3: Add failing backward-compatible metadata test**

```python
def test_legacy_turn_defaults_new_performance_metadata() -> None:
    turn = PrototypeTurnRecord.model_validate(legacy_payload)
    assert turn.router_type == ""
    assert turn.agent_iterations == 0
    assert turn.semantic_audit_performed is False
```

- [ ] **Step 4: Run focused tests and verify RED**

Run:

```bash
python -m pytest tests/prototypes/test_agent_call_budget.py tests/prototypes/test_llm_led_turn_contract.py -q
```

Expected: low-risk concept still performs semantic audit and new metadata fields do not exist.

- [ ] **Step 5: Implement risk policy and deterministic precheck**

Derive router type from `routing_guidance["router"]`, falling back to the classification trace for legacy orchestration results.

The low-risk predicate must require all of:

```python
router_type in {"deterministic_fast_path", "model_classifier_direct_path"}
classification.scope is RequestScope.K9
classification.is_new_problem is True
contract.response_mode is TeachingResponseMode.CONCEPT_EXPLANATION
not has_no_answer_request(student_message)
verification.status is not VerificationStatus.VERIFIED_INCORRECT
```

Before deciding on semantic audit, run `hard_boundary_recheck()` for low-risk outputs. If it returns violations, convert them into repair reasons and do not silently pass. If it returns none, skip `SemanticBoundaryAuditor`.

Keep the existing semantic-audit predicate for all non-low-risk concept/one-step/guided modes. Keep repair conditional on length, deterministic/claim warnings, or boundary violations.

Persist:

```python
router_type=resolved_router_type
agent_iterations=orchestration_result.agent_iterations if orchestration_result else 0
semantic_audit_performed=boundary_audit is not None
```

Do not disturb the existing Memory V2 shadow-admission changes in `logic.py`.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the Task 4 focused command. Confirm the existing high-risk tests still pass.

- [ ] **Step 7: Inspect scoped diff**

Run `git diff --check` and inspect only the Task 4 hunks, especially around the existing memory shadow writeback code.

---

### Task 5: Streamlit routing and performance display

**Files:**
- Modify: `tests/prototypes/test_llm_led_auth_ui.py`
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`

**Interfaces:**
- Consumes: `PrototypeTurnRecord.router_type`, `.agent_iterations`, `.semantic_audit_performed` with default values.
- Preserves: `_upgrade_turn_model()` and current session schema compatibility.

- [ ] **Step 1: Add failing UI source/behavior tests**

Add assertions for the four exact labels:

```python
expected = {
    "deterministic_fast_path": "确定性快速路由",
    "model_classifier_direct_path": "模型分类后直达生成",
    "dynamic_agent_path": "动态 Agent 工具路由",
    "memory_clarification_reply": "记忆澄清续接",
}
```

Extend the legacy hot-reload test to assert the new metadata defaults. Add a rendered-turn assertion that `agent_iterations=2` and `semantic_audit_performed=True` appear in the diagnostics without removing Token, timing, RAG, verifier, or repair sections.

- [ ] **Step 2: Run focused UI tests and verify RED**

Run:

```bash
python -m pytest tests/prototypes/test_llm_led_auth_ui.py -q
```

Expected: model-direct and dynamic route labels plus explicit iteration/audit metadata are absent.

- [ ] **Step 3: Implement minimal UI mapping and metadata display**

Prefer `turn.router_type`; fall back to classification trace for old records. Update labels exactly as specified. Add compact metrics/captions for:

- Agent iterations;
- Semantic audit performed/not performed;
- existing repair status;
- existing retrieval mode.

Do not alter the current `iter_display_chunks`/`st.write_stream` behavior or claim true model streaming.

- [ ] **Step 4: Run focused UI tests and verify GREEN**

Run the Task 5 command and confirm no Streamlit exceptions.

- [ ] **Step 5: Inspect scoped diff**

Confirm the existing session schema upgrade and memory diagnostics changes remain intact.

---

### Task 6: Call-budget regression matrix and full verification

**Files:**
- Modify: `tests/prototypes/test_agent_call_budget.py`
- Modify: `tests/prototypes/test_llm_led_orchestrator.py`
- Modify an affected production/test file only when a real regression demonstrates the need.

**Interfaces:**
- Verifies all interfaces produced by Tasks 1–5.

- [ ] **Step 1: Add/finish the three exact call-budget cases**

Use a single spy client whose `chat()` and `chat_json()` counters distinguish Classifier, ActionProvider, Teacher, and SemanticBoundaryAuditor schemas. Assert:

```text
Case 1: 勾股定理是什么
Classifier 0, ActionProvider 0, Teacher 1, SemanticAudit 0

Case 2: deterministic miss + classifier says ordinary K9
Classifier 1, ActionProvider 0, Teacher 1, SemanticAudit 0

Case 3: explicit latest/Web request
ActionProvider <= 2 and total calls materially below the old 7+ path
```

Do not make timing assertions against wall-clock seconds; validate call counts and stage metadata deterministically.

- [ ] **Step 2: Run the prototype-focused suite**

Run:

```bash
python -m pytest tests/prototypes -q
```

Read the entire failure summary. Fix production regressions; do not delete or loosen teaching-boundary, verifier, memory, RAG, or UI assertions.

- [ ] **Step 3: Run formatting/diff checks**

Run:

```bash
git diff --check
git diff --stat
git status --short
```

Confirm no runtime database, `.env`, API key, log, or generated vector-store file was modified by this task.

- [ ] **Step 4: Run full pytest verification**

Run:

```bash
python -m pytest
```

Expected: exit code 0 and zero failures.

- [ ] **Step 5: Run evaluation verification**

Run:

```bash
python scripts/run_eval.py
```

Expected: exit code 0 and all offline golden cases pass.

- [ ] **Step 6: Review requirements against the spec**

Check every design section: four router labels, on-demand RAG, narrowed Quality QA, L1 direct route, bounded L2, local-before-web, low-risk deterministic guard, high-risk audit, conditional repair, verifier preservation, metadata compatibility, and UI visibility.

- [ ] **Step 7: Report evidence and remaining limitations**

Report exact test/eval counts, scoped modified files, verified call budgets, and that real TTFT streaming remains a future phase. Do not claim completion if either full verification command fails.
