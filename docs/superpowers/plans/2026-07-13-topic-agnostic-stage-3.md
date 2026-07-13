# Topic-agnostic teaching harness — Stage 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default teaching path topic-agnostic while preserving the one-variable-linear-equation workflow as a compatibility adapter and adding cross-domain teaching smoke tests.

**Architecture:** `app/agent/workflow.py` becomes a narrow facade that runs a neutral `TeachingState` through generic analysis, deterministic policy, and guarded response generation. The current LangGraph workflow moves unchanged into `app/agent/linear_equation_adapter.py`; the facade delegates to it only when the adapter identifies an existing linear-equation turn. `run_turn` keeps returning `TeacherState` for legacy callers, while `run_teaching_turn` exposes the neutral primary contract.

**Tech Stack:** Python 3.13, Pydantic v2, LangGraph (legacy adapter only), FastAPI, pytest, offline mock evaluation.

## Global Constraints

- Do not change `app/agent/math_checker.py`; the adapter remains its only Stage 3 workflow consumer.
- Do not add a solver, data pack, RAG provider, or topic-specific policy branch for geometry, ratio, or statistics.
- Generic actions are `TeachingActionKind` values only; legacy action strings stay at the compatibility edge.
- The generic path does not call a retriever or graph reader and does not disclose final answers without verifier-backed permission.
- Preserve all existing API fields and all equation regression/evaluation cases.
- Do not call a paid API in tests; `DeepSeekClient` remains the only possible LLM integration point.

---

### Task 1: Define generic analysis, policy, and safe response contracts

**Files:**
- Modify: `app/agent/state.py`
- Create: `app/agent/math_analysis.py`
- Create: `app/agent/teaching_policy.py`
- Create: `app/agent/response_generation.py`
- Test: `tests/test_generic_teaching_policy.py`

**Interfaces:**
- Produces `TeachingIntent`, `analyze_turn(message) -> MathAnalysis`, `TeachingPolicy.choose(state) -> TeachingActionPlan`, and `generate_response(state) -> str`.
- `TeachingPolicy` only consumes `TeachingState`, `MathAnalysis`, and `VerificationResult`; it never imports a checker, retriever, or topic ID.

- [x] **Step 1: Write the failing policy and response contract tests**

```python
def test_unverified_problem_uses_a_hint_without_disclosing_an_answer() -> None:
    state = TeachingState(student_id="s", message="比例题不会做", problem_context="比例题不会做")
    state.action_plan = TeachingPolicy().choose(state)
    response = generate_response(state)
    assert state.action_plan.action is TeachingActionKind.GIVE_HINT
    assert state.answer_disclosure_allowed is False
    assert response.count("？") == 1

def test_verified_result_precedes_generic_mastery_signals() -> None:
    state = TeachingState(
        student_id="s", message="我这样算对吗", problem_context="某个数学问题",
        verification=VerificationResult(status=VerificationStatus.VERIFIED_CORRECT),
    )
    assert TeachingPolicy().choose(state).action is TeachingActionKind.CHECK_UNDERSTANDING
```

- [x] **Step 2: Run the focused test and verify it fails because the generic modules do not exist**

Run: `python -m pytest tests/test_generic_teaching_policy.py`

Expected: collection error naming `app.agent.teaching_policy` or `app.agent.response_generation`.

- [x] **Step 3: Add the smallest neutral contracts**

```python
class TeachingIntent(str, Enum):
    UNKNOWN = "unknown"
    ASKS_WHY = "asks_why"
    REQUESTS_EXAMPLE = "requests_example"
    REQUESTS_PRACTICE = "requests_practice"
    REQUESTS_ANSWER = "requests_answer"
    EXPRESSES_CONFUSION = "expresses_confusion"

class TeachingPolicy:
    def choose(self, state: TeachingState) -> TeachingActionPlan:
        if state.verification.status is VerificationStatus.VERIFIED_CORRECT:
            return TeachingActionPlan(action=TeachingActionKind.CHECK_UNDERSTANDING)
        if state.verification.status is VerificationStatus.VERIFIED_INCORRECT:
            return TeachingActionPlan(action=TeachingActionKind.EXPLAIN_STEP)
        if not state.problem_context:
            return TeachingActionPlan(action=TeachingActionKind.ASSESS)
        return TeachingActionPlan(action=TeachingActionKind.GIVE_HINT)
```

`math_analysis.py` uses only generic interaction phrases and returns safe fallback analysis. `response_generation.py` supplies one-question, no-answer templates for all neutral actions.

- [x] **Step 4: Run the focused test and verify it passes**

Run: `python -m pytest tests/test_generic_teaching_policy.py`

Expected: all focused tests pass.

### Task 2: Make the generic workflow the facade default

**Files:**
- Move: `app/agent/workflow.py` to `app/agent/linear_equation_adapter.py`
- Create: `app/agent/teaching_harness.py`
- Create: `app/agent/workflow.py`
- Modify: `app/agent/state.py`
- Test: `tests/test_topic_agnostic_workflow.py`

**Interfaces:**
- Produces `run_teaching_turn(student_id, message, conversation_id=None) -> TeachingState`.
- Keeps `run_turn(student_id, message, conversation_id=None) -> TeacherState` as a compatibility return type.
- `linear_equation_adapter.run_legacy_equation_turn` retains the prior LangGraph implementation and `supports_legacy_equation_turn` owns all equation recognition.

- [x] **Step 1: Write the failing generic-workflow test**

```python
def test_non_equation_turn_uses_generic_harness_without_a_legacy_concept() -> None:
    state = run_teaching_turn("ratio", "男生和女生人数的比是3:2，我不知道先想什么")
    assert state.action_plan is not None
    assert state.graph_context.available is False
    assert state.verification.status is VerificationStatus.NOT_APPLICABLE
    assert state.current_phase == state.action_plan.action.value
```

- [x] **Step 2: Run the focused test and verify it fails because `run_teaching_turn` is missing**

Run: `python -m pytest tests/test_topic_agnostic_workflow.py::test_non_equation_turn_uses_generic_harness_without_a_legacy_concept`

Expected: import error naming `run_teaching_turn`.

- [x] **Step 3: Relocate the old workflow mechanically and implement the facade**

```python
def run_teaching_turn(student_id: str, message: str, conversation_id: str | None = None) -> TeachingState:
    legacy = run_legacy_equation_turn_if_supported(student_id, message, conversation_id)
    if legacy is not None:
        return project_teacher_state(legacy)
    return run_generic_teaching_turn(student_id, message, conversation_id)

def run_turn(student_id: str, message: str, conversation_id: str | None = None) -> TeacherState:
    legacy = run_legacy_equation_turn_if_supported(student_id, message, conversation_id)
    if legacy is not None:
        return legacy
    return project_teaching_state_to_legacy(run_generic_teaching_turn(student_id, message, conversation_id))
```

The new `workflow.py` imports only state projections, the generic harness, and the adapter boundary. It must not import `math_checker`, `StepValidity`, a retriever, `DEFAULT_CONCEPT_ID`, or equation keyword rules.

- [x] **Step 4: Run generic and existing workflow tests**

Run: `python -m pytest tests/test_topic_agnostic_workflow.py tests/test_workflow.py tests/test_policy.py`

Expected: generic route is neutral; existing equation behavior is unchanged.

### Task 3: Add data-driven cross-domain smoke conversations and API coverage

**Files:**
- Create: `data/eval/topic_agnostic_smoke.jsonl`
- Modify: `tests/test_api.py`
- Create: `tests/test_topic_agnostic_smoke.py`

**Interfaces:**
- The JSONL stores only message/forbidden-answer/action expectations for ratio, geometry statement, and data interpretation prompts; it is not a domain knowledge package.
- API `/turn` preserves legacy fields and returns an additive neutral `teaching_state` for an unsupported topic.

- [x] **Step 1: Write the failing data-driven smoke test**

```python
@pytest.mark.parametrize("case", _load_smoke_cases())
def test_cross_domain_smoke_preserves_generic_teaching_invariants(case: dict) -> None:
    state = run_teaching_turn(case["student_id"], case["message"])
    assert state.action_plan.action.value in case["allowed_actions"]
    assert state.response.count("？") == 1
    assert all(token not in state.response for token in case["forbidden_answer_tokens"])
    assert state.graph_context.available is False
```

- [x] **Step 2: Run the smoke test and verify it fails because its JSONL fixture is absent**

Run: `python -m pytest tests/test_topic_agnostic_smoke.py`

Expected: file-not-found error for `data/eval/topic_agnostic_smoke.jsonl`.

- [x] **Step 3: Add three topic-neutral smoke inputs and API assertion**

```json
{"student_id":"smoke_ratio","message":"男生和女生人数的比是3:2，总数30人，我不知道先想什么。","allowed_actions":["give_hint","diagnose"],"forbidden_answer_tokens":["18","12"]}
```

Add analogous geometry-statement and data-interpretation entries. Assert `/turn` returns an empty legacy concept and a populated `teaching_state` with `graph_context.available == false` for the ratio request.

- [x] **Step 4: Run smoke and API tests**

Run: `python -m pytest tests/test_topic_agnostic_smoke.py tests/test_api.py`

Expected: three smoke cases and API compatibility pass.

### Task 4: Run regressions, inspect the boundary, and commit

**Files:**
- Verify only: `app/agent/workflow.py`, `app/agent/linear_equation_adapter.py`, all Stage 3 test/data files

- [x] **Step 1: Assert the generic facade has no equation-core imports or terms**

Run: `rg -n -S 'math_checker|StepValidity|DEFAULT_CONCEPT_ID|solve_for_x|target_equation|_KEYWORD_RULES|_classify_equation_structure' app/agent/workflow.py`

Expected: no matches.

- [x] **Step 2: Run full offline tests and evaluations**

Run:

```powershell
python -m pytest
python scripts/run_eval.py
python scripts/run_eval_modes.py --mode deepseek_only
python scripts/run_eval_modes.py --mode agent_no_graph
python scripts/run_eval_modes.py --mode agent_with_graph
```

Expected: all tests and strict mock eval pass; the known `deepseek_only` baseline remains recorded rather than changed.

- [x] **Step 3: Stage only Stage 3 files and commit**

```powershell
git add -- app/agent/state.py app/agent/math_analysis.py app/agent/teaching_policy.py app/agent/response_generation.py app/agent/teaching_harness.py app/agent/workflow.py app/agent/linear_equation_adapter.py app/api/main.py data/eval/topic_agnostic_smoke.jsonl tests/test_generic_teaching_policy.py tests/test_topic_agnostic_workflow.py tests/test_topic_agnostic_smoke.py tests/test_api.py docs/superpowers/plans/2026-07-13-topic-agnostic-stage-3.md
git commit -m "refactor: decouple teaching harness from linear equations"
```

Expected: the independent branch contains only the Stage 3 refactor; no merge, push, or deletion occurs.

## Plan self-review

- Spec coverage: Tasks 1–2 create neutral analysis/policy/response boundaries and isolate the equation workflow; Task 3 adds exactly three non-equation behavior smoke cases; Task 4 protects strict legacy behavior and checks the stated core-file boundary.
- Scope: no new solver, graph provider, or topic-specific package is added. Legacy graph behavior remains isolated in the compatibility adapter for Stage 4.
- Type consistency: `run_teaching_turn` always returns `TeachingState`; legacy `run_turn` always returns `TeacherState`; action mapping remains in `state.py`.
- Placeholder scan: no unassigned work, undefined interface, or deferred implementation marker remains.
