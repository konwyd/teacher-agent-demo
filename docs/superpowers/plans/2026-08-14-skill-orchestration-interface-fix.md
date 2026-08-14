# Skill Orchestration Interface Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Skill selection catalog-bound, idempotent, and deterministically recoverable so invalid model-generated names cannot consume the Agent Loop budget.

**Architecture:** Centralize the canonical Skill catalog and contract-to-Skill policy in a focused module. Pass that catalog into every action decision, enforce load idempotency in the tool registry, and let the orchestrator replace the first unknown Skill with a policy-selected canonical Skill without expanding the teaching contract.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, existing DeepSeek-compatible local client and Streamlit 8502 prototype.

## Global Constraints

- Do not expand the Skill catalog or add subject-specific Skills in this change.
- Do not change KV size, context limits, login, cache metrics, or UI layout.
- Do not bypass `math_checker`; without upstream verified error evidence, never auto-select `error_diagnosis`.
- Preserve teaching-contract disclosure permissions and current local/web retrieval boundaries.
- Preserve unrelated dirty worktree files and commit only files owned by each task.
- Run `python -m pytest` and `python scripts/run_eval.py` before completion.

---

### Task 1: Canonical Skill Policy

**Files:**
- Create: `prototypes/llm_led_teacher/skill_policy.py`
- Modify: `prototypes/llm_led_teacher/fast_plan.py`
- Test: `tests/prototypes/test_skill_policy.py`

**Interfaces:**
- Produces: `DEFAULT_SKILL_CATALOG: dict[str, str]`
- Produces: `available_skill_descriptors(catalog: Mapping[str, str]) -> list[dict[str, str]]`
- Produces: `select_skill_for_contract(contract: TeachingContract) -> str`
- Consumes: `TeachingContract.response_mode`

- [ ] **Step 1: Write failing policy tests**

```python
def test_available_skill_descriptors_are_sorted_and_complete() -> None:
    assert available_skill_descriptors({"z": "Z", "a": "A"}) == [
        {"name": "a", "description": "A"},
        {"name": "z", "description": "Z"},
    ]

@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("concept_explanation", "concept_explanation"),
        ("one_step", "socratic_guidance"),
        ("guided_solution", "socratic_guidance"),
        ("full_solution", "complete_solution"),
        ("brief_redirect", ""),
    ],
)
def test_contract_selects_only_canonical_skill(mode, expected) -> None:
    contract = _contract(mode)
    assert select_skill_for_contract(contract) == expected
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest tests/prototypes/test_skill_policy.py -q`

Expected: FAIL because `skill_policy` does not exist.

- [ ] **Step 3: Implement the focused policy module**

```python
DEFAULT_SKILL_CATALOG = {
    "socratic_guidance": "一次推进一个关键步骤，用一个必要问题检查理解。",
    "error_diagnosis": "指出具体错误证据，再给最小可执行修正。",
    "concept_explanation": "先讲核心因果，再用贴近学生的例子解释。",
    "complete_solution": "按明确授权给出完整步骤、关键理由与最终结论。",
}

def available_skill_descriptors(catalog):
    return [
        {"name": name, "description": catalog[name]}
        for name in sorted(catalog)
    ]

def select_skill_for_contract(contract):
    return {
        "concept_explanation": "concept_explanation",
        "one_step": "socratic_guidance",
        "guided_solution": "socratic_guidance",
        "full_solution": "complete_solution",
    }.get(contract.response_mode.value, "")
```

Update `fast_plan.py` to import `select_skill_for_contract` and remove its private duplicate.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_skill_policy.py tests/prototypes/test_fast_agent_routing.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add prototypes/llm_led_teacher/skill_policy.py prototypes/llm_led_teacher/fast_plan.py tests/prototypes/test_skill_policy.py
git commit -m "refactor: centralize teacher skill policy"
```

### Task 2: Idempotent Skill Tool Boundary

**Files:**
- Modify: `prototypes/llm_led_teacher/orchestration_tools.py`
- Test: `tests/prototypes/test_llm_led_orchestration_tools.py`

**Interfaces:**
- Adds: `OrchestrationRuntime.loaded_skill_names: set[str]`
- Adds: `OrchestrationRuntime.failed_skill_names: set[str]`
- Changes: `OrchestrationToolRegistry.execute("load_skill", arguments, runtime)`

- [ ] **Step 1: Write failing tool-boundary tests**

```python
def test_unknown_skill_is_rejected_once_then_marked_duplicate() -> None:
    runtime = _runtime()
    registry = OrchestrationToolRegistry()

    first = registry.execute("load_skill", {"name": "数学教学技能"}, runtime)
    second = registry.execute("load_skill", {"name": "数学教学技能"}, runtime)

    assert first.error == "unknown skill"
    assert second.error == "duplicate skill request"

def test_loaded_skill_is_idempotent() -> None:
    runtime = _runtime()
    registry = OrchestrationToolRegistry()

    first = registry.execute("load_skill", {"name": "concept_explanation"}, runtime)
    second = registry.execute("load_skill", {"name": "concept_explanation"}, runtime)

    assert first.ok and second.ok
    assert second.data["idempotent"] is True
    assert runtime.loaded_skill_names == {"concept_explanation"}
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest tests/prototypes/test_llm_led_orchestration_tools.py -k "skill" -q`

Expected: FAIL because repeated invalid and valid requests are not tracked.

- [ ] **Step 3: Implement registry state enforcement**

Add the two sets to `OrchestrationRuntime`. In the `load_skill` branch, return idempotent success for a loaded canonical name, reject a repeated failed name as `duplicate skill request`, record a first unknown name, and record a first successful canonical load.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_llm_led_orchestration_tools.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add prototypes/llm_led_teacher/orchestration_tools.py tests/prototypes/test_llm_led_orchestration_tools.py
git commit -m "fix: enforce idempotent skill loading"
```

### Task 3: Catalog-Bound Decisions and Unknown-Skill Recovery

**Files:**
- Modify: `prototypes/llm_led_teacher/orchestrator.py`
- Test: `tests/prototypes/test_llm_led_orchestrator.py`

**Interfaces:**
- Action payload adds: `available_skills: list[{"name": str, "description": str}]`
- Unknown-Skill trace uses one successful `tool_result` with `details.fallback_reason == "unknown_skill_fallback"`
- Empty catalog termination: `skill_catalog_unavailable`

- [ ] **Step 1: Write failing payload and nine-subject recovery tests**

```python
NINE_SUBJECT_CASES = [
    ("数学", "老师，2x + 3 = 9 怎么解？"),
    ("语文", "拟人这种修辞手法是什么意思？"),
    ("英语", "一般过去时什么时候使用？"),
    ("物理", "为什么惯性只与质量有关？"),
    ("化学", "怎么判断一个反应是不是置换反应？"),
    ("生物", "光合作用需要哪些条件？"),
    ("历史", "辛亥革命的主要影响是什么？"),
    ("地理", "怎么判断一个地方属于什么气候？"),
    ("道法", "权利和义务是什么关系？"),
]

@pytest.mark.parametrize(("subject", "message"), NINE_SUBJECT_CASES)
def test_unknown_skill_recovers_once_for_every_subject(subject, message) -> None:
    provider = _ScriptedActions([
        AgentDecision(
            kind="call_tool",
            tool_name="load_skill",
            arguments={"name": f"{subject}教学技能"},
            reason="加载教学方法",
        ),
        _finish(),
    ])
    result = MultisubjectAgentLoop(
        classifier=_Classifier(_k9(subject, new=True)),
        action_provider=provider,
        registry=OrchestrationToolRegistry(local_handler=lambda **kwargs: RagContext()),
        enable_fast_routing=False,
    ).run("", message, "", "")

    assert provider.calls == 2
    assert result.loaded_skills == ["socratic_guidance"]
    assert result.termination_reason == "model_finished"
    assert sum(
        event.details.get("fallback_reason") == "unknown_skill_fallback"
        for event in result.trace
    ) == 1
    assert all(event.summary != "load_skill 未执行：unknown skill" for event in result.trace)
    assert {item["name"] for item in provider.payloads[0]["available_skills"]} == {
        "socratic_guidance", "error_diagnosis",
        "concept_explanation", "complete_solution",
    }
```

Add separate cases for explicit full solution, “不要答案”, unverified wrong-step text, follow-up continuity, and empty catalog.

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest tests/prototypes/test_llm_led_orchestrator.py -k "skill or nine_subject" -q`

Expected: FAIL because the catalog is absent from the payload and unknown Skills consume another model iteration.

- [ ] **Step 3: Implement minimal orchestrator recovery**

Use `DEFAULT_SKILL_CATALOG` when creating the runtime. Add `available_skill_descriptors(runtime.skill_catalog)` to every action payload. When `load_skill` returns `unknown skill`, build the already-safe teaching contract, select its canonical Skill, execute that load once, retain the invalid observation for diagnostics, and emit one successful tool-result trace containing the requested name, selected name, and `unknown_skill_fallback`. If no canonical Skill is available, terminate safely without another provider call.

Update the system prompt to state that `load_skill.name` must be copied exactly from `available_skills`.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_llm_led_orchestrator.py tests/prototypes/test_fast_agent_routing.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add prototypes/llm_led_teacher/orchestrator.py tests/prototypes/test_llm_led_orchestrator.py
git commit -m "fix: bind agent decisions to canonical skills"
```

### Task 4: Regression, Evaluation, and Local-Model Smoke Matrix

**Files:**
- Modify only if a failing regression proves necessary: files directly responsible for that regression
- Runtime evidence: `data/runtime/conversations.sqlite3` and ignored logs; never commit these files

**Interfaces:**
- Consumes: existing `run_turn`, local DeepSeek client, 8502 deployment
- Produces: per-turn trace and usage evidence for twelve smoke scenarios

- [ ] **Step 1: Run focused orchestration suites**

Run:

```bash
python3 -m pytest tests/prototypes/test_skill_policy.py tests/prototypes/test_llm_led_orchestration_tools.py tests/prototypes/test_llm_led_orchestrator.py tests/prototypes/test_fast_agent_routing.py tests/prototypes/test_public_thinking.py -q
```

Expected: PASS with no repeated unknown-Skill public events.

- [ ] **Step 2: Run repository regression gates**

Run:

```bash
python3 -m pytest
python3 scripts/run_eval.py
```

Expected: both exit 0; evaluation preserves answer-disclosure and math-checker invariants.

- [ ] **Step 3: Deploy the changed backend to 8502**

Restart only the 8502 Streamlit prototype process through the existing launcher. Do not restart the model, Chroma, or alter KV arguments unless the launcher requires a coordinated safe restart.

- [ ] **Step 4: Execute the twelve local-model smoke prompts**

Run the nine questions in the approved design plus:

```text
请完整解答 3x-5=10，并说明每一步。
请引导我解 4x+1=13，但不要告诉我最后答案。
我觉得 2x+3=9 两边减3以后是 2x=5，对吗？
```

For every persisted turn, inspect trace and usage metadata. Assert no repeated signature, no unexplained `unknown skill`, `duplicate skill request`, `iteration_budget_exhausted`, Provider error, or tool error. Compare call count and tokens with the 9-call/8,453-token failure baseline. The wrong-step prompt must not auto-select `error_diagnosis` without upstream math-checker evidence.

- [ ] **Step 5: Run final mutation-oriented checks**

Confirm tests would fail if: `available_skills` is removed, failed-name tracking is removed, canonical fallback chooses `complete_solution` for guided mode, or fallback re-enters the provider loop without loading a valid Skill.

- [ ] **Step 6: Commit any test-backed correction and report evidence**

Commit only files required by a reproduced failure. Report exact test totals, evaluation totals, smoke outcomes, remaining independent errors, and before/after call/token measurements.

