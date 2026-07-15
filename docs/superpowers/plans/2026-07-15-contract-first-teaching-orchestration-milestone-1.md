# Contract-First Teaching Orchestration Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立统一教学动作、安全门、证据账本、策略能力注册和可审计 shadow planner，并让一条由现有 `math_checker` 验证的一元一次方程会话可通过功能开关完整运行在新编排器中。

**Architecture:** 保留一个 `run_teaching_turn` 编排入口。确定性 verifier 先形成证据，SafetyGate 再限定动作与披露边界，TeachingPlanner 只在合法候选中选一个主动作，TeachingBrief 是 ResponseGenerator 的唯一输入；旧方程运行时在里程碑内通过私有兼容枚举和输出投影继续存在，但不进入新核心决策。

**Tech Stack:** Python 3.13、Pydantic 2、FastAPI、现有 DeepSeek JSON mode client、现有 Fraction `math_checker`、pytest、JSONL 离线评测。

## Global Constraints

- 只使用 `refactor/topic-agnostic-teaching-harness`，不新建或切换开发分支。
- 不合并、不推送、不删除或改写历史提交。
- 不修改 `app/agent/math_checker.py` 的解析、步骤、计划或答案裁决算法。
- 新核心只使用九个统一动作：`ELICIT`、`FOCUS`、`SCAFFOLD`、`EXPLAIN`、`CORRECT`、`CHECK`、`MODEL`、`PRACTICE`、`CONSOLIDATE`。
- 一个回合恰好一个主教学动作；提示层级、讲解深度和练习差异使用类型化参数。
- `math_checker` 的确定性事实优先于 SafetyGate、Planner、策略、memory 和 RAG。
- 核心 SafetyGate、TeachingPlanner 和 TeachingPolicy 不增加方程组、几何、统计等知识点 `if/else`。
- RAG/GraphRAG、verifier 和长期 memory 均可缺失；无 Graph/RAG 时必须完成安全教学。
- 学生未请求最终答案且未自行解出时，不披露最终答案或完整数对。
- worked example 只允许三类触发：学生明确要例子、同一范围连续两次验证错误、学生明确表示完全不懂。
- 每轮回复恰好一个问句，并且问句位于结尾。
- 所有 LLM 调用只经过 `app/llm/deepseek_client.py`；结构化输出使用 JSON mode 和 Pydantic。
- 默认测试和评测完全离线，不设置 `USE_REAL_LLM_FOR_EVAL=true`，不调用真实付费 API。
- 不提交 `.env`、API key、`logs/`、`data/runtime/` 或用户提供的未跟踪交接文件。
- 每个 Task 先 RED、再最小 GREEN、再 focused regression、再独立提交；批次门额外运行完整离线回归。

## Scope Boundary

本计划覆盖设计规格的批次 0 至批次 3：契约盘点、统一动作、shadow 规划内核和功能开关控制的一元一次方程纵向切片。

设计规格的批次 4 与批次 5 不属于本计划：方程组、几何、数量建模、数据分析分别是独立能力子项目；兼容层删除依赖这些子项目的替代覆盖。里程碑 1 完成后，应依据实际 trace 和测试迁移清单，为每个领域分别生成规格与实施计划，最后再生成兼容层退出计划。

---

## File Responsibility Map

| File | Responsibility |
| --- | --- |
| `app/agent/teaching_actions.py` | 九动作枚举、类型化参数、披露层级和 SafetyConstraints。 |
| `app/agent/legacy_action_compat.py` | 临时保存旧动作枚举及纯投影；新核心禁止导入。 |
| `app/agent/safety_gate.py` | 把问题、意图和验证证据转换为允许动作与披露上限。 |
| `app/agent/student_model.py` | 从 episode 构建简单证据快照，并追加可追溯回合证据。 |
| `app/agent/strategy_capabilities.py` | 策略选择、能力协议、registry、unknown 与 symbolic 能力。 |
| `app/agent/teaching_planner.py` | 在 SafetyConstraints 内选择一个动作，校验或确定性降级。 |
| `app/agent/orchestration_trace.py` | 记录模块调用、跳过原因、LLM 调用数和 shadow/active 状态。 |
| `app/agent/teaching_brief.py` | 按优先级压缩证据，形成 ResponseGenerator 的唯一输入。 |
| `app/agent/response_generation.py` | 从 TeachingBrief 生成语言并执行 ResponseGuard。 |
| `app/agent/runtime_router.py` | 在兼容边界选择 legacy、generic 或开关控制的 unified linear slice。 |
| `app/agent/teaching_harness.py` | 串联 verifier 预检、按需 reasoner、策略、SafetyGate、Planner、Brief、Guard 和持久化。 |
| `app/memory/schemas.py` | 持久化学生证据、运行路径和最近 planning trace。 |
| `app/agent/state.py` | 聚合 TeachingState；legacy TeacherState 只保留兼容字段。 |
| `app/api/main.py` | 保留旧字段值，新增统一 action plan、constraints、evidence 和 trace。 |
| `app/ui/streamlit_app.py` | 在详情区展示统一动作、披露上限和模块调用原因。 |
| `tests/contracts/` | 永久安全与接口契约；禁止逐字依赖自然语言回复。 |
| `tests/behavior/` | 新教学质量多轮场景；断言推进、动作范围和安全行为。 |
| `data/eval/teaching_quality_conversations.jsonl` | 可重复的真实多轮教学质量 case。 |
| `app/evaluation/teaching_quality_evaluator.py` | 读取教学质量 case 并产生结构化离线结果。 |

---

### Task 1: Establish the Three-Layer Test Contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Create: `tests/contracts/__init__.py`
- Create: `tests/behavior/__init__.py`
- Create: `tests/contracts/test_contract_classification.py`
- Create: `tests/CONTRACT_LAYERS.md`

**Interfaces:**
- Consumes: pytest collection items.
- Produces: exactly one marker per test: `safety_contract`, `legacy_lock`, or `teaching_behavior`.
- Rule: existing tests directly under `tests/` are frozen legacy locks; new safety and behavior tests must live in their dedicated directories.

- [ ] **Step 1: Write the failing classification test**

Create `tests/contracts/test_contract_classification.py`:

```python
"""Enforce one migration layer for every collected test."""

import pytest


pytestmark = pytest.mark.safety_contract
_CONTRACT_MARKS = {"safety_contract", "legacy_lock", "teaching_behavior"}


def test_every_collected_test_has_exactly_one_contract_layer(request) -> None:
    failures: list[str] = []
    for item in request.session.items:
        active = _CONTRACT_MARKS.intersection(item.keywords)
        if len(active) != 1:
            failures.append(f"{item.nodeid}: {sorted(active)}")

    assert failures == []
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m pytest tests/contracts/test_contract_classification.py -v
```

Expected: FAIL listing existing tests with `[]`; warnings about an unregistered marker are acceptable only in this RED run.

- [ ] **Step 3: Register markers and classify collection**

Add to `[tool.pytest.ini_options]` in `pyproject.toml`:

```toml
markers = [
    "safety_contract: permanent safety and public-contract regression",
    "legacy_lock: frozen implementation-lock test; keep green and add no new cases",
    "teaching_behavior: new multi-turn teaching-quality behavior test",
]
```

Append to `tests/conftest.py`:

```python
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify tests by migration directory without editing frozen modules."""

    for item in items:
        path = Path(str(item.path)).as_posix()
        if "/tests/contracts/" in f"/{path}":
            item.add_marker(pytest.mark.safety_contract)
        elif "/tests/behavior/" in f"/{path}":
            item.add_marker(pytest.mark.teaching_behavior)
        else:
            item.add_marker(pytest.mark.legacy_lock)
```

Create empty package files in `tests/contracts/__init__.py` and `tests/behavior/__init__.py`. In `tests/CONTRACT_LAYERS.md`, state these enforceable rules:

```markdown
# Test Contract Layers

- `tests/contracts/`: permanent safety and public interfaces; assert structured behavior, never exact prose.
- `tests/behavior/`: new real multi-turn teaching behavior; assert allowed actions, progression, evidence, disclosure, and one final question.
- existing `tests/test_*.py`: frozen legacy locks; keep green and add no new cases.
- remove a legacy assertion only in the same commit that adds its replacement contract or behavior test.
```

- [ ] **Step 4: Run the classification test and both new marker selections**

Run:

```powershell
python -m pytest tests/contracts/test_contract_classification.py -v
python -m pytest -m safety_contract -q
python -m pytest -m teaching_behavior -q
```

Expected: classification PASS; safety selection contains the classification test; teaching behavior selection reports no tests collected with exit code 5 until Task 6 creates the first behavior test. Record that exit code as the expected empty-layer baseline rather than weakening pytest configuration.

- [ ] **Step 5: Run the frozen suite**

Run: `python -m pytest -m legacy_lock -q`

Expected: all existing tests pass with the current baseline count.

- [ ] **Step 6: Commit the contract layer**

```powershell
git add pyproject.toml tests/conftest.py tests/contracts/__init__.py tests/behavior/__init__.py tests/contracts/test_contract_classification.py tests/CONTRACT_LAYERS.md
git commit -m "test: classify teaching contracts and legacy locks"
```

---

### Task 2: Replace the Parallel Generic Action Enum with One Unified Action Contract

**Files:**
- Create: `app/agent/teaching_actions.py`
- Create: `app/agent/legacy_action_compat.py`
- Create: `tests/contracts/test_teaching_action_contract.py`
- Create: `tests/contracts/test_api_action_contract.py`
- Modify: `app/agent/state.py`
- Modify: `app/agent/teaching_policy.py`
- Modify: `app/agent/response_generation.py`
- Modify: `app/agent/linear_equation_adapter.py`
- Modify: `app/agent/socratic_policy.py`
- Modify: `app/evaluation/evaluator.py`
- Modify: `app/agent/workflow.py`
- Modify: `app/api/main.py`
- Modify mechanically where old enum imports occur: `tests/test_deepseek_integration.py`, `tests/test_evaluator.py`, `tests/test_flexible_steps.py`, `tests/test_grounding.py`, `tests/test_mastery.py`, `tests/test_pedagogical_rag.py`, `tests/test_policy.py`, `tests/test_socratic_policy.py`, `tests/test_targeted_fixes.py`, `tests/test_topic_agnostic_state.py`, `tests/test_workflow.py`
- Modify generic expectations: `tests/test_generic_response_generation.py`, `tests/test_generic_teaching_policy.py`, `tests/test_topic_agnostic_workflow.py`, `data/eval/topic_agnostic_smoke.jsonl`

**Interfaces:**
- Produces: `TeachingAction`, `TeachingActionPlan`, `SafetyConstraints`, `DisclosureLevel`, `ExplanationDepth`, `PracticeDifficulty`, `PracticeSimilarity` from `app.agent.teaching_actions`.
- Produces: temporary `LegacyTeachingAction`, `legacy_to_plan()`, and `plan_to_legacy_action()` from `app.agent.legacy_action_compat`.
- Preserves: existing API fields `teaching_action`, `teaching_action_kind`, and the compatibility representation inside `teaching_state`.
- Adds: top-level API field `teaching_action_plan` containing the unified contract.

- [ ] **Step 1: Write RED contract tests for the exact action set and typed parameters**

Create `tests/contracts/test_teaching_action_contract.py`:

```python
"""Permanent contract for the sole teaching-action model."""

import pytest
from pydantic import ValidationError

from app.agent.teaching_actions import (
    PracticeDifficulty,
    PracticeSimilarity,
    TeachingAction,
    TeachingActionPlan,
)


pytestmark = pytest.mark.safety_contract


def test_unified_action_set_is_exact() -> None:
    assert {action.value for action in TeachingAction} == {
        "elicit", "focus", "scaffold", "explain", "correct",
        "check", "model", "practice", "consolidate",
    }


def test_scaffold_requires_a_nonzero_level() -> None:
    with pytest.raises(ValidationError):
        TeachingActionPlan(action=TeachingAction.SCAFFOLD, scaffold_level=0)


def test_practice_requires_difficulty_and_similarity() -> None:
    plan = TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.SIMILAR,
        practice_similarity=PracticeSimilarity.SAME_STRATEGY,
    )
    assert plan.practice_difficulty is PracticeDifficulty.SIMILAR


def test_non_practice_action_rejects_practice_parameters() -> None:
    with pytest.raises(ValidationError):
        TeachingActionPlan(
            action=TeachingAction.EXPLAIN,
            practice_difficulty=PracticeDifficulty.HARDER,
            practice_similarity=PracticeSimilarity.TRANSFER,
        )
```

- [ ] **Step 2: Run the action contract test and verify RED**

Run: `python -m pytest tests/contracts/test_teaching_action_contract.py -v`

Expected: FAIL with `ModuleNotFoundError: app.agent.teaching_actions`.

- [ ] **Step 3: Implement the unified schema**

Create `app/agent/teaching_actions.py` with these public models:

```python
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class TeachingAction(str, Enum):
    ELICIT = "elicit"
    FOCUS = "focus"
    SCAFFOLD = "scaffold"
    EXPLAIN = "explain"
    CORRECT = "correct"
    CHECK = "check"
    MODEL = "model"
    PRACTICE = "practice"
    CONSOLIDATE = "consolidate"


class DisclosureLevel(str, Enum):
    NONE = "none"
    NEXT_STEP = "next_step"
    FINAL = "final"


class ExplanationDepth(str, Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


class PracticeDifficulty(str, Enum):
    NONE = "none"
    EASIER = "easier"
    SIMILAR = "similar"
    HARDER = "harder"


class PracticeSimilarity(str, Enum):
    NONE = "none"
    SAME_STRATEGY = "same_strategy"
    TRANSFER = "transfer"


class SafetyConstraints(BaseModel):
    allowed_actions: list[TeachingAction]
    disclosure_ceiling: DisclosureLevel = DisclosureLevel.NONE
    worked_example_allowed: bool = False
    new_problem_allowed: bool = False
    student_response_required: bool = True
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_candidates(self) -> "SafetyConstraints":
        if not self.allowed_actions:
            raise ValueError("allowed_actions must not be empty")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("allowed_actions must be unique")
        return self


class TeachingActionPlan(BaseModel):
    action: TeachingAction
    target: str = ""
    scaffold_level: int = Field(default=0, ge=0, le=3)
    explanation_depth: ExplanationDepth = ExplanationDepth.STANDARD
    practice_difficulty: PracticeDifficulty = PracticeDifficulty.NONE
    practice_similarity: PracticeSimilarity = PracticeSimilarity.NONE
    rationale: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    requires_student_response: bool = True
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action_parameters(self) -> "TeachingActionPlan":
        if self.action is TeachingAction.SCAFFOLD and self.scaffold_level == 0:
            raise ValueError("SCAFFOLD requires scaffold_level 1..3")
        if self.action is not TeachingAction.SCAFFOLD and self.scaffold_level != 0:
            raise ValueError("only SCAFFOLD may set scaffold_level")
        practice_fields_set = (
            self.practice_difficulty is not PracticeDifficulty.NONE
            or self.practice_similarity is not PracticeSimilarity.NONE
        )
        if self.action is TeachingAction.PRACTICE and not (
            self.practice_difficulty is not PracticeDifficulty.NONE
            and self.practice_similarity is not PracticeSimilarity.NONE
        ):
            raise ValueError("PRACTICE requires difficulty and similarity")
        if self.action is not TeachingAction.PRACTICE and practice_fields_set:
            raise ValueError("only PRACTICE may set practice parameters")
        return self
```

- [ ] **Step 4: Move the old enum behind an explicit compatibility boundary**

Create `app/agent/legacy_action_compat.py` by moving the existing thirteen-value enum unchanged and naming it `LegacyTeachingAction`. Define the complete legacy-to-unified mapping:

```python
_LEGACY_TO_PLAN = {
    LegacyTeachingAction.ASK_DIAGNOSTIC_QUESTION: TeachingActionPlan(action=TeachingAction.FOCUS),
    LegacyTeachingAction.EXPLAIN_CONCEPT: TeachingActionPlan(action=TeachingAction.EXPLAIN),
    LegacyTeachingAction.GIVE_HINT_LEVEL_1: TeachingActionPlan(action=TeachingAction.SCAFFOLD, scaffold_level=1),
    LegacyTeachingAction.GIVE_HINT_LEVEL_2: TeachingActionPlan(action=TeachingAction.SCAFFOLD, scaffold_level=2),
    LegacyTeachingAction.SHOW_WORKED_EXAMPLE: TeachingActionPlan(action=TeachingAction.MODEL),
    LegacyTeachingAction.GIVE_PRACTICE_QUESTION: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.SIMILAR,
        practice_similarity=PracticeSimilarity.SAME_STRATEGY,
    ),
    LegacyTeachingAction.GENERATE_PRACTICE_QUESTION: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.SIMILAR,
        practice_similarity=PracticeSimilarity.SAME_STRATEGY,
    ),
    LegacyTeachingAction.GENERATE_SIMILAR_EXERCISE: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.SIMILAR,
        practice_similarity=PracticeSimilarity.SAME_STRATEGY,
    ),
    LegacyTeachingAction.GENERATE_HARDER_EXERCISE: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.HARDER,
        practice_similarity=PracticeSimilarity.TRANSFER,
    ),
    LegacyTeachingAction.GENERATE_EASIER_EXERCISE: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.EASIER,
        practice_similarity=PracticeSimilarity.SAME_STRATEGY,
    ),
    LegacyTeachingAction.SUMMARIZE_THEN_PRACTICE: TeachingActionPlan(
        action=TeachingAction.PRACTICE,
        practice_difficulty=PracticeDifficulty.SIMILAR,
        practice_similarity=PracticeSimilarity.TRANSFER,
    ),
    LegacyTeachingAction.GRADE_AND_FEEDBACK: TeachingActionPlan(action=TeachingAction.CHECK),
    LegacyTeachingAction.SCHEDULE_REVIEW: TeachingActionPlan(action=TeachingAction.CONSOLIDATE),
}


def legacy_to_plan(
    action: LegacyTeachingAction,
    *,
    verified_incorrect: bool = False,
) -> TeachingActionPlan:
    if action is LegacyTeachingAction.GRADE_AND_FEEDBACK and verified_incorrect:
        return TeachingActionPlan(action=TeachingAction.CORRECT)
    return _LEGACY_TO_PLAN[action].model_copy(deep=True)


_SIMPLE_PLAN_TO_LEGACY = {
    TeachingAction.ELICIT: LegacyTeachingAction.ASK_DIAGNOSTIC_QUESTION,
    TeachingAction.FOCUS: LegacyTeachingAction.ASK_DIAGNOSTIC_QUESTION,
    TeachingAction.EXPLAIN: LegacyTeachingAction.EXPLAIN_CONCEPT,
    TeachingAction.CORRECT: LegacyTeachingAction.GRADE_AND_FEEDBACK,
    TeachingAction.CHECK: LegacyTeachingAction.GRADE_AND_FEEDBACK,
    TeachingAction.MODEL: LegacyTeachingAction.SHOW_WORKED_EXAMPLE,
    TeachingAction.CONSOLIDATE: LegacyTeachingAction.SCHEDULE_REVIEW,
}


def plan_to_legacy_action(plan: TeachingActionPlan) -> LegacyTeachingAction:
    if plan.action is TeachingAction.SCAFFOLD:
        return (
            LegacyTeachingAction.GIVE_HINT_LEVEL_2
            if plan.scaffold_level >= 2
            else LegacyTeachingAction.GIVE_HINT_LEVEL_1
        )
    if plan.action is TeachingAction.PRACTICE:
        if plan.practice_difficulty is PracticeDifficulty.EASIER:
            return LegacyTeachingAction.GENERATE_EASIER_EXERCISE
        if plan.practice_difficulty is PracticeDifficulty.HARDER:
            return LegacyTeachingAction.GENERATE_HARDER_EXERCISE
        if plan.practice_similarity is PracticeSimilarity.TRANSFER:
            return LegacyTeachingAction.SUMMARIZE_THEN_PRACTICE
        return LegacyTeachingAction.GIVE_PRACTICE_QUESTION
    return _SIMPLE_PLAN_TO_LEGACY[plan.action]


_LEGACY_KIND_CODE = {
    TeachingAction.ELICIT: "assess",
    TeachingAction.FOCUS: "diagnose",
    TeachingAction.SCAFFOLD: "give_hint",
    TeachingAction.EXPLAIN: "explain_step",
    TeachingAction.CORRECT: "explain_step",
    TeachingAction.CHECK: "check_understanding",
    TeachingAction.MODEL: "show_worked_example",
    TeachingAction.PRACTICE: "offer_practice",
    TeachingAction.CONSOLIDATE: "finish",
}


def legacy_kind_code(plan: TeachingActionPlan) -> str:
    return _LEGACY_KIND_CODE[plan.action]
```

New core files must not import `LegacyTeachingAction`; only `state.py`, `linear_equation_adapter.py`, legacy evaluator/policy modules, compatibility API projection, and frozen tests may import it. In `project_teacher_state()`, pass `verified_incorrect=True` only when the existing checker projection is `VERIFIED_INCORRECT`.

- [ ] **Step 5: Refactor state, generic policy, and fallback responses to the unified enum**

In `app/agent/state.py`:

- remove `TeachingActionKind`;
- import unified models from `teaching_actions.py`;
- type `TeachingState.action_plan` as the unified `TeachingActionPlan`;
- type `TeacherState.action` as `LegacyTeachingAction | None`;
- replace `_LEGACY_TO_TEACHING_ACTION` with `legacy_to_plan()`;
- replace `_TEACHING_TO_LEGACY_ACTION` with `plan_to_legacy_action()` at the output-only compatibility projection.

In `app/agent/teaching_policy.py`, keep the existing deterministic behavior but return these unified plans:

```python
verified_correct -> TeachingAction.CHECK
verified_incorrect -> TeachingAction.CORRECT
missing problem -> TeachingAction.ELICIT
needs clarification -> TeachingAction.FOCUS
authorized answer request -> TeachingAction.EXPLAIN
example request -> TeachingAction.MODEL
practice request -> TeachingAction.PRACTICE with SIMILAR + SAME_STRATEGY
asks why -> TeachingAction.EXPLAIN
available next step -> TeachingAction.SCAFFOLD with level 1
unknown intent -> TeachingAction.FOCUS
default -> TeachingAction.SCAFFOLD with level 1
```

Replace `_SAFE_RESPONSES` in `app/agent/response_generation.py` with the complete unified map:

```python
_SAFE_RESPONSES = {
    TeachingAction.ELICIT: "请把题目完整发来，并写出你已经想到的一步。你现在最不确定哪一部分？",
    TeachingAction.FOCUS: "我先不假设题型，也不直接给答案。你觉得自己卡在理解题意、选择方法还是计算哪一步？",
    TeachingAction.SCAFFOLD: "先把题目中的已知条件、所求内容和它们的关系分别写出来。你准备先整理哪一条信息？",
    TeachingAction.EXPLAIN: "可以先把这一步依据的数学关系说清楚，再把它对应回原题。你认为这一步依据是什么？",
    TeachingAction.CORRECT: "这一步已有明确的检查证据，可以先定位需要修正的位置。你愿意先重新检查这一处变化吗？",
    TeachingAction.CHECK: "这个结论已经有验证依据了。你能用题目中的条件说明它为什么成立吗？",
    TeachingAction.MODEL: "我可以按相同思路示范，但先不替代你完成原题。你希望我示范的是读题、列关系还是检查步骤？",
    TeachingAction.PRACTICE: "可以安排练习，不过先确定要巩固的思路。你想练习理解题意、建立关系还是检查结果？",
    TeachingAction.CONSOLIDATE: "我们先把刚才确认过的方法收束成一句话。你能说出下一次遇到同类关系时先看什么吗？",
}
```

- [ ] **Step 6: Preserve old API values while exposing the unified plan additively**

First make the facade execute a route once while retaining both views:

```python
@dataclass(frozen=True)
class TeachingTurnResult:
    teacher_state: TeacherState
    teaching_state: TeachingState


def execute_turn(student_id: str, message: str, conversation_id: str | None = None, **generic_dependencies) -> TeachingTurnResult:
    legacy_state = _run_legacy_turn_if_supported(student_id, message, conversation_id)
    if legacy_state is not None:
        return TeachingTurnResult(
            teacher_state=legacy_state,
            teaching_state=project_teacher_state(legacy_state),
        )
    teaching_state = run_generic_teaching_turn(
        student_id,
        message,
        conversation_id,
        **generic_dependencies,
    )
    return TeachingTurnResult(
        teacher_state=project_teaching_state_to_teacher_state(teaching_state),
        teaching_state=teaching_state,
    )
```

Make `run_teaching_turn()` and `run_turn()` thin wrappers over `execute_turn()` so a request never runs twice. In `app/api/main.py`, call `execute_turn()` and use its exact `teacher_state` for all old response fields and its exact `teaching_state` for the new unified fields.

Add to `TurnResponse`:

```python
teaching_action_plan: dict = Field(default_factory=dict)
safety_constraints: dict = Field(default_factory=dict)
orchestration_trace: dict = Field(default_factory=dict)
```

Before constructing `TurnResponse`, build a compatibility dump:

```python
teaching_state_payload = teaching_state.model_dump(mode="json")
unified_plan = teaching_state.action_plan
if unified_plan is not None:
    teaching_state_payload["action_plan"]["action"] = legacy_kind_code(unified_plan)
```

Return the old `teaching_action_kind` through `legacy_kind_code()` and return the real unified plan through `teaching_action_plan=unified_plan.model_dump(mode="json")`. Existing clients therefore retain `give_hint`, `explain_step`, and other old strings while new consumers use `scaffold`, `explain`, and the typed parameters.

Create `tests/contracts/test_api_action_contract.py`:

```python
import pytest
from fastapi.testclient import TestClient

from app.api.main import app


pytestmark = pytest.mark.safety_contract
client = TestClient(app)


def test_api_preserves_legacy_action_codes_and_adds_unified_plan() -> None:
    data = client.post(
        "/turn",
        json={"student_id": "api-action-contract", "message": "我不会解2x+3=7"},
    ).json()
    assert data["teaching_action"] == "give_hint_level_1"
    assert data["teaching_action_kind"] == "give_hint"
    assert data["teaching_state"]["action_plan"]["action"] == "give_hint"
    assert data["teaching_action_plan"]["action"] == "scaffold"
    assert data["teaching_action_plan"]["scaffold_level"] == 1
```

- [ ] **Step 7: Update imports and expectations without adding legacy cases**

Mechanically change old-runtime imports to:

```python
from app.agent.legacy_action_compat import LegacyTeachingAction as TeachingAction
```

Change generic tests to import `TeachingAction` from `app.agent.teaching_actions` and map expectations as follows: `GIVE_HINT -> SCAFFOLD`, `EXPLAIN_STEP -> EXPLAIN`, `CHECK_UNDERSTANDING -> CHECK`, `SHOW_WORKED_EXAMPLE -> MODEL`, `OFFER_PRACTICE -> PRACTICE`. Update `topic_agnostic_smoke.jsonl` expected action strings with the same mapping; do not change case inputs or forbidden-answer assertions.

- [ ] **Step 8: Run action, state, policy, response, API, and legacy focused tests**

Run:

```powershell
python -m pytest tests/contracts/test_teaching_action_contract.py tests/contracts/test_api_action_contract.py tests/test_topic_agnostic_state.py tests/test_generic_teaching_policy.py tests/test_generic_response_generation.py tests/test_api.py tests/test_policy.py tests/test_workflow.py -v
```

Expected: PASS; old API assertions remain unchanged, and the new top-level `teaching_action_plan.action` is one of the nine unified values.

- [ ] **Step 9: Run the batch-1 full offline gate**

Run:

```powershell
python -m pytest
python scripts/run_eval.py
python scripts/run_eval_modes.py --mode deepseek_only
python scripts/run_eval_modes.py --mode agent_no_graph
python scripts/run_eval_modes.py --mode agent_with_graph
```

Expected: pytest and strict `run_eval` have zero failures; `deepseek_only` keeps its recorded comparison baseline while both Agent modes keep all cases passing.

- [ ] **Step 10: Commit the unified action contract**

Stage only the files listed in this Task and commit:

```powershell
git commit -m "refactor: unify teaching action contract"
```

---

### Task 3: Add the Deterministic SafetyGate

**Files:**
- Create: `app/agent/safety_gate.py`
- Create: `tests/contracts/test_safety_gate.py`

**Interfaces:**
- Consumes: `SafetyInput(problem_context, intent, verification_status, candidate_confidence, candidate_available, consecutive_verified_errors, has_verified_error, has_confirmed_progress, expresses_total_confusion)`.
- Produces: `SafetyConstraints` with at least one allowed action.
- Does not consume: topic ID, equation variables, strategy ID, RAG content, mastery score.

- [ ] **Step 1: Write RED tests for disclosure, examples, and verifier precedence**

Create `tests/contracts/test_safety_gate.py`:

```python
import pytest

from app.agent.safety_gate import SafetyGate, SafetyInput
from app.agent.state import TeachingIntent, VerificationStatus
from app.agent.teaching_actions import DisclosureLevel, TeachingAction


pytestmark = pytest.mark.safety_contract


def test_missing_problem_forces_elicit_and_no_disclosure() -> None:
    result = SafetyGate().evaluate(SafetyInput(problem_context=""))
    assert result.allowed_actions == [TeachingAction.ELICIT]
    assert result.disclosure_ceiling is DisclosureLevel.NONE


def test_unrequested_candidate_never_receives_final_disclosure() -> None:
    result = SafetyGate().evaluate(
        SafetyInput(problem_context="3x+2y=5，6x-2y=3", candidate_available=True, candidate_confidence=0.99)
    )
    assert result.disclosure_ceiling is DisclosureLevel.NEXT_STEP


def test_verified_error_limits_actions_without_forcing_one() -> None:
    result = SafetyGate().evaluate(
        SafetyInput(
            problem_context="2x+3=7",
            verification_status=VerificationStatus.VERIFIED_INCORRECT,
            has_verified_error=True,
        )
    )
    assert result.allowed_actions == [
        TeachingAction.FOCUS,
        TeachingAction.SCAFFOLD,
        TeachingAction.CORRECT,
        TeachingAction.EXPLAIN,
    ]


def test_model_requires_one_of_the_three_approved_triggers() -> None:
    ordinary = SafetyGate().evaluate(SafetyInput(problem_context="一个数学问题"))
    requested = SafetyGate().evaluate(
        SafetyInput(problem_context="一个数学问题", intent=TeachingIntent.REQUESTS_EXAMPLE)
    )
    assert TeachingAction.MODEL not in ordinary.allowed_actions
    assert requested.worked_example_allowed is True
    assert TeachingAction.MODEL in requested.allowed_actions
```

- [ ] **Step 2: Run the safety tests and verify RED**

Run: `python -m pytest tests/contracts/test_safety_gate.py -v`

Expected: FAIL with `ModuleNotFoundError: app.agent.safety_gate`.

- [ ] **Step 3: Implement pure candidate restriction rules**

Create `SafetyInput` as a Pydantic model with the fields in the interface. Implement `SafetyGate.evaluate()` in this exact decision order:

```python
if not context.problem_context.strip():
    return SafetyConstraints(
        allowed_actions=[TeachingAction.ELICIT],
        disclosure_ceiling=DisclosureLevel.NONE,
        reason_codes=["missing_problem_context"],
    )

example_allowed = (
    context.intent is TeachingIntent.REQUESTS_EXAMPLE
    or context.consecutive_verified_errors >= 2
    or context.expresses_total_confusion
)

if context.verification_status is VerificationStatus.VERIFIED_INCORRECT:
    allowed = [FOCUS, SCAFFOLD, CORRECT, EXPLAIN]
elif context.verification_status is VerificationStatus.VERIFIED_CORRECT:
    allowed = [CHECK, EXPLAIN, CONSOLIDATE, PRACTICE]
elif context.intent is TeachingIntent.REQUESTS_PRACTICE:
    allowed = [PRACTICE, CHECK]
else:
    allowed = [FOCUS, SCAFFOLD, EXPLAIN, CHECK]

if example_allowed:
    allowed.append(MODEL)

disclosure = DisclosureLevel.NEXT_STEP
if (
    context.intent is TeachingIntent.REQUESTS_ANSWER
    and context.candidate_available
    and context.candidate_confidence >= 0.8
):
    disclosure = DisclosureLevel.FINAL
```

Use enum-qualified values in production code. Set `new_problem_allowed=True` for `REQUESTS_PRACTICE` or a `VERIFIED_CORRECT` turn; otherwise false. This permits PRACTICE only after an explicit practice request or confirmed progress. Set `student_response_required=True` for every milestone-1 branch. Add deterministic `reason_codes` for verifier status, example trigger, new-problem permission, answer authorization, and blocked final disclosure.

- [ ] **Step 4: Add the explicit “completely confused” boundary test**

Extend the same test file with:

```python
@pytest.mark.parametrize("message", ["我完全不懂", "我一点都不会", "我完全不会做"])
def test_total_confusion_markers_are_narrow(message: str) -> None:
    context = SafetyInput.from_turn(problem_context="一个数学问题", message=message)
    assert context.expresses_total_confusion is True


def test_ordinary_method_uncertainty_does_not_unlock_model() -> None:
    context = SafetyInput.from_turn(problem_context="一个数学问题", message="我不知道怎么选择方法")
    assert context.expresses_total_confusion is False
```

Implement `from_turn()` with exactly the three markers above; do not treat generic “不会”“不知道” as the complete-confusion trigger.

- [ ] **Step 5: Run the safety and existing policy tests**

Run:

```powershell
python -m pytest tests/contracts/test_safety_gate.py tests/test_generic_teaching_policy.py tests/test_student_requests.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit SafetyGate**

```powershell
git add app/agent/safety_gate.py tests/contracts/test_safety_gate.py
git commit -m "feat: constrain teaching actions with safety gate"
```

---

### Task 4: Add the Minimal Evidence-Ledger StudentModel

**Files:**
- Modify: `app/memory/schemas.py`
- Create: `app/agent/student_model.py`
- Create: `tests/contracts/test_student_model.py`
- Modify: `tests/test_three_layer_memory.py` only for serialization compatibility if the new default field appears in an exact dictionary assertion.

**Interfaces:**
- Produces: `StudentEvidenceKind`, `StudentEvidenceRecord` in `app.memory.schemas`.
- Produces: `StudentEvidenceSnapshot` in `app.memory.schemas` and `StudentModel.snapshot()` / `StudentModel.record_turn()` in `app.agent.student_model`.
- Persists: `EpisodeState.student_evidence`, capped at 40 records.
- Forbids: broad ability, personality, or preferred-style inference.

- [ ] **Step 1: Write RED evidence-ledger tests**

Create `tests/contracts/test_student_model.py`:

```python
import pytest

from app.agent.student_model import StudentModel
from app.agent.teaching_actions import TeachingAction, TeachingActionPlan
from app.agent.state import MathAnalysis, VerificationResult, VerificationStatus
from app.memory.schemas import EpisodeState, StudentEvidenceKind


pytestmark = pytest.mark.safety_contract


def _episode() -> EpisodeState:
    return EpisodeState(conversation_id="evidence-conv", student_id="student")


def test_verified_error_is_stored_with_source_and_verifier() -> None:
    episode = _episode()
    StudentModel().record_turn(
        episode,
        verification=VerificationResult(
            status=VerificationStatus.VERIFIED_INCORRECT,
            evidence=["移项时符号没有改变"],
            verifier_name="linear_equation_fraction_checker",
        ),
        action_plan=TeachingActionPlan(action=TeachingAction.FOCUS),
        analysis=MathAnalysis(),
        source_turn=2,
        scope_id="linear-equation",
    )
    record = episode.student_evidence[0]
    assert record.kind is StudentEvidenceKind.VERIFIED_ERROR
    assert record.source_turn == 2
    assert record.verifier_name == "linear_equation_fraction_checker"


def test_snapshot_contains_evidence_not_personality_claims() -> None:
    snapshot = StudentModel().snapshot(_episode(), scope_id="linear-equation")
    dumped = snapshot.model_dump(mode="json")
    assert "preferred_teaching_style" not in dumped
    assert "ability_level" not in dumped
    assert "confirmed_correct" in dumped
    assert "verified_errors" in dumped


def test_correct_verification_after_scaffold_records_progress() -> None:
    episode = _episode()
    model = StudentModel()
    model.record_turn(
        episode,
        verification=VerificationResult(),
        action_plan=TeachingActionPlan(action=TeachingAction.SCAFFOLD, scaffold_level=1),
        analysis=MathAnalysis(),
        source_turn=1,
        scope_id="linear-equation",
    )
    model.record_turn(
        episode,
        verification=VerificationResult(status=VerificationStatus.VERIFIED_CORRECT, evidence=["2x=4"]),
        action_plan=TeachingActionPlan(action=TeachingAction.CHECK),
        analysis=MathAnalysis(),
        source_turn=2,
        scope_id="linear-equation",
    )
    assert model.snapshot(episode, scope_id="linear-equation").progress_after_latest_hint is True
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/contracts/test_student_model.py -v`

Expected: FAIL because `app.agent.student_model` and evidence schemas do not exist.

- [ ] **Step 3: Add persistent evidence schemas**

Add to `app/memory/schemas.py` before `EpisodeState`:

```python
from enum import Enum


class StudentEvidenceKind(str, Enum):
    CONFIRMED_CORRECT = "confirmed_correct"
    VERIFIED_ERROR = "verified_error"
    UNRESOLVED_OBSTACLE = "unresolved_obstacle"
    TEACHING_ACTION = "teaching_action"
    HINT_OUTCOME = "hint_outcome"
    SOFT_CLAIM = "soft_claim"


class StudentEvidenceRecord(BaseModel):
    evidence_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = Field(default_factory=now_iso)
    source_turn: int = Field(ge=1)
    scope_id: str = ""
    kind: StudentEvidenceKind
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    verifier_name: str = ""
    teaching_action_code: str = ""
    scaffold_level: int = Field(default=0, ge=0, le=3)
    progressed_after_hint: bool | None = None
```

Add `student_evidence: list[StudentEvidenceRecord] = Field(default_factory=list)` and `last_orchestration_trace: dict[str, Any] = Field(default_factory=dict)` to `EpisodeState`.

- [ ] **Step 4: Implement snapshot and record rules**

Add `StudentEvidenceSnapshot` to `app/memory/schemas.py`:

```python
class StudentEvidenceSnapshot(BaseModel):
    records: list[StudentEvidenceRecord] = Field(default_factory=list)
    confirmed_correct: list[StudentEvidenceRecord] = Field(default_factory=list)
    verified_errors: list[StudentEvidenceRecord] = Field(default_factory=list)
    unresolved_obstacles: list[StudentEvidenceRecord] = Field(default_factory=list)
    recent_action_codes: list[str] = Field(default_factory=list)
    latest_scaffold_level: int = Field(default=0, ge=0, le=3)
    latest_verified_step: str = ""
    progress_after_latest_hint: bool | None = None
    consecutive_verified_errors: int = Field(default=0, ge=0)
```

Create `app/agent/student_model.py`. `record_turn()` must append one `TEACHING_ACTION` record each turn; append verified records only when status is `VERIFIED_CORRECT` or `VERIFIED_INCORRECT`; append an `UNRESOLVED_OBSTACLE` only when `analysis.likely_obstacle` is non-empty, with confidence no greater than `analysis.confidence`; and append `HINT_OUTCOME(progressed_after_hint=True)` when the most recent prior action is `scaffold` and the current status is verified correct. Keep only the last 40 records.

`snapshot()` filters by `scope_id` when provided. `latest_verified_step` is the text of the newest `CONFIRMED_CORRECT` record or an empty string. Count consecutive verified errors by walking matching records backward until the first non-error mathematical outcome.

- [ ] **Step 5: Run schema, store, and student-model tests**

Run:

```powershell
python -m pytest tests/contracts/test_student_model.py tests/test_three_layer_memory.py tests/test_topic_agnostic_state.py -v
```

Expected: PASS; no test or production branch reads `preferred_teaching_style` to choose an action.

- [ ] **Step 6: Commit the evidence ledger**

```powershell
git add app/memory/schemas.py app/agent/student_model.py tests/contracts/test_student_model.py tests/test_three_layer_memory.py
git commit -m "feat: record minimal student evidence ledger"
```

---

### Task 5: Add Testable Strategy Capabilities Without Extra LLM Calls

**Files:**
- Create: `app/agent/strategy_capabilities.py`
- Create: `tests/contracts/test_strategy_capabilities.py`
- Modify: `app/agent/state.py`
- Modify: `app/agent/generic_math_reasoning.py`
- Modify: `tests/test_generic_math_reasoning.py`

**Interfaces:**
- Produces: `StrategyId`, `StrategyCandidate`, `StrategySelection`, `StrategyContext`, `StrategyGuidance`, `StrategyBundle`, `StrategyCapability`, `StrategyRegistry`.
- Initial capabilities: `unknown` and `symbolic_transformation` only.
- `GenericMathReasoning.strategy_selection` carries selection in the existing reasoner JSON call.
- Registry makes zero LLM calls.

- [ ] **Step 1: Write RED registry and normalization tests**

Create `tests/contracts/test_strategy_capabilities.py` with:

```python
import pytest

from app.agent.strategy_capabilities import (
    StrategyCandidate, StrategyContext, StrategyId, StrategyRegistry,
    StrategySelection, normalize_strategy_selection,
)


pytestmark = pytest.mark.safety_contract


def test_low_confidence_primary_becomes_unknown() -> None:
    normalized = normalize_strategy_selection(
        StrategySelection(primary=StrategyCandidate(strategy_id=StrategyId.SYMBOLIC_TRANSFORMATION, confidence=0.59))
    )
    assert normalized.primary.strategy_id is StrategyId.UNKNOWN


def test_supporting_is_unique_and_capped_at_two() -> None:
    selection = StrategySelection(
        primary=StrategyCandidate(strategy_id=StrategyId.SYMBOLIC_TRANSFORMATION, confidence=0.9),
        supporting=[
            StrategyCandidate(strategy_id=StrategyId.QUANTITATIVE_MODELING, confidence=0.9),
            StrategyCandidate(strategy_id=StrategyId.QUANTITATIVE_MODELING, confidence=0.8),
            StrategyCandidate(strategy_id=StrategyId.GEOMETRIC_REASONING, confidence=0.7),
            StrategyCandidate(strategy_id=StrategyId.DATA_ANALYSIS, confidence=0.7),
        ],
    )
    normalized = normalize_strategy_selection(selection)
    assert [item.strategy_id for item in normalized.supporting] == [
        StrategyId.QUANTITATIVE_MODELING,
        StrategyId.GEOMETRIC_REASONING,
    ]


def test_registry_returns_one_primary_and_ignores_missing_supporting_plugins() -> None:
    bundle = StrategyRegistry.default().advise(
        StrategySelection(primary=StrategyCandidate(strategy_id=StrategyId.SYMBOLIC_TRANSFORMATION, confidence=0.9)),
        StrategyContext(problem_context="2x+3=7", student_message="我不知道第一步"),
    )
    assert bundle.primary.source_strategy is StrategyId.SYMBOLIC_TRANSFORMATION
    assert bundle.supporting == []
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/contracts/test_strategy_capabilities.py -v`

Expected: FAIL with missing strategy module.

- [ ] **Step 3: Implement schemas, normalization, and the initial registry**

Define the five stable IDs now, even though only two plugins are registered:

```python
class StrategyId(str, Enum):
    SYMBOLIC_TRANSFORMATION = "symbolic_transformation"
    QUANTITATIVE_MODELING = "quantitative_modeling"
    GEOMETRIC_REASONING = "geometric_reasoning"
    DATA_ANALYSIS = "data_analysis"
    UNKNOWN = "unknown"


class StrategyCandidate(BaseModel):
    strategy_id: StrategyId = StrategyId.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class StrategySelection(BaseModel):
    primary: StrategyCandidate = Field(default_factory=StrategyCandidate)
    supporting: list[StrategyCandidate] = Field(default_factory=list)


class StrategyContext(BaseModel):
    problem_context: str
    student_message: str
    reasoning_next_step: str = ""
    verifier_name: str = ""
    verified_evidence: list[str] = Field(default_factory=list)
    unresolved_obstacles: list[str] = Field(default_factory=list)


class StrategyGuidance(BaseModel):
    source_strategy: StrategyId
    representation: str = ""
    operation_types: list[str] = Field(default_factory=list)
    diagnostic_target: str = ""
    misconception_checks: list[str] = Field(default_factory=list)
    verifier_name: str = ""
    normalizers: list[str] = Field(default_factory=list)
    primary_next_step: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class StrategyBundle(BaseModel):
    primary: StrategyGuidance = Field(
        default_factory=lambda: StrategyGuidance(source_strategy=StrategyId.UNKNOWN)
    )
    supporting: list[StrategyGuidance] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


class StrategyCapability(Protocol):
    strategy_id: StrategyId

    def advise(self, context: StrategyContext) -> StrategyGuidance:
        """Return deterministic guidance without choosing an action."""
```

Use `0.60` as the minimum confidence. Implement normalization exactly:

```python
def normalize_strategy_selection(selection: StrategySelection) -> StrategySelection:
    primary = selection.primary
    if primary.confidence < 0.60:
        primary = StrategyCandidate(
            strategy_id=StrategyId.UNKNOWN,
            confidence=primary.confidence,
            rationale="primary_below_confidence_threshold",
        )

    supporting: list[StrategyCandidate] = []
    seen = {primary.strategy_id}
    for candidate in selection.supporting:
        if candidate.confidence < 0.60 or candidate.strategy_id in seen:
            continue
        seen.add(candidate.strategy_id)
        supporting.append(candidate)
        if len(supporting) == 2:
            break
    return StrategySelection(primary=primary, supporting=supporting)
```

`SymbolicTransformationCapability.advise()` may use `StrategyContext.reasoning_next_step` but must not solve the problem. Its deterministic fallback is:

```python
StrategyGuidance(
    source_strategy=StrategyId.SYMBOLIC_TRANSFORMATION,
    representation="使用题目已有的等式或代数关系表示当前状态。",
    operation_types=["比较关系", "选择等价变换", "检查变换前后是否保持关系"],
    diagnostic_target="学生能说明下一步变换的对象和依据。",
    misconception_checks=["是否只改变了等式一侧", "是否把候选结论当成已验证事实"],
    verifier_name=context.verifier_name,
    normalizers=["保留原题符号", "一次只提出一个主要下一步"],
    primary_next_step=context.reasoning_next_step,
    confidence=0.8,
)
```

`UnknownCapability` returns a guidance object with `source_strategy=UNKNOWN`, an empty next step, and confidence `0.0`. `StrategyRegistry.default()` registers only `UnknownCapability` and `SymbolicTransformationCapability`. `advise()` first normalizes selection, invokes the selected primary once, invokes only registered supporting plugins, catches each plugin's `Exception` separately, and records `<strategy_id>:<exception type>` in `failure_reasons`. If primary is missing or fails, invoke `UnknownCapability`; never call an LLM.

- [ ] **Step 4: Extend the existing reasoner schema without adding a call**

Add `strategy_selection: StrategySelection = Field(default_factory=StrategySelection)` to `GenericMathReasoning`. Update `GenericMathReasoner` system prompt to request one primary and at most two supporting capability IDs in the same JSON result. The offline/unconfigured and exception fallbacks use the default `unknown` selection.

Add to `tests/test_generic_math_reasoning.py` an assertion that the fake client receives exactly one `chat_json` call and the returned strategy selection survives Pydantic validation.

- [ ] **Step 5: Run strategy and reasoner tests**

Run:

```powershell
python -m pytest tests/contracts/test_strategy_capabilities.py tests/test_generic_math_reasoning.py -v
```

Expected: PASS and exactly one reasoner JSON call.

- [ ] **Step 6: Commit the strategy capability boundary**

```powershell
git add app/agent/strategy_capabilities.py app/agent/state.py app/agent/generic_math_reasoning.py tests/contracts/test_strategy_capabilities.py tests/test_generic_math_reasoning.py
git commit -m "feat: add testable strategy capability registry"
```

---

### Task 6: Add the Adaptive Planner and Record It in Shadow Mode

**Files:**
- Create: `app/agent/orchestration_trace.py`
- Create: `app/agent/teaching_planner.py`
- Create: `tests/contracts/test_teaching_planner.py`
- Create: `tests/contracts/test_orchestration_degradation.py`
- Create: `tests/behavior/test_shadow_planning.py`
- Modify: `app/agent/state.py`
- Modify: `app/agent/optional_verifier.py`
- Modify: `app/agent/teaching_harness.py`
- Modify: `app/agent/workflow.py`

**Interfaces:**
- Produces: `PlannerInput`, `PlannerResult`, `TeachingPlanner.plan()`.
- Produces: `ModuleDecision` and `OrchestrationTrace` with `llm_call_count`, `shadow_mode`, active plan, shadow plan, and call/skip reasons.
- In this Task: planner result is shadow-only and cannot affect response text or active action.

- [ ] **Step 1: Write RED planner validation and call-budget tests**

Create `tests/contracts/test_teaching_planner.py`:

```python
import pytest

from app.agent.teaching_actions import SafetyConstraints, TeachingAction, TeachingActionPlan
from app.agent.teaching_planner import PlannerInput, TeachingPlanner


pytestmark = pytest.mark.safety_contract


def _input(actions: list[TeachingAction], **updates) -> PlannerInput:
    base = PlannerInput(
        problem_context="2x+3=7",
        current_student_request="我不知道下一步",
        constraints=SafetyConstraints(allowed_actions=actions),
    )
    return base.model_copy(update=updates)


def test_single_candidate_skips_llm() -> None:
    result = TeachingPlanner().plan(_input([TeachingAction.ELICIT]))
    assert result.plan.action is TeachingAction.ELICIT
    assert result.used_llm is False
    assert result.reason == "single_allowed_action"


def test_invalid_model_action_falls_back_inside_allowed_set() -> None:
    class Client:
        def chat_json(self, **kwargs):
            return TeachingActionPlan(action=TeachingAction.MODEL)

    result = TeachingPlanner(client=Client()).plan(
        _input([TeachingAction.FOCUS, TeachingAction.SCAFFOLD])
    )
    assert result.plan.action in {TeachingAction.FOCUS, TeachingAction.SCAFFOLD}
    assert result.reason == "planner_output_rejected"


def test_correct_requires_verified_error_evidence() -> None:
    result = TeachingPlanner().validate_or_fallback(
        TeachingActionPlan(action=TeachingAction.CORRECT),
        _input([TeachingAction.CORRECT, TeachingAction.FOCUS], has_verified_error=False),
    )
    assert result.plan.action is TeachingAction.FOCUS
```

- [ ] **Step 2: Run and verify planner RED**

Run: `python -m pytest tests/contracts/test_teaching_planner.py -v`

Expected: FAIL with missing planner module.

- [ ] **Step 3: Implement planner validation and deterministic fallback**

Define the input and result models exactly:

```python
class PlannerInput(BaseModel):
    problem_context: str
    current_student_request: str
    constraints: SafetyConstraints
    verified_facts: list[str] = Field(default_factory=list)
    student_evidence: dict = Field(default_factory=dict)
    strategy_bundle: dict = Field(default_factory=dict)
    has_verified_error: bool = False
    has_confirmed_progress: bool = False


class PlannerResult(BaseModel):
    plan: TeachingActionPlan
    used_llm: bool = False
    reason: str = ""
```

The fallback priority must be deterministic and always filtered by `allowed_actions`:

```python
_FALLBACK_PRIORITY = (
    TeachingAction.ELICIT,
    TeachingAction.FOCUS,
    TeachingAction.SCAFFOLD,
    TeachingAction.CHECK,
    TeachingAction.EXPLAIN,
    TeachingAction.CORRECT,
    TeachingAction.CONSOLIDATE,
    TeachingAction.PRACTICE,
    TeachingAction.MODEL,
)
```

Construct valid defaults: SCAFFOLD gets level 1; PRACTICE gets SIMILAR and SAME_STRATEGY; all other actions keep zero/none parameters. Reject an action outside the allowed set; reject MODEL unless `worked_example_allowed`; reject PRACTICE unless `new_problem_allowed`; reject CORRECT without verified error; reject CONSOLIDATE without confirmed progress.

If more than one action remains and a client or configured DeepSeek is available, call `chat_json(schema=TeachingActionPlan, temperature=0.0, max_tokens=300)`. Catch `DeepSeekError`, `ValueError`, and Pydantic validation errors, then fallback. No other client is legal.

- [ ] **Step 4: Write the shadow non-interference behavior test**

Create `tests/behavior/test_shadow_planning.py`:

```python
import pytest

from app.agent.state import GenericMathReasoning
from app.agent.teaching_actions import TeachingAction, TeachingActionPlan
from app.agent.teaching_planner import TeachingPlanner
from app.agent.workflow import run_teaching_turn


pytestmark = pytest.mark.teaching_behavior


class Reasoner:
    def analyze(self, message, problem_context, analysis):
        return GenericMathReasoning(next_step_hint="先整理已知关系。", confidence=0.9)


class PlannerClient:
    def chat_json(self, **kwargs):
        return TeachingActionPlan(action=TeachingAction.EXPLAIN)


class ResponseClient:
    def chat(self, system, user, temperature=0.2, max_tokens=None):
        return "先整理题目中已经明确的关系。你准备先写哪一条关系？"


def test_shadow_plan_is_recorded_but_does_not_control_response() -> None:
    state = run_teaching_turn(
        "shadow-student",
        "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
        reasoner=Reasoner(),
        planner=TeachingPlanner(client=PlannerClient()),
        response_client=ResponseClient(),
        analysis_allow_llm=False,
    )
    assert state.action_plan.action is TeachingAction.SCAFFOLD
    assert state.orchestration_trace.shadow_plan.action is TeachingAction.EXPLAIN
    assert state.orchestration_trace.shadow_mode is True
    assert state.response == "先整理题目中已经明确的关系。你准备先写哪一条关系？"
```

- [ ] **Step 5: Assemble evidence in the harness with verifier preflight**

Update `run_generic_teaching_turn()` to accept `planner: TeachingPlanner | None = None`, `activate_planner: bool = False`, and an injectable `episode_store=None` used only for deterministic storage-failure tests. Update `run_teaching_turn()` to accept and forward `planner`; the compatibility `run_turn()` signature remains unchanged.

Create the `VerificationRequest` with:

```python
student_attempt = message if had_problem_context else ""
request = VerificationRequest(
    problem_context=problem_context,
    student_attempt=student_attempt,
    previous_verified_step=evidence_snapshot.latest_verified_step,
)
```

Call `verifier.supports(request)` before `verify()`. If unsupported, use `NoopVerifier`. If verification is correct or incorrect and includes evidence, skip GenericMathReasoner and record `called=False, reason="verifier_evidence_sufficient"`; otherwise run it once. Assemble strategy guidance, SafetyInput, constraints, and shadow PlannerInput. Set `has_verified_error` when either the current status is verified incorrect or the snapshot contains a verified error; set `has_confirmed_progress` when either the current status is verified correct or the snapshot contains confirmed correct evidence. Keep `state.action_plan = TeachingPolicy().choose(state)` while `activate_planner=False`; store the planner result only in `state.orchestration_trace.shadow_plan`.

Add these fields to TeachingState with safe default factories:

```python
safety_constraints: SafetyConstraints = Field(
    default_factory=lambda: SafetyConstraints(allowed_actions=[TeachingAction.ELICIT])
)
student_evidence_snapshot: StudentEvidenceSnapshot = Field(
    default_factory=StudentEvidenceSnapshot
)
strategy_bundle: StrategyBundle = Field(default_factory=StrategyBundle)
orchestration_trace: OrchestrationTrace = Field(default_factory=OrchestrationTrace)
```

- [ ] **Step 6: Track call and skip reasons**

Define trace models with serializable defaults:

```python
class ModuleDecision(BaseModel):
    module: str
    called: bool
    reason: str
    llm_calls: int = Field(default=0, ge=0)


class OrchestrationTrace(BaseModel):
    decisions: list[ModuleDecision] = Field(default_factory=list)
    llm_call_count: int = Field(default=0, ge=0, le=3)
    shadow_mode: bool = True
    active_plan: TeachingActionPlan | None = None
    shadow_plan: TeachingActionPlan | None = None
    fallback_reasons: list[str] = Field(default_factory=list)
```

Record context resolution, verifier, reasoner, strategy registry, graph provider, planner, response generation, guard, and persistence. Sum `llm_calls`; assert it never exceeds 3. After response generation, call `StudentModel.record_turn()`, store `episode.last_orchestration_trace`, and attempt one episode save. A save error appends `persistence_failed` to `fallback_reasons` and returns the already-guarded response; it does not mutate a fresh episode on the next call.

Add these defaulted fields to `TurnLog` so existing legacy constructors remain valid and future log readers have one schema:

```python
unified_action_plan: dict = Field(default_factory=dict)
orchestration_trace: dict = Field(default_factory=dict)
```

- [ ] **Step 7: Write and pass optional-service degradation tests**

Create `tests/contracts/test_orchestration_degradation.py`:

```python
import pytest

from app.agent.state import GraphContext, VerificationStatus
from app.agent.teaching_harness import run_generic_teaching_turn
from app.memory.schemas import EpisodeState


pytestmark = pytest.mark.safety_contract


class BrokenVerifier:
    name = "broken"

    def supports(self, request):
        return True

    def verify(self, request):
        raise RuntimeError("verifier unavailable")


class BrokenGraph:
    name = "broken"
    available = False

    def get_context(self, request):
        raise RuntimeError("graph unavailable")


class FailingEpisodeStore:
    def get(self, student_id, conversation_id=None):
        return EpisodeState(conversation_id=conversation_id or student_id, student_id=student_id)

    def save(self, episode):
        raise OSError("read-only store")


def test_optional_failures_return_a_guarded_response() -> None:
    state = run_generic_teaching_turn(
        "degraded-student",
        "三角形内角和为什么是180度？",
        verifier=BrokenVerifier(),
        graph_context_provider=BrokenGraph(),
        analysis_allow_llm=False,
    )
    assert state.verification.status is VerificationStatus.NOT_APPLICABLE
    assert state.graph_context == GraphContext()
    assert state.response.count("？") == 1
    assert {"verifier_failed", "graph_context_failed"}.issubset(
        set(state.orchestration_trace.fallback_reasons)
    )


def test_persistence_failure_does_not_block_safe_response() -> None:
    state = run_generic_teaching_turn(
        "storage-student",
        "三角形内角和为什么是180度？",
        episode_store=FailingEpisodeStore(),
        analysis_allow_llm=False,
    )
    assert state.response.count("？") == 1
    assert "persistence_failed" in state.orchestration_trace.fallback_reasons
```

Wrap verifier, graph, injected reasoner, and store calls separately. Verifier failure becomes `NoopVerifier` evidence; graph failure becomes empty `GraphContext`; reasoner failure becomes clarification reasoning; storage failure follows the behavior above. Never catch `KeyboardInterrupt` or `SystemExit`.

- [ ] **Step 8: Run planner, degradation, shadow, verifier, and workflow tests**

Run:

```powershell
python -m pytest tests/contracts/test_teaching_planner.py tests/contracts/test_orchestration_degradation.py tests/behavior/test_shadow_planning.py tests/test_optional_verifier.py tests/test_topic_agnostic_workflow.py -v
```

Expected: PASS; active generic replies remain controlled by the deterministic compatibility policy.

- [ ] **Step 9: Run the batch-2 full offline gate**

Run:

```powershell
python -m pytest
python scripts/run_eval.py
python scripts/run_eval_modes.py --mode deepseek_only
python scripts/run_eval_modes.py --mode agent_no_graph
python scripts/run_eval_modes.py --mode agent_with_graph
```

Expected: strict gates pass; no paid API call; trace call count is at most 3 in new tests.

- [ ] **Step 10: Commit the shadow planner**

```powershell
git add app/agent/orchestration_trace.py app/agent/teaching_planner.py app/agent/state.py app/agent/optional_verifier.py app/agent/teaching_harness.py app/agent/workflow.py tests/contracts/test_teaching_planner.py tests/contracts/test_orchestration_degradation.py tests/behavior/test_shadow_planning.py
git commit -m "feat: record adaptive teaching planner in shadow mode"
```

---

### Task 7: Make TeachingBrief the Sole Response Input

**Files:**
- Create: `app/agent/teaching_brief.py`
- Create: `tests/contracts/test_teaching_brief.py`
- Modify: `app/agent/response_generation.py`
- Modify: `app/agent/teaching_harness.py`
- Modify: `app/agent/state.py`
- Modify: `tests/test_generic_response_generation.py`
- Modify: `tests/test_topic_agnostic_workflow.py`
- Modify: `app/api/main.py`
- Modify: `app/ui/streamlit_app.py`
- Modify: `tests/contracts/test_api_action_contract.py`

**Interfaces:**
- Produces: `TeachingBrief`, `TeachingBriefComposer`, `ResponseGuard`, and `generate_response(brief, client=None)`.
- ResponseGenerator cannot accept `TeachingState`, raw reasoning, raw RAG context, or raw student profile.
- Preserves: exact old public API fields via compatibility serialization.

- [ ] **Step 1: Write RED brief-boundary tests**

Create `tests/contracts/test_teaching_brief.py`:

```python
import pytest

from app.agent.teaching_actions import DisclosureLevel, SafetyConstraints, TeachingAction, TeachingActionPlan
from app.agent.teaching_brief import TeachingBrief, TeachingBriefComposer


pytestmark = pytest.mark.safety_contract


def test_brief_contains_one_next_step_and_one_closing_task() -> None:
    brief = TeachingBriefComposer().compose(
        problem_context="2x+3=7",
        current_student_request="我不知道下一步",
        verified_facts=[],
        student_evidence=[],
        action_plan=TeachingActionPlan(action=TeachingAction.SCAFFOLD, scaffold_level=1),
        constraints=SafetyConstraints(
            allowed_actions=[TeachingAction.SCAFFOLD],
            disclosure_ceiling=DisclosureLevel.NEXT_STEP,
        ),
        next_step_candidates=["先观察常数项", "直接写出最终答案"],
        candidate_conclusion="x=2",
    )
    assert brief.primary_next_step == "先观察常数项"
    assert brief.closing_task
    assert "x=2" in brief.forbidden_disclosures


def test_brief_rejects_plan_outside_constraints() -> None:
    with pytest.raises(ValueError, match="allowed_actions"):
        TeachingBriefComposer().compose(
            problem_context="一个问题",
            current_student_request="请解释",
            verified_facts=[],
            student_evidence=[],
            action_plan=TeachingActionPlan(action=TeachingAction.MODEL),
            constraints=SafetyConstraints(allowed_actions=[TeachingAction.EXPLAIN]),
            next_step_candidates=[],
            candidate_conclusion="",
        )
```

- [ ] **Step 2: Run and verify brief RED**

Run: `python -m pytest tests/contracts/test_teaching_brief.py -v`

Expected: FAIL with missing brief module.

- [ ] **Step 3: Implement the brief schema and composer**

`TeachingBrief` fields must be exactly:

```python
problem_context: str
current_student_request: str
verified_facts: list[str]
student_evidence: list[str]
action_plan: TeachingActionPlan
disclosure_ceiling: DisclosureLevel
primary_next_step: str
allowed_content: list[str]
forbidden_disclosures: list[str]
closing_task: str
```

The composer validates the action against `constraints.allowed_actions`; selects only the first non-empty next-step candidate; includes candidate conclusion in `allowed_content` only when disclosure ceiling is FINAL and otherwise places it in `forbidden_disclosures`; and adds no raw Graph/RAG text. Use this complete closing-task map:

```python
_CLOSING_TASKS = {
    TeachingAction.ELICIT: "你能先把完整题目和已经想到的一步发给我吗？",
    TeachingAction.FOCUS: "你愿意先指出自己最不确定的那一处吗？",
    TeachingAction.SCAFFOLD: "你能按这个提示先写出下一小步吗？",
    TeachingAction.EXPLAIN: "你能用自己的话复述这一步的依据吗？",
    TeachingAction.CORRECT: "你愿意先重新检查这处变化吗？",
    TeachingAction.CHECK: "你能说明这个结论为什么符合原题条件吗？",
    TeachingAction.MODEL: "你能把示范中的方法对应回原题第一步吗？",
    TeachingAction.PRACTICE: "你愿意先独立完成这道练习的第一步吗？",
    TeachingAction.CONSOLIDATE: "你能说出下次遇到同类关系时先看什么吗？",
}
```

- [ ] **Step 4: Refactor response generation around TeachingBrief**

Change the public signature to:

```python
def generate_response(
    brief: TeachingBrief,
    *,
    client: _ResponseClient | None = None,
) -> str:
```

Build the prompt only from `brief.model_dump(mode="json")`. Define `GroundingGuard = Callable[[str, TeachingBrief], bool]` and `ResponseGuard.validate(response, brief, grounding_guard=None)`. It returns false when:

- response is empty;
- question count is not exactly one;
- response does not end in `？` or `?`;
- any normalized `forbidden_disclosures` item is present;
- response introduces a new practice problem while action is not MODEL or PRACTICE.

The composer already makes a MODEL brief impossible when MODEL is outside the constraints. The new-problem check uses the injected `grounding_guard`; the default generic guard returns true without equation parsing. The linear-equation slice in Task 8 supplies a callback around the existing `math_checker.uses_new_example` check through its verifier capability.

Keep one deterministic fallback for each of the nine actions. The fallback must itself pass ResponseGuard.

- [ ] **Step 5: Compose an active brief even while the new planner is shadow-only**

Add `teaching_brief: TeachingBrief | None = None` to TeachingState. In `run_generic_teaching_turn()`, compose the active brief from `state.action_plan` and the already-computed constraints. Save it to `state.teaching_brief`, then call `generate_response(state.teaching_brief, client=response_client)`. Do not pass state, raw strategy bundle, candidate conclusion, or graph evidence into ResponseGenerator.

- [ ] **Step 6: Expose additive API and UI observability**

Return these unified values in the new API fields:

```python
teaching_action_plan=teaching_state.action_plan.model_dump(mode="json")
safety_constraints=teaching_state.safety_constraints.model_dump(mode="json")
orchestration_trace=teaching_state.orchestration_trace.model_dump(mode="json")
```

In `render_meta()`, read those three dictionaries and display unified action, scaffold level, disclosure ceiling, shadow action, LLM call count, and each module decision. Do not remove any existing detail field.

Extend `tests/contracts/test_api_action_contract.py` to assert that old `teaching_action_kind` remains `give_hint`, `teaching_action_plan.action == "scaffold"`, `safety_constraints.allowed_actions` is non-empty, and `orchestration_trace.decisions` is a list.

- [ ] **Step 7: Run brief, response, API, UI-import, and workflow tests**

Run:

```powershell
python -m pytest tests/contracts/test_teaching_brief.py tests/contracts/test_api_action_contract.py tests/test_generic_response_generation.py tests/test_topic_agnostic_workflow.py tests/test_api.py -v
python -m py_compile app/ui/streamlit_app.py
```

Expected: tests PASS; UI source compiles without requiring the optional Streamlit runtime; generic response clients receive only a brief-derived payload.

- [ ] **Step 8: Commit the brief boundary**

```powershell
git add app/agent/teaching_brief.py app/agent/response_generation.py app/agent/teaching_harness.py app/agent/state.py app/api/main.py app/ui/streamlit_app.py tests/contracts/test_teaching_brief.py tests/contracts/test_api_action_contract.py tests/test_generic_response_generation.py tests/test_topic_agnostic_workflow.py
git commit -m "refactor: generate responses only from teaching briefs"
```

---

### Task 8: Activate a Feature-Flagged Linear-Equation Vertical Slice

**Files:**
- Create: `app/agent/runtime_router.py`
- Create: `tests/behavior/test_linear_equation_vertical_slice.py`
- Modify: `app/memory/schemas.py`
- Modify: `app/agent/optional_verifier.py`
- Modify: `app/agent/linear_equation_adapter.py`
- Modify: `app/agent/workflow.py`
- Modify: `app/agent/teaching_harness.py`
- Modify: `app/agent/state.py`

**Interfaces:**
- Produces: `RuntimePath`, `RuntimeDecision`, `resolve_runtime_path()`.
- Adds: `EpisodeState.orchestration_path` so every follow-up remains on the path selected at episode start.
- Activates: `TEACHER_UNIFIED_LINEAR_SLICE=true`; default false preserves existing API and frozen tests.
- Uses: `LinearEquationVerifier`, which delegates to unmodified `math_checker`.

- [ ] **Step 1: Write the real multi-turn RED test**

Create `tests/behavior/test_linear_equation_vertical_slice.py`:

```python
import pytest

from app.agent.state import VerificationStatus
from app.agent.teaching_actions import TeachingAction
from app.agent.workflow import run_teaching_turn


pytestmark = pytest.mark.teaching_behavior


def test_linear_equation_slice_uses_new_orchestrator_and_existing_checker(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_UNIFIED_LINEAR_SLICE", "true")
    conversation_id = "unified-linear-slice"

    first = run_teaching_turn(
        "linear-slice-student",
        "老师，2x+3=7怎么解？",
        conversation_id=conversation_id,
        analysis_allow_llm=False,
    )
    second = run_teaching_turn(
        "linear-slice-student",
        "我得到2x=4",
        conversation_id=conversation_id,
        analysis_allow_llm=False,
    )

    assert first.episode_state.orchestration_path == "unified_linear_slice"
    assert second.episode_state.orchestration_path == "unified_linear_slice"
    assert second.verification.status is VerificationStatus.VERIFIED_CORRECT
    assert second.verification.verifier_name == "linear_equation_fraction_checker"
    assert second.orchestration_trace.shadow_mode is False
    assert second.action_plan.action in {
        TeachingAction.CHECK,
        TeachingAction.EXPLAIN,
        TeachingAction.CONSOLIDATE,
        TeachingAction.PRACTICE,
    }
    assert "x=2" not in first.response
    assert "x=2" not in second.response
    assert first.response.count("？") == 1
    assert second.response.count("？") == 1
```

- [ ] **Step 2: Run and verify routing RED**

Run: `python -m pytest tests/behavior/test_linear_equation_vertical_slice.py -v`

Expected: FAIL because the existing router sends the first turn to the legacy equation runtime.

- [ ] **Step 3: Implement a boundary-only runtime router**

Create `RuntimePath` values `LEGACY_EQUATION`, `GENERIC`, and `UNIFIED_LINEAR_SLICE`. `resolve_runtime_path()` follows this order:

```python
if episode and episode.target_equation:
    return LEGACY_EQUATION  # existing legacy episode never changes owner
if episode and episode.orchestration_path:
    return RuntimePath(episode.orchestration_path)
if episode and episode.problem_context:
    return GENERIC
if is_legacy_concept_request(message):
    return LEGACY_EQUATION
if unified_linear_slice_enabled() and LinearEquationVerifier().supports(
    VerificationRequest(problem_context=message)
):
    return UNIFIED_LINEAR_SLICE
if supports_legacy_equation_turn(student_id, message, conversation_id):
    return LEGACY_EQUATION
return GENERIC
```

Add `orchestration_path: str = ""` to `EpisodeState`. Expose this compatibility-boundary helper from `linear_equation_adapter.py`:

```python
def is_legacy_concept_request(message: str) -> bool:
    return any(marker in message for marker in _LEGACY_CONCEPT_REQUEST_MARKERS)
```

Make `supports_legacy_equation_turn()` call that helper rather than duplicating its expression. `unified_linear_slice_enabled()` reads `TEACHER_UNIFIED_LINEAR_SLICE` and defaults false. Topic recognition remains inside the router/verifier boundary; SafetyGate and TeachingPlanner receive no equation-specific marker.

- [ ] **Step 4: Make verifier selection automatic only for the activated slice**

When route is UNIFIED_LINEAR_SLICE, call the generic harness with `verifier=LinearEquationVerifier()` and `activate_planner=True`. On the first turn, pass an empty `student_attempt` so the problem statement is not graded as a student step. On follow-ups, pass the current message and the latest confirmed step from StudentModel. Supply this grounding callback to ResponseGuard:

```python
def linear_grounding_guard(response: str, brief: TeachingBrief) -> bool:
    introduced_example = mc.uses_new_example(response, brief.problem_context)
    return not introduced_example or brief.action_plan.action in {
        TeachingAction.MODEL,
        TeachingAction.PRACTICE,
    }
```

Persist `episode.orchestration_path = route.value` before response generation. Existing episodes with `target_equation` remain legacy even when the flag is true.

- [ ] **Step 5: Activate the new planner only after validation**

When `activate_planner=True`, set:

```python
state.action_plan = planner_result.plan
state.orchestration_trace.shadow_mode = False
state.orchestration_trace.active_plan = planner_result.plan
state.orchestration_trace.shadow_plan = None
```

Re-compose TeachingBrief from the active plan. For verified correct evidence, candidate conclusion remains forbidden unless SafetyGate grants FINAL disclosure. `math_checker` evidence may be included in `verified_facts`; strategy or reasoner candidates may not be labeled verified.

- [ ] **Step 6: Add rollback and path-stability contract tests**

Extend the behavior file:

```python
def test_linear_slice_flag_off_keeps_legacy_runtime(monkeypatch) -> None:
    monkeypatch.delenv("TEACHER_UNIFIED_LINEAR_SLICE", raising=False)
    state = run_teaching_turn("legacy-student", "我不会解2x+3=7")
    assert state.episode_state.target_equation


def test_existing_legacy_episode_is_not_moved_by_flag(monkeypatch) -> None:
    conversation_id = "existing-legacy-owner"
    run_teaching_turn("legacy-owner", "我不会解2x+3=7", conversation_id=conversation_id)
    monkeypatch.setenv("TEACHER_UNIFIED_LINEAR_SLICE", "true")
    state = run_teaching_turn("legacy-owner", "我得到2x=4", conversation_id=conversation_id)
    assert state.episode_state.target_equation


def test_linear_slice_flag_does_not_capture_an_equation_system(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_UNIFIED_LINEAR_SLICE", "true")
    state = run_teaching_turn(
        "equation-system-owner",
        "老师，3x+2y=5，6x-2y=3怎么解？",
        conversation_id="equation-system-owner",
        analysis_allow_llm=False,
    )
    assert state.episode_state.orchestration_path == "generic"
    assert state.verification.status is VerificationStatus.NOT_APPLICABLE
```

Tighten only `LinearEquationVerifier.supports()`—not `math_checker`—to require `len(mc.find_equations(request.problem_context)) == 1`. This preserves the original equation-system routing fix even when the slice flag is enabled.

- [ ] **Step 7: Run the vertical slice and checker regressions**

Run:

```powershell
python -m pytest tests/behavior/test_linear_equation_vertical_slice.py tests/test_math_checker.py tests/test_flexible_steps.py tests/test_grounding.py tests/test_topic_agnostic_workflow.py -v
```

Expected: PASS; new flagged path uses `linear_equation_fraction_checker`; unflagged and existing legacy episodes behave exactly as before.

- [ ] **Step 8: Run the batch-3 full offline gate**

Run:

```powershell
python -m pytest
python scripts/run_eval.py
python scripts/run_eval_modes.py --mode deepseek_only
python scripts/run_eval_modes.py --mode agent_no_graph
python scripts/run_eval_modes.py --mode agent_with_graph
```

Expected: strict commands pass; the three comparison modes keep their recorded baselines because the feature flag defaults false.

- [ ] **Step 9: Commit the vertical slice and pause at the batch-3 gate**

```powershell
git add app/agent/runtime_router.py app/memory/schemas.py app/agent/optional_verifier.py app/agent/linear_equation_adapter.py app/agent/workflow.py app/agent/teaching_harness.py app/agent/state.py tests/behavior/test_linear_equation_vertical_slice.py
git commit -m "feat: add guarded linear equation orchestration slice"
```

Pause after this commit and report batch-3 evidence. Proceed to Task 9 only after explicit review approval. Do not enable the flag by default and do not begin domain expansion.

---

### Task 9: Add Teaching-Quality Offline Evaluation and Final Milestone Evidence

**Files:**
- Create: `data/eval/teaching_quality_conversations.jsonl`
- Create: `app/evaluation/teaching_quality_evaluator.py`
- Create: `tests/behavior/test_teaching_quality_evaluator.py`
- Modify: `scripts/run_eval.py`
- Modify: `app/ui/streamlit_app.py` only if evaluator trace fields reveal a missing display key.

**Interfaces:**
- Produces: `TeachingQualityCase`, `TeachingQualityResult`, `load_teaching_quality_cases()`, `run_teaching_quality_eval()`.
- Extends: strict offline `scripts/run_eval.py`; does not change three-mode command exit semantics.
- Evaluates: structured action range, progression, evidence, disclosure, one final question, context reuse, and optional-service degradation.

- [ ] **Step 1: Write the evaluator RED test**

Create `tests/behavior/test_teaching_quality_evaluator.py`:

```python
import pytest

from app.evaluation.teaching_quality_evaluator import (
    load_teaching_quality_cases,
    run_teaching_quality_eval,
)


pytestmark = pytest.mark.teaching_behavior


def test_quality_data_contains_real_multiturn_cases() -> None:
    cases = load_teaching_quality_cases()
    assert {case.domain for case in cases} >= {"linear_equation", "equation_system", "geometry", "data_analysis"}
    assert all(len(case.messages) >= 2 for case in cases)


def test_quality_eval_checks_progress_without_exact_prose() -> None:
    results = run_teaching_quality_eval()
    assert results
    assert all(result.passed for result in results)
    assert all("exact_response" not in result.checks for result in results)
```

- [ ] **Step 2: Run and verify evaluator RED**

Run: `python -m pytest tests/behavior/test_teaching_quality_evaluator.py -v`

Expected: FAIL with missing evaluator module.

- [ ] **Step 3: Create four deterministic multi-turn cases**

Write these four complete JSONL records, one per physical line:

```json
{"id":"linear_verified_progress","domain":"linear_equation","messages":["老师，2x+3=7怎么解？","我得到2x=4"],"planner_actions":["scaffold","check"],"allowed_actions_by_turn":[["focus","scaffold","explain","check"],["check","explain","consolidate","practice"]],"forbidden_tokens":["x=2"],"feature_flags":{"TEACHER_UNIFIED_LINEAR_SLICE":"true"},"use_workflow":true,"requires_context_reuse":true,"requires_action_change":true,"requires_response_change":true}
{"id":"equation_system_method_choice","domain":"equation_system","messages":["老师，3x+2y=5，6x-2y=3怎么解？","具体要怎么做呢？","我不知道要怎么选择方法"],"planner_actions":["scaffold","explain","focus"],"allowed_actions_by_turn":[["focus","scaffold","explain","check"],["focus","scaffold","explain","check"],["focus","scaffold","explain","check"]],"forbidden_tokens":["x=8/9","y=7/6"],"feature_flags":{"USE_GRAPH_CONTEXT":"false"},"use_workflow":false,"requires_context_reuse":true,"requires_action_change":true,"requires_response_change":true}
{"id":"geometry_explanation_progress","domain":"geometry","messages":["为什么三角形内角和是180°？","我不知道辅助线怎么想"],"planner_actions":["explain","scaffold"],"allowed_actions_by_turn":[["focus","scaffold","explain","check"],["focus","scaffold","explain","check"]],"forbidden_tokens":["正方形面积"],"feature_flags":{"USE_GRAPH_CONTEXT":"false"},"use_workflow":false,"requires_context_reuse":true,"requires_action_change":true,"requires_response_change":true}
{"id":"data_outlier_reasoning","domain":"data_analysis","messages":["平均数能代表所有数据吗？","如果有一个特别大的数怎么办"],"planner_actions":["explain","focus"],"allowed_actions_by_turn":[["focus","scaffold","explain","check"],["focus","scaffold","explain","check"]],"forbidden_tokens":["平均数一定正确"],"feature_flags":{"USE_GRAPH_CONTEXT":"false"},"use_workflow":false,"requires_context_reuse":true,"requires_action_change":true,"requires_response_change":true}
```

Do not store an exact expected response string.

- [ ] **Step 4: Implement deterministic fixture boundaries and structured checks**

Define the schemas exactly:

```python
class TeachingQualityCase(BaseModel):
    id: str
    domain: str
    messages: list[str]
    planner_actions: list[TeachingAction]
    allowed_actions_by_turn: list[list[TeachingAction]]
    forbidden_tokens: list[str] = Field(default_factory=list)
    feature_flags: dict[str, str] = Field(default_factory=dict)
    use_workflow: bool = False
    requires_context_reuse: bool = True
    requires_action_change: bool = False
    requires_response_change: bool = False

    @model_validator(mode="after")
    def validate_turn_lengths(self) -> "TeachingQualityCase":
        lengths = {
            len(self.messages),
            len(self.planner_actions),
            len(self.allowed_actions_by_turn),
        }
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("messages, planner_actions, and allowed actions must align")
        return self


class TeachingQualityResult(BaseModel):
    case_id: str
    case_type: str = "teaching_quality"
    domain: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    responses: list[str] = Field(default_factory=list)
    final_trace: dict = Field(default_factory=dict)
```

Validate that `messages`, `planner_actions`, and `allowed_actions_by_turn` have equal nonzero lengths. Use `student_id=f"quality-{case.id}"` and `conversation_id=f"quality-{case.id}-conversation"` for every turn in a case. The evaluator injects a fixture reasoner whose next-step hint is `"先确认当前关系，再完成一个可检查的小步骤。"` and whose candidate conclusion is the first `forbidden_tokens` item; this deliberately exercises the disclosure guard. It also injects a scripted planner client that returns the declared `planner_actions` in order, with `scaffold_level=1` for SCAFFOLD. For `use_workflow=true`, call `run_teaching_turn`; otherwise call `run_generic_teaching_turn(..., activate_planner=True)` so generic cases exercise the active new planner without changing production routing. Fixture responses come from the production deterministic fallback templates, not from exact-response fields in JSONL. Apply feature flags with a context manager that restores every prior environment value after each case.

For every turn record checks:

```python
checks = {
    "action_allowed": action in case.allowed_actions_by_turn[index],
    "single_final_question": question_count == 1 and response.endswith(("？", "?")),
    "no_forbidden_disclosure": not any(token in response for token in case.forbidden_tokens),
    "context_reused": state.problem_context == case.messages[0],
    "evidence_trace_present": bool(state.orchestration_trace.decisions),
}
```

At conversation end, add `action_changed` and `response_advanced` checks when requested. Set `passed=all(checks.values())`; include failed check names, action sequence, response sequence, and final trace in the result. This evaluator never invokes a real LLM because both reasoner and planner are injected deterministic fixtures and response generation uses offline fallbacks.

- [ ] **Step 5: Add the evaluator to strict run_eval**

Import `run_teaching_quality_eval` inside `main()` after offline environment setup and change:

```python
results = run_all() + run_generic_math_eval() + run_teaching_quality_eval()
```

Do not change existing result schemas or make case failures non-fatal.

- [ ] **Step 6: Run evaluator and strict offline tests**

Run:

```powershell
python -m pytest tests/behavior/test_teaching_quality_evaluator.py tests/test_generic_math_evaluator.py -v
python scripts/run_eval.py
```

Expected: all new quality cases pass and appear with case type `teaching_quality`; strict command exits 0.

- [ ] **Step 7: Run final complete verification**

Run:

```powershell
python -m pytest
python scripts/run_eval.py
python scripts/run_eval_modes.py --mode deepseek_only
python scripts/run_eval_modes.py --mode agent_no_graph
python scripts/run_eval_modes.py --mode agent_with_graph
git diff --check
git status --short
git log --oneline -12
```

Record exact counts for pytest, strict eval, and all three modes. Confirm only the pre-existing untracked `teacher-agent-strategy-family-handoff.md` remains outside commits.

- [ ] **Step 8: Commit evaluation and stop at milestone 1**

```powershell
git add data/eval/teaching_quality_conversations.jsonl app/evaluation/teaching_quality_evaluator.py tests/behavior/test_teaching_quality_evaluator.py scripts/run_eval.py app/ui/streamlit_app.py
git commit -m "test: evaluate contract-first teaching progression"
```

If `app/ui/streamlit_app.py` was unchanged, omit it from `git add`. Stop after the commit. Do not implement equation-system, geometry, modeling, or data-analysis capability plugins and do not delete the legacy compatibility layer.

---

## Milestone 1 Completion Report

The implementation worker must report:

- current branch;
- actual modified files grouped by Task;
- RED command and failure reason for every Task;
- GREEN focused-test evidence for every Task;
- exact complete pytest result;
- exact strict `run_eval` result;
- exact results for `deepseek_only`, `agent_no_graph`, and `agent_with_graph`;
- every new commit hash and subject;
- final `git status --short`;
- confirmation that no real paid API was called;
- confirmation that the feature flag defaults off;
- replacement-test mapping for any legacy assertion changed or removed;
- blockers discovered for separate batch-4 domain plans;
- blockers discovered for the eventual batch-5 compatibility-layer exit.

## Follow-On Planning Gates

After Milestone 1 is reviewed, create separate specs and plans in this order:

1. symbolic-transformation activation for equation systems, using the approved three-turn conversation;
2. geometric-reasoning plus quantitative-modeling composition;
3. data-analysis capability;
4. remaining legacy linear-equation behavior migration, split by explanation, hints/checker feedback, worked examples, practice, and review;
5. compatibility-layer exit after every legacy test has an explicit replacement mapping.

None of these follow-on items is authorized or implemented by this plan.
