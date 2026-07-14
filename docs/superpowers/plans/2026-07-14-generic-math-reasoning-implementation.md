# 通用数学推理 Harness 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让非一元一次方程的初中数学问题可通过 DeepSeek 的结构化推理获得受教学约束的提示、解释或获授权答案，同时保持线性方程 checker 和现有离线回归不变。

**Architecture:** `MathAnalysis` 保留交互诊断职责；新增 `GenericMathReasoning` 仅输出题意、可教学步骤、候选结论和不确定性。Harness 将分析、可选验证、策略与受控回复串联，只有策略显式允许时才向回复层传递候选结论；一元一次方程仍走 `linear_equation_adapter.py`。

**Tech Stack:** Python 3.11、Pydantic、DeepSeek OpenAI-compatible client、pytest、现有 JSONL 评测。

## 全局约束

- 只在 `refactor/topic-agnostic-teaching-harness` 分支增量修改，不合并或改写 main/master。
- 所有实际 LLM 调用只能通过 `app/llm/deepseek_client.py`；结构化输出必须经 Pydantic JSON 校验。
- `math_checker.py` 仍是线性方程步骤、计划和最终答案的唯一确定性裁决来源；通用 LLM 推理不得写入 `verified_*` 状态。
- 普通通用路径不依赖 RAG/GraphRAG；缺失 provider、模型配置或模型响应时必须返回单问句安全回复。
- 学生未明确请求完整答案时不得暴露 `candidate_conclusion`；回复围绕 `problem_context`，至多一个结尾问句。
- 所有 pytest 与 `scripts/run_eval.py` 默认为离线；真实 smoke 必须显式 `--live`、至多 12 样本、每样本至多 reasoning 和 response 两次调用、输出上限 450/350 tokens、无重试，预算硬上限 2 元。
- 不新建知识包、领域分支或通用符号求解器；不改动线性方程 adapter、math checker 算法、RAG 数据或既有 golden case 的期望。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `app/agent/state.py` | 声明通用推理 schema，并把它加入 `TeachingState`、双向兼容投影。 |
| `app/llm/deepseek_client.py` | 为已有 chat/chat_json 增加可选的 `max_tokens`，不改变未传参数时的行为。 |
| `app/agent/generic_math_reasoning.py` | 单一职责：请求或降级为 `GenericMathReasoning`，不选择教学动作。 |
| `app/agent/math_analysis.py` | 允许 Harness 在预算 smoke 中关闭分析 LLM，以保证每个样本最多两次调用。 |
| `app/agent/teaching_policy.py` | 用验证优先、澄清优先和显式授权规则选择动作及答案披露位。 |
| `app/agent/response_generation.py` | 只向模型交付策略批准的事实，标准化为一条结尾问句，并在失败时回退安全模板。 |
| `app/agent/teaching_harness.py` | 注入 reasoner/response client，按分析→推理→验证→策略→回复顺序编排。 |
| `app/evaluation/generic_math_evaluator.py` | 用离线 fake LLM 对跨领域教学守卫做确定性检查。 |
| `data/eval/topic_agnostic_smoke.jsonl` | 保存 12 条跨比例、代数、几何、统计、概率的 smoke 样本；是评测样本，不是知识包。 |
| `scripts/run_eval.py` | 合并既有方程回归和新的离线通用 Harness 评测。 |
| `scripts/run_generic_math_smoke.py` | 仅在 `--live` 下按预算上限运行人工审阅用 smoke 并写 JSONL 日志。 |
| `tests/test_generic_math_reasoning.py` | reasoner、schema、预算参数和降级的单元测试。 |
| `tests/test_generic_teaching_policy.py` | 回归并扩展披露门控与验证优先级测试。 |
| `tests/test_generic_response_generation.py` | 验证受控上下文、单问句、泄露拦截和 fallback。 |
| `tests/test_generic_math_evaluator.py` | 验证离线跨领域评测和 live runner 的拒绝/上限行为。 |

## Task 1: 通用推理 schema 与 DeepSeek token 上限

**Files:**
- Modify: `app/agent/state.py`
- Modify: `app/llm/deepseek_client.py`
- Modify: `tests/test_topic_agnostic_state.py`
- Modify: `tests/test_deepseek_integration.py`

**Interfaces:**
- Produces `GenericMathReasoning` 和 `TeachingState.math_reasoning`。
- Produces `DeepSeekClient.chat(system, user, temperature=0.7, max_tokens: int | None = None)` 与 `chat_json(system, user, schema, temperature=0.0, max_tokens: int | None = None)`；未提供时 API 参数中不出现 `max_tokens`。

- [ ] **Step 1: 写失败 schema 与 token 参数测试。**

```python
from app.agent.state import GenericMathReasoning, TeachingState


def test_teaching_state_carries_generic_reasoning_without_topic_id() -> None:
    state = TeachingState(
        student_id="ratio-student",
        message="比是3:2，总数30人。",
        math_reasoning=GenericMathReasoning(
            problem_interpretation="总份数是5份。",
            reasoning_steps=["先求总份数 3+2。"],
            next_step_hint="把30按5份平均分。",
            candidate_conclusion="男生18人，女生12人。",
            confidence=0.9,
        ),
    )

    dumped = state.model_dump(mode="json")
    assert dumped["math_reasoning"]["candidate_conclusion"] == "男生18人，女生12人。"
    assert "concept_id" not in dumped["math_reasoning"]
```

```python
def test_chat_json_passes_optional_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    client = _client_with_recording_completion(monkeypatch, calls)

    client.chat_json("system", "user", _ReplySchema, max_tokens=450)

    assert calls[-1]["max_tokens"] == 450
```

- [ ] **Step 2: 运行指定测试，确认因类型/参数尚不存在而失败。**

Run: `python -m pytest tests/test_topic_agnostic_state.py::test_teaching_state_carries_generic_reasoning_without_topic_id tests/test_deepseek_integration.py::test_chat_json_passes_optional_max_tokens -v`

Expected: FAIL，提示 `GenericMathReasoning` 或 `max_tokens` 尚不存在。

- [ ] **Step 3: 实现最小 schema 和向后兼容 token 参数。**

```python
class GenericMathReasoning(BaseModel):
    problem_interpretation: str = ""
    student_attempt_assessment: str = ""
    reasoning_steps: list[str] = Field(default_factory=list)
    next_step_hint: str = ""
    candidate_conclusion: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    uncertainty_note: str = ""


# Insert immediately after the existing `math_analysis` field in `TeachingState`.
math_reasoning: GenericMathReasoning = Field(default_factory=GenericMathReasoning)
```

```python
def _completion_options(max_tokens: int | None) -> dict[str, int]:
    return {"max_tokens": max_tokens} if max_tokens is not None else {}


def chat(self, system: str, user: str, temperature: float = 0.7,
         max_tokens: int | None = None) -> str:
    resp = self._client.chat.completions.create(
        model=self.settings.deepseek_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        **_completion_options(max_tokens),
    )
```

Apply the same optional expansion to `chat_json`, preserving `response_format={"type": "json_object"}`.

- [ ] **Step 4: 运行新增测试与既有 client 测试。**

Run: `python -m pytest tests/test_topic_agnostic_state.py tests/test_deepseek_integration.py -v`

Expected: PASS；不访问网络。

- [ ] **Step 5: 提交本任务。**

```bash
git add app/agent/state.py app/llm/deepseek_client.py tests/test_topic_agnostic_state.py tests/test_deepseek_integration.py
git commit -m "feat: add generic math reasoning state"
```

## Task 2: 可注入的结构化通用数学推理器

**Files:**
- Create: `app/agent/generic_math_reasoning.py`
- Modify: `app/agent/math_analysis.py`
- Create: `tests/test_generic_math_reasoning.py`

**Interfaces:**
- Produces `GenericMathReasoner.analyze(message, problem_context, analysis) -> GenericMathReasoning`。
- Consumes a client with `chat_json(system, user, schema, temperature=0.0, max_tokens=450)`。
- `analyze_turn(message, problem_context="", allow_llm: bool = True)` must return its current deterministic fallback when `allow_llm=False`。

- [ ] **Step 1: 写失败测试，固定推理器最小契约。**

```python
class _ReasoningClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.calls.append({"system": system, "user": user, "schema": schema, "max_tokens": max_tokens})
        return schema(
            problem_interpretation="总份数为5。",
            reasoning_steps=["把3:2看成5个相等份。"],
            next_step_hint="先计算每份是多少。",
            candidate_conclusion="男生18人，女生12人。",
            confidence=0.9,
        )


def test_reasoner_returns_validated_reasoning_with_a_token_cap() -> None:
    client = _ReasoningClient()
    result = GenericMathReasoner(client=client).analyze(
        "我不会做。", "男生和女生人数的比是3:2，总数30人。", MathAnalysis()
    )

    assert result.confidence == 0.9
    assert result.reasoning_steps == ["把3:2看成5个相等份。"]
    assert client.calls[-1]["max_tokens"] == 450
    assert "不要选择教学动作" in client.calls[-1]["system"]


def test_reasoner_declines_ambiguous_context_without_calling_client() -> None:
    client = _ReasoningClient()
    result = GenericMathReasoner(client=client).analyze("怎么算", "", MathAnalysis(needs_clarification=True))

    assert result.needs_clarification is True
    assert result.reasoning_steps == []
    assert client.calls == []
```

- [ ] **Step 2: 运行测试，确认因模块不存在而失败。**

Run: `python -m pytest tests/test_generic_math_reasoning.py -v`

Expected: FAIL，提示 `app.agent.generic_math_reasoning` 不存在。

- [ ] **Step 3: 实现 reasoner 与分析预算开关。**

```python
class GenericMathReasoner:
    def __init__(self, client=None) -> None:
        self._client = client

    def analyze(self, message: str, problem_context: str,
                analysis: MathAnalysis) -> GenericMathReasoning:
        if not problem_context.strip() or analysis.needs_clarification:
            return GenericMathReasoning(
                needs_clarification=True,
                uncertainty_note="缺少足以形成可靠推理的题目上下文。",
            )
        if self._client is None and not is_configured():
            return GenericMathReasoning(
                needs_clarification=True,
                uncertainty_note="未配置通用数学推理模型。",
            )
        client = self._client or DeepSeekClient()
        try:
            return client.chat_json(
                system=("你是初中数学推理器。给出短小、可教学的推理；"
                        "不选择教学动作，不声称 verifier 已验证，不清楚时标记 needs_clarification。"),
                user=f"题目：{problem_context}\n学生消息：{message}",
                schema=GenericMathReasoning,
                temperature=0.0,
                max_tokens=450,
            )
        except (DeepSeekError, ValueError):
            return GenericMathReasoning(needs_clarification=True,
                uncertainty_note="通用数学推理不可用，已降级为澄清教学。")
```

In `analyze_turn`, change the guard to `if not allow_llm or not is_configured(): return fallback`; keep its default `allow_llm=True`.

- [ ] **Step 4: 运行 reasoner 与分析回归测试。**

Run: `python -m pytest tests/test_generic_math_reasoning.py tests/test_generic_teaching_policy.py -v`

Expected: PASS；fake client 被调用一次，ambiguous 路径零调用。

- [ ] **Step 5: 提交本任务。**

```bash
git add app/agent/generic_math_reasoning.py app/agent/math_analysis.py tests/test_generic_math_reasoning.py
git commit -m "feat: add structured generic math reasoner"
```

## Task 3: 策略披露位和守卫式回复生成

**Files:**
- Modify: `app/agent/state.py`
- Modify: `app/agent/teaching_policy.py`
- Modify: `app/agent/response_generation.py`
- Modify: `tests/test_generic_teaching_policy.py`
- Create: `tests/test_generic_response_generation.py`

**Interfaces:**
- `TeachingActionPlan.answer_disclosure_allowed: bool = False` 是唯一由 policy 产生的披露许可。
- `generate_response(state, *, client=None) -> str` 只向 client 传递已批准字段；client 非空时使用 `chat(..., max_tokens=350)`。

- [ ] **Step 1: 写失败的 policy 和 response 测试。**

```python
def test_explicit_answer_request_allows_high_confidence_candidate_only() -> None:
    state = TeachingState(
        student_id="ratio", message="直接告诉我完整答案", problem_context="比是3:2，总数30人",
        math_analysis= MathAnalysis(intent=TeachingIntent.REQUESTS_ANSWER),
        math_reasoning=GenericMathReasoning(candidate_conclusion="男生18人，女生12人", confidence=0.8),
    )

    plan = TeachingPolicy().choose(state)

    assert plan.action is TeachingActionKind.EXPLAIN_STEP
    assert plan.answer_disclosure_allowed is True


def test_hint_request_never_allows_candidate_conclusion() -> None:
    state = TeachingState(
        student_id="ratio", message="给个提示", problem_context="比是3:2，总数30人",
        math_reasoning=GenericMathReasoning(next_step_hint="先求总份数。", candidate_conclusion="男生18人", confidence=0.99),
    )

    plan = TeachingPolicy().choose(state)

    assert plan.answer_disclosure_allowed is False
```

```python
class _ResponseClient:
    def chat(self, system, user, temperature=0.2, max_tokens=None):
        self.user = user
        self.max_tokens = max_tokens
        return "先把比的两项相加，得到总份数。现在能求出每一份是多少吗？"


def test_response_omits_candidate_from_unapproved_context() -> None:
    client = _ResponseClient()
    state = _hint_state(candidate="男生18人，女生12人")

    response = generate_response(state, client=client)

    assert "男生18人" not in client.user
    assert "男生18人" not in response
    assert client.max_tokens == 350
    assert response.count("？") == 1
```

- [ ] **Step 2: 运行测试，确认因披露字段和 client 参数尚不存在而失败。**

Run: `python -m pytest tests/test_generic_teaching_policy.py tests/test_generic_response_generation.py -v`

Expected: FAIL，提示 `answer_disclosure_allowed` 或 `client` 参数尚不存在。

- [ ] **Step 3: 实现最小 policy 与生成器守卫。**

```python
if state.math_analysis.needs_clarification or state.math_reasoning.needs_clarification:
    return TeachingActionPlan(action=TeachingActionKind.DIAGNOSE,
                              rationale="题意或推理不可靠，先澄清。")
if (intent is TeachingIntent.REQUESTS_ANSWER
        and state.math_reasoning.confidence >= 0.8
        and state.math_reasoning.candidate_conclusion):
    return TeachingActionPlan(action=TeachingActionKind.EXPLAIN_STEP,
                              answer_disclosure_allowed=True,
                              rationale="学生明确请求答案且推理置信度足够。")
```

`response_generation.py` must construct a `facts` mapping containing `problem_context`, action, hint level, reasoning steps and hint. Add `candidate_conclusion` only when `state.answer_disclosure_allowed` is true. On no configured/injected client, empty reasoning, `DeepSeekError`, multiple questions, or occurrence of an unapproved candidate in output, return the existing `_SAFE_RESPONSES[action]`. A `_one_final_question` helper must retain prose before the final question and append the action-appropriate single question if none exists.

- [ ] **Step 4: 运行 policy/response 测试与现有回归。**

Run: `python -m pytest tests/test_generic_teaching_policy.py tests/test_generic_response_generation.py tests/test_topic_agnostic_workflow.py -v`

Expected: PASS；无授权候选结论不进 prompt、不进回复；验证状态仍优先。

- [ ] **Step 5: 提交本任务。**

```bash
git add app/agent/state.py app/agent/teaching_policy.py app/agent/response_generation.py tests/test_generic_teaching_policy.py tests/test_generic_response_generation.py
git commit -m "feat: guard generic teaching responses"
```

## Task 4: 将通用推理接入默认 Harness 与 API 兼容投影

**Files:**
- Modify: `app/agent/teaching_harness.py`
- Modify: `app/agent/state.py`
- Modify: `tests/test_topic_agnostic_workflow.py`
- Modify: `tests/test_optional_verifier.py`

**Interfaces:**
- `run_generic_teaching_turn(student_id, message, conversation_id=None, *, verifier=None, graph_context_provider=None, reasoner=None, response_client=None, analysis_allow_llm=True) -> TeachingState`。
- 全部 optional service 缺失时不抛出；`TeachingState.trace` 依次包含 `generic_analysis`、`generic_math_reasoning`、`optional_verifier`、`optional_graph_context`、`generic_teaching_policy`、`generic_response_generation`。

- [ ] **Step 1: 写失败的端到端 Harness 测试。**

```python
def test_non_equation_turn_uses_injected_reasoning_without_leaking_answer() -> None:
    state = run_generic_teaching_turn(
        "ratio-student", "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
        reasoner=_FakeReasoner(candidate="男生18人，女生12人", confidence=0.95),
        response_client=_FakeResponseClient("先求总份数。你愿意先写出3和2相加的结果吗？"),
    )

    assert state.action_plan.action is TeachingActionKind.GIVE_HINT
    assert state.answer_disclosure_allowed is False
    assert "18" not in state.response
    assert state.math_reasoning.reasoning_steps
    assert state.trace[:2] == ["generic_analysis", "generic_math_reasoning"]
```

- [ ] **Step 2: 运行测试，确认因 Harness 尚未调用 reasoner 而失败。**

Run: `python -m pytest tests/test_topic_agnostic_workflow.py::test_non_equation_turn_uses_injected_reasoning_without_leaking_answer -v`

Expected: FAIL，提示 `reasoner` 参数未被接受或 `math_reasoning` 未填充。

- [ ] **Step 3: 实现接线并保留 legacy 边界。**

```python
analysis = analyze_turn(message, problem_context, allow_llm=analysis_allow_llm)
active_reasoner = reasoner or GenericMathReasoner()
reasoning = active_reasoner.analyze(message, problem_context, analysis)
state = TeachingState(
    student_id=student_id,
    message=message,
    conversation_id=conversation_id,
    problem_context=problem_context,
    math_analysis=analysis,
    math_reasoning=reasoning,
    verification=verification,
    graph_context=graph_context,
)
plan = TeachingPolicy().choose(state)
state.action_plan = plan
state.answer_disclosure_allowed = plan.answer_disclosure_allowed
state.response = generate_response(state, client=response_client)
```

Update both projections in `state.py` so a generic `TeachingState` round-tripped through legacy `run_turn` and API still serializes `math_reasoning`; do not add a fake `detected_concept_id` or `target_equation`.

- [ ] **Step 4: 运行 Harness、workflow、API 和 optional verifier 回归。**

Run: `python -m pytest tests/test_topic_agnostic_workflow.py tests/test_optional_verifier.py tests/test_api.py -v`

Expected: PASS；方程测试仍由 legacy adapter 保持原行为，非方程 state 公开 `math_reasoning`。

- [ ] **Step 5: 提交本任务。**

```bash
git add app/agent/teaching_harness.py app/agent/state.py tests/test_topic_agnostic_workflow.py tests/test_optional_verifier.py
git commit -m "feat: route generic turns through math reasoning"
```

## Task 5: 离线跨领域评测和有预算的真实 smoke runner

**Files:**
- Modify: `data/eval/topic_agnostic_smoke.jsonl`
- Create: `app/evaluation/generic_math_evaluator.py`
- Modify: `scripts/run_eval.py`
- Create: `scripts/run_generic_math_smoke.py`
- Create: `tests/test_generic_math_evaluator.py`

**Interfaces:**
- `run_generic_math_eval() -> list[GenericMathCaseResult]`，每个结果有 `case_id`、`passed`、`failures`。
- `python scripts/run_generic_math_smoke.py` 非零退出且不调用模型；`--live --max-cases N` 仅接受 `1 <= N <= 12`。

- [ ] **Step 1: 写失败评测与 runner 安全测试。**

```python
def test_offline_generic_eval_covers_five_domains_without_answer_leakage() -> None:
    results = run_generic_math_eval()

    assert {item.domain for item in results} >= {"ratio", "algebra", "geometry", "statistics", "probability"}
    assert all(item.passed for item in results)


def test_live_smoke_refuses_without_explicit_flag(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["run_generic_math_smoke.py"])

    assert main() == 2
    assert "--live" in capsys.readouterr().out
```

- [ ] **Step 2: 运行测试，确认模块不存在而失败。**

Run: `python -m pytest tests/test_generic_math_evaluator.py -v`

Expected: FAIL，提示 generic evaluator 或 smoke runner 尚不存在。

- [ ] **Step 3: 实现离线评测、样本和拒绝优先 runner。**

The JSONL must contain exactly 12 cases across the five named domains; each line declares `id`, `domain`, `message`, `candidate_conclusion`, `student_requests_answer`, and `expected_action`. The evaluator supplies fake analysis/reasoning/response clients and checks: non-empty response, at most one question, candidate absent without explicit answer request, candidate available only when request plus `confidence >= 0.8`, and verifier status not in `verified_correct|verified_incorrect`.

```python
parser.add_argument("--live", action="store_true")
parser.add_argument("--max-cases", type=int)
if not args.live or args.max_cases is None:
    print("拒绝运行：必须同时提供 --live 和 --max-cases。")
    return 2
if not 1 <= args.max_cases <= 12:
    print("拒绝运行：--max-cases 必须在 1 到 12 之间。")
    return 2
```

For each live case call `run_generic_teaching_turn(case.student_id, case.message, analysis_allow_llm=False)` once. It permits at most the injected/default reasoner and response generator calls, writes one JSON object to `logs/generic_math_live_smoke.jsonl`, records `case_id`, `model_call_count`, `error`, `confidence`, `verification_status`, `action`, `response`, and the deterministic safety-check results. Do not retry or invoke an LLM judge.

- [ ] **Step 4: 运行离线评测、完整 pytest 和既有 mock 回归。**

Run: `$env:TMP = (Join-Path $PWD 'logs'); $env:TEMP = $env:TMP; python -m pytest`

Expected: PASS，所有测试离线。

Run: `$env:TEACHER_FORCE_MOCK = '1'; python scripts/run_eval.py`

Expected: PASS，既有 golden 回归及新增 generic evaluator 都通过；无网络调用。

- [ ] **Step 5: 仅在用户再次明确批准时执行一次真实 smoke 并人工审阅。**

Run: `python scripts/run_generic_math_smoke.py --live --max-cases 12`

Expected: 最多 24 次模型调用，输出包含实际调用数；执行总预算不超过 2 元。逐条人工检查题意、下一步、答案披露、原题锚定；结果不作为 CI 门槛。

- [ ] **Step 6: 提交本任务。**

```bash
git add data/eval/topic_agnostic_smoke.jsonl app/evaluation/generic_math_evaluator.py scripts/run_eval.py scripts/run_generic_math_smoke.py tests/test_generic_math_evaluator.py
git commit -m "test: add generic math smoke evaluation"
```

## 实施后验收

1. 五个初中数学领域的离线 fake 推理样本能进入通用 Harness，且没有知识点 if/else 或知识包。
2. 普通提示不会包含 `candidate_conclusion`；只有明确索要、置信度至少 0.8 的情形才允许完整讲解。
3. 无 verifier 时 generic `VerificationResult` 保持 `not_applicable` 或 `unable_to_verify`，不会伪称数学结论已验证。
4. API 的 `teaching_state` 可观察 `math_reasoning`，但非方程返回不伪造线性方程字段。
5. `python -m pytest`、`python scripts/run_eval.py` 和三种原有模式保持通过；真实 smoke 默认拒绝运行、显式运行符合 2 元硬限制。
