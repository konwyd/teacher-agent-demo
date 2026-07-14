# 可组合数学教学策略族实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. This repository explicitly forbids subagents for this handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先修复通用数学会话的能力路由与稳定原题，再以影子选择、影子顾问、TeachingBrief 激活和跨策略评测逐步引入可组合策略族。

**Architecture:** 旧一元一次方程继续通过 `linear_equation_adapter.py` 和 `math_checker.py` 处理；其它题目进入稳定 `problem_context` 的通用 Harness。策略族位于 reasoner 与回复之间，服从 verifier 和 `TeachingPolicy`，最终只通过 `TeachingBrief` 向回复层提供一个获准下一步。

**Tech Stack:** Python 3.13、Pydantic、LangGraph 兼容 facade、DeepSeek OpenAI-compatible client、pytest、JSONL 离线评测。

## Global Constraints

- 只使用 `refactor/topic-agnostic-teaching-harness`，不新建或切换开发分支。
- 不合并、不推送、不改写或删除历史提交。
- 阶段 0 和阶段 1 分别提交；阶段 2 及以后每个阶段也必须单独授权、TDD、验证和提交。
- 不修改 `math_checker.py` 的解析、步骤、计划或答案裁决算法。
- 核心 `TeachingPolicy` 不增加二元一次方程、几何、统计等知识点 if/else。
- RAG/GraphRAG 与 verifier 继续可选；memory 和 mastery 继续是软约束。
- 未获授权不得披露最终答案；每轮恰好一个结尾问题。
- 所有 LLM 调用只经过 `app/llm/deepseek_client.py`；本计划的默认验证全部离线。
- 不运行真实付费 API，不提交 `.env`、API key、`logs/` 或 `data/runtime/`。
- 用户提供的未跟踪交接文件不是自动暂存对象。

---

## 文件职责

### 阶段 1 修改面

| 文件 | 职责 |
| --- | --- |
| `tests/test_topic_agnostic_workflow.py` | 三轮方程组公共接口回归，验证路由、上下文、披露和回复连续性。 |
| `tests/test_api.py` | 验证 `/turn` 对方程组返回空概念和稳定原题投影。 |
| `app/agent/linear_equation_adapter.py` | 只根据 episode 所属路径和 checker 的实际解析能力决定旧流程。 |
| `app/memory/schemas.py` | 为 `EpisodeState` 增加中性 `problem_context`。 |
| `app/agent/state.py` | 让中性 `TeachingState` 携带 episode 与 `ContextResolution`，兼容投影不伪造方程字段。 |
| `app/agent/teaching_harness.py` | 读取/初始化/保存稳定通用原题及本轮会话元数据。 |

### 后续阶段预定新增文件

| 文件 | 首次使用阶段 | 职责 |
| --- | --- | --- |
| `app/agent/strategy_family.py` | 2 | 策略 schema、allowlist 和纯函数规范化。 |
| `tests/test_strategy_family.py` | 2 | 非法、低置信度、去重和最多两个 supporting 的契约测试。 |
| `app/agent/strategy_advisors.py` | 3 | Advisor protocol、registry、symbolic 与 unknown guidance。 |
| `tests/test_strategy_advisors.py` | 3 | registry 和 advisor 边界测试。 |
| `app/agent/teaching_brief.py` | 4 | 按优先级组合 verifier、Policy、策略和可选图证据。 |
| `tests/test_teaching_brief.py` | 4 | 唯一动作、单一下一步、披露和冲突降级测试。 |
| `tests/test_strategy_composition.py` | 5 | primary + supporting 的跨表示组合测试。 |
| `app/evaluation/strategy_family_evaluator.py` | 6 | 策略族离线契约与跨族多轮结果汇总。 |
| `data/eval/strategy_family_conversations.jsonl` | 6 | 非知识包的行为回归样本。 |

## 阶段 0：设计与基线

- [x] 完整阅读仓库规则、交接、旧通用推理设计、旧连续性设计、旧实施计划和架构审计。
- [x] 核对分支、状态和最近 12 个提交；保留用户未跟踪交接文件。
- [x] 运行 `python -m pytest`、`python scripts/run_eval.py` 和三种离线模式。
- [x] 新建 `docs/superpowers/specs/2026-07-14-composable-strategy-family-design.md`。
- [x] 新建本实施计划，并声明部分取代 `f00eb4b` 的设计关系。
- [ ] 自审两份文档：无占位符、无范围矛盾、没有阶段 2 运行时实现。
- [ ] 只暂存两份新文档并提交 `docs: design composable teaching strategy family`。

## Task 1: 阶段 1 三轮会话 tracer bullet

**Files:**
- Modify: `tests/test_topic_agnostic_workflow.py`

**Interfaces:**
- Consumes: `run_teaching_turn(student_id, message, conversation_id, *, reasoner, response_client)`。
- Produces: 一个通过公共 facade 重放真实三轮输入的回归测试；LLM 只以注入的边界 fake 替代。

- [ ] **Step 1: 写一个失败的三轮行为测试**

在测试文件中增加一个记录 `(message, problem_context)` 的 fake reasoner，并让 fake response client 依次返回三条不同、各含一个结尾问句且不含完整数对的回复。测试按同一 `conversation_id` 调用：

```python
messages = [
    "老师，3x+2y=5，6x-2y=3怎么解？",
    "具体要怎么做呢？",
    "我不知道要怎么选择方法",
]
states = [
    run_teaching_turn(
        "equation-system-student",
        message,
        conversation_id="equation-system-three-turns",
        reasoner=reasoner,
        response_client=response_client,
        analysis_allow_llm=False,
    )
    for message in messages
]

assert [state.problem_context for state in states] == [messages[0]] * 3
assert [call[1] for call in reasoner.calls] == [messages[0]] * 3
assert [state.context_resolution.is_reply_to_previous_teacher for state in states] == [False, True, True]
assert all(state.episode_state and state.episode_state.problem_context == messages[0] for state in states)
assert all("我先不假设题型" not in state.response for state in states[1:])
assert len({state.response for state in states[1:]}) == 2
assert all(state.response.count("？") == 1 for state in states)
assert all("x=8/9" not in state.response and "y=7/6" not in state.response for state in states)
```

- [ ] **Step 2: 运行并确认 RED**

Run:

```powershell
python -m pytest tests/test_topic_agnostic_workflow.py::test_equation_system_follow_ups_reuse_the_first_problem_context -v
```

Expected: FAIL。当前首轮被旧 marker 路由接管，或 `TeachingState` 尚无 episode/context projection，或后续上下文等于短追问。失败必须由现有 Bug 引起，而不是导入、拼写或 fixture 错误。

## Task 2: 收紧旧方程能力路由

**Files:**
- Modify: `app/agent/linear_equation_adapter.py`
- Test: `tests/test_topic_agnostic_workflow.py`

**Interfaces:**
- Consumes: `get_episode_store().find(...)`、`math_checker.find_equations(message)`。
- Produces: `supports_legacy_equation_turn(...) -> bool` 的确定性路径归属判断。

- [ ] **Step 1: 实现最小能力路由**

判断顺序固定为：

```python
episode = get_episode_store().find(student_id, conversation_id)
if episode and episode.target_equation:
    return True
if episode and episode.problem_context:
    return False
return len(mc.find_equations(message)) == 1
```

删除 marker 作为新 episode 的判据；不修改 `math_checker`。已存在通用原题的 episode 即使后续出现 `9x=8` 也保持通用路径。

- [ ] **Step 2: 运行 tracer bullet**

Run: `python -m pytest tests/test_topic_agnostic_workflow.py::test_equation_system_follow_ups_reuse_the_first_problem_context -v`

Expected: 仍可能因 `problem_context` 尚未持久化而 FAIL，但首轮不再进入旧适配器。只在确认失败推进到上下文层后进入 Task 3。

## Task 3: 持久化通用 problem_context

**Files:**
- Modify: `app/memory/schemas.py`
- Modify: `app/agent/state.py`
- Modify: `app/agent/teaching_harness.py`
- Test: `tests/test_topic_agnostic_workflow.py`

**Interfaces:**
- Produces: `EpisodeState.problem_context: str | None = None`。
- Produces: `TeachingState.episode_state: EpisodeState | None` 和 `TeachingState.context_resolution: ContextResolution`。
- Preserves: `project_teaching_state_to_teacher_state()` 把通用 `problem_context` 投影到兼容 `original_problem`，不填 `target_equation` 或 concept。

- [ ] **Step 1: 增加中性 episode 字段和投影字段**

在 `EpisodeState` 中把 `problem_context` 放在 `target_equation` 前；在 `TeachingState` 中携带 episode 和 context resolution，均使用现有 Pydantic 类型与默认值。

- [ ] **Step 2: 在通用 Harness 解析稳定上下文**

一轮开始时：

```python
episode_store = get_episode_store()
episode = episode_store.get(student_id, conversation_id)
had_problem_context = bool(episode.problem_context)
if not had_problem_context:
    episode.problem_context = message.strip()
    episode_store.save(episode)
problem_context = episode.problem_context or message.strip()
```

创建 `ContextResolution`：首轮 `is_reply_to_previous_teacher=False`；已有通用原题的后续轮为真，reason 明确说明复用了会话原题。分析和 reasoner 均接收稳定原题加当前消息。

- [ ] **Step 3: 保存通用轮次元数据**

回复生成后保存：`current_phase`、`last_teacher_question`、`expected_student_response` 和有界 `recent_turns`。保存失败沿用 store 现有失败语义，不切换到旧流程、不伪造结论。

- [ ] **Step 4: 验证 GREEN**

Run: `python -m pytest tests/test_topic_agnostic_workflow.py::test_equation_system_follow_ups_reuse_the_first_problem_context -v`

Expected: PASS；三轮稳定原题、后两轮 reply 标记、三条不同安全回复均成立。

- [ ] **Step 5: 运行 focused workflow 回归**

Run: `python -m pytest tests/test_topic_agnostic_workflow.py tests/test_three_layer_memory.py -v`

Expected: PASS；`2x+3=7` 与已有旧 episode follow-up 仍走 checker。

## Task 4: API 投影的第二个 RED→GREEN

**Files:**
- Modify: `tests/test_api.py`
- Modify only if required by the failing public assertion: `app/agent/state.py` or `app/api/main.py`

**Interfaces:**
- Consumes: `POST /turn` 现有请求与响应 schema。
- Produces: 方程组响应中的空 `detected_concept`、完整 `original_problem` 和 `teaching_state.problem_context`。

- [ ] **Step 1: 写 API 行为测试**

```python
def test_turn_keeps_an_equation_system_in_the_generic_conversation() -> None:
    problem = "老师，3x+2y=5，6x-2y=3怎么解？"
    response = client.post(
        "/turn",
        json={
            "student_id": "api-equation-system",
            "conversation_id": "api-equation-system-conversation",
            "message": problem,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["detected_concept"] is None
    assert data["original_problem"] == problem
    assert data["teaching_state"]["problem_context"] == problem
```

- [ ] **Step 2: 运行 RED 或确认由前一 tracer bullet 已自然满足**

Run: `python -m pytest tests/test_api.py::test_turn_keeps_an_equation_system_in_the_generic_conversation -v`

如果立即 PASS，记录它是前一公共行为修复的覆盖扩展，不再改生产代码；如果因 projection 缺字段而 FAIL，只做使上述公共断言通过的最小投影修改。

- [ ] **Step 3: 运行 API 与旧 checker focused tests**

Run: `python -m pytest tests/test_api.py tests/test_math_checker.py tests/test_flexible_steps.py -v`

Expected: PASS；不修改 checker 算法或放宽旧断言。

## Task 5: 阶段 1 完整验证与提交

**Files:**
- Review only: all files changed by Tasks 1-4

- [ ] **Step 1: 运行目标测试**

Run: `python -m pytest tests/test_topic_agnostic_workflow.py tests/test_api.py tests/test_math_checker.py tests/test_flexible_steps.py -v`

- [ ] **Step 2: 运行完整 pytest**

在受限环境把 `TMP`、`TEMP` 和 cache 指向 `logs/`，不设置 `TEACHER_FORCE_MOCK`，以免遮蔽 fake-key 集成测试。

Run: `python -m pytest -o cache_dir='D:\teacher\logs\.pytest_cache'`

Expected: 0 failed；测试总数包含新增用例。

- [ ] **Step 3: 运行严格离线评测**

Run: `python scripts/run_eval.py`

Expected: `38/38` 或更高 case 总数全部通过。

- [ ] **Step 4: 运行三模式离线验证**

Run:

```powershell
python scripts/run_eval_modes.py --mode deepseek_only
python scripts/run_eval_modes.py --mode agent_no_graph
python scripts/run_eval_modes.py --mode agent_with_graph
```

Expected: `deepseek_only` 保持历史 baseline `6/26` 且命令成功；两个 Agent 模式均 `26/26`。

- [ ] **Step 5: 检查范围与状态**

Run:

```powershell
git diff --check
git diff --stat
git status --short
```

确认没有策略族运行时代码、`.env`、日志、runtime 数据或用户交接文件进入暂存区。

- [ ] **Step 6: 只提交阶段 1 文件**

```powershell
git add app/agent/linear_equation_adapter.py app/agent/teaching_harness.py app/agent/state.py app/memory/schemas.py tests/test_topic_agnostic_workflow.py tests/test_api.py
git commit -m "fix: preserve generic math conversation context"
```

提交后停止；不得执行下面的延期阶段。

---

## 延期阶段 2：影子策略选择

授权后单独执行：

1. 在 `tests/test_strategy_family.py` 先写非法 ID、`confidence < 0.60`、supporting 去重和最多两个的失败测试；
2. 在 `app/agent/strategy_family.py` 增加 `StrategyFamilyId`、`StrategyCandidate`、`StrategySelection` 和纯函数 `normalize_strategy_selection()`；
3. 扩展 `GenericMathReasoning`，使现有一次 `chat_json()` 同时返回策略选择，不增加模型调用；
4. 把规范化结果写入 `TeachingState` 与 trace，固定 `shadow_mode=True`；
5. 断言相同 fake reasoning 下回复与阶段 1 完全相同；
6. 跑完整离线门并提交 `feat: record shadow strategy selection`，然后停止。

## 延期阶段 3：影子策略顾问

授权后单独执行：

1. 在 `tests/test_strategy_advisors.py` 先写 registry、unknown fallback、advisor 异常降级和禁止决策字段测试；
2. 在 `app/agent/strategy_advisors.py` 定义 advisor protocol、`StrategyGuidance`、registry、`symbolic_transformation` 与 `unknown`；
3. 让 Harness 在 Policy 后调用 advisor，但只记录 guidance，不传给 response generator；
4. 断言 verifier、action、披露位和回复均不受 shadow guidance 影响；
5. 跑完整离线门并提交 `feat: add shadow strategy advisors`，然后停止。

## 延期阶段 4：TeachingBrief 激活

授权后单独执行：

1. 在 `tests/test_teaching_brief.py` 先写优先级、单一下一步、禁止披露、冲突丢弃和一个结尾任务测试；
2. 在 `app/agent/teaching_brief.py` 增加 `TeachingBrief` 与 `TeachingBriefComposer`；
3. 让 `response_generation.py` 只读取 brief，不直接读取未规范化 guidance；
4. 用方程组三轮回归作为第一个 `symbolic_transformation` 垂直切片；
5. 保持 `2x+3=7` 完全绕过策略族并继续 checker；
6. 跑完整离线门并提交 `feat: compose guarded teaching briefs`，然后停止。

## 延期阶段 5A：几何与数量建模组合

授权后单独执行：

1. 为锐角三角函数应用题写 primary `geometric_reasoning` + supporting `quantitative_modeling` 的失败测试；
2. 增加两个 advisor，只输出表示、诊断目标、下一步、误区信号和工具建议；
3. 断言 composer 仍只产生一个主要下一步，supporting 不覆盖 Policy；
4. 跑完整离线门并提交 `feat: compose geometry and modeling guidance`，然后停止。

## 延期阶段 5B：数据分析策略

授权后单独执行：

1. 为统计量解释和图表读取写 `data_analysis` 失败测试；
2. 增加 data advisor，验证无需 symbolic primary 也能形成 brief；
3. 断言无 verifier 时不伪称数据结论已验证；
4. 跑完整离线门并提交 `feat: add data analysis guidance`，然后停止。

## 延期阶段 6：评测与可观测性

授权后单独执行：

1. 为 TurnLog 策略 ID、置信度、fallback reason 和 shadow/active 状态写失败测试；
2. 增加 `app/evaluation/strategy_family_evaluator.py` 与行为型 JSONL 样本；
3. 记录但不逐字断言自然语言回复，重点检查优先级、稳定原题、披露、单问题和降级；
4. 让三模式继续离线运行，Graph 开关只改变可选证据；
5. 跑完整离线门并提交 `test: evaluate composable strategy families`，然后停止。

## 最终执行边界

本次只勾选并执行阶段 0、Task 1-5。阶段 2-6 保持未执行；即使阶段 1 全绿也不得自动开始延期阶段。
