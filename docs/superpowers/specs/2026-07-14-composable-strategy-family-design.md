# 可组合数学教学策略族设计

状态：已批准，阶段 0 设计基线

日期：2026-07-14

适用分支：`refactor/topic-agnostic-teaching-harness`

## 1. 目标

在现有知识点无关 Teaching Harness 上增加可组合的数学教学策略族，使同一套教学编排能够处理大部分初中数学问题，同时继续保留一元一次方程的确定性 `math_checker` 兼容路径。

本设计采用一个 primary strategy、最多两个 supporting strategies、可选 verifier 和可选 Graph/RAG。策略族提供数学表示与教学建议，但不获得教学动作、答案披露、提示等级、会话结束或数学裁决权。

本轮实施只覆盖：

1. 阶段 0：固化本设计与实施计划，记录离线基线；
2. 阶段 1：修复旧方程路由和通用会话 `problem_context` 连续性。

阶段 2 及以后只在本文和实施计划中定义，不在本轮实现。

## 2. 与既有设计的关系

本文及配套实施计划**部分取代**提交 `f00eb4b` 中的 `docs/superpowers/specs/2026-07-14-generic-math-conversation-continuity-design.md`。

继续有效的部分：

- 旧一元一次方程路径必须以 `math_checker` 的实际解析能力为边界；
- 通用会话必须稳定保存首轮原题，短追问不得覆盖它；
- 已有 `target_equation` 的旧 episode 继续走旧 checker；
- 核心 `TeachingPolicy` 不按知识点写分支；
- verifier、Graph/RAG 和 memory 都保持可选或软约束；
- 未获授权不得披露最终答案，每轮恰好一个结尾问题。

被扩展或取代的部分：

- 通用链路不再止于“通用推理 + Policy + 回复”，而是增加可审计的策略选择、策略顾问和 `TeachingBrief`；
- `GenericMathReasoner` 后续在同一次结构化调用中输出推理与影子策略选择，不增加额外路由 LLM 调用；
- 回复层后续只消费压缩后的 `TeachingBrief`，不直接读取未规范化策略建议；
- 方程组只作为首个垂直回归切片，不成为核心策略中的知识点特判。

旧设计文件和历史提交必须保留，便于审计阶段 1 的路由与上下文决策。

## 3. 不变式与优先级

### 3.1 不可破坏的不变式

- `math_checker.py` 仍是一元一次方程步骤、计划和最终答案的唯一确定性裁决来源。
- LLM、策略族、memory 或 RAG 均不得推翻 verifier 的明确结论。
- checker 分支优先于 mastery；mastery 只调整解释深浅、提示粒度、复习和变式题。
- 普通通用路径不依赖 RAG/GraphRAG；provider 缺失或失败时仍能完成一轮安全教学。
- 未明确索要最终答案且学生未自行解出时，不向回复层传递候选最终结论。
- 回复围绕稳定 `problem_context`，不无故引入新题；每轮恰好一个结尾问题。
- 例题仅由显式例题动作触发，策略族不能自行插入例题。
- 所有真实 LLM 调用仍只经过 `app/llm/deepseek_client.py`，结构化输出使用 JSON mode 和 Pydantic。

### 3.2 决策优先级

从高到低固定为：

1. verifier 的确定性证据；
2. `TeachingPolicy` 的动作与答案披露决策；
3. primary strategy 的数学教学建议；
4. supporting strategies 的补充建议；
5. 可选 Graph/RAG 的教学证据。

下层只能补充上层未决定的内容，不能覆盖上层结论。

## 4. 当前架构与阶段 1 缺口

现有通用链路为：

```text
MathAnalysis
  -> GenericMathReasoner
  -> OptionalVerifier
  -> Optional GraphContext
  -> TeachingPolicy
  -> Guarded ResponseGeneration
```

旧一元一次方程实现保留在 `linear_equation_adapter.py` 与 `math_checker.py`。当前缺口有两个：

1. `supports_legacy_equation_turn()` 只要看见 `x`、`=` 或“方程”等 marker 就进入旧流程，导致方程组被误标为 `c_linear`；
2. `run_generic_teaching_turn()` 每轮都把当前消息写成 `problem_context`，短追问会丢失首轮方程组。

阶段 1 的修复是能力路由和会话状态修复，不引入任何策略族运行时代码。

## 5. 策略族模型

### 5.1 首批策略 ID

- `symbolic_transformation`：等式、代数式、函数关系等符号表示与等价变换；
- `quantitative_modeling`：从文字条件建立数量关系、单位关系或模型；
- `geometric_reasoning`：图形性质、构造、空间关系与证明链；
- `data_analysis`：统计量、数据表示、趋势和不确定性解释；
- `unknown`：证据不足、非法选择、低置信度或分析失败时的保守降级。

这些是教学推理方式，不是知识点包，也不等同于题型枚举。

### 5.2 选择结构

`StrategyCandidate` 包含：

- `strategy_id`：上述 allowlist 中的 ID；
- `confidence`：`0.0` 到 `1.0`；
- `rationale`：面向审计的简短选择依据，不含最终答案。

`StrategySelection` 包含：

- `primary: StrategyCandidate | None`；
- `supporting: list[StrategyCandidate]`，去重后最多两个；
- `fallback_reason: str`；
- `shadow_mode: bool`，阶段 2 和阶段 3 固定为真。

规范化规则：

- 非法 ID 变为 `unknown`；
- `confidence < 0.60` 的 primary 变为 `unknown`；
- supporting 必须置信度至少 `0.60`、不得与 primary 重复、最多保留两个；
- 分析失败或没有候选时产生 `unknown`，不得回退成某个知识点默认策略；
- 规范化必须是确定性纯函数，可独立单测。

### 5.3 策略顾问输出

`StrategyGuidance` 只允许包含：

- 推荐数学表示；
- 本轮诊断目标；
- 一个可采用的主要下一步；
- 误区信号；
- 可能需要的工具；
- 置信度和来源 strategy ID。

它明确禁止：

- 选择或覆盖 `TeachingAction`；
- 决定 `answer_disclosure_allowed`；
- 独立控制提示等级或结束会话；
- 直接生成教师最终回复；
- 将未验证推理标记为正确或错误；
- 推翻 verifier、Policy 或原题锚定。

## 6. 目标数据流

```text
student message + persisted episode
  -> resolve stable problem_context
  -> MathAnalysis
  -> GenericMathReasoning + shadow StrategySelection
  -> OptionalVerifier
  -> TeachingPolicy
  -> StrategyRegistry
       -> primary StrategyAdvisor
       -> up to two supporting StrategyAdvisors
  -> Optional GraphContext
  -> TeachingBriefComposer
  -> Guarded ResponseGeneration
  -> persist episode/state
  -> audit trace
```

`GenericMathReasoner` 在既有一次 JSON 调用中附带策略选择，避免为路由新增一次模型调用。`StrategyRegistry` 只按已规范化 ID 查找顾问；核心 `TeachingPolicy` 不读取 `x`、`y`、“几何”“统计”等题型关键词。

## 7. TeachingBrief 边界

`TeachingBrief` 是回复层唯一允许消费的策略组合结果，只包含：

- `problem_context`：稳定原题；
- `current_student_request`：本轮消息；
- `action`：Policy 已选定的唯一动作；
- `primary_next_step`：一个主要下一步；
- `verified_facts`：仅来自 verifier 的已验证事实；
- `allowed_reasoning`：Policy 与披露门允许使用的推理；
- `forbidden_disclosures`：候选答案、未经验证结论或其它禁止内容；
- `closing_task`：一个学生可执行的结尾任务。

组合规则：

- primary guidance 决定建议的主要数学表示和下一步候选；
- supporting guidance 只能补充诊断角度、误区信号或工具建议；
- 多个建议冲突时，以 verifier 和 Policy 为准，策略建议降级为不进入 brief；
- Graph/RAG 仅在 brief 仍有空缺且证据与原题相关时补充；
- brief 中只保留一个主要下一步和一个结尾任务，避免多问题和多路线同时倾倒。

## 8. 状态、会话与可观测性

- `EpisodeState.problem_context` 保存通用会话首轮完整原题，与旧 `original_problem`、`target_equation` 并存但不互相覆盖。
- 已有 `target_equation` 的 episode 始终优先走旧 checker；已有通用 `problem_context` 且无 `target_equation` 的 episode 始终留在通用路径。
- 阶段 2 起，`TeachingState` 和 trace 记录规范化后的策略选择；阶段 2/3 的选择不影响回复。
- 阶段 6 才把策略 ID、置信度、降级原因和组合结果加入持久化 `TurnLog` 与评测汇总。
- 长期 mastery 的跨知识点 taxonomy 仍未定，本设计不在策略 ID 上直接累计掌握度。

## 9. 失败与降级

- reasoner 不可用或 JSON 校验失败：选择 `unknown`，继续现有安全通用教学。
- 非法、低置信度或重复策略：由规范化器确定性修正并记录原因。
- advisor 缺失或抛错：忽略该 guidance；primary 失败时使用 `unknown` guidance，不阻断 Policy。
- verifier 不适用：保持 `not_applicable` 或 `unable_to_verify`，策略不得伪造已验证事实。
- Graph/RAG 缺失或失败：返回空上下文，不改变动作和披露门。
- brief 无可靠数学下一步：保守请求澄清或给读题任务，不能编造结论。

## 10. 增量阶段与停止门

### 阶段 0：设计和基线

新建设计文档与实施计划，记录当前离线基线，不修改运行行为。

### 阶段 1：路由和跨轮上下文

收紧旧适配器能力边界，增加 `EpisodeState.problem_context`，让三轮方程组回归稳定复用首轮原题；不引入策略族。

### 阶段 2：影子策略选择

增加策略选择 schema 和规范化器，reasoner 在现有 JSON 调用中附带选择；只记录 state/trace。

### 阶段 3：影子策略顾问

增加 registry、`symbolic_transformation` 与 `unknown` advisor；产生 guidance 但不进入回复。

### 阶段 4：TeachingBrief 激活

增加 brief composer，首先让方程组垂直切片使用 symbolic guidance；旧一元一次方程仍走 checker。

### 阶段 5：多策略组合

先验证 `geometric_reasoning + quantitative_modeling`，再独立验证 `data_analysis`；两个子阶段分别 TDD、验证和提交。

### 阶段 6：评测和可观测性

扩展 TurnLog、离线评测和跨族多轮 case，记录策略选择、置信度与降级原因。

每个阶段必须独立 TDD、独立提交、完整离线验证并停止等待确认。任何阶段不得自动进入下一阶段。

## 11. 阶段 0 离线基线

2026-07-14 在 Python 3.13.13、离线 mock 环境运行：

| 命令 | 结果 |
| --- | --- |
| `python -m pytest` | 242 passed，1 warning，0 failed |
| `python scripts/run_eval.py` | 38/38 passed |
| `python scripts/run_eval_modes.py --mode deepseek_only` | 6/26 passed，20 个预期 baseline 能力差异，命令成功 |
| `python scripts/run_eval_modes.py --mode agent_no_graph` | 26/26 passed |
| `python scripts/run_eval_modes.py --mode agent_with_graph` | 26/26 passed |

受限环境中 pytest 需要把 `TMP`、`TEMP` 与 cache 目录指向仓库内被忽略的 `logs/`。测试未调用真实 DeepSeek；真实 smoke 不属于本轮验收。

## 12. 本轮验收

阶段 1 完成后必须满足：

- 方程组首轮不再返回“一元一次方程”或 `c_linear`；
- 三轮 `problem_context` 都是首轮完整题目；
- 后两轮标记为复用上一题上下文，且不重复固定 DIAGNOSE 文案；
- 未索要答案时不泄露 `x=8/9，y=7/6` 等完整数对；
- `2x+3=7` 及所有旧 checker 回归不变；
- pytest、run_eval 和三模式离线验证达到基线；
- 不存在阶段 2 的运行时代码、提交、合并或推送。
