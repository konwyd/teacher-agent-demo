# 通用数学推理与守卫式教学回复设计

## 目标

为知识点无关的 Teaching Harness 增加首个可验证的通用数学能力切片：在比例、整式与代数、基础几何、统计和概率等初中数学问题上，底座 LLM 能提供结构化推理和围绕原题的教学回复；Harness 继续控制教学动作、答案披露和不确定性降级。

这不是通用符号计算系统，也不宣称在这一切片完成后已可靠覆盖全部初中数学。

## 已确认约束

- Runtime LLM 继续只经由 `app/llm/deepseek_client.py` 调用 DeepSeek；结构化输出使用 JSON mode 和 Pydantic 校验。
- 不为比例、几何、统计或其它知识点创建知识包，也不在核心 `TeachingPolicy` 中添加按知识点分支。
- `math_checker.py` 继续是一元一次方程步骤、计划和最终答案的唯一确定性判定来源；它的明确结论优先于 LLM。
- RAG 和 GraphRAG 不参与普通通用数学教学的必要路径；无 provider 时流程仍能教学。
- 学生未明确索要完整答案且未自行解出时，不展示最终答案；每轮最多一个收尾问题；普通回复围绕 `problem_context`。
- 离线 pytest 与既有 mock 评测必须保持通过。
- 真实 DeepSeek smoke 总预算上限为 2 元：只在显式 `--live` 开关下执行，最多 12 题、每题最多两次调用，并限制单次输出 token。运行记录要写明实际题数、调用数、模型错误和每题结果。

## 不做的内容

- 不接入 Neo4j、Qdrant、教材语料、外部数学数据集或新的数学领域专用 verifier。
- 不替换、放宽或删除一元一次方程的 golden cases、math checker、记忆流程或现有 API 兼容字段。
- 不用 LLM 的高置信度替代确定性 verifier 的 `verified_correct` 或 `verified_incorrect` 结论。
- 不在常规测试、默认 API 请求或 `scripts/run_eval.py` 中调用真实 DeepSeek。
- 不把真实 smoke 的模型回复当作自动化正确性证明；它是有限、可审计的开发信号。

## 方案比较

### 方案 A：单次自由回答

让模型直接生成教师回复，最快但无法区分“模型认为数学正确”“教师可以透露哪些信息”和“学生是否真的理解”。无法复用现有答案泄露和 verifier 优先级，故不采用。

### 方案 B：结构化推理加守卫式生成（采用）

先生成不决定教学动作的 `GenericMathReasoning`，再由现有 Policy 结合 verifier、学生意图与置信度选择动作，最后只将获准披露的事实交给回复生成器。它把模型的数学能力与 Harness 的教学约束分开，且能在无 verifier 的领域保守降级。

### 方案 C：先建设通用求解与验证工具

可获得更强确定性，但比例、整式、几何、统计和概率需要不同的表示与验证器，不符合本轮“快速形成跨领域教学能力”的目标。它保留为后续可选增强。

## 架构

```text
student message + problem_context
  -> MathAnalysis                    # 意图、卡点；不判数学正误
  -> GenericMathReasoner             # LLM JSON，输出推理、候选结论和置信度
  -> OptionalVerifier                # 有适配 verifier 时产生强数学结论
  -> TeachingPolicy                  # 先 verifier，后意图/置信度，选择中性动作
  -> GuardedResponseGenerator        # LLM 或安全模板，只看获准披露的事实
  -> TeachingState + audit trace
```

所有节点都以 `TeachingState` 为中心。推理器、验证器和图上下文都可以不可用；任一不可用时，Policy 和回复生成必须仍能完成一轮安全教学。

## 数据模型与职责

### `GenericMathReasoning`

在 `app/agent/state.py` 中添加下列 Pydantic 模型，字段不包含知识点 ID、图节点 ID 或题型枚举：

- `problem_interpretation: str`：LLM 对题意和所求的简短转述；无法可靠理解时为空。
- `student_attempt_assessment: str`：仅描述学生给出的尝试或缺失的信息，不宣称确定对错。
- `reasoning_steps: list[str]`：可用于教学的短步骤；每一步只描述一个数学关系或运算。
- `next_step_hint: str`：不包含最终结论的下一步提示。
- `candidate_conclusion: str`：模型候选结论，仅在答案披露许可时可传给回复生成器。
- `confidence: float`：0 到 1 的整体推理置信度。
- `needs_clarification: bool`：题目缺条件、表达含糊或模型不能形成可靠推理时为真。
- `uncertainty_note: str`：面向日志和 Policy 的不确定原因；不会直接冒充数学结论。

`TeachingState` 增加 `math_reasoning: GenericMathReasoning`，默认空对象。它与已有 `MathAnalysis` 并列：前者提供数学内容建议，后者提供交互/教学分析。它不替代 `VerificationResult`。

### `GenericMathReasoner`

在 `app/agent/generic_math_reasoning.py` 新建单一职责类和函数：

- 仅在 `problem_context` 非空且 `MathAnalysis.needs_clarification` 为假时请求模型。
- 通过 `DeepSeekClient.chat_json()` 请求 `GenericMathReasoning`；system prompt 要求列出可教学步骤、显式置信度、无法确定时设置 `needs_clarification=true`，且禁止选择教学动作或使用知识点标签。
- 无 API key、JSON 校验失败、网络错误或空推理时，返回空 reasoning，并把 `needs_clarification=true` 与原因写入 `uncertainty_note`。
- 允许测试注入一个拥有 `chat_json()` 的客户端；生产默认使用 `DeepSeekClient`。
- 真实 smoke 中单次 reasoning 输出最多 450 tokens。

### Policy 与披露门控

`TeachingPolicy` 保持确定性，不解析题目也不生成数学文本。优先级固定为：

1. `VerificationStatus.VERIFIED_CORRECT` 和 `VERIFIED_INCORRECT` 继续优先。
2. `MathAnalysis.needs_clarification` 或 `GenericMathReasoning.needs_clarification` 为真时选择 `ASSESS` 或 `DIAGNOSE`。
3. 学生明确请求答案、reasoning 置信度至少 0.8 且 reasoning 有候选结论时，可选择 `EXPLAIN_STEP` 并将 `answer_disclosure_allowed` 设为真。
4. 未明确索要答案时，即使 reasoning 置信度很高，也只选择 `GIVE_HINT` 或 `EXPLAIN_STEP`，并将 `answer_disclosure_allowed` 设为假。
5. 无 verifier 的高置信度 reasoning 不能写入 `VerificationResult` 的正确或错误状态；该状态仍为 `not_applicable` 或 `unable_to_verify`。

这一规则保证模型能在学生明确要求时提供完整解法，但不会把候选结论泄露给普通提示或纠错回复。

### `GuardedResponseGenerator`

扩展 `app/agent/response_generation.py`，保留现有安全模板作为无模型、低置信度和模型失败时的 fallback。

生产生成器的输入只包括：`problem_context`、教学动作、hint level、已获准的 `reasoning_steps`/`next_step_hint`、以及在 `answer_disclosure_allowed=true` 时才包含的 `candidate_conclusion`。system prompt 强制：

- 回复围绕原题，不引入新题或类比例题。
- 未获准时不得写候选结论或等价的最终答案。
- 一次只给一个学生可执行的下一步，并以恰好一个问题收尾。
- 确定性 verifier 已有结论时，只能以该结论为准。
- reasoning 缺失或低置信度时，承认需要更多信息或给通用读题提示，不编造步骤。

回复生成器允许测试注入拥有 `chat()` 的客户端；真实 smoke 中单次回复最多 350 tokens。生成或后处理失败时回退现有 `_SAFE_RESPONSES`。

## Runtime 接线

`run_generic_teaching_turn()` 按下列顺序接线：

1. 调用现有 `analyze_turn()`。
2. 调用 `GenericMathReasoner`，得到 `math_reasoning`。
3. 调用可选 verifier；默认空 verifier。
4. 构造 `TeachingState`，写入 analysis、reasoning、verification 和空/可选 graph context。
5. 调用 `TeachingPolicy`，根据其 action 与 reasoning 置信度设置 `answer_disclosure_allowed`。
6. 调用守卫式回复生成器；任何 LLM 错误回退安全模板。

一元一次方程仍先走 `linear_equation_adapter.py`，不迁移既有专用工作流。非方程题才使用上述路径。

## 评测与真实 smoke

### 离线契约测试

新增跨领域 fixture，仅作为评测样本而非知识包，覆盖：比例、整式/代数、基础几何、统计、概率。每类至少包含：

- 学生只求提示，断言候选最终结论不出现在回复中。
- 学生问“为什么”，断言解释使用 reasoning 的关系而不是固定模板。
- 学生明确请求完整答案，且 reasoning 置信度足够高，断言回复能使用候选结论。
- reasoning 无法确定或 client 报错，断言动作降级、无候选答案泄露且仍有一个收尾问题。

所有离线测试使用 fake `chat_json()`/`chat()` clients；`python -m pytest` 与 `python scripts/run_eval.py` 不访问网络。

### 真实 smoke runner

新增 `scripts/run_generic_math_smoke.py`。它默认退出并说明必须传 `--live`；只有同时提供 `--live` 和 `--max-cases` 时才可调用 DeepSeek。参数约束如下：

- `--max-cases` 必须为 1 到 12；超过 12 立即报错且不调用 API。
- 运行器从固定的 12 条跨领域 smoke 样本中取前 N 条；每条最多调用 reasoning 和 response 各一次。
- 每次调用用上述 450/350 token 上限；运行器不重试，不进行 LLM-as-a-judge。
- 结果写入 `logs/generic_math_live_smoke.jsonl`，每行包含 case ID、调用次数、模型是否出错、reasoning 置信度、verification 状态、教学动作、回复和基础安全检查结果。
- 真实运行前输出“最多 2N 次模型调用”和预算上限 2 元；运行完成后输出实际调用数。运行器不承诺从 API 响应计算精确人民币费用，因为当前 client 不暴露账单计量。

真实 smoke 样本用于人工审阅：检查题意理解、下一步是否正确、是否在未授权时泄露答案、是否始终围绕原题。它不改变离线 CI 通过标准。

## 失败处理与风险

- LLM 的数学错误：无 verifier 时不赋予 `verified_*` 状态；低置信度和不清晰题目走澄清/模板回退；真实 smoke 保存证据供人工审阅。
- JSON 不符合 schema：reasoner 返回空 reasoning，不影响 Policy 或回复结束。
- 回复违反披露约束：生成器只接收允许披露的字段，失败回退模板；离线测试覆盖不泄露路径。
- API 成本和网络：runner 默认不联网、无重试、12 条硬上限、每题最多两次调用和输出 token 限制。
- 既有方程能力回归：不修改 `math_checker.py` 的算法或 `linear_equation_adapter.py` 的方程判断；完整 pytest 与已有 mock eval 是合入门槛。

## 验收标准

1. 非方程题进入通用路径后，不再始终返回固定 `_SAFE_RESPONSES`；注入的 reasoning 能改变提示或解释内容。
2. 无 verifier 时，任何通用题目的 `VerificationResult` 不会被标为 `verified_correct` 或 `verified_incorrect`。
3. 未明确请求答案的跨领域样本不泄露 `candidate_conclusion`；明确请求、置信度至少 0.8 的样本可获得结构化完整讲解。
4. reasoning 或 response client 失败时，流程返回非空、安全、单问题回复。
5. 一元一次方程原有 pytest、mock eval 和三模式结果不退化。
6. `python -m pytest` 和 `python scripts/run_eval.py` 通过；真实 smoke 只有在用户明确命令后才运行，次数不超过 12。
