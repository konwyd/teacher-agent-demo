# Hybrid Student Simulator Evaluation 设计

日期：2026-07-10  
状态：已实施并于 2026-07-13 完成离线验证  
范围：evaluation 层增量能力，不修改 Teacher 核心 workflow、math_checker 或 retriever

## 1. 背景与目标

现有 evaluation 主要回放预先编写的学生消息，适合保护确定性教学不变式，但无法观察 Teacher 的回复如何改变学生后续行为。Hybrid Student Simulator Evaluation 增加一个带隐藏状态的确定性学生模拟器，让学生消息在每轮 Teacher 回复之后动态产生。

本设计的目标是：

- 构建 18 个学生场景：6 个概念乘以低、中、高三档初始掌握度。
- 对现有三种 Teacher mode 分别运行全部 18 个场景，完整实验共 54 段对话。
- 用确定性状态机决定学生状态变化与数学意图，保证默认离线运行可复现。
- 允许真实 DeepSeek 仅改写学生措辞，不允许它改变隐藏状态、数学意图或步骤正确性。
- 保存完整对话和每轮状态变化后，再进行确定性评分。
- 保持现有 golden evaluation 及其严格回归门不变。

三种 Teacher mode 为：

- `deepseek_only`
- `agent_no_graph`
- `agent_with_graph`

## 2. 非目标与硬约束

本功能不做以下事情：

- 不修改 `app/agent/workflow.py` 的教学决策。
- 不修改或绕过 `app/agent/math_checker.py`。
- 不修改 retriever 或 Pedagogical Graph 的语义。
- 不把学生模拟器的隐藏 mastery 写入 Teacher memory。
- 不让学生模拟器读取 Teacher mode、诊断、教学动作、retrieval 结果或 memory。
- 不用 LLM 判断学生步骤对错、状态转移或终止条件。
- 不把预先写好的完整聊天记录倒灌给 Teacher。
- 不用 LLM-as-judge 进行评分。
- 不替换或弱化 `scripts/run_eval.py`、golden cases 或现有 mode comparison。

所有新增 LLM 调用必须经过 `app/llm/deepseek_client.py`。结构化输出必须使用 JSON mode 和 Pydantic 校验。默认运行不访问真实 API。

## 3. 总体架构

新增独立的 Hybrid Evaluation 子系统：

```text
scenario JSONL
    |
    v
StudentSimulator --student message--> HybridRunner --message--> Teacher mode
    ^                                      |
    |                                      v
    +------ teacher response text ---------+
                    |
                    v
          HybridConversationTrace
                    |
             先持久化 traces
                    |
                    v
              HybridScorer
                    |
          scores JSONL + summary JSON
```

文件职责：

- `app/evaluation/student_simulator.py`
  - 定义场景、隐藏状态、Teacher 文本观察、学生意图和状态转移模型。
  - 从 Teacher 回复文本提取确定性信号。
  - 选择下一条学生数学意图。
  - 用模板生成默认学生措辞。
  - 可选调用 DeepSeek 改写措辞，并验证改写没有改变语义。

- `app/evaluation/hybrid_runner.py`
  - 为每个 scenario 和 mode 创建隔离会话。
  - 逐轮交替调用 Teacher 和 Student Simulator。
  - 强制最多 8 个 Teacher turn。
  - 保存每轮完整输入、输出、观察信号和状态快照。
  - 只负责生成 trace，不负责评分。

- `app/evaluation/hybrid_scorer.py`
  - 从已持久化 trace 计算确定性教学行为与轨迹指标。
  - 生成逐对话 score 和按 mode、concept、mastery 聚合的 summary。

- `data/eval/student_simulator_scenarios.jsonl`
  - 保存 18 个场景定义，不保存完整对话。

- `scripts/run_hybrid_eval.py`
  - 提供完整和分批运行入口。
  - 先生成并关闭 trace 文件，再调用 scorer 读取该文件。

- `tests/test_student_simulator.py`
  - 覆盖场景、状态转移、数学意图与措辞保护。

- `tests/test_hybrid_runner.py`
  - 覆盖 mode 隔离、轮数限制、终止和持久化顺序。

- `tests/test_hybrid_scorer.py`
  - 覆盖固定轨迹评分、聚合和 54 组端到端输出。

允许对 `app/evaluation/mode_runner.py` 做仅限 evaluation 内部复用的微小公开接口整理，例如为单轮 mode 响应提供稳定入口；不得改变现有三种 mode 的行为或结果 schema。

## 4. 场景设计

### 4.1 概念集合

18 个场景必须完整覆盖以下笛卡尔积：

```text
concept_id in {
  linear_equation,
  equation_properties,
  move_terms,
  divide_by_coefficient,
  remove_parentheses,
  fractional_coefficient_equation
}

mastery_level in {low, medium, high}
```

默认场景文件加载时必须验证：

- 总数恰好为 18。
- `scenario_id` 唯一。
- `(concept_id, mastery_level)` 组合唯一且无缺失。
- 每个 concept 的低、中、高场景使用同一道 `original_problem`，只改变初始隐藏状态和行为倾向。
- 每个数学步骤模板均能被 math_checker 判定为场景声明的正确步骤、错误步骤或正确最终答案。

### 4.2 场景 schema

每行 JSON 对应一个 `StudentScenario`：

```python
class StudentScenario(BaseModel):
    scenario_id: str
    concept_id: str
    mastery_level: Literal["low", "medium", "high"]
    original_problem: str
    expected_final_answer: str
    initial_message: str
    initial_state: StudentHiddenState
    solution_steps: list[StudentMathIntent]
    verification_intent: StudentMathIntent
    misconception_intent: StudentMathIntent | None
    misconception_markers: list[str]
    targeted_support_markers: list[str]
    utterance_templates: dict[str, list[str]]
```

`solution_steps` 是数学意图序列，不是聊天记录。模拟器根据 Teacher 回复和隐藏状态动态选择是否前进、重试、提问或给出错误步骤。
其中最后一项必须是 `FINAL_ANSWER`，此前各项是中间正确步骤；`verification_intent` 独立于该序列，不计入 solution progress。

### 4.3 初始档位

为保证跨概念和跨 mode 可比，三档使用固定基准：

| mastery_level | mastery | confidence | understanding_status | 默认 help_seeking_style | active_misconception |
|---|---:|---:|---|---|---|
| low | 0.20 | 0.30 | `confused` | `asks_hint` | 当前概念典型误区 |
| medium | 0.50 | 0.55 | `partial` | `guesses` | 当前概念典型误区 |
| high | 0.80 | 0.80 | `partial` | `direct` | `null` |

场景可以为概念选择不同的具体误区和措辞模板，但不得改变上述数值基准。`mastery` 是模拟器私有真值，不预先写入 Teacher Agent 的 memory。

## 5. 隐藏状态与领域模型

### 5.1 隐藏状态

```python
class StudentHiddenState(BaseModel):
    mastery: float = Field(ge=0.0, le=1.0)
    active_misconception: str | None
    current_solution_step: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    help_seeking_style: HelpSeekingStyle
    consecutive_failures: int = Field(ge=0)
    hint_exposure: int = Field(ge=0)
    understanding_status: UnderstandingStatus
```

枚举：

```python
class HelpSeekingStyle(str, Enum):
    ASKS_WHY = "asks_why"
    ASKS_HINT = "asks_hint"
    GUESSES = "guesses"
    DIRECT = "direct"

class UnderstandingStatus(str, Enum):
    CONFUSED = "confused"
    PARTIAL = "partial"
    UNDERSTOOD = "understood"
    VERIFIED = "verified"
```

状态模型不包含 Teacher mode。对每个 mode 运行时，必须从同一个 `initial_state.model_copy(deep=True)` 开始。

### 5.2 Teacher 文本观察

Student Simulator 的公开转移入口只接收：

```python
advance(
    scenario: StudentScenario,
    state: StudentHiddenState,
    teacher_response: str,
    teacher_turn_index: int,
) -> StudentTransition
```

不接受 mode、TeacherState、diagnosis、action、memory 或 retrieval 参数。

Teacher 文本被确定性分类为 `TeacherObservation`：

```python
class TeacherObservation(BaseModel):
    acknowledges_correct: bool
    rejects_incorrect: bool
    gives_hint: bool
    addresses_misconception: bool
    asks_for_verification: bool
    confirms_verification: bool
    leaks_final_answer: bool
    asks_question: bool
    grounded_in_original_problem: bool
    matched_markers: list[str]
```

信号来自固定词语、场景 marker、方程提取与 math_checker，不调用 LLM。`leaks_final_answer` 仅在学生尚未给出答案时成立。

### 5.3 学生意图

```python
class StudentIntentType(str, Enum):
    ASK_HELP = "ask_help"
    ASK_WHY = "ask_why"
    RESTATE_CONFUSION = "restate_confusion"
    INCORRECT_STEP = "incorrect_step"
    CORRECT_STEP = "correct_step"
    FINAL_ANSWER = "final_answer"
    VERIFICATION = "verification"
    GIVE_UP = "give_up"

class StudentMathIntent(BaseModel):
    intent_type: StudentIntentType
    canonical_content: str
    expected_step_validity: str | None
    required_math_anchors: list[str]
```

凡 `expected_step_validity` 非空，运行时和测试都必须调用 math_checker 验证 `canonical_content`。LLM 无权产生或修改 `StudentMathIntent`。

## 6. 确定性状态转移

### 6.1 更新顺序

每个 Teacher turn 后严格按以下顺序处理：

1. 从 Teacher 回复提取 `TeacherObservation`。
2. 检查当前学生是否已给出正确最终答案或验算，以及 Teacher 是否正确认可。
3. 更新 hint、误区、理解、confidence 和 mastery。
4. 检查终止条件。
5. 若未终止，选择下一条 `StudentMathIntent`。
6. 用 math_checker 验证数学意图。
7. 生成学生措辞。
8. 记录完整 `StudentTransition`。

`mastery` 只反映模拟学生内部学习变化，不控制 Teacher workflow。所有浮点更新都截断到 `[0, 1]`。

### 6.2 状态更新规则

规则按下列优先级执行：

1. **答案泄露**
   - 若 Teacher 在学生尚未给出最终答案时泄露答案，不增加步骤进度、understanding 或 mastery。
   - `confidence -= 0.05`。
   - 下一意图按 help style 选择 `ASK_WHY` 或 `RESTATE_CONFUSION`，不得直接复制泄露答案。

2. **针对性处理误区**
   - `addresses_misconception=True` 时清空 `active_misconception`。
   - `mastery += 0.10`，`confidence += 0.10`。
   - `confused -> partial`，`partial -> understood`。
   - 下一意图优先选择当前正确步骤。

3. **一般提示**
   - `gives_hint=True` 时 `hint_exposure += 1`。
   - `mastery += 0.05`，`confidence += 0.05`。
   - 高掌握学生直接选择当前正确步骤。
   - 中掌握学生在第一次提示后选择当前正确步骤。
   - 低掌握学生只有在误区已清除，或累计收到第二次提示后，才选择当前正确步骤；否则重述困惑或给出典型错误步骤。

4. **正确纠错反馈**
   - 模拟学生上一意图为 `INCORRECT_STEP` 且 Teacher 明确拒绝时，`confidence -= 0.05`。
   - 若同时给出针对性支持，按规则 2 恢复；否则下一意图为求助或再次尝试。

5. **无有效支持**
   - Teacher 只有泛化追问、未处理当前误区时，不提升 mastery 或 understanding。
   - 高掌握学生仍可独立给出当前正确步骤。
   - 中掌握学生按当前 confidence 选择正确步骤或猜测。
   - 低掌握学生选择典型错误步骤或重述困惑。

6. **学生数学表现**
   - 发出 `CORRECT_STEP` 后，只有 math_checker 判定正确才将 `current_solution_step += 1`、`consecutive_failures = 0`、`confidence += 0.05`。
   - 发出 `INCORRECT_STEP` 后，只有 math_checker 判定错误才将 `consecutive_failures += 1`、`confidence -= 0.10`。
   - 正确最终答案使 `understanding_status=understood`，但尚不直接终止。
   - 正确验算意图得到 Teacher 后续认可后，设置 `understanding_status=verified`。

### 6.3 终止条件

终止条件按以下优先级检查：

1. `solved_and_verified`
   - 学生已经给出由 math_checker 判定正确的最终答案；并且
   - 学生已完成场景定义的验算意图，Teacher 随后明确认可正确或确认验算。

2. `student_gave_up`
   - `confidence <= 0.10`；并且
   - 当前行为规则选择 `GIVE_UP`。

3. `student_stuck`
   - `consecutive_failures >= 3`；并且
   - 自最近三个失败起 `current_solution_step` 没有增加。

4. `max_turns`
   - 完成第 8 个 Teacher turn 后仍未满足前三项。

`max_turns` 是硬上限，CLI 不允许配置超过 8。终止后不得再生成学生消息或调用 Teacher。

## 7. 学生语言生成

### 7.1 默认模板

默认模式从场景的 `utterance_templates[intent_type]` 中确定性选择模板。选择索引由 `teacher_turn_index` 和固定 scenario 顺序决定，不使用随机数。

模板只能填入 `canonical_content` 和允许的简短风格前后缀。数学内容保持原样。

### 7.2 可选 DeepSeek 措辞

环境变量：

- `USE_REAL_LLM_FOR_EVAL=true`：控制 Teacher 是否使用真实 DeepSeek，沿用现有含义。
- `USE_REAL_STUDENT_SIMULATOR_LLM=true`：仅控制学生措辞是否调用真实 DeepSeek。

两者相互独立，默认均关闭。

开关组合的环境隔离规则为：

- 两者都关闭：移除当前进程的 API key，并设置 `TEACHER_FORCE_MOCK=1`。
- Teacher 关闭、Student wording 开启：保留 API key，Teacher 调用期间保持 `TEACHER_FORCE_MOCK=1`；Student wording 通过显式构造 `DeepSeekClient` 调用，不以 `is_configured()` 决定是否执行。
- Teacher 开启、Student wording 关闭：保留 API key，不设置 `TEACHER_FORCE_MOCK`，但学生始终使用模板。
- 两者都开启：保留 API key，Teacher 和 Student wording 分别走各自入口。

这样 Student wording 的付费开关不会意外打开 Teacher LLM，Teacher 的开关也不会强制开启学生改写。

学生措辞调用使用 `DeepSeekClient.chat_json(..., schema=StudentWording)`：

```python
class StudentWording(BaseModel):
    text: str = Field(min_length=1, max_length=200)
```

Prompt 只包含：当前学生意图、canonical content、required anchors、help style 和简短语气要求。不得包含 Teacher mode 或 Teacher 内部元数据。

改写结果必须通过全部校验：

- 包含所有 `required_math_anchors`。
- 未引入 canonical content 以外的新方程。
- math_checker 对改写文本的步骤判定与 `expected_step_validity` 一致。
- 非数学意图不得新增方程或最终答案。
- `GIVE_UP`、`ASK_HELP`、`ASK_WHY` 等意图仍包含对应语言信号。
- 文本长度不超过 200 字。

任何 API 错误、JSON/Pydantic 错误或语义校验失败，都静默回退到模板，并在 trace 中记录 `wording_fallback_reason`；不得让措辞失败改变状态转移。

## 8. Teacher mode 隔离与公平性

每个 `(scenario_id, mode)` 使用：

- 同一个 scenario 初始隐藏状态的深拷贝。
- 独立的 `student_id` 和 `conversation_id`。
- 独立的 evaluation runtime 路径，避免跨 mode memory 污染。
- 相同的最多 8 个 Teacher turn。

Agent mode 通过现有 `run_turn` 进入完整 workflow。`agent_no_graph` 和 `agent_with_graph` 继续使用现有 graph 开关语义。`deepseek_only` 继续使用现有直接 DeepSeek baseline，不调用 workflow、math_checker、memory 或 retrieval 作为 Teacher 组件。

这里的“学生数学意图必须由 math_checker 验证”属于 evaluation harness 对模拟数据的真实性保护，不表示 `deepseek_only` Teacher 获得 math_checker 能力。

Student Simulator 的调用边界只传 Teacher 回复文本。Trace 可以在模拟器返回之后记录 mode 和 Teacher 元数据，但这些信息不得进入模拟器状态转移或措辞 Prompt。

## 9. 对话轨迹与两阶段持久化

### 9.1 Trace schema

```python
class HybridTurnTrace(BaseModel):
    teacher_turn_index: int
    student_state_before: StudentHiddenState
    student_message: str
    teacher_response: str
    teacher_metadata: dict[str, Any]
    teacher_observation: TeacherObservation
    selected_student_intent: StudentMathIntent | None
    generated_student_message: str | None
    student_state_after: StudentHiddenState
    used_real_student_wording_llm: bool
    wording_fallback_reason: str

class HybridConversationTrace(BaseModel):
    schema_version: str
    run_id: str
    scenario_id: str
    concept_id: str
    mastery_level: str
    mode: EvalMode
    original_problem: str
    initial_student_state: StudentHiddenState
    final_student_state: StudentHiddenState
    initial_student_message: str
    turns: list[HybridTurnTrace]
    transcript: list[dict[str, str]]
    termination_reason: TerminationReason
    teacher_turn_count: int
```

`transcript` 按实际顺序保存完整 student/teacher 文本。`turns` 保存可审计的状态变化和信号。二者均不可只保存摘要。

### 9.2 文件流程

每次脚本运行创建独立目录：

```text
<TEACHER_LOG_DIR>/hybrid_eval/<run_id>/
  manifest.json
  hybrid_traces.jsonl
  hybrid_scores.jsonl
  hybrid_summary.json
```

执行顺序必须是：

1. 运行选定的 scenario-mode 组合。
2. 完整写入并关闭 `hybrid_traces.jsonl`。
3. 写入 manifest 中的 `trace_count`。
4. scorer 从磁盘重新读取 `hybrid_traces.jsonl`。
5. 写入 `hybrid_scores.jsonl`。
6. 写入聚合 `hybrid_summary.json`。

不得在 trace 持久化完成前把内存对象直接交给 scorer。这样可以证明评分针对的是实际保存的研究数据。

完整默认运行必须验证 trace 和 score 都恰好为 54 条。分批运行只验证其过滤后的期望笛卡尔积数量，并在 manifest 保存过滤条件。

## 10. 确定性评分

评分对象是已保存的完整 trace。评分不读取 Teacher mode 私有能力来加分，只使用对话文本、可验证数学行为和学生轨迹；mode 仅用于结果分组。

### 10.1 逐对话指标

```python
class HybridConversationScore(BaseModel):
    run_id: str
    scenario_id: str
    concept_id: str
    mastery_level: str
    mode: EvalMode
    teacher_quality_pass_rate: float
    teacher_quality_checks: dict[str, bool]
    task_success: bool
    solution_progress: float
    misconception_resolved: bool | None
    understanding_gain: int
    confidence_change: float
    teacher_turns: int
    hint_exposure: int
    termination_reason: TerminationReason
```

指标定义：

- `teacher_quality_pass_rate`
  - 对每个 Teacher 回复运行现有确定性教学不变式检查：保持原题、无无关方程、未提前泄答案、最多一个问句、正确步骤被认可、错误步骤被纠正、why 问题得到等式性质解释。
  - 所有适用检查的通过数除以检查总数。

- `task_success`
  - 仅当终止原因为 `solved_and_verified` 时为 `true`。

- `solution_progress`
  - `min(final.current_solution_step / len(solution_steps), 1.0)`。

- `misconception_resolved`
  - 初始无误区时为 `null`。
  - 初始有误区且最终 `active_misconception is None` 时为 `true`。

- `understanding_gain`
  - 按 `confused=0, partial=1, understood=2, verified=3` 计算最终值减初始值。

- `confidence_change`
  - 最终 confidence 减初始 confidence，保留四位小数。

- `teacher_turns`
  - 实际 Teacher turn 数，范围 1 到 8。

- `hint_exposure`
  - 最终状态的 hint exposure。

不在第一版引入单一加权总分。研究比较保留上述指标向量，避免未经验证的权重掩盖“答对但教学差”或“过程改善但未在 8 轮内完成”的差异。

### 10.2 聚合指标

`hybrid_summary.json` 分别按下列维度聚合：

- mode
- mode + concept
- mode + mastery
- mode + concept + mastery

每组报告：

- conversation count
- teacher quality 平均通过率
- solved_and_verified 成功率
- 平均 solution progress
- 可适用场景的 misconception resolution rate
- 平均 understanding gain
- 平均 confidence change
- 平均 Teacher turns
- 平均 hint exposure
- 四种 termination reason 的数量与比例

## 11. CLI 与运行语义

默认完整离线运行：

```powershell
python scripts/run_hybrid_eval.py
```

支持过滤：

```powershell
python scripts/run_hybrid_eval.py --mode agent_with_graph
python scripts/run_hybrid_eval.py --concept move_terms
python scripts/run_hybrid_eval.py --mastery low
python scripts/run_hybrid_eval.py --mode agent_no_graph --concept remove_parentheses --mastery medium
```

参数：

- `--mode`：`all` 或三个 EvalMode，默认 `all`。
- `--concept`：默认全部六个概念。
- `--mastery`：`all|low|medium|high`，默认 `all`。
- `--max-turns`：1 到 8，默认 8，禁止大于 8。
- `--output-dir`：可选；默认使用带 run_id 的日志目录。
- `--run-id`：可选；默认用 UTC 时间戳和短 UUID。

环境变量控制真实 LLM。脚本在默认离线模式下必须移除 API key 并设置 `TEACHER_FORCE_MOCK=1`，沿用现有评测的双保险。研究对话得分较低不导致进程失败；schema、场景覆盖、持久化数量或运行异常会导致非零退出码。

## 12. 错误处理

- 场景 schema、覆盖组合或数学意图校验失败：运行前失败，不生成部分实验结果。
- 任一 Teacher 调用失败：中止本次生成，保留错误上下文但不伪造四种学生终止状态，不启动 scorer，并以非零状态退出。已经写入的内容只作为不完整运行诊断，manifest 标记 `complete=false`，不得纳入 54 组正式数据。
- 学生措辞 LLM 失败：回退模板，不中断对话。
- trace 写入失败：不启动 scorer，进程非零退出。
- scorer 遇到无效 trace：不写 summary，进程非零退出。
- 过滤后没有场景：命令行参数错误，进程非零退出。

任何错误日志不得包含 API key 或完整环境变量。

## 13. 测试策略与 TDD 顺序

实施严格按测试先行推进。

### 13.1 场景与状态模型

- 默认场景文件恰好加载 18 个场景。
- 六个 concept 和三档 mastery 的笛卡尔积完整且唯一。
- 同 concept 三档使用相同 original problem。
- 所有正确、错误、最终答案 intent 均经 math_checker 得到声明的判定。
- Pydantic 拒绝越界 mastery/confidence、负计数和非法枚举。

### 13.2 状态转移

- targeted support 清除误区并提升理解。
- generic hint 增加 hint exposure，但不会无条件清除误区。
- premature answer leak 不推进步骤或理解。
- correct step 只在 math_checker 判定正确后推进索引。
- incorrect step 增加 consecutive failures。
- 连续三个失败且无进展终止为 student_stuck。
- 低 confidence 触发 student_gave_up。
- 正确答案、验算和 Teacher 认可后终止为 solved_and_verified。
- 第 8 个 Teacher turn 后强制 max_turns，且没有第 9 次调用。

### 13.3 措辞安全

- 默认模板输出确定且保留 canonical math。
- DeepSeek 返回新增方程时回退。
- DeepSeek 改变步骤正确性时回退。
- DeepSeek 把求助改成答案时回退。
- API、JSON 和 Pydantic 错误均回退且不改变状态。

### 13.4 Mode 隔离与 runner

- 三个 mode 获得值相等但对象独立的初始状态。
- 模拟器公开入口和措辞 Prompt 不包含 mode。
- 每个 mode 使用独立 student_id、conversation_id 和 runtime path。
- runner 每轮只把 Teacher 回复文本传给模拟器。
- Agent mode 调用 workflow，deepseek_only 不调用 workflow。
- no-graph 与 with-graph 保持现有开关语义。
- runner 生成完整 transcript 和每轮前后状态。
- scorer 只能从已关闭并重新读取的 trace 文件启动。

### 13.5 Scorer 与端到端

- 用手工构造的固定 trace 验证每个指标的精确值。
- 聚合按 mode、concept、mastery 得到正确分母。
- 默认离线完整运行得到 54 条 trace、54 条 score。
- 每个 scenario 在三个 mode 中的 initial_student_state 完全相等。
- 每段对话 Teacher turn 不超过 8。
- 现有 pytest 与 golden evaluation 继续全绿。

最终验证命令：

```powershell
python -m pytest
python scripts/run_eval.py
python scripts/run_hybrid_eval.py
```

真实 DeepSeek 验证是可选的付费实验，不属于默认完成门：

```powershell
$env:USE_REAL_LLM_FOR_EVAL = "true"
$env:USE_REAL_STUDENT_SIMULATOR_LLM = "true"
python scripts/run_hybrid_eval.py
```

## 14. 验收标准

实现满足以下条件才算完成：

- 默认场景集精确包含 18 个场景，覆盖 6×3 组合。
- 默认完整运行精确产生 54 条 trace 和 54 条 score。
- 每种 mode 对同一场景从相同学生隐藏初态开始。
- 模拟器没有接收或推断 Teacher mode 的接口。
- 所有状态转移和终止决定均为确定性规则。
- 所有学生数学意图经 math_checker 验证。
- DeepSeek 仅能改写措辞，失败时安全回退模板。
- 每段对话最多 8 个 Teacher turn。
- trace 包含完整对话和每轮状态变化。
- trace 已写入并关闭后才开始评分。
- 评分不使用 LLM-as-judge。
- 未修改核心 workflow、math_checker 或 retriever 行为。
- `python -m pytest`、`python scripts/run_eval.py` 和完整离线 Hybrid Evaluation 均通过各自的运行完整性检查。

## 15. 实施边界总结

本设计把 Student Simulator 作为 evaluation-only 的研究参与者。它模拟的是“同一隐藏学生状态面对不同 Teacher 回复时如何继续”，而不是另一个自由聊天模型。确定性状态机负责因果与数学意图，DeepSeek 最多负责表面语言。Teacher 仍按原有 mode 运行，现有回归门仍保护教学不变式。最终产物既能逐轮审计，也能在 54 个配对实验上比较 Teacher quality、学习进展和终止结果。
