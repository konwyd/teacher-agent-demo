# 通用数学会话连续性与旧方程路由收紧设计

## 目标

修复二元一次方程组被错误送入一元一次方程兼容流程、以及后续短追问丢失原题的问题。修改后，旧流程只处理它能够由 `math_checker` 确定解析的一元一次方程；其它数学问题进入通用 Teaching Harness，并在同一 `conversation_id` 内持续使用首轮原题上下文。

本设计不新增二元一次方程知识包、题型枚举或核心教学策略分支。二元一次方程组只是验证通用能力边界的首个回归样本。

## 已确认行为

- `2x+3=7` 这类 `math_checker` 能确定解析的单个一元一次方程继续走旧兼容流程，保留精确步骤判定、掌握度、评测和 GraphRAG 行为。
- `3x+2y=5, 6x-2y=3` 这类旧 checker 无法处理的问题进入通用数学推理流程，不再显示 `c_linear` 或“一元一次方程”。
- 通用问题首轮保存完整原题；同一会话中的“具体要怎么做”“我不知道怎么选择方法”等短追问继续使用该原题。
- “具体要怎么做”授权给出可执行的下一步及其依据，但不自动授权披露最终答案。
- 学生明确请求完整答案时，继续遵循现有 `TeachingPolicy` 的答案披露门控。
- UI 继续传递现有 `conversation_id`，不修改前端会话协议。

## 方案比较

### 方案 A：严格能力路由并复用 EpisodeState（采用）

旧适配器通过 `math_checker.find_equations()` 判断当前输入是否只有一个可解析方程，并优先遵守已有 episode 的运行路径。通用流程在现有 `EpisodeState` 中保存中性的 `problem_context`，避免另建存储。

优点是改动局限在路由、通用编排、状态模型和回归测试；旧一元一次方程实现不用迁移。代价是同一通用 episode 默认围绕首轮原题，切换新题应使用现有“新建会话”能力。

### 方案 B：只移除 `x` 和 `=` 路由标记

可以修复首轮误分类，但不保存通用原题，第二轮仍会把“具体要怎么做”当成独立题目，因此不采用。

### 方案 C：新建完整通用对话存储与语义路由器

能够更细致地区分同一会话中的新题和追问，但会与现有 EpisodeState 重叠，并引入额外 LLM 路由调用和迁移成本。本轮问题不需要这一复杂度，因此不采用。

## 架构边界

### 旧方程兼容路由

`supports_legacy_equation_turn()` 按以下顺序判断：

1. 已有 episode 含 `target_equation`：继续旧流程，保证学生的短回答、步骤和练习请求仍由 checker 驱动。
2. 已有 episode 含通用 `problem_context` 且无 `target_equation`：继续通用流程，避免方程组学生写出 `9x=8` 后突然切入旧流程。
3. 新 episode 仅在 `math_checker.find_equations(message)` 恰好返回一个可解析方程时进入旧流程。
4. 其它输入进入通用流程。

这里的判断是能力检测，不是知识点分类。核心 `TeachingPolicy` 不读取 `x`、`y`、“二元”或其它题型关键词。

### 通用问题上下文

`EpisodeState` 增加可选字段 `problem_context`。它保存学生首轮提交的完整通用数学问题，与旧流程的 `original_problem`、`target_equation` 并存但不互相替代。

`run_generic_teaching_turn()` 在分析前读取 episode：

- episode 没有 `problem_context` 时，以当前消息初始化并保存；
- episode 已有 `problem_context` 时，将其作为原题，把当前消息只作为学生本轮请求或尝试；
- `TeachingState.problem_context` 始终是稳定原题，不被短追问覆盖。

一轮结束后，通用流程保存 `current_phase`、教师最后一个问题、预期学生响应和最近轮次摘要。保存失败不得改走旧方程流程，也不得伪造数学结论；本地 JSON store 的现有失败语义保持不变。

### 会话投影与 UI 元数据

通用 `TeachingState` 携带当前 `EpisodeState` 和中性的 `ContextResolution`。投影回兼容 `TeacherState` 时保留这两个字段：

- 首轮 `is_reply_to_previous_teacher=false`；
- 已有通用 `problem_context` 的后续轮次为 `true`，reason 说明复用了会话原题；
- `detected_concept_id` 和 `detected_concept_name` 继续为空，不用未经验证的题型标签填充 UI。

API 的现有字段和 Streamlit 请求格式不变。`original_problem` 兼容输出可继续映射为通用 `problem_context`，便于现有详情区显示稳定原题。

## 数据流

```text
student message + conversation_id
  -> inspect existing EpisodeState
  -> legacy capability check
       -> supported single linear equation: legacy adapter + math_checker
       -> otherwise: stable generic problem_context
            -> MathAnalysis(current message, stable problem)
            -> GenericMathReasoner(current message, stable problem)
            -> OptionalVerifier / Optional GraphContext
            -> TeachingPolicy
            -> GuardedResponseGenerator
            -> persist generic episode metadata
            -> TeachingState / compatible TeacherState
```

## 回答质量约束

对于方程组回归会话，底座 LLM 负责理解消元方法，Harness 只提供稳定原题、当前学生请求和披露许可。预期教学行为如下：

1. 首轮可以指出系数互为相反数，建议消去一个未知数。
2. 学生追问“具体要怎么做”时，应给出“将两式对应相加，使相反数项抵消”这样的可执行步骤。
3. 学生说“不知道怎么选择方法”时，应解释根据未知数系数关系选择加减消元，而不是重复询问其卡点。
4. 未明确索要最终答案时，不直接输出完整解和值。
5. 每轮仍恰好保留一个结尾问题。

离线测试用注入的 fake reasoner 和 response client 验证上述 Harness 数据流，不把固定自然语言模板写成产品知识包，也不要求真实模型逐字输出。

## 测试设计

### 路由单元测试

- 单个可解析一元一次方程继续返回 `c_linear` 和规范化 `target_equation`。
- 含两个未知数的方程组走通用 trace，`detected_concept_id` 为空。
- 通用 episode 后续出现可解析的中间式时仍留在通用流程。
- 旧一元一次方程 episode 的无公式追问仍留在旧流程。

### 会话连续性测试

使用同一 `conversation_id` 连续运行：

1. `老师，3x+2y=5，6x-2y=3怎么解？`
2. `具体要怎么做呢？`
3. `我不知道要怎么选择方法`

断言三轮的 `problem_context` 都是首轮完整题目，后两轮的 reasoner 均接收“稳定原题 + 当前消息”，后两轮不会返回现有固定 DIAGNOSE 文案，且 `context_resolution.is_reply_to_previous_teacher` 为真。

### API 与回归验证

- `/turn` 对方程组返回空 `detected_concept`，并在 `teaching_state.problem_context` 和兼容 `original_problem` 中返回完整原题。
- `python -m pytest` 全部通过。
- `python scripts/run_eval.py` 默认离线评测全部通过。
- 三种现有评测模式继续离线运行，旧一元一次方程 golden cases 不降低断言。
- 不运行真实付费 API；真实 DeepSeek 行为由用户在改动完成后自行启动或另行明确授权 smoke。

## 明确不做

- 不修改 `math_checker` 使其求解二元一次方程组。
- 不在 `TeachingPolicy` 增加二元一次方程、几何、概率等知识点判断。
- 不新增方程组 Graph 节点、RAG 数据、知识包或固定解题模板。
- 不让 LLM 推翻 `math_checker` 对受支持一元一次方程的确定性结论。
- 不重写现有 memory、API、UI 或 evaluation 层。
- 不自动合并、推送或删除当前分支和历史提交。

## 风险与缓解

- **同一通用会话中切换新题：** 本轮维持“一个 conversation 对应一个稳定原题”的现有教学主线；用户可通过 UI 新建会话切题，避免用脆弱题型规则猜测。
- **方程组中间步骤形似一元一次方程：** 已存在通用 `problem_context` 时优先保持通用路由，避免中途切换状态机。
- **模型仍可能给出低质量步骤：** structured reasoning、答案披露守卫和单问题后处理继续生效；本轮只修 Harness 上下文，不宣称新增确定性方程组 verifier。
- **旧方程回归：** 已有 `target_equation` 的 episode 优先保留旧路径，并运行完整 pytest、mock eval 和模式回归。

## 验收标准

1. 截图中的方程组首轮不再被标记为“一元一次方程”。
2. 三轮回归中原题上下文保持不变，后两轮不再重复固定诊断回复。
3. 旧一元一次方程步骤检查、跨轮 episode、mastery、RAG 和 golden evaluation 不退化。
4. 核心 TeachingPolicy 不包含任何新增知识点专用分支。
5. 工作区只包含本设计范围内的代码、测试和文档变更，完整离线验证通过。
