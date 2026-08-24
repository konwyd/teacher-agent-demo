# Teacher Agent 快速路由与 Fast Miss 执行链路优化设计

## 1. 目标

在保留现有 TeachingContract、数学 deterministic verifier、学生记忆、RAG、Agent、安全审计、执行 trace、性能指标和 Streamlit UI 的前提下，减少普通 K9 请求的串行 LLM 调用与后端延迟。

本次实现严格限于任务 prompt 的范围，不新增封闭式短答案模式，不重构现有数据模型，不实现真正 token streaming。

目标调用预算：

- 确定性路由命中的低风险概念解释：约 1 次，只进行 Teacher generation。
- Fast Router 未命中、模型分类器确认是普通 K9：约 2 次，即 Classifier + Teacher。
- 明确需要动态工具的请求：尽量控制在 3～4 次，ActionProvider 最多调用 2 次。

## 2. 已确认的当前调用链

当前主链路为：

```text
用户消息
→ run_turn()
→ MultisubjectAgentLoop.run()
→ choose_fast_route()
→ RequestClassifier（Fast miss 时）
→ run_fast_plan() 或 ActionProvider loop
→ RAG / tools
→ Teacher generation
→ SemanticBoundaryAuditor
→ conditional repair
→ UI
```

当前成本来源：

- `RequestClassifier.classify()` 调用一次 `chat_json()`。
- `DeepSeekActionProvider.decide()` 每执行一次调用一次 `chat_json()`；当前最多执行 4 轮。
- Teacher generation 调用一次 `chat()`。
- `SemanticBoundaryAuditor.audit()` 每执行一次调用一次 `chat_json()`。
- `_repair()` 每执行一次调用一次 `chat()`；高风险修复后还可能再次进行 semantic audit。
- local RAG、QA Cache、Memory 和 deterministic verifier 本身不调用 DeepSeek。
- `DeepSeekClient` 汇总 `llm_call_count` 与 token usage；`logic.py` 记录 stage timing 和 backend total time。

因此当前普通 Fast 请求通常为 Teacher + Audit 两次调用；Fast miss 可能形成 Classifier + 最多 4 次 ActionProvider + Teacher + Audit 的 7 次常态链路，修复时更多。

## 3. 设计原则

### 3.1 路由置信度、推理复杂度与工具需求分离

“复杂问题”不等于“需要动态 Agent Tool Loop”。证明、详细解释、多步骤推理可以由 Teacher 一次生成；只有需要联网、实时资料、官网来源或动态证据补充时才进入 Agent Loop。

### 3.2 Fast Miss 不等于 Deep Agent

Fast Router 未命中后先调用模型分类器。分类器一旦确认请求属于普通 K9 且不需要动态工具，后续立即使用确定性执行计划，不调用 ActionProvider。

### 3.3 检索从默认执行改为明确按需

新题或概念请求不再自动触发 local RAG。只有学生明确要求教材、课标、原文、页码、题库等本地证据时才执行 local retrieval；联网类请求仍遵守 local-before-web 边界。

### 3.4 安全检查按风险分层

低风险概念解释跳过 LLM Semantic Audit，但必须执行 deterministic hard boundary check。当前题续接、多步骤引导、禁止答案、verifier 冲突或已出现 warning 的场景继续保留语义审计和条件修复。

## 4. 路由架构

对外只使用任务 prompt 指定的路由标签：

```text
deterministic_fast_path
model_classifier_direct_path
dynamic_agent_path
memory_clarification_reply
```

### 4.1 L0：deterministic_fast_path

`choose_fast_route()` 使用高置信度 subject、stage、continuity 和概念模式进行确定性分类。命中后直接进入确定性计划。

### 4.2 L1：model_classifier_direct_path

Fast Router miss 时调用 `RequestClassifier`。若分类结果满足以下条件，则直接复用确定性计划：

- `scope == K9`；
- `subject` 已确定；
- 当前请求不需要动态工具。

该路径不调用 `DeepSeekActionProvider.decide()`。

### 4.3 L2：dynamic_agent_path

只有请求明确需要联网、官网、网页、最新或实时来源等动态工具能力时进入 bounded Agent Loop。

进入前由 Python 确定执行已有信息可以决定的动作：

- 构建 TeachingContract；
- 选择并加载 skill；
- 根据 evidence policy 决定 local retrieval；
- 根据 Quality QA policy 决定历史优质问答检索。

ActionProvider 只负责剩余动态决策，例如在 local retrieval 成功且仍有证据缺口时选择 Web 或结束。默认最大迭代次数由 4 调整为 2。

### 4.4 memory_clarification_reply

保持现有学生记忆歧义澄清直返逻辑。普通上下文续接沿用确定性快速路由，不引入新的 router label。

## 5. 组件修改

### 5.1 `fast_routing.py`

- 保留 `FastRoute` 数据类。
- 将 subject cue 组织为 `dict[Subject, tuple[str, ...]]` 或等价显式映射，消除数组下标依赖。
- 添加任务 prompt 要求的数学高置信度知识点。
- 平衡九学科 cue，只添加跨学科歧义低的词。
- 明确的物理、化学知识点可以推断 `SchoolStage.MIDDLE`；其它未知学段不得强制设为初中。
- 保持表达式、显式学科和上下文续接能力。
- 让“那为什么呢”等 continuation 继承 active subject/stage，并设置 `is_new_problem=False`。
- 从 Quality QA marker 中移除“为什么、为啥、怎么解释、怎么理解、区别、思路”等普通解释词；保留技巧、记忆、常见错误、批改、作文、阅读理解等经验型需求。
- 新增纯策略函数 `needs_dynamic_tool_loop(message, classification)`，只依据明确动态证据需求判断，不因证明、详细解释、多步骤或消息长度进入 Agent Loop。

### 5.2 `fast_plan.py`

- 新增或复用纯策略函数 `needs_local_evidence(message)`。
- 普通新题和概念解释默认 `RetrievalMode.NONE`。
- 教材、课本、课标、课程标准、原文、原句、出处、页码、练习册、题库、教材例题或原题等明确需求允许 local retrieval。
- 当前题 continuation 默认不重复检索，除非当前消息明确提出证据要求。
- 保留 `load_skill`、`search_local_knowledge`、`search_quality_qa` 和 `search_web` 能力；只改变触发策略。
- 确定性执行结果通过传入的 router label 区分 `deterministic_fast_path` 与 `model_classifier_direct_path`。

### 5.3 `orchestrator.py`

- 保留 L0 `choose_fast_route()`。
- Fast Router miss 后调用分类器，但不自动进入 ActionProvider loop。
- 对普通 K9 构建模型分类后直达的 `FastRoute` 或等价确定性计划输入，并返回 `model_classifier_direct_path`。
- 只有 `needs_dynamic_tool_loop()` 为真时进入 `dynamic_agent_path`。
- 动态路径在循环前确定性构建 contract、加载 skill，并按 evidence policy 执行 local/Quality QA retrieval。
- 默认 `max_iterations=2`。
- 保留 local max 1、Quality QA max 1、Web max 1、tool timeout 和 local-before-web。
- 保留分类失败、provider error、工具失败的 fail-safe 行为，不让 Memory 或 RAG 成为硬流程控制。

### 5.4 `logic.py` 与 `boundary_audit.py`

- 抽取 `is_low_risk_fast_generation(...)` 或等价纯策略函数。
- 低风险条件至少包括：K9、新概念问题、确定性或模型分类后直达路径、`concept_explanation`、非 active problem 敏感续接、没有显式禁止答案、没有 verifier 冲突和 deterministic warning。
- 低风险输出先执行 `hard_boundary_recheck()`；无 violation 时直接结束，不调用 semantic auditor。
- deterministic guard 发现 violation 时才进入 repair；不得静默放行。
- `one_step`、`guided_solution`、当前题续接、复杂 disclosure contract、禁止答案和 verifier 风险继续调用 semantic auditor。
- `_repair()` 仅在 length、boundary violation、verifier conflict 或真实 warning 时执行。
- 高风险修复后按现有安全策略进行必要的 post-repair audit；低风险路径不固定追加 audit。
- 不修改 `LinearEquationVerifier`、`VerificationRequest`、claim audit 以及 verified correct/incorrect 的优先级。

### 5.5 元数据与兼容性

每轮至少可观测：

```text
router_type
llm_call_count
backend_total_seconds
stage_timings
retrieval_mode
agent_iterations
repair_performed
semantic_audit_performed
```

实现优先复用现有 `agent_termination_reason`、`routing_guidance`、trace、retrieval decision 和 model metadata。确需增加 `PrototypeTurnRecord` 字段时必须提供默认值，确保旧 `ConversationRecord`、旧 `PrototypeEpisode` 和 Streamlit 热重载对象仍可通过 Pydantic 校验。

### 5.6 Streamlit UI

- 保留现有决策、工具与验证、Agent 执行链、处理耗时、Token 和模型调用次数展示。
- 将四个 router label 映射为：确定性快速路由、模型分类后直达生成、动态 Agent 工具路由、记忆澄清续接。
- 展示或复用 retrieval mode、Agent iterations、repair 和 semantic audit 状态。
- 不修改当前伪流式展示架构，不声明实现真正 TTFT/token streaming。

## 6. 错误处理与边界

- 分类器格式失败继续安全降级，不能绕过 Pydantic 校验。
- 模型分类结果为 social/off-topic 时继续使用直接响应和既有范围边界。
- local retrieval 失败不能自动放宽 Web 权限；Web 仍要求 local search 成功。
- Memory、RAG 和 Quality QA 失败均 fail-open 到不使用该辅助信息，但不得修改 checker/verifier 结论。
- Agent 迭代预算耗尽时使用已有证据和安全 TeachingContract 收口。
- 任何调用预算优化都不得通过删除 trace、安全审计实现。

## 7. 测试设计

采用 TDD，先写失败测试并确认失败原因，再实现最小代码。

### 7.1 Fast Router

- `勾股定理是什么`、`什么是平方根`、`一次函数是什么`、`一元二次方程怎么理解`、`全等三角形是什么`、`圆周角是什么` 命中 K9/Math FastRoute。
- 明确数学主题推断正确学段或保持符合现有上下文的学段。
- 物理 cue 不再错误读取语文 cue。
- 其它学科高置信 cue 不使用模糊单字误判。

### 7.2 Continuity

在勾股定理的数学/初中 active context 下，`那为什么呢？` 命中 FastRoute、继承 Math/Middle，并设置 `is_new_problem=False`。

### 7.3 Retrieval policy

- `勾股定理是什么` 不调用 local、Quality QA 或 Web。
- `根据教材解释一下勾股定理` 调用 local retrieval。
- `为什么勾股定理成立？` 不因“为什么”启用 Quality QA。
- continuation 默认不重复 retrieval。

### 7.4 Fast miss direct path

使用 fake classifier 返回 K9/Math/Middle，验证普通冷门概念请求调用 classifier 一次、ActionProvider 零次，并产生 `model_classifier_direct_path`。

### 7.5 Dynamic path

`联网查一下最新的初中数学课程标准` 进入 `dynamic_agent_path`，并验证 ActionProvider 调用不超过两次、local-before-web 仍有效。

### 7.6 Semantic audit

- 低风险 Fast concept：Teacher 一次、SemanticBoundaryAuditor 零次，并执行 deterministic hard-boundary check。
- 模型分类后直达的低风险概念保持 Classifier + Teacher 两次预算。
- 高风险 continuation 继续执行 semantic audit。
- 只有真实 violation/warning 才 repair。

### 7.7 兼容与 UI

- 旧 turn/episode JSON 缺少新增 metadata 时仍可加载。
- UI 四种路由标签映射正确。
- 原有性能、RAG、verifier 和 token 展示仍存在。

### 7.8 完整回归

最终必须运行并通过：

```bash
python -m pytest
python scripts/run_eval.py
```

测试失败时修复真实行为问题，不放宽或删除现有评测断言。

## 8. 非目标

- 不新增短答案或封闭式计算请求类型。
- 不修复 `2+2=几？` 的教学契约策略。
- 不接入新的 RAG、Web 或模型服务。
- 不重构 DeepSeek client。
- 不实现真正 token streaming。
- 不删除 Agent、Semantic Audit、TeachingContract、Memory、RAG、数学 verifier、trace 或 UI 调试能力。
