# 契约优先的数学教学编排重构设计

状态：已批准

日期：2026-07-15

适用分支：`refactor/topic-agnostic-teaching-harness`

## 1. 决策摘要

Teacher Agent 继续采用“单编排器 + 可测试专家模块”，不改成多个自主 Agent 互相对话。重构目标是让系统更会教，而不是搭建通用多 Agent 平台。

本设计采用“契约优先、纵向切片收敛”路线：先用安全契约和真实多轮教学场景定义行为，再把现有两个重叠的教学动作模型合并成一个 `TeachingAction`；随后以 shadow mode 引入新的教学规划内核，先迁移一条完整的一元一次方程流程，最终让旧流程和通用流程收敛到同一个官方运行时。

核心决定如下：

- 一个回合只选择一个主教学动作，差异由类型化参数表达；
- 确定性安全规则限定可选动作和答案披露边界，LLM 只在允许集合内选择；
- `math_checker` 保持可判定代数步骤的最高裁决权；
- 学生模型第一版只保存可追溯的证据，不推断宽泛人格或长期能力标签；
- 策略专家是能力插件，不是第二套动作系统、完整解题器或独立 Agent；
- RAG/GraphRAG、复杂掌握度模型和多 Agent 消息总线都不是必选路径；
- 所有迁移严格执行 TDD、离线验证、独立提交和阶段停止门。

## 2. 与既有设计的关系

本文部分取代：

- `docs/superpowers/specs/2026-07-14-composable-strategy-family-design.md`
- 提交 `7cbda2f` 中与双重动作模型、策略顾问输出和阶段排序有关的设计；
- 提交 `f00eb4b` 中把旧方程运行时与通用 Teaching Harness 长期并列的过渡性设计。

旧文件和历史提交必须保留，不删除、不改写，以便审计阶段 0、阶段 1 和设计演进。

继续有效的既有决定包括：

- `problem_context` 首轮锚定并跨轮复用；
- 已被 `math_checker` 支持的旧一元一次方程行为不能退化；
- 核心教学策略不按方程组、几何、统计等知识点编写 `if/else`；
- verifier、memory 和 RAG 都不能推翻确定性检查结果；
- 未获授权时不泄露最终答案，每轮恰好一个结尾问题；
- 所有真实 LLM 调用继续经过 `app/llm/deepseek_client.py`，结构化输出使用 JSON mode 和 Pydantic 校验。

## 3. 目标与非目标

### 3.1 目标

1. 让连续教学回合根据学生的实际进展选择更合适的教学动作，而不是重复固定诊断文案。
2. 用统一动作契约消除 legacy `TeachingAction` 与通用 `TeachingActionKind` 的职责重叠。
3. 让安全规则、数学验证、教学规划和语言生成各自拥有清晰边界并可独立测试。
4. 保留一元一次方程的确定性优势，同时让方程组、几何、数据分析等能力通过同一编排器增量接入。
5. 让每次决策可追溯：知道使用了哪些证据、跳过了哪些模块、为什么选这个动作、为什么允许这个披露层级。

### 3.2 非目标

- 不构建多个拥有独立上下文和自主决策权的 LLM Agent；
- 不在第一版引入 BKT、知识追踪网络或复杂学生画像；
- 不把 RAG/GraphRAG 设为正常教学的前置条件；
- 不引入可绕过 `math_checker` 的通用 LLM 判题器；
- 不为每个策略插件增加一次 LLM 调用；
- 不一次性重写全部旧运行时或全部旧测试；
- 本文定稿和后续实施计划不自动授权任何后续运行时代码实施。

## 4. 开源项目借鉴边界

本设计借鉴组件思想，不复制完整架构：

| 项目 | 借鉴内容 | 明确不照搬 |
| --- | --- | --- |
| [DeepTutor](https://github.com/HKUDS/DeepTutor) | 单编排入口、能力注册、持久证据、可观测调用 | 多自主 Agent 协商、完整通用导师平台 |
| [MathDial](https://github.com/eth-nlped/mathdial) | 教师动作标注、学生困惑类型、真实多轮评测思路 | 直接把英文数据或标签当作中文初中教学标准 |
| [SocraticMath](https://github.com/ECNU-ICALK/SocraticMath) | 苏格拉底式教学场景、分步引导数据组织 | 严格限制为只提问、不允许直接解释的教学风格 |
| [Math-Verify](https://github.com/huggingface/Math-Verify) | 抽取、规范化、验证、比较的职责分离 | 替换现有 Fraction `math_checker` 或改变旧判定语义 |
| [OATutor](https://github.com/CAHLR/OATutor) | 提示路径、学习证据、练习递进 | 第一版 BKT 掌握度和强制路径控制 |
| [Socratic Tutor](https://github.com/gpoesia/socratic-tutor) | 领域能力、可用动作、完成条件之间的显式契约 | 将插件做成独立完整解题器 |

当前只把这些仓库作为设计参考。若后续需要复制代码、提示词或数据，必须先核对当时的许可证、署名、商业使用和衍生作品要求；默认不把第三方源码 vendoring 到本仓库。

## 5. 总体架构

```text
Student Message + Persisted Episode
        |
        v
ContextResolver
        |
        v
EvidenceAssembly
  |- StudentModel
  |- MathReasoner (on demand)
  |- StrategyRegistry (1 primary, up to 2 supporting)
  |- OptionalVerifier
  `- Optional Graph/RAG
        |
        v
SafetyGate
  |- allowed_actions
  |- disclosure_ceiling
  |- worked_example_allowed
  |- new_problem_allowed
  `- student_response_required
        |
        v
TeachingPlanner (LLM on demand)
        |
        v
TeachingBriefComposer
        |
        v
ResponseGenerator
        |
        v
ResponseGuard
        |
        v
Persist Evidence + Episode + Trace
```

这是一个单编排器运行时。所谓“专家”只是有明确输入输出的模块：`math_checker` 是确定性验证专家，StudentModel 是证据账本，StrategyRegistry 提供数学教学能力，Graph/RAG 是可选证据提供者。它们不互相聊天，也不独立生成最终教师回复。

决策优先级固定为：

1. verifier 的确定性事实；
2. SafetyGate 的安全与披露限制；
3. TeachingPlanner 的动作选择；
4. primary strategy 的主要教学建议；
5. supporting strategies 的补充建议；
6. Graph/RAG 的可选教学证据。

低优先级模块只能填补空缺，不能覆盖高优先级结论。

## 6. 统一教学动作契约

### 6.1 唯一动作枚举

新 `TeachingAction` 只保留九个稳定主动作：

- `ELICIT`：获取缺失的题目、目标或学生想法；
- `FOCUS`：把注意力集中到关键条件、错误位置或局部关系；
- `SCAFFOLD`：提供仍需学生参与的分层提示；
- `EXPLAIN`：解释概念、理由或方法；
- `CORRECT`：基于已验证错误进行明确纠正；
- `CHECK`：检查学生理解、步骤或理由；
- `MODEL`：在满足例题触发条件时示范一个 worked example；
- `PRACTICE`：提供受控的新练习或变式；
- `CONSOLIDATE`：总结已确认的进展和可迁移方法。

旧 `TeachingAction` 与 `TeachingActionKind` 不再长期并存。迁移期可以有单向兼容适配器，但新核心、日志和新测试只认统一动作模型。

### 6.2 SafetyConstraints

SafetyGate 先输出：

```text
SafetyConstraints
  allowed_actions
  disclosure_ceiling: NONE | NEXT_STEP | FINAL
  worked_example_allowed
  new_problem_allowed
  student_response_required
  reason_codes
```

披露权属于 SafetyGate，不属于 LLM。默认规则包括：

- 缺少可用题目时强制 `ELICIT`；
- 学生未求最终答案且未自行解出时，披露上限为 `NONE` 或受控的 `NEXT_STEP`；
- 不满足例题触发条件时禁止 `MODEL`；
- 已验证错误时允许 `FOCUS`、`SCAFFOLD`、`CORRECT`、`EXPLAIN`；
- 已验证正确时允许 `CHECK`、`EXPLAIN`、`CONSOLIDATE`、`PRACTICE`；
- 安全规则通常缩小候选集合，只在缺题或只剩一个合法动作时直接决定动作。

### 6.3 TeachingActionPlan

Planner 在允许集合内选择一个动作，并输出类型化参数：

```text
TeachingActionPlan
  action
  target
  scaffold_level: 0..3
  explanation_depth: brief | standard | detailed
  practice_difficulty: easier | similar | harder | none
  practice_similarity: same_strategy | transfer | none
  rationale
  confidence
```

参数按动作校验：`CORRECT` 必须有已验证错误，`MODEL` 必须有合法触发，`CONSOLIDATE` 必须有确认过的学生进展。非法组合或低置信度不能直接进入回复层，而是走确定性降级。

## 7. TeachingBrief 边界

`TeachingBrief` 是 ResponseGenerator 唯一允许消费的规划结果：

```text
TeachingBrief
  problem_context
  current_student_request
  verified_facts
  student_evidence
  action_plan
  primary_next_step
  allowed_content
  forbidden_disclosures
  closing_task
```

ResponseGenerator 不直接读取策略插件的原始输出、完整候选解、原始 RAG 文本、未验证推理或宽泛学生画像。`closing_task` 只能有一个，ResponseGuard 负责保证最终回复恰好一个结尾问题。

## 8. 简化学生模型

第一版 StudentModel 是可追溯证据账本，只保存：

- 学生已明确写出且验证正确的步骤或解释；
- verifier 确认的错误及错误位置；
- 尚未解决的障碍；
- 最近使用过的教学动作和提示层级；
- 学生在提示后是否产生可观察进展；
- 每条证据的来源回合、时间和置信度。

StudentModel 不保存“代数很差”“喜欢直接讲解”等未经验证的结论。学生自述但尚未验证的内容可以作为软证据，必须带来源和较低置信度，不能改变 checker 分支或答案披露边界。

## 9. 策略能力插件

策略插件提供可测试的数学教学能力，而不是 prompt card 或完整 solver。统一接口最多暴露：

- 推荐表示方式；
- 可选操作类型；
- 本轮诊断目标；
- 常见误区检查；
- 可用 verifier 或工具；
- 输入输出规范化器；
- 一个主要下一步候选及其证据来源。

每轮选择一个 primary strategy，最多两个 supporting strategies。插件本身不获得动作选择权、披露权、数学裁决权或最终回复生成权，也不各自触发 LLM 调用。

策略 ID 描述推理能力而非知识点，例如符号变换、数量建模、几何推理和数据分析。方程组可以选择符号变换能力，但核心 TeachingPolicy 不出现“如果是方程组就……”的分支。

## 10. 数据流与自适应 LLM 预算

正常回合按以下顺序执行：

1. ContextResolver 稳定解析首轮 `problem_context` 和当前请求；
2. StudentModel 生成当前证据快照；
3. OptionalVerifier 对当前学生步骤做确定性预检；不适用时返回 `not_applicable`，不能阻断教学；
4. MathReasoner 在 verifier 证据不足时按需理解题目、分析尝试、选择策略和候选下一步；
5. StrategyRegistry 运行一个 primary 和最多两个 supporting 插件；
6. 可选 Graph/RAG 只补充与当前教学决策直接相关的证据；
7. SafetyGate 计算允许动作、披露上限以及例题和新题权限；
8. TeachingPlanner 按需选择一个动作及参数；
9. TeachingBriefComposer 压缩并隔离下游输入；
10. ResponseGenerator 生成回复；
11. ResponseGuard 检查 grounding、答案泄露、例题资格和问句数量；
12. 持久化 episode、学生证据和 trace。

调用预算是自适应的：

- checker 或其它确定性工具已经足够时跳过 MathReasoner；
- SafetyGate 只留下一个动作时跳过 LLM Planner；
- 简单回合通常一次 LLM 调用；
- 常见回合约两次；
- 复杂回合最多三次；
- trace 必须记录每个模块被调用或跳过的原因。

## 11. 失败与降级

- MathReasoner 失败：策略为 `unknown`，不得伪造题意或结论；
- 单个策略插件失败：忽略该插件，不能阻断整轮；
- verifier 不适用：没有 `verified_facts`，候选推理不得冒充事实；
- Graph/RAG 失败：使用空证据继续，不改变动作或披露边界；
- Planner 失败：只能从 `allowed_actions` 中确定性降级；依次优先缺信息时的 `ELICIT`、确认错误时的 `FOCUS`、其余场景的一级 `SCAFFOLD`，若首选动作不在允许集合则选择允许集合中预先定义的最低披露动作；
- Generator 或 ResponseGuard 失败：使用按动作定义的确定性安全模板；
- 持久化失败：可以返回已通过 Guard 的安全回复，并尽力向独立日志或指标通道发出失败信号；下一轮不得假装状态已经保存。

任何降级路径仍须满足原题锚定、不泄露未授权答案、例题触发限制和恰好一个结尾问题。

## 12. 测试体系

### 12.1 永久安全契约

永久保留并强化：

- `math_checker` 的 Fraction 精确判定；
- original problem grounding；
- 未授权答案不泄露；
- 每轮恰好一个结尾问题；
- worked example 的三类触发条件：学生明确要例子、同一知识点连续两次答错、学生表示完全不懂；
- 跨轮 `problem_context`；
- API、TurnLog 和离线评测兼容性。

### 12.2 冻结的旧实现测试

依赖旧枚举、旧私有方法或固定文案的测试标记为 legacy，并在迁移期保持全绿，但不再增加新用例。删除任何旧测试前，必须在替代清单中指出覆盖同一风险的新契约或新场景测试。

### 12.3 新教学质量场景

新测试使用真实多轮对话，断言结构化教学行为而非逐字回复：

- 动作是否属于 SafetyGate 允许集合；
- 提示是否逐步加深而非机械重复；
- 是否准确定位和纠正已验证错误；
- 学生进步后是否更新证据并改变后续动作；
- 是否保留学生参与空间；
- 短追问是否继续围绕首轮原题并推进教学；
- 是否遵守答案披露、例题资格和结尾问题不变式。

Planner 和 Generator 使用 mock LLM。每个迁移批次必须依次完成：失败测试、确认因真实缺口失败、最小实现、目标测试、完整 `pytest`、`run_eval`、三模式离线验证。三模式指 `deepseek_only`、`agent_no_graph` 和 `agent_with_graph`；它们都走离线 mock 路径。不得通过删除断言或放宽安全要求制造 GREEN。

## 13. 实施批次与停止门

### 批次 0：契约盘点

给现有测试建立永久、legacy 和新行为三类清单；记录每个旧测试保护的风险，不改变运行时。

退出条件：所有旧动作耦合测试均有分类，永久安全契约可以单独运行。

### 批次 1：统一动作模型

建立九动作枚举、类型化参数、`SafetyConstraints`、`TeachingActionPlan` 和 `TeachingBrief`；通过单向适配器保持旧入口兼容。

退出条件：新核心只认统一动作；旧回归全绿；没有第二套新动作模型。

### 批次 2：新规划内核 shadow mode

接入 EvidenceAssembly、SafetyGate、Planner、BriefComposer 和 trace。新规划只记录，不接管用户可见回复。

退出条件：离线场景中可以解释候选动作、披露上限和模块跳过原因；旧回复行为不变。

### 批次 3：一元一次方程纵向切片

让 `2x+3=7`、学生步骤 `2x=4` 等一条完整流程经过新 Orchestrator、Planner 和 TeachingBrief；实际数学判断仍由 `math_checker` 完成。

退出条件：永久契约、新教学场景、旧 checker 回归和三模式离线验证全部通过，并有明确回退开关。

### 批次 4：能力扩展

先迁移其余一元一次方程路径，再依次启用方程组、几何和数据分析策略插件。每个领域是独立 TDD 子批次，不允许一次性全量切换。

退出条件：每个领域具备多轮场景、失败降级和无 RAG 模式覆盖。

### 批次 5：兼容层退出

只有旧入口、动作投影和旧测试都具备替代覆盖，并且 shadow/正式路径对安全契约无差异时，才删除兼容代码。

退出条件：仓库只剩一个官方教学编排运行时；删除列表和替代测试映射可审计。

每个批次独立 TDD、独立提交、完整离线验证并停止等待确认。本文不自动授权执行任何批次。

## 14. 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| 迁移期两套入口造成长期复杂度 | 每个适配器写明替代覆盖和删除条件；legacy 测试只冻结不扩张 |
| 新测试过度依赖 LLM 文案 | 断言动作、证据、披露和对话推进等结构化行为 |
| LLM Planner 选择不稳定 | SafetyGate 先限候选；单候选跳过 LLM；非法输出确定性降级 |
| 策略插件演化成知识点分支或完整 solver | 接口只暴露能力与教学证据；核心 Policy 不读取知识点关键词 |
| 学生模型把猜测当事实 | 只保存带来源和置信度的证据；checker 事实优先 |
| RAG 内容污染原题 | RAG 只能补 brief 空缺；普通回复不接收检索例题或新题 |
| 调用次数和延迟增加 | 确定性预处理、按需 Reasoner、单候选免 Planner、最多三次调用 |
| 第三方许可证或数据适配问题 | 默认只参考设计；复制任何源码或数据前单独审查许可证和适用性 |

## 15. 整体验收标准

完成整条路线时必须满足：

- 仓库只有一个统一 `TeachingAction` 契约和一个官方教学编排入口；
- 新旧一元一次方程场景继续由 `math_checker` 提供确定性判断；
- 方程组、几何和数据分析通过能力插件接入，核心 Policy 没有知识点 `if/else`；
- 三轮及更长对话稳定复用原题并根据学生证据推进，不重复固定诊断文案；
- 未索要最终答案时不泄露完整答案或完整数对；
- 每轮恰好一个结尾问题；
- RAG/GraphRAG 缺失时仍能完成安全教学；
- 学生模型不包含未经证实的长期能力或偏好标签；
- `python -m pytest`、`python scripts/run_eval.py` 和三模式离线验证达到对应批次基线；
- 全部验证使用 mock/offline 路径，不调用真实付费 API。

## 16. 已解决的设计问题

本规格没有待定的架构分支。已确定：单编排器、统一九动作模型、安全规则限定候选、LLM 在候选内选择、简化证据账本、自适应调用预算、契约优先测试迁移、shadow mode 和一元一次方程纵向切片。后续若改变其中任何决定，应先更新本规格并重新获得批准。
