# Teacher Agent 动态 Reasoning Trace 设计

> 日期：2026-08-21  
> Active subsystem：Agent Runtime / Product UI  
> Acting owner：3322  
> 状态：已由 owner 确认，待实施  

## 1. 背景

8502 当前已经具备安全的公开思考摘要能力，但前端把摘要固定组织为“题意 / 写法 / 资料 / 成稿”四个槽位，并以“备课批注”展示。该结构能够保存真实 Agent 事件，却更像执行日志，不能清楚表达 Teacher Agent 如何理解任务、判断知识需求、选择工具、制定教学策略和验证输出。

本次改造将现有公开摘要升级为 `Reasoning Trace`。Reasoning Trace 是对真实运行事实的安全、结构化解释，不是模型隐藏 Chain of Thought，也不读取或展示 `reasoning_content`、系统提示词、原始模型草稿或私有推理 token。

## 2. 目标

- 将左侧“备课批注”升级为“🧠 AI教学决策过程”。
- 阶段按照本轮实际执行路径动态出现，不固定为四个阶段。
- 让用户理解本轮发生了哪些可复核判断，以及这些判断为什么支持最终回答。
- 复用现有事件队列，实现阶段逐步出现和运行中更新。
- 公开摘要按实际执行顺序连续出现，始终可见，不使用阶段卡片或折叠交互，并有轻量进入、进行中和完成动画。
- Reasoning Trace 随 `PrototypeTurnRecord` 持久化，历史回合可完整重放。
- 不新增 LLM 调用，不改变 Teacher generation、RAG、Memory、verifier、TeachingContract、audit 或 repair 的权威边界。
- 保持旧会话兼容，不进行数据库迁移。

## 3. 非目标

- 不展示真实 Chain of Thought。
- 不让 Teacher 模型在回答正文中输出“我的思考是……”。
- 不新增 reasoning-summary LLM 调用。
- 不实现未经审计的模型 token 直出。
- 不把 UI 展示阶段变成新的固定 Agent 执行流水线。
- 不修改 8501/8000 legacy baseline。
- 不改变现有路由、调用预算、RAG 权限、Memory 语义或数学 checker 结论。
- 不移除详情区已有的 Agent 执行链、工具结果和性能诊断。

## 4. 当前链路

当前公开思考链路为：

```text
orchestrator / verifier / audit 产生运行事实
  → logic.py 调用 PublicThinkingRecorder
  → PublicThinkingUpdate(event, sections)
  → run_with_event_stream() 跨线程传递快照
  → LiveThinkingPanel
  → Streamlit 左侧实时渲染
  → PrototypeTurnRecord.public_thinking_events + sections
  → SQLite Conversation Archive
  → 历史回合重新渲染
```

当前固定结构定义在 `public_thinking.py::PublicThinkingSection`，固定标题为“题意 / 写法 / 资料 / 成稿”；HTML 与标题“备课批注”定义在 `desk_theme.py::render_thinking_sections()`。

现有基础设施已经满足以下能力，因此本次不重写：

- 后端同步教学工作与 UI 线程之间的安全事件队列；
- 每个事件到达后替换最新结构化快照；
- 每回合独立的展开状态；
- 回合完成后保存公开摘要；
- 旧会话的 Pydantic 默认值升级；
- 最终回答只有审计完成后才进入学生可见区。

## 5. 设计原则

### 5.1 动态阶段是实际事实的投影

Reasoning Trace 不预先创建固定数量的空阶段。阶段只在对应事实真实出现时创建，顺序按照首次出现顺序保持稳定；同一语义阶段的后续事件只更新该阶段，不重复追加同名卡片。

例如：

```text
普通概念题
  问题分析 → 知识判断 → 教学策略 → 回答生成 → 质量验证

明确联网请求
  问题分析 → 知识判断 → 工具执行 → 教学策略 → 回答生成 → 质量验证

学生步骤续答
  问题分析 → 学习上下文 → 步骤验证 → 教学策略 → 回答生成 → 质量验证

触发修正
  ... → 质量验证 → 回答修正 → 最终复核
```

这里的序列只是可能结果，不是固定执行模板。没有 Memory、RAG、工具、verifier、semantic audit 或 repair 时，不创建对应阶段。

### 5.2 只使用可公开、可复核字段

每一条公开内容必须来自以下安全来源之一：

- 已验证的 `RequestClassification` 枚举和布尔字段；
- `routing_guidance.router` 等受限路由标签；
- `RetrievalDecision.mode` 与经过限制的公开原因；
- Tool 名、成功状态、命中数和耗时；
- 已加载的 canonical Skill 名及本地白名单展示名；
- `TeachingContract` 的 response mode 与权限布尔值；
- deterministic verifier 状态；
- semantic audit 是否执行、是否通过；
- repair 是否执行、最终 warning 数；
- 后端阶段耗时。

禁止读取或展示：

- `reasoning_content`；
- system prompt / user payload 原文；
- ActionProvider 的自由推理文本；
- 被拒绝的初稿或 repair 前草稿；
- RAG 文档中的任意指令性内容；
- Memory 中未经白名单处理的自由文本。

### 5.3 展示层不扩大运行时权威

- Math checker 在适用范围内继续是数学正确性的权威。
- Semantic boundary audit 只表示教学契约和披露边界检查，不表示全学科事实正确。
- 未运行 verifier 时不得展示“概念准确”或“公式正确”。
- 未调用 RAG 时明确展示“无需外部资料”或不显示工具阶段，不伪装成执行过检索。
- 低风险路径跳过 semantic audit 时，只能展示“已执行确定性边界检查”。

## 6. 数据模型

### 6.1 动态阶段模型

在 `public_thinking.py` 增加：

```python
class ReasoningTraceStageId(str, Enum):
    REQUEST_ANALYSIS = "request_analysis"
    LEARNING_CONTEXT = "learning_context"
    KNOWLEDGE_DECISION = "knowledge_decision"
    TOOL_EXECUTION = "tool_execution"
    STEP_VERIFICATION = "step_verification"
    TEACHING_STRATEGY = "teaching_strategy"
    ANSWER_GENERATION = "answer_generation"
    QUALITY_VALIDATION = "quality_validation"
    ANSWER_REPAIR = "answer_repair"
    FINAL_RECHECK = "final_recheck"


class ReasoningTraceSource(str, Enum):
    REQUEST_CLASSIFICATION = "RequestClassification"
    MEMORY_RESOLUTION = "StudentMemoryResolution"
    RETRIEVAL_DECISION = "RetrievalDecision"
    RAG_CONTEXT = "RagContext"
    AGENT_TRACE = "AgentTraceEvent"
    TEACHING_CONTRACT = "TeachingContract"
    VERIFICATION_RESULT = "VerificationResult"
    BOUNDARY_AUDIT = "BoundaryAudit"
    RUNTIME_STAGE = "RuntimeStage"


class ReasoningTraceEvidence(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    value: str = Field(min_length=1, max_length=160)
    source: ReasoningTraceSource
    source_field: str = Field(min_length=1, max_length=80)


class ReasoningTraceStage(BaseModel):
    sequence: int = Field(ge=1)
    stage_id: ReasoningTraceStageId
    stage: str = Field(min_length=1, max_length=24)
    icon: str = Field(min_length=1, max_length=8)
    summary: str = Field(min_length=1, max_length=240)
    content: list[ReasoningTraceEvidence] = Field(default_factory=list, max_length=6)
    status: PublicThinkingStatus
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
```

`stage_id` 是稳定接口；`stage` 与 `icon` 来自本地白名单注册表，不接受模型自定义值。`sequence` 记录首次出现顺序，后续更新不得改变。每一条 `content` 都必须标记结构化来源类型和准确字段路径，例如 `RequestClassification.subject`、`RetrievalDecision.mode`、`RagContext.hits`、`TeachingContract.response_mode`、`VerificationResult.status` 或 `BoundaryAudit.passed`。`source_field` 必须以对应 `source` 名称开头，由数据模型执行一致性校验。UI 展示的是这些真实决策字段的安全映射，不是普通文本总结。

### 6.2 更新快照

`PublicThinkingUpdate` 新增：

```python
reasoning_trace: list[ReasoningTraceStage] = Field(default_factory=list)
```

现有 `event` 保留，用于 legacy fallback 和诊断。现有 `sections` 在兼容期保留，但新 UI 不再以它作为主数据源。

### 6.3 回合持久化

`PrototypeTurnRecord` 新增：

```python
reasoning_trace: list[ReasoningTraceStage] = Field(default_factory=list)
```

新回合同时保存 `reasoning_trace` 和现有 `public_thinking_events`。`sections` 暂不删除，以避免破坏旧消费者；新代码不依赖它继续增长。

旧记录缺少 `reasoning_trace` 时：

1. Pydantic 使用空列表默认值加载；
2. UI 若发现 `reasoning_trace` 为空但 `sections` 非空，则在显示层进行 legacy 映射；
3. legacy 映射只用于展示，不回写数据库、不伪造新运行事实。

## 7. 动态阶段注册表

注册表由本地代码维护，定义稳定标题和图标：

| Stage ID | 标题 | 图标 | 创建条件 |
|---|---|---:|---|
| `request_analysis` | 问题分析 | 🔍 | 收到分类开始事件；正常教学回合均可出现 |
| `learning_context` | 学习上下文 | 🧠 | 本轮实际执行 Memory resolution、恢复主题或要求澄清 |
| `knowledge_decision` | 知识判断 | 📚 | 产生 retrieval decision，或确认无需外部资料 |
| `tool_execution` | 工具执行 | 🛠️ | 实际开始或完成 Skill、Local RAG、Quality-QA、Web Tool |
| `step_verification` | 步骤验证 | 🧮 | deterministic verifier 实际适用或发现可验证学生步骤 |
| `teaching_strategy` | 教学策略 | 📝 | TeachingContract 和回答模式已确定 |
| `answer_generation` | 回答生成 | ✍️ | Teacher generation 开始或回答结构已确定 |
| `quality_validation` | 质量验证 | ✅ | hard boundary、claim audit 或 semantic audit 实际执行 |
| `answer_repair` | 回答修正 | 🩹 | length、boundary、verifier conflict 或 warning 触发 repair |
| `final_recheck` | 最终复核 | 🛡️ | repair 后执行 post-repair audit / hard recheck / fallback |

完成状态不额外创建固定“完成”阶段，由外层“AI 推理完成（用时 …）”表示。

## 8. Recorder 行为

`PublicThinkingRecorder` 从固定 `_SECTION_ORDER` / `_SLOT_ORDER` 改为按 `stage_id` upsert：

```text
第一次收到 stage_id
  → 分配 sequence
  → 添加 in_progress 阶段
  → 发送完整 reasoning_trace 快照

同一 stage_id 后续更新
  → 保持 sequence
  → 更新 summary / content / status / elapsed
  → 发送完整 reasoning_trace 快照
```

每次快照深拷贝，避免工作线程继续修改已进入 UI 队列的对象。

### 8.1 问题分析

从 classification 构建，例如：

```text
识别任务：数学概念解释
用户目标：理解当前知识点
适用水平：初中
问题关系：新问题
路由方式：确定性快速路由
```

用户目标必须从受限 answer mode / TeachingContract 映射，不能自由摘要学生原话。

### 8.2 学习上下文

仅在 Memory resolution 实际发生时出现，例如：

```text
学生记忆：找到唯一相关主题
上下文处理：继续已有学习主题
```

不得展示学生隐私字段、Memory 原始文本或未经允许的个人事实。

### 8.3 知识判断

从 `RetrievalDecision` 和实际 RAG 状态构建：

```text
知识来源：模型已有课程知识
RAG：未调用
判断依据：当前请求未明确要求外部证据
```

或：

```text
知识来源：本地教材 + 官网资料
RAG：本地后联网
资料结果：本地 3 条，联网 2 条
```

### 8.4 工具执行

只有工具实际开始或返回时出现。按工具签名聚合，避免重复显示同一幂等调用：

```text
教学 Skill：概念解释
本地资料检索：完成，命中 3 条
联网检索：完成，命中 2 条
```

不得展示 ActionProvider 自由文本 reason。

### 8.5 教学策略

从 `TeachingContract`、canonical Skill 和 pacing 构建：

```text
教学方式：概念解释
回答目标：解释概念，不套用当前活动题
答案边界：不提前披露最终答案
回答结构：核心概念 + 公式 + 简单例子
```

“回答结构”使用本地 response-mode 映射，不要求模型额外生成计划。

### 8.6 步骤验证与质量验证

步骤验证仅表达 checker 的实际结论。质量验证根据真实路径显示：

```text
边界检查：确定性检查已完成
语义审计：按低风险策略跳过
数学验证：当前题型不适用
公开警告：0
```

或：

```text
语义审计：已执行并通过
数学验证：学生步骤存在确定性错误
回答修正：已触发
```

若无法证明事实正确，不显示“概念准确”“公式正确”。

## 9. 前端设计

### 9.1 外层区域

- 运行中标题：`🧠 AI教学决策中…（已用时 N 秒）`
- 完成标题：`🧠 AI教学决策完成（用时 N 秒）`
- 失败标题：`🧠 AI教学决策中断`
- 运行中、完成和历史回放均直接展示公开摘要，不提供外层折叠状态。
- rerun 后从持久化的 `reasoning_trace` 恢复同一条连续轨迹。

### 9.2 连续公开摘要

每个真实阶段投影为一行安全公开摘要，按照事件发生顺序追加到连续阅读流：

```html
<div class="desk-reasoning-flow" role="list">
  <p class="desk-reasoning-line is-completed" role="listitem">
    已确认这是初中数学概念解释任务。
  </p>
</div>
```

- 主 Reasoning Trace 不显示 `source`、`source_field`、内部枚举或技术标签；这些证据仍保留在持久化记录和下方诊断详情中。
- 阶段标题只作为无障碍标签，视觉区域只呈现经过整理的公开自然语言摘要。
- 运行中的最新摘要显示轻量光标或状态点；完成后保持所有摘要直接可见。
- 所有可见摘要和无障碍标签先经 `html.escape()`，禁止插入任意 HTML。

### 9.3 动画

只使用 CSS 动画，不增加前端依赖：

- 新阶段：`opacity + translateY` 淡入，约 220–300ms；
- `in_progress`：标题旁轻量脉冲圆点；
- `completed`：绿色勾选淡入；
- `warning`：朱砂色提示，不闪烁；
- 遵守 `prefers-reduced-motion: reduce`，关闭非必要动画。

动画只表示 UI 状态，不伪造实际模型 token streaming。

### 9.4 实时更新

继续使用：

```text
thinking_update_callback
  → Queue
  → run_with_event_stream
  → LiveThinkingPanel.append
  → reasoning_trace snapshot renderer
```

每次收到快照时只替换左侧 trace 容器，不重建回答区域，也不触发页面 rerun。阶段会随真实后端事件逐步出现。

### 9.5 历史回放

- 新回合从 `turn.reasoning_trace` 渲染为连续公开摘要。
- 旧回合从 `turn.sections` 做只读 legacy 映射，同样直接展示而不折叠。
- 历史回放不重新播放动画计时，也不重新调用后端。
- 会话切换继续由 `conversation_id + turn_id` 隔离状态。

## 10. 数据流

```text
RequestClassification / MemoryResolution / AgentTrace / TeachingContract
RetrievalDecision / VerificationResult / BoundaryAudit / Repair status
  → PublicThinkingRecorder（仅白名单字段）
  → 动态 upsert ReasoningTraceStage
  → PublicThinkingUpdate(reasoning_trace snapshot)
  → UI event queue
  → 左侧阶段逐步出现
  → turn.reasoning_trace
  → Conversation Archive
  → 历史回放
```

最终回答仍保持：

```text
Teacher draft
  → verifier / audit / conditional repair
  → safe final response
  → answer display stream
```

Reasoning Trace 不改变最终回答的安全展示时机。

## 11. 错误处理

- 公开 trace callback 抛异常：继续 fail-open，不影响教学回答。
- 未知 stage / tool / enum：使用本地中性文案或忽略，不显示原始值。
- Tool 失败：阶段显示“未完成，使用已有证据继续”，不暴露 exception details。
- Memory 失败：仅显示“学习上下文暂不可用”，不阻塞回答。
- semantic audit 失败：按现有安全修正与 fallback 处理，trace 只显示公开状态。
- 历史 `reasoning_trace` 缺失或校验失败：回退 legacy `sections`；两者均无则回退 `public_thinking_events`。
- UI renderer 失败不得修改 `PrototypeTurnRecord` 或回答正文。

## 12. Shared contract 影响

### Owner

- `ReasoningTraceStage`、`PublicThinkingUpdate`、`PrototypeTurnRecord`：Agent Runtime / 3322。
- Streamlit UI 与公开展示：Product / 3322。

### Producer

- `PublicThinkingRecorder`。

### Consumers

- `logic.py` 回合构造；
- `live_progress.py` / `streamlit_app.py` 实时展示；
- Conversation Archive 的 episode JSON；
- 历史会话加载；
- smoke、UI、streaming、turn-contract 测试。

### Current behavior

- 保存 `public_thinking_events` 和固定四槽 `sections`；
- UI 显示“备课批注”。

### Proposed behavior

- 新增动态 `reasoning_trace`；
- UI 优先显示真实发生阶段；
- `sections` 保留为兼容字段。

### Schema compatibility

- 只新增带默认值字段；
- 不删除、不改名、不改变既有字段类型；
- 旧 JSON 可直接加载。

### Persistence compatibility

- SQLite 不新增列；字段继续存在 episode JSON 内；
- 不需要迁移或批量回写旧记录。

### Evaluation impact

- 不改变 teacher answer、checker、golden label 或评测指标；
- 新增 public trace 安全性、阶段动态性、流式快照和历史兼容测试；
- 完整 `pytest` 与 `run_eval.py` 继续作为回归门禁。

### Migration/default behavior

- `reasoning_trace=[]`；
- 新 UI 读取优先级：`reasoning_trace → legacy sections → public events`。

### Rollback behavior

- 回滚 UI 后旧代码仍可读取保留的 `sections` 和 `public_thinking_events`；
- 新字段被旧消费者忽略；
- 不需要回滚数据库。

### Mandatory co-review

- Product：UI-visible turn schema / public trace；
- Evaluation：observable behavior 和安全性测试；
- Memory：仅当展示内容读取或改变 Memory contract 时需要；本设计不改变 Memory contract；
- Retrieval：仅当改变 RetrievalDecision/RagContext 语义时需要；本设计只读取现有字段，不改变其语义。

## 13. 最小修改范围

生产代码：

- `prototypes/llm_led_teacher/public_thinking.py`
  - 新增动态 trace schema 和 recorder upsert；
  - 保留 legacy events/sections。
- `prototypes/llm_led_teacher/logic.py`
  - 把结构化运行事实传给 recorder；
  - 持久化 `reasoning_trace`。
- `prototypes/llm_led_teacher/live_progress.py`
  - 更新运行中/完成标题，不改变事件桥接结构。
- `prototypes/llm_led_teacher/streamlit_app.py`
  - 优先渲染 `reasoning_trace`；
  - 历史记录兼容 fallback。
- `prototypes/llm_led_teacher/desk_theme.py`
  - Reasoning Trace `<details>` UI 和动画。

测试：

- `tests/prototypes/test_public_thinking.py`
- `tests/prototypes/test_llm_led_streaming.py`
- `tests/prototypes/test_llm_led_turn_contract.py`
- `tests/prototypes/test_llm_led_auth_ui.py`

文档状态：

- 只更新 `PROGRESS.md` 的 Agent Runtime section；
- 不重复改写 `DECISIONS.md` 中已有的按需 Agent Loop 和私有推理边界。

## 14. TDD 验收标准

### 数据模型与安全

1. `ReasoningTraceStage` 能序列化并随旧 episode 安全加载。
2. 每条 evidence 都包含可枚举的 `source` 与准确 `source_field`，并关联真实 Agent 决策字段。
3. 未知 stage、原始 `reasoning_content`、system prompt、Agent 自由 reason 不进入 trace。
4. 新字段缺失时旧 turn 能通过 Pydantic 校验。
5. legacy `sections` 能映射为历史兼容显示，且不回写存储。

### 动态阶段

5. 普通概念题不出现 Memory、Tool、Step Verification 或 Repair 阶段。
6. Local RAG 请求出现知识判断和工具执行阶段。
7. Web 请求在实际 Web Tool 成功后更新工具执行阶段。
8. Memory ambiguity 只在实际 resolution 时出现学习上下文阶段。
9. verifier 适用时出现步骤验证；不适用时不伪造验证通过。
10. repair 只在真实 repair 时出现，post-repair 检查产生最终复核阶段。
11. 同一 stage 多次事件只更新，不产生重复卡片。
12. 阶段顺序按首次出现稳定，不由固定四槽排序覆盖。

### 流式与历史

13. 后台任务结束前，UI 能收到至少一个 reasoning trace 快照。
14. 每次快照是独立深拷贝，不被后续 recorder 更新污染。
15. 运行中摘要逐步出现，完成后全部保持可见。
16. 历史回合刷新后显示与持久化 trace 一致。
17. conversation / turn 的历史轨迹互不串联。

### UI

18. 页面不再显示“备课批注 · 题意 / 写法 / 资料 / 成稿”。
19. 标题显示“🧠 AI教学决策过程”或“🧠 Agent Reasoning Trace”，不得使用“思维链”或 Chain of Thought。
20. 主 Reasoning Trace 不生成 `<details>` / `<summary>`，不显示技术来源字段，也不提供折叠交互。
21. CSS 包含进入、进行中、完成动画和 reduced-motion fallback。
22. 所有 stage/summary/content 均 HTML escape。

### 回归

23. 不增加 LLM 调用次数。
24. 不改变最终 answer、安全 audit、repair、checker、RAG 和 Memory 行为。
25. 运行并通过：

```bash
python3 -m pytest tests/prototypes/test_public_thinking.py -q
python3 -m pytest tests/prototypes/test_llm_led_streaming.py -q
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
python3 -m pytest tests/prototypes/test_llm_led_auth_ui.py -q
python3 -m pytest -q
python3 scripts/run_eval.py
git diff --check
```

## 15. 明确取舍

本设计选择“实际运行事实的确定性动态投影 + 连续公开自然语言摘要”，而不是额外调用 LLM 生成解释。这样可以：

- 保持 fast-route 优化后的调用预算；
- 防止推理摘要与真实执行不一致；
- 避免暴露模型隐藏推理；
- 复用现有流式、历史与安全边界；
- 以最小修改把固定四槽日志升级为论文/项目展示可用的 Teacher Agent Reasoning Trace。
