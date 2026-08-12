# 8502 多学科 Agent Loop 改造设计

日期：2026-08-12

## 目标

把当前 8502 LLM-led Teacher 的固定单轮流水线改成有步数上限的 Agent
Loop。模型根据当前消息和对话状态识别 K9 学科，按需加载教学 Skill、调用本地
知识库或联网工具，最后提交受确定性安全层约束的回答。

改造后继续使用 8502 作为唯一真人体验入口，并保留现有登录、历史会话、待重试
消息、逐步进度、数学验证和回答审计能力。

## 已确认的产品行为

- 支持数学、语文、英语、物理、化学、生物、历史、地理和道法自动切换。
- 对当前题的追问默认继承 active problem、学段和学科。
- 用户提出新的自包含问题时重新识别学段和学科。
- K9 学科问题优先检索本地教材和题库；本地证据不足时才允许联网。
- 非 K9 普通问题不调用本地知识库或联网搜索，只做简短回答并引导回学习内容。
- 数学 checker 等确定性规则不能由 Agent 跳过。

## 当前问题

8502 当前由 prototypes/llm_led_teacher/logic.py 的 run_turn() 固定调用 RAG。
rag_hook.retrieve_context() 又把所有请求硬编码为“初中数学”，并让底层 RAG
根据“达到 0.65 的命中数是否至少为 topk”自动触发联网。

这造成两个职责倒置：

1. 学科识别和联网决策被藏在检索器里，而不是由编排层决定；
2. RRF 融合分沿用了 raw cosine 的阈值，且要求五条结果同时过线，导致相关数学
   教材已经命中时仍频繁联网。

## 方案选择

采用渐进式接管，而不是整体替换页面：

- 保留 8502 的 Streamlit、认证、会话存档和展示层；
- 复用独立 tool_loop_teacher 原型中已经验证的 Tool Registry、结构化动作协议和
  bounded loop 思路；
- 新建面向 8502 的多学科编排适配层；
- 逐步把固定的 observer、RAG、生成和审计调用收进 Agent Loop；
- 原有确定性安全能力作为 loop 外围护栏继续生效。

不直接用 8503 原型覆盖 8502，因为它没有现有页面的认证、历史记录、RAG
可观测性和完整教学回合契约。

## 总体架构

    用户消息 + 对话状态
            │
            ├─ 强制前置规则
            │    ├─ 恢复 active problem / pending message
            │    └─ 数学题型支持时运行 math checker
            │
            └─ Bounded Agent Loop
                 ├─ classify_request
                 ├─ load_skill
                 ├─ inspect_student_context
                 ├─ search_local_knowledge
                 ├─ search_web
                 └─ finish
                        │
                        ├─ Teaching Plan Validator
                        ├─ Math / disclosure / pacing audit
                        ├─ 必要时修正
                        └─ 保存并展示最终回答

Agent Loop 负责“是否调用、调用什么、下一步做什么”；Tool 负责执行一个边界清楚的
动作；Safety Gate 负责拒绝不允许的动作或回答。

## 请求分类与对话连续性

classify_request 返回结构化结果：

- scope：k9 或 off_topic；
- subject：九个受支持学科之一，off-topic 时为空；
- stage：小学、初中或未知；
- is_new_problem：是否替换当前 active problem；
- answer_mode：guided、complete 或 brief_redirect；
- reason：简短、可审计的路由理由。

首次消息必须分类。追问优先继承 episode 中的 active_subject 和
active_stage，只有分类结果明确表示新问题时才覆盖。会话存档随 episode 保存这
两个字段，重新登录或切换历史对话后仍能恢复学科上下文。

分类结果只能影响工具选择，不能覆盖用户原始问题。模型返回未知学科、非法枚举或
互相矛盾的字段时，动作校验失败并允许 loop 修正一次；再次失败则以 off-topic
简短引导安全收口。

## Tools

### classify_request

输入当前消息、active problem 和最近对话摘要，输出上述分类结构。它不检索、不
生成最终回答。

### search_local_knowledge

参数包括 query、subject、stage、top_k，以及 textbook/exercise 来源选择。

返回：

- 命中文本和引用；
- collection、学科、学段等元数据；
- raw vector similarity、BM25 分、RRF rank 和融合排序分；
- 命中数量、过滤条件和耗时；
- 工具级错误，不包含任何联网结果。

该工具不再使用 online_mode=auto。它不得自行调用 Tavily，也不得用“少于
top_k 条过阈值”替 Agent 做联网决定。RRF 只负责排序，raw vector、关键词证据和
实际片段同时返回给 Agent 判断证据是否覆盖问题。

### search_web

只有以下条件全部满足时才允许执行：

1. 分类结果属于 K9 学科；
2. 本轮已经成功调用一次 search_local_knowledge；
3. Agent 给出具体证据缺口，例如本地无结果、内容与问题不匹配或问题要求时效信息；
4. 本轮尚未调用过联网工具。

非 K9 请求、重复联网或没有先查本地时，Tool Registry 返回可恢复的拒绝结果，
由 Agent 选择重新规划或 finish。

### load_skill

按学科和教学意图加载本地 Skill。第一阶段复用已有苏格拉底引导、错误诊断和概念
解释 Skill；学科专属 Skill 可以以后独立增加，不阻塞本次改造。

### inspect_student_context

按 active problem 读取学生证据。它是可选工具，不能把估计掌握度变成绕过安全规则
的理由。

### verify_math_attempt

数学题且现有 verifier 支持时在 loop 前强制运行。Agent 可读取结果，但不能决定
是否运行，也不能覆盖 verifier 的确定性结论。

## Agent Loop 协议

每一步只能返回一个结构化动作：call_tool(name, arguments, reason) 或
finish(plan, response)。

默认最大六步，其中：

- 分类最多一次，校验失败时允许修正一次；
- 本地知识检索最多两次，第二次必须改变 query 或过滤条件并说明原因；
- 联网最多一次；
- Skill 和学生上下文按需调用；
- 达到步数上限时使用已有证据生成受限回答，不继续隐式调用工具。

工具异常作为 ToolResult 返回 loop，不让 Streamlit 进程退出。Agent 可以选择
降级回答；数学审计或最终安全检查失败时仍走现有修正流程。

## 学科路由

学科名称在 Tool 边界统一为中文枚举，由本地检索适配器转换成各 collection 的实际
metadata，例如教材的“数学”和题库的“Math”。映射只做格式转换，不做学科判断。

当某学科没有对应 collection 或命中为空时，本地工具明确返回空结果和实际过滤条件。
Agent 随后可以在 K9 范围内调用联网工具。不能把空结果伪装为
retrieval_error，也不能把联网引用标成 local。

## 非 K9 请求

scope=off_topic 时，Safety Gate 禁止调用本地知识库和联网工具，强制
answer_mode=brief_redirect。最终回答允许用模型已有知识给出一小段概括，但不做
详尽研究、引用或实时事实承诺，结尾邀请用户提出 K9 学习问题。

这保证“网易邮箱”类问题不会再拿初中数学教材做伪检索，也不会为了普通闲聊消耗
联网搜索。

## 8502 页面与可观测性

保留现有进度框，但内容改为真实 loop trace，例如：

- “识别为初中物理新题”；
- “加载概念解释 Skill”；
- “检索本地物理教材，命中 4 条”；
- “本地证据未覆盖时效部分，补充联网检索”；
- “数学 verifier 已执行”；
- “最终回答通过审计”。

页面详细信息展示工具名、参数摘要、耗时、命中来源和联网理由。不得展示完整系统
提示、密钥、内部思维链或未经审计的草稿。

历史记录新增分类结果和工具轨迹时必须兼容旧 episode；旧记录缺少新字段时使用
空值迁移，不能导致登录后加载失败。

## 错误处理

- 分类失败：允许一次格式修正，再降级为 off-topic 简短引导；
- 本地 RAG 不可用：K9 请求可调用一次联网，页面明确标记本地工具错误；
- 联网不可用：使用本地证据或明确表示证据不足，不阻断主回答；
- 单个 Tool 超时：返回失败结果，不终止 8502 WebSocket；
- Agent 超步数：使用现有证据安全收口；
- 最终生成失败：保留 pending message，继续沿用现有重新生成机制；
- 数学验证或回答审计失败：不得绕过，沿用修正或失败提示。

## 数据与兼容性

现有用户库和会话库保持原路径。Episode/turn schema 新字段均提供默认值：

- active subject 和 stage；
- request classification；
- agent actions；
- tool results 摘要；
- local/web retrieval diagnostics。

旧会话加载测试必须覆盖缺失字段；新记录仍由原 conversation archive 负责用户隔离。

## 测试策略

先写失败测试，再实施。离线测试不得访问 Tavily、真实模型或付费 API。

必须覆盖：

1. 九个 K9 学科能产生合法分类和正确的 collection metadata 映射；
2. 数学追问继承 active subject，明确的新物理题切换学科；
3. off-topic 请求不能调用 local/web Tool，并生成简短引导；
4. K9 请求必须先查本地，才能调用一次联网；
5. 本地相关片段已覆盖问题时 Agent 可以直接 finish，不要求五条都超过 0.65；
6. RAG Tool 返回 raw vector、BM25、RRF 和引用，且绝不隐式联网；
7. 数学 verifier 在支持题型上强制运行；
8. Tool 异常、超时和非法参数不会结束 Streamlit 进程；
9. Loop 达到预算后安全结束，不会无限调用；
10. 旧会话缺少多学科字段时仍可加载；
11. 页面 trace 与实际 ToolResult 一致；
12. 现有登录、历史、pending message、数学审计和完整测试不回归。

集成验证至少包含：

- 本地数学教材可回答“移项为什么要变号”且不误触发联网；
- 本地物理或化学题能使用对应学科过滤；
- 一个本地确实缺少证据的 K9 问题能够先本地、后联网；
- “网易企业邮箱是什么”不检索数学库、不联网，并简短引导回学习内容；
- 8502 连续完成多学科切换且 WebSocket 保持健康。

## 验收标准

- 8502 是唯一需要用户访问的入口；
- 学科识别和联网决策出现在 Agent trace 中，不再藏在 RAG 内部；
- 本地库相关命中不会因固定“5 条均过 0.65”规则被一律判不足；
- K9 学科切换、追问继承和 off-topic 行为符合已确认规则；
- 数学确定性安全层无法被 Tool 选择绕过；
- python3 -m pytest 全部通过；
- python3 scripts/run_eval.py 全部通过；
- 使用本地 DeepSeek 和 Chroma 完成上述五类真实回归；
- 8080、8002、8502 在真实回归后继续健康。

## 非目标

- 不增加子 Agent、并行委派、Skill 自动学习或长期自主任务；
- 不把非 K9 普通知识问答扩展成通用搜索助手；
- 不更换 Chroma、bge-m3、llama.cpp 或本地模型权重；
- 不删除现有 verifier、审计或 conversation archive；
- 不在本次改造中清理 Git LFS 历史对象或重建整套向量库。
