# 知识点无关初中数学 Teaching Harness：阶段一审计与最小重构设计

日期：2026-07-13
范围：仅审计与设计；本文件不实施核心业务重构。

## 1. 结论摘要

当前系统的教学安全目标有价值，也已经形成了可离线回归的基础：原题锚定、单问句、答案泄露控制、学生进度、掌握度、日志和评测都可成为知识点无关 Harness 的组成部分。

但是，正常 Agent 路径目前不是知识点无关的。`app/agent/workflow.py` 直接以一元一次方程作为问题表示、进度表示、概念识别、提示生成、练习生成、验算和错误归因的共同基础；`app/agent/math_checker.py` 既是验证器，也是下一步规划器和教学话术生成器。结果是教学状态机、数学求解轨迹和一元一次方程知识图数据彼此耦合。

建议采用增量替换而非重写：保留 `app/api`、`app/llm`、`app/memory`、`app/evaluation` 的目录边界与现有 `run_turn` 外部入口；在 `app/agent` 内逐步建立知识点无关的状态、动作、数学分析、策略、回复生成和可选验证/检索接口。现有一元一次方程 checker 先作为一个兼容实现被隔离，不再作为每轮教学的流程前提。

本审计发现的首要风险是：`agent_no_graph` 并不是真正的无检索 Agent。它只通过 `USE_PEDAGOGICAL_RAG=false` 跳过新版 Pedagogical Graph traversal，但 `workflow.py` 的概念检测、前置知识、练习和误区读取仍无条件依赖旧的本地 JSON 图。阶段四必须把这两层依赖同时改为可选降级。

## 2. 当前架构摘要

当前单轮真实执行顺序是七个节点，而不是文档中曾描述的六个节点：

```text
context_resolver
  -> diagnose
  -> retrieve
  -> teaching_policy
  -> teach
  -> memory_update
  -> log
```

主要职责如下：

| 层/文件 | 当前职责 | 审计判断 |
| --- | --- | --- |
| `app/api/main.py` | `/turn` 请求、响应投影、学生/对话接口 | API 入口大体可保留；响应投影暴露了许多方程和图谱专用字段。 |
| `app/agent/state.py` | `TeacherState`、诊断、动作、检索、日志 schema | 同时承载通用教学状态和方程专用轨迹，需拆出中性表示。 |
| `app/agent/workflow.py` | 诊断、检索、策略、模板、提示、练习、持久化编排 | 最大耦合点；应作为有限边界内的增量拆分目标。 |
| `app/agent/math_checker.py` | 一元一次方程解析、同解判断、步骤模拟、建议下一步、误因分类 | 领域实现，不是通用数学验证接口。 |
| `app/retrieval/graph_retriever.py` | 旧概念 JSON 与新版 Pedagogical Graph traversal | 两个概念图 reader 共存，前者仍处于正常流程的硬路径。 |
| `app/retrieval/pedagogical_retriever.py` | 另一份 Pedagogical Graph 兼容 reader | 主要是兼容/测试接口；正常 workflow 未把它作为唯一检索边界。 |
| `app/memory/*` | 掌握度、episode、事件、长期 profile | 数据结构有可泛化部分，但现有 concept/skill ID 和部分字段仍指向方程。 |
| `app/evaluation/*` | golden 回放、三模式比较、Hybrid 学生模拟 | 保障教学不变式的价值高；目前 case、判据和模拟器明显绑定方程。 |

`DeepSeekClient` 是唯一 LLM 出口，符合项目约束。当前 LLM 的两类调用是：

1. `_llm_diagnosis()`：一次 JSON 调用同时判断意图、六个固定概念之一、学生卡点和置信度。
2. `_deepseek_response()`：一次自由文本调用同时接收原题、当前方程、预期方程变形、checker 结果、动作、掌握度与检索上下文，生成教师话语。

这两个调用之间没有独立、知识点无关的 `MathAnalysis` 契约。

## 3. Git 与基线

### 3.1 审计开始时的 Git 状态

- 分支：`refactor/topic-agnostic-teaching-harness`
- HEAD：`70177be feat: add pedagogical graph traversal and evaluation modes`
- 工作区在审计开始前已有未提交内容：`t` 被修改；`app/evaluation/hybrid_runner.py`、`app/evaluation/hybrid_scorer.py`、`app/evaluation/student_simulator.py`、`data/eval/student_simulator_scenarios.jsonl`、`docs/superpowers/`、多个 Hybrid 脚本和测试文件未跟踪。
- 这些内容没有被覆盖、编辑、暂存或纳入本阶段提交。本次测试结果是在该工作区状态下取得的。
- 当前 checkout 为普通 Git checkout（`.git` 与 git common dir 相同），但已位于用户指定的独立分支。

### 3.2 已识别命令

`pyproject.toml` 只配置 pytest；仓库未发现独立的 Ruff、Flake8、Black、mypy、pyright、tox、Makefile 或 npm lint/type-check 入口。

| 命令 | 类型 | 是否执行 | 结果 |
| --- | --- | --- | --- |
| `python -m pytest` | 全离线测试 | 是 | 179 passed，1 个 Starlette `TestClient` 弃用警告，0 failed。首次在受限沙箱内因系统临时目录权限被拒绝；允许写入临时目录后重跑通过。 |
| `python scripts/run_eval.py` | 严格 mock 回归评测 | 是 | mock LLM，26/26 passed，0 failed。 |
| `python scripts/run_eval_modes.py --mode deepseek_only` | 三模式比较基线 | 是 | 命令正常结束；26 case 中 6 passed、20 failed。失败是该 baseline 没有 workflow、checker、memory 和 retrieval 所导致的预期能力差异，不是脚本启动失败。 |
| `python scripts/run_eval_modes.py --mode agent_no_graph` | 三模式比较基线 | 是 | 命令正常结束；26/26 passed。workflow、checker、memory 开启；新版 Pedagogical Graph flag 关闭。 |
| `python scripts/run_eval_modes.py --mode agent_with_graph` | 三模式比较基线 | 是 | 命令正常结束；26/26 passed。workflow、checker、memory、新版 Pedagogical Graph traversal 开启。 |
| `python scripts/test_deepseek.py` | 真实 DeepSeek 连通性 | 否 | 需要真实凭证/API，不属于本阶段离线基线。 |
| `python scripts/run_hybrid_eval.py` | Hybrid 开发评测 | 未单独执行 | 相关测试已被 `pytest` 覆盖；本阶段用严格回归评测和三模式启动检查作为基线，未额外运行 54 trace 的开发实验。 |

三种模式都能正常启动。其真实差异见第 6.6 节：`agent_no_graph` 与 `agent_with_graph` 的严格回归结果相同，不能证明图谱对通用教学已经可选。

## 4. 一元一次方程绑定清单

本节按位置列出可操作的绑定，而不是只搜索 `linear equation` 字符串。

### 4.1 核心 workflow 与 prompt

| 位置 | 绑定证据 | 影响 |
| --- | --- | --- |
| `app/agent/workflow.py:52, 64-72, 108-142` | `DEFAULT_CONCEPT_ID = "c_linear"`、固定 `_KEYWORD_RULES`、`_classify_equation_structure()` 依据括号/分数系数/两边含 `x` 返回 `c_bracket`、`c_frac`、`c_move` 或 `c_linear`。 | 知识点识别和默认回退都是方程专用 if/else。 |
| `app/agent/workflow.py:155-191` | LLM 诊断 prompt 限制 `detected_concept` 只能从六个一元一次方程概念中选择。 | LLM 数学理解被预先收窄为题库，而不是返回一般分析。 |
| `app/agent/workflow.py:286-295` | `_init_problem()` 把问题设为方程，`lesson_goal = "solve_for_x"`，并立刻调用 `mc.suggest_next_step()`。 | 新问题只能进入“解 x 的线性方程”轨迹。 |
| `app/agent/workflow.py:297-529` | 首轮抽取、续轮判定和多步计划均以 `mc.extract_equation`、`check_step`、`check_multi_step_plan`、`check_final_answer` 为前提。 | 普通教学流程必须先是可解析方程，无法自然降级为 LLM 数学分析。 |
| `app/agent/workflow.py:815-868` | 练习难度预设为 `x + 4 = 9`、`3x + 4 = 13`、`5x - 2 = 3x + 6`。 | 即使不使用图谱，练习仍硬编码为方程。 |
| `app/agent/workflow.py:1007-1038` | teach context 包含“当前推进到”“预期下一步”“math checker 权威判定”“等式两边同时”等方程信息。 | LLM 回复生成接收的是方程轨迹，而非中性数学上下文。 |
| `app/agent/workflow.py:1051-1292` | 多步反馈、正确/错误步骤反馈、为什么移项变号、提示、mock 回复、泄答案检测都调用 `mc.next_question`、`suggest_next_step`、`solve_str` 或直接使用“代回原方程”“两边相等”。 | 回复生成、验证与一元一次方程教学语言混在同一模块。 |
| `app/agent/workflow.py:1351-1539` | mastery 推荐、技能推断和 learning event 使用“等式两边同时”“移项”“去括号”“除以系数”“solve_linear_equation”。 | 进度事件和长期技能 ID 也有领域绑定。 |
| `app/agent/workflow.py:1542-1713` | `log_node()` 同时写日志、用方程步骤更新 conversation、写 episode 的 `current_solution_state`、推断下一步和验算阶段。 | 记录职责与方程状态转换合并，增加替换风险。 |

### 4.2 数学 checker 与状态 schema

| 位置 | 绑定证据 | 影响 |
| --- | --- | --- |
| `app/agent/math_checker.py` 全文件 | 数据模型固定为 `(a*x+b)=(a*x+b)`；解析字符集限定 `x`；含括号展开、移项、系数化为 1、最终 `x=常数`、代回、移项不变号、中文“加减乘除/移”操作模拟。 | 是完整的一元一次方程领域实现，不能被称作通用数学 verifier。 |
| `app/agent/math_checker.py:333-371` | `suggest_next_step()` 和 `next_question()` 直接规划“移项、消常数、两边除系数、代回”。 | checker 同时充当流程规划器和教师提示器。 |
| `app/agent/math_checker.py:373-392` | `classify_misconception()` 只识别“移项时忘记变号”。 | 错误类型是方程专用。 |
| `app/agent/state.py:14-31` | 当前 `TeachingAction` 混合通用动作和题目生成细分动作，命名带 `LEVEL_1/2` 与多种 `GENERATE_*_EXERCISE`。 | action 有可保留的意图，但缺少中性最小词表。 |
| `app/agent/state.py:172-256` | `target_equation`、`current_step`、`expected_next_step`、`student_equation`、`student_step_validity`、`response_kind`、`solved_steps`。 | 这些字段直接编码方程的文本与解题轨迹。 |
| `app/memory/conversation_state.py` 与 `app/memory/schemas.py` | `target_equation`、`current_solution_state`、`expected_next_step`、`solved_steps` 在短期和 episode 状态重复出现。 | 状态重复加大迁移一致性风险；其中 `target_equation` 必须替换。 |
| `app/api/main.py` | `/turn` 暴露 `original_problem`、`student_step_validity`、检索细节和许多方程轨迹关联字段。 | API 需要兼容投影策略，不能在核心重构时直接删字段。 |

### 4.3 RAG、GraphRAG 与样本数据

| 位置 | 绑定证据 | 影响 |
| --- | --- | --- |
| `app/retrieval/graph_retriever.py:15-75` | 旧 JSON reader 读取 `concepts.json`、`exercises.json`、`misconceptions.json`；正常 workflow 用它进行概念检测和基础 retrieve。 | 旧本地图不是可选增强，而是 Agent 的硬依赖。 |
| `app/retrieval/graph_retriever.py:122-139` | alias 固定映射 `c_linear`、`c_eq_prop`、`c_move`、`c_bracket`、`c_frac` 到六个方程概念。 | 新旧数据模型均锁定一元一次方程。 |
| `app/retrieval/graph_retriever.py:243-386` | Pedagogical evidence 依赖当前 concept ID、误区 marker、提示层级、方程练习难度。 | 这是可关闭的新版 traversal，但数据仍是专题图。 |
| `app/retrieval/pedagogical_retriever.py` | 另一套 legacy alias 和 `PedagogicalConceptContext` reader。 | 双实现增加将来泛化时的维护面。 |
| `data/sample/concepts.json` | 6 个 legacy 概念：等式性质、合并同类项、移项、去括号、一元一次方程、分数系数方程。 | 旧检索数据是主题知识包。 |
| `data/sample/exercises.json`、`misconceptions.json` | 8 道方程练习和 8 条方程误区。 | 题目、答案和误因均为专题样本。 |
| `data/sample/pedagogical_graph.json` | graph ID 为 `pedagogical_graph_linear_equation_mvp`；6 concept、7 skill、12 misconception、12 diagnostic question、18 hint、6 worked example、12 practice problem、4 strategy、6 threshold、132 edge。 | GraphRAG 明确为线性方程 MVP 图。 |
| `docs/pedagogical_graph_schema.md` | 验证规则要求六个 MVP concepts 与 legacy IDs 永远可用。 | 文档把领域数据当成 schema 不变量，应在后续改为样例而非核心契约。 |

### 4.4 mock、评测、fixture 与文档

| 位置 | 绑定证据 | 影响 |
| --- | --- | --- |
| `app/evaluation/evaluator.py` | `mc.check_step`、`mc.is_equivalent`、`mc.find_equations` 是原题保持、无关方程、步骤认可、泄答案和最终摘要检查的基础。 | 教学不变式与方程 verifier 没有分离。 |
| `app/evaluation/mode_runner.py` | `deepseek_only` 离线 mock 是固定通用话术；Agent mode 的质量检查仍借助 equation checker。 | 模式对比能启动，但对无方程领域的质量度量尚不存在。 |
| `data/eval/golden_conversations.jsonl` | 15 个 golden case，均为一元一次方程、移项、去括号或分数系数的对话。 | 严格回归门是专题回归门。 |
| `data/eval/three_layer_memory_conversations.jsonl` | 5 个 memory/请求 case，仍围绕方程、验算和方程练习。 | memory 行为未有非方程 smoke coverage。 |
| `data/eval/single_turn_cases.jsonl` | 6 个单轮 case，expected concept/action 固定为方程概念和等级提示。 | action 回归带有 topic 识别依赖。 |
| `app/evaluation/student_simulator.py` 与 `data/eval/student_simulator_scenarios.jsonl` | `SUPPORTED_CONCEPTS` 固定为六概念，默认场景要求 6×3=18，所有数学意图必须由 checker 验证。 | Hybrid 开发评测也是专题模拟器；它是评测层绑定，不应反向定义 Harness 核心。 |
| `tests/test_math_checker.py`、`tests/test_flexible_steps.py`、`tests/test_grounding.py`、`tests/test_targeted_fixes.py`、`tests/test_workflow.py`、`tests/test_policy.py`、`tests/test_retriever.py`、`tests/test_pedagogical_rag.py`、`tests/test_graph_traversal.py` | fixture 和断言包含 `3x+6=10`、`c_linear`、移项、去括号、分数系数、固定提示语和图谱节点。 | 是最容易因无意更改而退化的回归面。 |
| `README.md`、`CODEX_HANDOFF.md`、`docs/pedagogical_rag_design.md`、`docs/pedagogical_graph_schema.md`、`docs/kele_comparison.md` | 均将当前 MVP、checker 或 graph schema 描述为一元一次方程。 | 文档可保留为历史/MVP 描述，但不能继续被解释为 Harness 的长期核心约束。 |

## 5. 四类逻辑分类

### A. 通用教学逻辑

以下逻辑应保留为 Harness 候选，去掉其中的方程措辞和硬编码条件即可：

- `ContextResolver` 的 pending question、练习/复习/总结/换主题请求、off-topic 处理和会话延续意图。
- `TeachingAction` 背后的教学意图：诊断、解释、提示、例题、练习、反馈、复习、收束。
- `_choose_action()` 的优先级原则：已验证的学生尝试优先于掌握度；掌握度只调节深浅；连续失败才升级支持；学生可跳步或直接回答。
- `postprocess_teacher_response()` 的每轮至多一个问句、在结尾给出下一个学生任务、非允许动作不引入例题、答案披露受控。
- `SocraticDecision` 的阶段/动作作为顾问信号，而非数学判定来源。
- conversation、episode、profile、learning event 的持久化边界，以及 TurnLog 的可审计性。

这些逻辑当前散落在 `workflow.py`、`context_resolver.py`、`socratic_policy.py`、`memory/*` 和 `evaluator.py` 中。泛化时应优先移动职责边界，而不是扩大规则表。

### B. 通用数学处理逻辑

目前真正可复用的通用数学处理较少，主要是可抽取的底层能力，而不是现成的通用模块：

- `Fraction` 精确数值表示、全角符号标准化和 Pydantic 校验是通用实现技巧。
- `DeepSeekClient.chat_json()` 提供了 JSON mode 与 schema 验证，可用于未来 `MathAnalysis`。
- “验证结果优先于生成模型叙述”“验证器可以返回无法判断”是通用的工具协调原则。
- 等价性、答案、步骤和误因的具体算法目前都假设一元一次方程；它们应归入 C，而不能误标为通用数学引擎。

### C. 一元一次方程领域逻辑

- `math_checker.py` 全部解析、求解、同解、下一步、问题、误因与中文操作模拟。
- workflow 中的默认概念、关键词表、方程结构分类、`solve_for_x` 目标、当前方程步骤、练习 presets、为什么移项变号、代回验算、系数化为 1。
- `target_equation`、`student_equation`、`expected_next_step`、`StepValidity` 及与方程形式相关的 response kind。
- legacy concept IDs、两套 alias、方程图节点/边、专题题目/答案/误区/提示和 mastery strategy 文案。
- profile 中的 `solve_linear_equation`、`move_terms`、`divide_by_coefficient`、`remove_parentheses` 等技能名。

### D. 测试和 mock 逻辑

- 20 个 strict golden/memory conversation case、6 个 single-turn case 与 18 个 Hybrid scenario 都集中在方程域。
- evaluator 和 mode runner 为保证教学质量，直接依赖 checker 的方程同解与答案判断。
- 断言并不要求整段自然语言逐字一致，但依赖固定的中文认可/纠错/等式/练习关键词、问号数量、特定方程文本、固定 concept ID、动作枚举和部分练习结构。
- `deepseek_only` 离线 mock 固定输出三类简短话术；它避免真实 API，但不产生可替代多领域数学理解的 baseline。

## 6. 当前架构风险分析

### 6.1 当前核心 Agent 是否依赖固定的一元一次方程流程？

是。`diagnose_node()` 的新题初始化要求可抽取 equation，随后立即把目标设为 `solve_for_x` 并计算 `expected_next_step`；续轮的步骤分派、反馈、策略、提示、练习和状态回写都调用同一个 checker。即使学生只请求一般解释，概念检测也回退到 `c_linear`。这不是“LLM 负责数学、Harness 负责教学”的架构，而是“方程求解器驱动教学流程，LLM 补充诊断和表述”。

### 6.2 哪些状态字段可以直接泛化？哪些必须替换？

可直接泛化或小幅改名：`student_id`、`conversation_id`、`message`、`original_problem`（改为问题上下文）、`lesson_goal`（改为学习目标）、`solved_steps`（改为进度证据）、`pending_teacher_question`、`current_phase`、`expected_student_response`、`action`、`response`、`mastery_update`、`known_misconceptions`、`learning_events_this_turn`、`trace`。

必须替换或移出核心：`target_equation`、`current_step` 的方程文本语义、`expected_next_step` 的可计算变形语义、`student_equation`、`student_step_validity`、`response_kind` 中的 `intermediate_equation/final_answer`、`lesson_goal="solve_for_x"`、`StepValidity` 作为主流程状态。它们可在一元一次方程 verifier 的兼容元数据中保留，不应继续定义 Harness 主状态。

`TeachingAction` 也需要收敛：现有枚举混合了通用意图、提示编号和四种“生成某类练习”的实现细节。外部兼容期可保留旧值；新 policy 应选择更小的中性动作集，再由 request metadata 决定难度、相似性或例题形式。

### 6.3 数学分析和教学策略是否混在同一个 LLM 调用中？

部分混合，且缺少独立数学分析对象。

- 诊断 LLM 调用把意图、概念选择、卡点和置信度合在一个 JSON 中，并被限制到六个概念；这把数学/任务理解和教学诊断合并了。
- TeachingPolicy 本身目前是确定性函数，和 LLM 生成分开；这是一项可保留的优势。
- 教师回复调用同时接收 checker 判定、方程下一步、掌握度、检索、动作和顾问建议。它主要做表达，但输入中包含太多未标准化的数学/策略细节。

最小改进不是让 LLM 直接选择动作，而是先令 LLM 或工具产生 `MathAnalysis`，再由 `TeachingPolicy` 读状态、分析和验证结果选择动作。

### 6.4 最终回复生成和状态更新是否职责混乱？

`teach_node()` 本身不直接写存储，但职责仍偏重：既有 DeepSeek prompt，也有方程专用确定性反馈、mock 模板、验算文本、例题和练习文本。更明显的混合发生在 `log_node()`：它在写审计日志的同时生成 learning events、更新 profile、回写 conversation、修改 episode、推断下一待答类型。`memory_update_node()` 又更新另一套 mastery store。

因此问题不是“回复生成直接改状态”，而是状态更新分散在 memory 和 log 两个节点，而 log 同时承担审计与业务持久化。重构时应保持“生成回复不写状态；状态提交只发生一次；日志从已提交状态投影”的顺序。

### 6.5 三种模式的真实差异在哪里？

| 模式 | workflow/checker/memory | Pedagogical Graph traversal | 旧 `JsonGraphRetriever` | 观察结果 |
| --- | --- | --- | --- | --- |
| `deepseek_only` | 全关闭 | 关闭 | 不调用 | 6/26；无步骤认可、无 memory/episode 行为是设计上的 baseline 差异。 |
| `agent_no_graph` | 全开启 | 由 flag 关闭 | 仍调用 | 26/26；不是完全无检索。 |
| `agent_with_graph` | 全开启 | 开启 `LocalPedagogicalGraphRetriever.retrieve_evidence()` | 仍调用 | 26/26；额外记录 `local_json_graph` evidence。 |

具体原因是 `retrieve_node()` 无论 flag 如何都先调用 `get_retriever()`、`get_prerequisites()`、`get_related_exercises()` 和 `get_misconceptions()`；`_detect_concept()` 也无条件调用旧 reader。flag 只包住新版 local pedagogical graph traversal。

### 6.6 GraphRAG 是否已经成为强制依赖？

新版 Pedagogical GraphRAG 是可关闭的；旧本地概念图不是。因为 concept detection 和 normal retrieve 必须读取 `data/sample/concepts.json`、`exercises.json`、`misconceptions.json`，图数据丢失或不可用时 workflow 无法完成正常教学。这与“RAG / GraphRAG 是可选增强”的目标不一致。

### 6.7 当前测试是否依赖完全一致的自然语言输出？

不要求整段回复逐字一致，但存在明显的词汇和形式耦合：

- 认可依赖“正确/对/很好/思路”等 marker；纠错依赖“不对/错误/不是/先停”等 marker。
- why 问题依赖“两边同时/等式/保持相等”；低掌握度依赖基础语言 marker。
- 多步认可检查“除以”，练习检查“练习/来一题/变式”，部分断言检查原方程文本或最终答案文本。
- single-turn 检查固定 expected concept 和 action；数据集固定 case ID、方程和 forbidden patterns。

这比精确字符串断言更宽松，但对于重构仍然脆弱。阶段三应保留这些作为一元一次方程回归门，同时新增只检验教学行为、不要求专题术语的多领域 smoke tests。

### 6.8 最容易造成行为退化的模块

1. `app/agent/workflow.py`：集中了原题锚定、checker 优先级、提示、泄答案防护和持久化。
2. `app/agent/math_checker.py`：当前 evaluator、workflow、student simulator 都直接导入它。
3. `app/evaluation/evaluator.py` 与 `app/evaluation/mode_runner.py`：目前把方程验证和教学质量混合在同一组断言。
4. `app/memory/conversation_state.py`、`app/memory/schemas.py`、`log_node()`：同一会话状态在多处复制回写。
5. `app/retrieval/graph_retriever.py` 和 `data/sample/*`：旧 reader 的隐式硬依赖易被忽略。

### 6.9 应小步修改还是替换模块？

应小步修改，但需要替换一个明确边界内的职责：将 `workflow.py` 中“领域求解轨迹 + 教学编排 + 回复模板 + 持久化”逐步拆为中性编排边界和领域 adapter。不要重写 FastAPI、LLM client、memory store 或 evaluation 框架；不要先删除一元一次方程实现。第一阶段之后仍应能通过现有严格回归。

## 7. 最小重构设计

### 7.1 目标数据流

```text
Turn input + persisted TeachingState
  -> Context resolution
  -> MathAnalysis (LLM / safe fallback)
  -> OptionalVerifier (only when applicable)
  -> OptionalRetriever or GraphContextProvider (best effort)
  -> TeachingPolicy
  -> ResponseGeneration
  -> one state commit
  -> audit log
```

任何 optional 组件不可用时，流程仍必须到达 `TeachingPolicy` 与 `ResponseGeneration`。在验证器没有结论时，policy 可以要求澄清、给非答案性提示或检查理解；它不能把“未验证”伪装成“错误”。

### 7.2 TeachingState

在 `app/agent/state.py` 中逐步引入中性主状态，并在兼容期把现有 `TeacherState` 作为 API/日志投影或别名承载。中性字段应表达教学事实，不表达某种解题算法：

- 学生与会话：`student_id`、`conversation_id`、本轮消息。
- 问题上下文：`problem_context`、`learning_goal`、`problem_status`，不使用 `target_equation`。
- 教学进度：`progress_summary`、已确认的学生贡献、待答问题、当前教学阶段、答案披露许可。
- 学习者模型：掌握度摘要、重复误区、最近事件、偏好，不要求固定 concept ID 才能存在。
- 编排结果：`math_analysis`、`verification_result`、`graph_context`、`action`、`response_plan`、最终回复和审计信息。

一元一次方程的当前方程、规范化解、步骤序列可以放进 `verification_result.evidence` 或兼容 metadata，不能继续决定核心状态的字段集合。

### 7.3 TeachingAction

新的 policy 词表应使用知识点无关动作：`ASSESS`、`DIAGNOSE`、`GIVE_HINT`、`EXPLAIN_STEP`、`CHECK_UNDERSTANDING`、`SHOW_WORKED_EXAMPLE`、`OFFER_PRACTICE`、`FINISH`。动作参数携带提示层级、是否是学生主动练习请求、难度方向和例题许可；不要继续为“更难/更易/相似”各设一个核心动作。

兼容映射可让旧 API 在阶段二/三仍返回既有 `give_hint_level_1`、`generate_harder_exercise` 等字符串。映射只在边缘层存在，不能把 topic 条件重新塞回 policy。

### 7.4 MathAnalysis

`MathAnalysis` 是分析而非判决，使用现有 `DeepSeekClient.chat_json()` 和 Pydantic：

- `problem_summary`：学生正在处理什么数学对象，空值表示无法可靠确定。
- `attempt_summary`：学生是否给出尝试、陈述、理由或答案。
- `likely_obstacle`：用自由文本或有限的通用类别描述卡点；不从六个概念表强选。
- `reasoning_evidence` 与 `confidence`：让 policy 可审计地判断是否先澄清。
- `suggested_next_learning_focus`：描述性建议，不包含最终答案，也不直接选择教学 action。

现有 `_llm_diagnosis()` 可以先成为该对象的实现来源；其固定 concept 列表和方程规则必须在阶段三移出核心调用。若 LLM 不可用或低置信度，返回“需要澄清”的安全分析，而不是回退到 `c_linear`。

### 7.5 TeachingPolicy

TeachingPolicy 继续保持确定性、可单测。它读取 `TeachingState`、`MathAnalysis`、可选 verifier 结果和可选图上下文，按以下优先级选 action：

1. 明确的 verifier 结论先于 mastery 和 retrieval。
2. 已回答、直接答案、多步尝试和 why 问题优先于通用引导。
3. 低置信度或未确定问题时先 `ASSESS/DIAGNOSE`，不假设题型。
4. mastery 只调整提示粒度、解释深度、复习与练习难度。
5. 例题和新练习必须是显式 action，且由答案披露/原题状态 gate 控制。

Policy 不解析方程、不调用图数据库、不生成回复文本，也不让 LLM 覆盖 verifier 的明确结论。

### 7.6 ResponseGeneration

回复生成只消费动作、结构化教学上下文和允许披露的数学事实；它不写 memory，也不请求 RAG。它可以由 DeepSeek 或 deterministic fallback 实现。保留的跨领域后处理是：

- 每轮最多一个收尾问题。
- 未获准时不披露最终答案。
- 未获准时不引入类比例题或新练习。
- 回复围绕 `problem_context`，或在没有可靠问题时明确要求学生补充。

当前的方程同解检测不能直接充当通用“跑题检测”。阶段四应让 verifier 在能判断时提供 response-grounding evidence；没有 verifier 时只执行较弱但通用的题目引用和披露门控，不伪造数学保证。

### 7.7 OptionalVerifier

定义一个小接口，例如 `verify(problem_context, student_attempt, progress_context) -> VerificationResult`。返回值至少区分：`verified_correct`、`verified_incorrect`、`not_applicable`、`unable_to_verify`，并带可展示的证据/解释和置信边界。

现有 `math_checker.py` 在兼容期实现“一元一次方程 verifier”。其 `suggest_next_step`、`next_question`、方程专用误因与话术必须留在 adapter 内，不再被通用 policy 或 response 层直接导入。验证器不在每轮强制调用：只有检测到适配格式或学生明确要求核验时才尝试。

### 7.8 OptionalRetriever / GraphContextProvider

定义只读、best-effort 的 `GraphContextProvider`，输入应是 `MathAnalysis` 的可选主题/关键词与教学目的，输出是解释、误区、提示或练习候选及其来源。没有 provider、没有匹配或 provider 失败时返回空上下文并记录 backend 状态；不阻断默认教学。

`agent_no_graph` 的含义必须收紧为“不读取任何 graph/retrieval 数据”。现有 legacy `JsonGraphRetriever` 与新版 `LocalPedagogicalGraphRetriever` 都应从核心 detect/policy 路径移至这个可选边界。

## 8. 后续实施阶段

### 阶段二：通用教学状态、动作和接口骨架

**目标**：在不改变现有一元一次方程教学行为的前提下，建立 `TeachingState`、中性 `TeachingAction`、`MathAnalysis`、`VerificationResult` 和 `GraphContext` 的 schema 与兼容投影；将 policy/response 的输入输出边界写清楚。

**明确不做**：不删除旧 `TeacherState` 字段；不改 math checker 算法；不删除图数据；不改 prompt 行为；不新增数学领域；不把 LLM 分析接入正式决策路径。

**预计修改文件**：

- 修改：`app/agent/state.py`、`app/agent/socratic_policy.py`、`app/api/main.py`、`app/agent/workflow.py` 的类型接线与兼容投影。
- 新增：仅在现有 `app/agent/` 下增加职责明确的 schema/adapter 文件；名称以实际最小接口为准，不新建知识点目录。
- 测试：新增一个中性 schema/policy contract 测试文件；补充 `tests/test_policy.py`、`tests/test_api.py` 的兼容断言。

**测试方法**：先跑新增 schema/policy focused tests，再跑 `python -m pytest`、`python scripts/run_eval.py` 和三条 `run_eval_modes.py` 命令。

**验收标准**：旧 API response 和 26 个严格评测 case 均保持通过；新中性 schema 可以表示“没有可验证数学对象”的会话；任何核心 action 都不要求 concept ID 或 equation 字段。

**主要风险**：同时维护 `TeacherState` 与 `TeachingState` 可能造成字段漂移。缓解方式是在单一适配函数中做投影，并为其写精确测试，不在每个 node 自己拼装两种状态。

### 阶段三：移除一元一次方程核心绑定，并添加多领域 smoke tests

**目标**：让 workflow 的默认路径只依赖中性状态、分析和 policy，不再含 `DEFAULT_CONCEPT_ID`、固定关键词/结构分类、`solve_for_x`、方程练习 presets 或直接 `math_checker` 导入；一元一次方程通过兼容 verifier 继续受支持。

**明确不做**：不实现几何、函数、统计或任何新领域的专用 solver；不为各知识点建立数据包；不扩大 GraphRAG；不删除当前方程回归用例；不声称无 verifier 时能数学判真伪。

**预计修改文件**：

- 修改：`app/agent/workflow.py`、`app/agent/context_resolver.py`、`app/agent/state.py`、`app/agent/socratic_policy.py`、`app/memory/conversation_state.py`、`app/memory/schemas.py`、`app/api/main.py`。
- 新增：中性 math-analysis、response-generation 和 compatibility-adapter 模块，仍位于 `app/agent/`。
- 测试：保留并更新现有方程回归；新增面向比例、几何陈述和数据解释的 smoke conversations。它们只验证诊断、单问句、无提前答案、无无关例题、状态推进和 no-verifier 降级，不要求 Harness 给出这些领域的确定性答案。
- 评测数据：在 `data/eval/` 增加少量中性行为 smoke cases，不复制成知识点包，也不包含专用规则表。

**测试方法**：focused workflow/policy/response tests、完整 pytest、strict mock eval、三模式对比；验证无 verifier 的 smoke cases 可到达 response 而不回退到 `c_linear`。

**验收标准**：核心 `workflow.py` 不包含方程结构关键词、固定 equation presets、`solve_for_x` 或 topic-specific action branches；现有一元一次方程 case 通过兼容 verifier；至少三个非方程题型的 smoke case 验证通用教学不变式而非答案正确性。

**主要风险**：移除 equation guard 后可能削弱现有原题锚定和泄答案保护。缓解方式是先把这些保护提升为状态/动作级 gate，再仅对一元一次方程继续用 verifier 做强数学 grounding。

### 阶段四：整理数学验证接口及 RAG/GraphRAG 的可选降级

**目标**：把一元一次方程 checker 收敛为 `OptionalVerifier` 的一个实现；让 legacy JSON graph 和 Pedagogical Graph 都通过 `GraphContextProvider` 可选接入；使 `agent_no_graph` 成为真正无检索的 Agent。

**明确不做**：不接入 Neo4j、Qdrant、教材语料或真实模型大规模评测；不删除一元一次方程 checker；不把 verifier registry 变成按知识点硬编码的核心路由器。

**预计修改文件**：

- 修改：`app/agent/math_checker.py` 的公开适配面、`app/agent/workflow.py`、`app/evaluation/evaluator.py`、`app/evaluation/mode_runner.py`、`app/retrieval/graph_retriever.py`、`app/retrieval/pedagogical_retriever.py`、`app/api/main.py`。
- 新增：`app/agent/` 内的 verifier protocol/adapter 与 `app/retrieval/` 内的 provider protocol/empty provider；不创建专题知识目录。
- 测试：`tests/test_math_checker.py`、`tests/test_flexible_steps.py`、`tests/test_eval_modes.py`、`tests/test_evaluator.py`、`tests/test_retriever.py`、`tests/test_pedagogical_rag.py`，以及新的 no-verifier/no-retriever degradation tests。

**测试方法**：所有既有离线门；显式运行三种 mode，并断言 `agent_no_graph` 不调用任一 graph reader；通过注入空 verifier/空 provider 验证默认教学不崩溃；检查日志能区分 `not_applicable` 与 `unable_to_verify`。

**验收标准**：图数据不可用时默认 Agent 仍可完成诊断、策略和回复；方程 verifier 不适用时不产生错误判定；`agent_with_graph` 只是在相同教学流程上增加可审计上下文，而不是开启另一个必需流程。

**主要风险**：现有 evaluator 直接调用 checker，若同时改评测会掩盖回归。缓解方式是先保留方程 evaluator 原样作为 compatibility suite，再新增独立的通用教学行为 checks，最后才将两者组合展示。

## 9. 预计修改文件总览

以下是后续阶段的预期影响面，不是本阶段实际修改清单：

| 优先级 | 文件 | 变化原因 |
| --- | --- | --- |
| 高 | `app/agent/workflow.py` | 从方程流程聚合器收敛为中性 orchestration facade。 |
| 高 | `app/agent/state.py` | 承载 TeachingState、动作、分析、验证和检索上下文的兼容 schema。 |
| 高 | `app/agent/math_checker.py` | 通过 adapter 对外暴露为一元一次方程 OptionalVerifier，而非核心流程 API。 |
| 高 | `app/evaluation/evaluator.py` | 分离通用教学行为检查与方程 compatibility checks。 |
| 高 | `app/evaluation/mode_runner.py` | 让 no-graph 禁用所有 reader，并报告 verifier/provider 状态。 |
| 中 | `app/agent/context_resolver.py`、`app/agent/socratic_policy.py` | 只依赖中性会话/动作信息。 |
| 中 | `app/memory/conversation_state.py`、`app/memory/schemas.py`、`app/memory/profile_updater.py` | 移除 equation 作为核心短期状态，保留可审计学习事件。 |
| 中 | `app/retrieval/graph_retriever.py`、`app/retrieval/pedagogical_retriever.py` | 建立 optional provider 与 empty fallback，处理双 reader 兼容。 |
| 中 | `app/api/main.py`、`app/ui/streamlit_app.py` | 在不破坏现有展示的前提下，投影通用状态并保留兼容字段。 |
| 中 | `tests/*`、`data/eval/*` | 添加无领域 smoke cases，保留一元一次方程 compatibility suite。 |
| 低 | `README.md`、`CODEX_HANDOFF.md`、RAG/架构文档 | 阶段二至四完成后更新项目定位与 mode 语义；阶段一不改历史说明。 |

## 10. 本阶段和后续均不应实施的内容

- 不重写整个项目或替换 FastAPI、LangGraph、DeepSeek client、memory 持久化和 evaluation 目录。
- 不为比例、几何、函数、统计等知识点新增核心 if/else、专用 policy 分支或“知识包”目录。
- 不让 LLM 自由判定当前一元一次方程学生步骤，也不在 teach 阶段推翻该 checker 的明确结果。
- 不把 RAG、GraphRAG、Neo4j、Qdrant 或教材语料变成默认教学的前置条件。
- 不以删除、放宽或改写现有方程 golden cases 的方式获得通过。
- 不在本次阶段一修改 prompt、核心 state schema、RAG、GraphRAG、业务代码、测试或样本数据。
- 不运行真实 DeepSeek 的大规模评测，不合并到 main/master，不自动进入阶段二。

## 11. 尚待确认的问题与证据缺口

1. **外部 API 兼容窗口**：仓库内可见的 `/turn` consumer 是 Streamlit UI 和测试；没有发现公开 API 版本或外部客户端清单。因此无法确认何时可以移除 `target_equation`、`student_step_validity` 等响应字段。阶段二必须先确定是否以兼容字段、API 版本或迁移公告处理。
2. **无 verifier 时的答案泄露检测强度**：现有强检测依赖方程可求解与最终答案文本。对于无法验证的数学对象，当前证据不足以定义同等强度的自动泄露判定。阶段三的非方程 smoke cases只能先保证“未获准不生成明确最终结论”和“保持问题上下文”，不能假称数学等价性已验证。
3. **MathAnalysis 的真实模型稳定性**：本阶段没有调用付费 API，只有 mock LLM 基线；因此没有证据证明 DeepSeek 对开放式 `MathAnalysis` schema 的字段稳定性。阶段二应先用 stub/fixture 验证 policy 契约，后续再以小规模、明确预算的真实 smoke test 收集证据。
4. **跨领域掌握度键的粒度**：现有 profile 同时包含 `concept_id` 与 `skill_id`，但它们来自方程图和硬编码推断。尚无证据表明应以课程概念、可观察技能、学习目标还是 LLM 提取标签作为通用 key。阶段二只应支持空或文本摘要，避免把未决 taxonomy 写进核心策略。
5. **两套检索 reader 的长期兼容策略**：`graph_retriever.py` 与 `pedagogical_retriever.py` 都读取同一主题图却提供不同接口，当前 workflow 主要使用前者。没有证据显示外部用户依赖后者；阶段四应先以调用点与测试覆盖确认，再决定保留一个 compatibility adapter 还是双接口。

## 12. 阶段一交付物

本阶段只新增本审计与设计文档：

`docs/topic-agnostic-teaching-harness-audit.md`

未修改核心业务代码、prompt、state schema、RAG/GraphRAG、测试、评测数据或现有一元一次方程稳定实现。
