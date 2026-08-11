# CODEX_HANDOFF.md — 当前项目交接

> 本文是修改本仓库前必须阅读的工程快照。还必须阅读 `AGENTS.md` 与
> `CLAUDE.md`。运行时 LLM 是 DeepSeek；Claude/Codex 仅是开发助手。

## 1. 当前结论

当前 canonical runtime 是 `prototypes/llm_led_teacher/` 的 LLM-led Teacher
Agent：DeepSeek 自主选择教学动作、讲解深度和措辞；确定性外壳负责保持题目
连续性、验证已支持的数学内容、提供可选且有边界的证据，并审计最终回复。

`TeachingPlanner`、`StrategyRegistry` 和由掌握度驱动的硬流程控制不属于
canonical loop。`app/agent` 的旧结构化工作流仍保留，服务于兼容性、回归研究和
架构对比，但不能作为新 runtime 的规划器或控制器重新接回该循环。

## 2. Canonical Teacher Agent 架构

核心回合按以下顺序运行：

```text
观察消息与软节奏
  -> 建立或保留 active problem
  -> 验证已支持的学生数学内容
  -> 检索或复用有边界的可选证据
  -> 让 DeepSeek 生成教学回复
  -> 提取并验证已支持的教师数学断言
  -> 仅当硬审计失败时修复一次
  -> 持久化已审计回合与安全的检索元数据
```

实现归属如下：

canonical loop 面向通用 K9 数学对话；确定性验证的权威覆盖在当前已支持的
一元一次方程上最强。对其他主题，DeepSeek 可以继续教学，但不能把教学性推理说成
已经由确定性 checker 验证。

- `prototypes/llm_led_teacher/logic.py`：episode 状态、观察/软节奏、验证、
  DeepSeek 调用、断言审计、一次性 repair 与回合记录。
- `prototypes/llm_led_teacher/rag_adapter.py`：canonical loop 对检索层的唯一
  适配边界，负责例题曝光许可、缓存/刷新和有界 payload。
- `prototypes/llm_led_teacher/streamlit_app.py`：canonical Streamlit UI 与安全
  的诊断、检索元数据显示。
- `app/agent/math_checker.py` 与 `app/agent/optional_verifier.py`：已支持数学的
  确定性权威；LLM 和 RAG 都不得推翻其结论。
- `app/retrieval/semantic_provider.py` 与
  `app/retrieval/chroma_semantic_provider.py`：可复用、非阻塞的语义检索合同与
  Chroma 向量实现。
- `app/llm/deepseek_client.py`：所有 DeepSeek 调用的唯一出口。

## 3. 回合数据流与信任边界

`problem_context` 是 active problem，首轮建立后跨回合保留；观察器只能提供软
节奏，不能重写题目或裁定数学正误。学生尝试与教师可验证断言均交给确定性
verifier。确定性验证对当前已支持的一元一次方程最强；其他通用 K9 数学主题仍可由
DeepSeek 正常教学，但不能伪装为已经验证。

检索位于学生验证之后、教师生成之前。`retrieval_reference` 是不可信、可选的
辅助证据，必须与 `problem_context`、`deterministic_verification`、
`turn_guidance` 分开传入。它不能覆盖题目、对话历史、节奏或 verifier，也不能
单独证明学生或教师的数学结论。观察器和 verifier payload 不得含检索文本。

审计会从教师输出提取已支持的数学断言；仅硬审计冲突触发一次 repair。回合记录
保存审计后的回复与安全检索元数据，不保存链式思维、凭据、绝对本地路径或原始
异常。

## 4. 关键模块

`PrototypeEpisode` 维护 active problem、上一步已验证内容、回合历史及当前问题的
检索缓存。`run_turn()` 负责完整 canonical 回合；UI 只显示已审计的回复以及
verifier、RAG 状态、来源、相似度、例题许可和阶段耗时。

`math_checker` 使用精确计算处理受支持的一元一次方程等内容。`optional_verifier`
以统一请求/结果合同界定“已验证”和“不适用”，因此模型可以在不适用时继续教，
但不得声称确定性验证已经发生。

`semantic_provider` 的稳定合同是 `SemanticRetrievalRequest`、`RetrievalHit`、
`SemanticRetrievalResult` 与非抛出的 provider。调用方必须接受所有检索状态，
且不得因此改变教学控制流。

## 5. RAG 架构与配置

RAG 默认关闭、显式 opt-in、非阻塞，并且始终与 active problem 分离。检索状态为
`disabled`、`success`、`empty` 或 `error`；任一状态都必须继续教学。正常回合只
请求 `reference` 内容。`example` 内容只有在学生明确要例子/练习、同一 active
problem 连续两次被 verifier 确认答错、或学生明确表示完全不懂时才能请求和展示。
这些 gates 只限制检索内容，绝不选择教学动作或强制学生按脚本作答。

新题会清除前题缓存并检索一次；普通追问复用当前问题缓存。若例题后来首次获准且
此前缓存尚未标记为允许例题，则可刷新一次请求例题（无论旧结果是 reference、
disabled、empty 还是 error）。一旦缓存已经在允许例题的条件下生成，后续回合会
复用它，不承诺对 cached disabled、empty 或 error 自动重试；下一个新题才会清除
缓存并开始新的检索。检索结果不会写入 `problem_context` 或
`previous_verified_step`。

Chroma provider 是惰性加载的向量检索实现，支持 `CHROMA_SERVER` 指定的 HTTP
服务，或以 `TEACHER_RAG_VECTORDB_PATH` 指定持久化路径；两者同时设置时 HTTP
优先。嵌入模型由 `TEACHER_RAG_EMBED_MODEL` 提供，采用与 bge-m3 兼容的
sentence-transformers 模型 ID 或本地模型目录。检索保持方程、LaTeX、分数、根式和
数学符号，按阈值过滤、按稳定来源去重并按相似度排序；不接入 BM25/RRF 或在线搜索。

配置名（不要把机器特定值写进文档或代码）包括：

- `TEACHER_RAG_ENABLED`
- `CHROMA_SERVER` 或 `TEACHER_RAG_VECTORDB_PATH`
- `TEACHER_RAG_EMBED_MODEL`
- `TEACHER_RAG_REFERENCE_COLLECTIONS` 与 `TEACHER_RAG_EXERCISE_COLLECTIONS`
- `TEACHER_RAG_TOPK` 与 `TEACHER_RAG_MIN_SIMILARITY`
- `TEACHER_RAG_SUBJECT_FILTER` 与 `TEACHER_RAG_EXERCISE_SUBJECT_FILTER`
- `TEACHER_RAG_DEVICE` 与 `TEACHER_RAG_CPU_THREADS`

离线测试使用 fake provider 覆盖合同、失败降级、例题 gates、缓存/刷新与 payload
隔离；这不等同于真实 Chroma、嵌入模型或 DeepSeek 的验证。外部依赖不可用时必须
报告为不可用/未运行或失败，不能当作真实服务通过。

## 6. Legacy baseline 的位置

`app/agent/workflow.py`、`app/api/main.py`、`app/ui/streamlit_app.py` 和现有的
mastery persistence 都保留用于兼容性、回归研究及与 canonical 架构的比较。它们
不是 primary runtime，不能被并回 LLM-led loop 形成硬规划或 mastery 控制；memory
和 RAG 始终只是讲解深浅、提示、复习和变式选择的软约束。

## 7. 教学行为不变式

- original/active problem 首轮建立后不得被跨轮覆盖；普通讲解始终围绕它。
- 数学正确性由 `math_checker` / `optional_verifier` 裁定，判定分支优先于任何
  LLM、memory 或 RAG 偏好。
- 非明确例题或练习动作不得让检索例题污染原题；检索内容不改变题目语境。
- 学生可单步、多步、直接答案、跳步或问“为什么”；不得以 mastery 或检索强制
  一步一答。
- 学生未解出且未明确索要时，不泄露最终答案；有明确“不需要答案”请求时同样
  不得泄露。
- `show_worked_example` 的内容曝光只能由明确要例子/练习、连续两次已确认错误或
  明确完全不懂触发。
- `single_move` 对已支持的多步求解维持一次只推进一个产生新结果的步骤；概念、
  原因、直接事实或明确完整解答可使用完整回答。

## 8. 运行与验证

安装 UI 与可选 RAG runtime：

```powershell
pip install -e ".[dev,ui,rag]"
Copy-Item .env.example .env
```

将 `DEEPSEEK_API_KEY` 仅放在 `.env` 或启动 shell 中。启用检索时设置
`TEACHER_RAG_ENABLED=true`，并选择 Chroma HTTP 服务或持久化路径之一。启动
canonical UI：

```powershell
streamlit run prototypes/llm_led_teacher/streamlit_app.py
```

每次修改后的两个必跑离线 gate 是：

```powershell
python -m pytest
python scripts/run_eval.py
```

真实模型评估是可选的，并会产生 API 成本：

```powershell
$env:USE_REAL_LLM_FOR_EVAL = "true"; python scripts/run_eval.py
Remove-Item Env:USE_REAL_LLM_FOR_EVAL
```

真实 Chroma 和嵌入模型检查也必须单独配置并报告，不会被 fake-provider 离线结果
替代。

## 9. 当前验证状态

离线 gate 默认使用 mock LLM，且不要求 RAG 可选依赖或外部服务；它们是提交前的
回归门。RAG 单元/集成覆盖使用 fake provider，因此只证明本地合同和非阻塞行为。
真实 Chroma、bge-m3-compatible embedding 和 DeepSeek 检查依赖本机配置、可用服务
与成本授权，应分别记录 PASS、FAIL 或 NOT RUN；外部服务缺失绝不等价于 PASS。

本次完整五份文档刷新及其最终修复后的文档状态，已在最终提交前运行完整离线 gate：
`python -m pytest` 为 `388 passed, 1 warning`，`python scripts/run_eval.py` 在离线 mock
模式为 `38/38 passed`。此外执行了 focused 文档标记检查、所有文档路径验证、占位符扫描
和 `git diff --check`；这些文档检查不表示 runtime 或真实外部服务验证。

## 10. 已知问题与下一步

以下 runtime 问题仍未解决，不能在文档中声称已修复：

- 回答被接受后，多个问号目前仍只是 advisory。
- `single_move` 的文字仍可能泄露第二个操作。
- 检索到的 worked example 需要相对于 original problem 的独立审计范围。
- 可选外部服务可能不可用，必须如实报告其状态。

先阅读 [single-move turn contract](docs/superpowers/specs/2026-08-10-single-move-turn-contract-design.md)
和 [canonical documentation refresh design](docs/superpowers/specs/2026-08-11-canonical-documentation-refresh-design.md)，
再设计这些修复。不要把修复作为恢复旧 Agent loop、硬 mastery 控制或阻塞 RAG 的理由。

## 11. Git consolidation 记录

- canonical source before consolidation：`eee8059cbb52e7abd995583c666ec6b01fb62f53`
- server RAG snapshot：`20af21b76a8d965f37958889e7588e45ae7f20fd`
- historical shared-remote RAG base：`1e535c131a3edf3d010a78abbe10bce701c4829c`
- completed RAG integration state：`9a30b0c16d1e1eda167f9f1f811a6f1b70614256`

RAG 通过人工适配并入 canonical runtime，而不是合并任一旧 Agent loop；两个历史
RAG 分支中的旧 workflow、生成结果、向量数据库工件和机器特定路径均不应回流。

## 12. 接手规则

做增量修改，保留 `app/api`、`app/agent`、`app/llm`、`app/retrieval`、`app/memory`
和 `app/evaluation` 的分层。不要删除或弱化 `data/eval/`、`app/evaluation/` 或
`scripts/run_eval.py`；新增教学能力必须有相应 golden case 和 evaluator 行为检查。

所有 LLM 调用必须经过 `app/llm/deepseek_client.py`，结构化输出用 JSON mode 与
Pydantic 校验。不要提交 `.env`、API key、向量数据库工件、runtime logs 或
`data/runtime` 内容；也不要打印凭据。修改 workflow、prompt、模板、checker 或
schema 时，按 `AGENTS.md` 完成其相应测试与同步检查。
