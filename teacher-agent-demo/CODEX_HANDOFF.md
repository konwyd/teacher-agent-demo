# CODEX_HANDOFF.md — 项目交接文档

> 交接自 Claude（开发助手），2026-07。接手前请完整读完本文件 + AGENTS.md。

## 一、项目目标

- 本项目是**中文初中数学 Teacher Agent 的 research demo**（MVP 范围：一元一次方程相关知识点）。
- **runtime LLM 是 DeepSeek API（OpenAI-compatible），不是 Claude/Codex**。Claude/Codex 只是开发助手，任何代码里不应出现对开发助手模型的依赖。
- 目标是 **Graph-grounded Adaptive Tutor Agent**：诊断学生卡点 → 检索知识图谱 → 选择教学动作 → 分层引导 → 更新学生掌握度，**不是普通聊天机器人**。
- 核心研究命题：确定性规则（math checker、教学策略）+ LLM 生成 的混合架构，能否比纯 LLM 更稳定地"像老师一样教"。

## 二、已完成的功能

| 功能 | 状态 | 位置 |
|---|---|---|
| FastAPI 后端（/health, /turn） | ✅ | `app/api/main.py` |
| Streamlit 聊天 UI（含诊断详情展开区） | ✅ | `app/ui/streamlit_app.py` |
| DeepSeek 接入（chat + JSON mode，超时/异常处理，.env 加载） | ✅ | `app/llm/deepseek_client.py` |
| 六节点 structured teaching workflow（LangGraph） | ✅ | `app/agent/workflow.py` |
| original_problem / target_equation 主线锚定（跨轮不覆盖） | ✅ | `app/memory/conversation_state.py` + workflow |
| math_checker（方程解析/步骤判定/误因分类，纯 fractions 无 sympy） | ✅ | `app/agent/math_checker.py` |
| 学生多步回答支持（单步/多步计划/直接最终答案/问为什么） | ✅ | math_checker + workflow diagnose 分派 |
| Student memory / mastery tracking（JSON 持久化） | ✅ | `app/memory/student_state.py` |
| mock Graph RAG（本地 JSON 概念图/练习/误解） | ✅ | `app/retrieval/graph_retriever.py` + `data/sample/` |
| Evaluation harness（9 项确定性行为检查） | ✅ | `app/evaluation/evaluator.py` + `scripts/run_eval.py` |
| Golden test cases（15 组多轮 + 6 个单轮） | ✅ | `data/eval/*.jsonl` |
| pytest 全离线测试（~80 个，默认 mock LLM） | ✅ | `tests/` |

六节点 workflow：`diagnose → retrieve → teaching_policy → teach → memory_update → log`。
每轮日志写 `logs/turn_logs.jsonl`（含 grounding 字段），供研究分析。

## 三、当前架构

```
app/api/            FastAPI 路由。/turn 返回诊断/动作/回复/掌握度变化/log_id
app/agent/
  state.py          全部 Pydantic schema（TeacherState、DiagnosisResult、TurnLog…）
  workflow.py       六节点 LangGraph 状态机 + 教学策略 + 回复模板 + postprocess
  math_checker.py   纯 stdlib 数学判定器（方程解析、步骤/多步计划/最终答案检查）
app/llm/            deepseek_client.py：唯一的 LLM 出口，禁止绕过
app/retrieval/      graph_retriever.py：JSON mock 的 Graph RAG（接口稳定，可平替 Neo4j）
app/memory/
  conversation_state.py  跨轮对话状态（original_problem、current_step、solved_steps…，进程内存）
  student_state.py       掌握度档案（data/runtime/student_profiles.json 持久化）
app/evaluation/     evaluator.py：golden 对话回放 + 行为检查
data/sample/        概念/练习/误解 mock 数据集（6 概念 8 练习 8 误解）
data/eval/          golden_conversations.jsonl（15）+ single_turn_cases.jsonl（6）
scripts/            run_eval.py（评测入口）、test_deepseek.py（连通性）
tests/              pytest；conftest.py 全局移除 API key + 隔离日志/档案到 tmp
```

关键机制速查：
- **意图分派**（diagnose_node）：why → confused → 索要答案 → `classify_student_math_response()`（multi_step_plan / single_step / intermediate_equation / final_answer）→ 普通规则诊断。带求助词（怎么/不会/吗…）的消息不当步骤处理。
- **策略**（`_choose_action`）：checker 判定分支永远最优先；show_worked_example 只有三种触发（学生要例子/连续两次答错/完全不懂）；mastery <0.3 转诊断式+基础语言、≥0.7 给变式题。
- **LLM 兜底**（`postprocess_teacher_response`）：无关新方程/提前泄答案 → 降级为确定性模板；多问句 → 只留最后一个；无问句 → 补收尾问题；低掌握度诊断问必须含"两边同时"类基础语言。
- **LLM 诊断锚定**：概念按方程结构确定性覆盖（括号→去括号、分数系数→分数系数方程、两边有 x→移项）；LLM 猜的卡点（procedure/calculation/prerequisite）无关键词或掌握度证据时回退规则（`_anchor_gap`）。

## 四、关键设计原则（不可破坏）

1. **original_problem 是教学主线**，首轮抽取后跨轮不覆盖；retrieved examples 只能辅助（且仅 show_worked_example / give_practice_question 可引入新题），用完必须回到原题。
2. **math_checker 的判定优先于 LLM 自由判断**：学生给出步骤/计划/答案时，对错由 checker（Fraction 精确计算）裁决，回复用确定性模板，LLM 无权改判。
3. **memory 只影响教学深浅**（解释详略、类比、复习推荐、变式题），不得拒绝跳步、不得强制一步一答、不得覆盖 checker 的正确判定。
4. **学生可以四种方式回答**：单步（"3x=3"/"先减4"）、多步计划（"先都减四，然后同时除以3"）、直接最终答案（"x=1"/"1"/"最后等于1"）、问为什么——都必须被正确理解。
5. **每轮最多问一个问题**，且结尾必须给学生一个任务（postprocess 兜底）。
6. **不过早给最终答案**：只有学生已解出、或明确索要时才允许出现。
7. **不无故引入新题**：非例题动作的回复中，所有方程必须与原题同解（或引自学生消息）。

## 五、已修过的重要 bug（不要复发）

| Bug | 根因 | 修复 |
|---|---|---|
| 2x+3=7 等 mock 例题污染主线 | 检索例题无条件进 LLM 上下文 | 例题仅 worked_example 时入 prompt + postprocess 检测无关方程降级 |
| 老师过早给最终答案 | prompt 无强约束 | prompt 收紧 + postprocess 检测泄答案降级为模板 |
| 每轮问 2-4 个问题 | LLM 习惯连环追问 | `_keep_single_question()` 只保留最后一个问句 |
| 3(x-2)=9 被识别成一元一次方程 | LLM 概念误诊 | `_classify_equation_structure()` 按方程结构确定性锚定 |
| memory 让老师机械化（卡住 expected_next_step） | 只认单方程回答 | `classify_student_math_response` + 多步计划整体模拟 |
| 学生多步回答无法识别 | 同上 | `check_multi_step_plan()` 中文操作解析 + 逐项模拟 |
| "mock 评测"实际调了真实 DeepSeek | run_eval 先 pop key 后 import，deepseek_client 模块级 load_dotenv 又把 key 加回来 | import 顺序修正 + `TEACHER_FORCE_MOCK=1` 双保险 |
| 错误步骤反馈泄露正确答案 | checker 解释含解 | 解释不再包含原方程的解 |
| x=4/3 被判成"分数系数方程" | 结构分类看了常数项 | 只看 x 的系数 |
| .env / API key 入库风险 | — | `.gitignore` 已含 .env、logs/、data/runtime/；key 只从环境读取，永不打印 |

## 六、运行方式（Windows PowerShell）

```powershell
# 环境（conda）
conda create -n teacher-agent python=3.11 -y
conda activate teacher-agent
pip install -e ".[dev,ui]"

# 配置
copy .env.example .env    # 填 DEEPSEEK_API_KEY

# 测试与评测（每次改动后必跑）
python -m pytest
python scripts/run_eval.py                     # mock 模式，预期 21/21

# 真实 DeepSeek 评测（会产生 API 费用）
$env:USE_REAL_LLM_FOR_EVAL = "true"; python scripts/run_eval.py
Remove-Item Env:USE_REAL_LLM_FOR_EVAL          # 跑完关闭

# 运行 demo（两个终端）
python -m uvicorn app.api.main:app --reload    # 后端 :8000
streamlit run app/ui/streamlit_app.py          # 前端 :8501

# 工具
python scripts/test_deepseek.py                # DeepSeek 连通性
del data\runtime\student_profiles.json         # 清空全部学生记忆
```

## 七、环境变量（.env，参考 .env.example，禁止提交真实 key）

| 变量 | 说明 | 默认 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API key（必填才走真实 LLM，否则 mock） | 空 |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 模型名 | `deepseek-chat` |
| `USE_REAL_LLM_FOR_EVAL` | `true` 时评测调真实 DeepSeek | 未设置（mock） |
| `TEACHER_FORCE_MOCK` | `1` 强制 mock（评测/调试用） | 未设置 |
| `TEACHER_LOG_DIR` | 日志目录 | `logs` |
| `TEACHER_PROFILE_PATH` | 学生档案 JSON 路径 | `data/runtime/student_profiles.json` |

## 八、测试与评测

- `python -m pytest`：~80 个测试，全离线（conftest 自动移除 API key、隔离日志与学生档案到 tmp）。
- `python scripts/run_eval.py`：回放 15 组 golden 多轮对话 + 6 个单轮 case，9 项确定性行为检查（保持原题、无关新题、步骤识别、每轮一问、不泄答案、hallucination、mastery 行为等）。默认 mock 不调真实 API。
- `USE_REAL_LLM_FOR_EVAL=true` 才调真实 DeepSeek（有随机性，postprocess 已兜底主要违规类型）。
- 结果追加写 `logs/eval_results.jsonl`（含 use_real_llm 标记与逐 case 失败原因）。
- **规则：每次大改必须先跑 pytest + run_eval，全绿才算完成。**

## 九、下一步开发计划（按优先级）

1. **继续稳定 student memory**：让掌握度更像软约束（如衰减、置信区间），观察真实对话日志调阈值。
2. **接入真实小型教材 RAG**：先从 1-2 章人教版初一教材文本开始建 vector 检索，不要一开始全量。
3. **接入 K12-KGraph 或自建小型知识图谱**：替换 `graph_retriever` 的 JSON mock（接口已稳定，实现 `Neo4jGraphRetriever` 即可）。
4. **接入 Math23K / Ape210K 小子集**：扩充 exercises，先抽样 100-500 道一元一次方程题。
5. **加 Neo4j / Qdrant**：配置在 `.env.example` 已预留（NEO4J_URI、QDRANT_URL）。
6. **baseline 对比实验**：DeepSeek only vs DeepSeek+vector RAG vs DeepSeek+Graph RAG+teaching workflow，用同一套 golden cases + eval harness 打分。
7. **写技术报告 / 小论文**：`logs/turn_logs.jsonl` 和 `logs/eval_results.jsonl` 就是实验数据来源。

## 十、给 Codex 的工作规则

见 AGENTS.md（Codex 会自动读取）。核心：不要重构整个项目、不要删评测、不要绕过 math_checker、不要把 memory 变成硬流程、不要提交 .env、改完必跑 pytest + run_eval、测试失败优先修真实问题而不是降低测试标准。
