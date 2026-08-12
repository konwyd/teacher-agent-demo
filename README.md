# LLM-led Teacher — 中文 K9 多学科教师原型

这是一个面向中文小学、初中九学科的研究型教学原型。当前主入口是 **LLM-led Teacher Prototype**：有界 Agent Loop 识别学科并按需调用 Skill、本地知识和联网工具，DeepSeek 负责自然语言教学，可选的确定性 verifier 为其支持的数学步骤提供证据。

项目关注的不是让模型无条件一次性给出答案，而是让每轮回答服从明确的教学契约：概念提问只讲通用概念，普通求解只推进允许的步骤，只有学生明确索要完整解答时才开放最终答案。

## 当前主版本

- 网页入口：`http://localhost:8502`
- 一键启动：`python3 scripts/run_llm_led_teacher.py`
- 教师模型：本地 DeepSeek-V4 Flash GGUF（llama.cpp OpenAI-compatible 服务）
- 本地检索：bge-m3 + Chroma Server + BM25 + RRF
- 学科范围：数学、语文、英语、物理、化学、生物、历史、地理、道法
- 教学防线：六步 Agent Loop + 工具门禁 + disclosure-aware `TeachingContract` + 本地语义边界审计 + 可选确定性 verifier
- 展示方式：先完成审计，再将最终安全回答逐字符显示

旧的结构化 Teacher Agent 仍保留为研究 baseline，入口为 Streamlit `8501` 和 FastAPI `8000`，但它不是当前默认演示。

## 单轮处理流程

```text
学生消息
  ↓
Agent 强制分类学科、学段、是否新题
  ↓
有界 Loop 按需加载 Skill、检索本地知识，必要时受控联网
  ↓
Agent 提交 TeachingContract：目标、回答模式、允许/禁止内容、当前题应用与答案披露权限
  ↓
本地策略根据学生原始请求收紧契约；可选 verifier 只检查其支持的数学步骤
  ↓
DeepSeek 按契约生成教师初稿
  ↓
本地模型做语义边界审计；越界时按公开修正指令重写，并做高置信字面复查
  ↓
将公式规范化为 MathJax 兼容 Markdown，持久化本轮契约、公开 Agent trace 与审计结果
  ↓
只把审计后的回答逐字符展示给学生
```

8502 的 Agent 路径不再以实验性 Observer 作为节奏来源；Observer 仅留在未启用 Agent Loop 的兼容路径。Agent 必须以 `concept_explanation`、`one_step`、`guided_solution`、`full_solution` 或 `brief_redirect` 提交教学契约。本地策略再以学生原始措辞为可信依据收紧权限；旧的 `single_move` / `complete_response` 只是由契约派生的兼容字段，不再决定披露边界。

例如，追问“求根公式是什么”会规范化为 `concept_explanation`，不允许把公式代入 active problem，也不允许披露该题答案；学生明确要求“完整步骤和答案”时才可使用 `full_solution` 并开放这两项权限。语义边界审计检查的是回答是否服从该目标、应用权限、披露权限和步骤范围，**不是**对数学、物理或其他学科陈述做通用事实验证。

确定性 verifier 是可选的学科工具，而不是生成回答的前置条件。当前实现只把既有的 Fraction 一元一次方程 checker 暴露为 verifier；超出支持域时记录 `not_applicable` / `unable_to_verify`，回答仍受教学契约与边界审计约束。不要把“边界审计通过”解释成“所有事实均已验证正确”。

## RAG 的实际工作方式

RAG 在服务端运行，用户浏览器不需要下载向量库或嵌入模型。

```text
学生题目
  ├─ bge-m3：在应用进程生成查询向量，可使用 GPU
  ├─ Chroma Server：通过服务器内部 HTTP API 查询本地 HNSW 索引
  ├─ BM25：关键词召回
  ├─ RRF：融合语义与关键词排序
  └─ Agent 判断存在具体证据缺口时：通过门禁后至多调用一次 Tavily
```

- `bge-m3` 是本地 embedding 模型，不是 Chroma 自带模型，也不是付费 embedding API。
- Chroma 既是 Python 包，也能作为独立服务运行。本项目由启动脚本执行 `chroma run`，应用再通过 `chromadb.HttpClient` 访问它。
- Chroma 的 HNSW 检索当前主要使用 CPU；GPU 用于加速 bge-m3 查询向量和联网结果重排。
- 8502 的本地工具复用 `llm-test/rag_retriever.py` 的教材/题库、BM25、bge-m3 和 RRF，但不应用固定阈值，也绝不会隐式联网；raw vector、BM25、RRF 和融合分会作为诊断返回 Agent。
- 联网只能用于 K9 请求，并且本轮必须已经成功执行本地检索、提供具体证据缺口；每轮最多一次。非 K9 请求不调用本地库或联网，只简短回答并引导回学习内容。

RAG 是有边界的参考数据：它只能通过 Agent 的受控工具调用进入教师生成请求，不能改变请求分类、`TeachingContract` 或当前 `active problem`。检索内容与题目无关时，教师可以忽略。

## 化学分子结构图增强（MolScribe + RDKit）

现有 OCR/RAG 数据中的 `metadata.images` 和图片存储保持不变。可选的数据预处理命令只对化学块中命中“结构式/结构简式/分子结构/化学式/分子式/电子式/分子模型/球棍模型”等上下文的图片运行 MolScribe，并把经 RDKit 校验后的规范 SMILES、分子式和官能团描述追加到文本中。它不处理实验装置图、曲线图或普通插图，也不会覆盖原始 OCR。

MolScribe 1.1.1 要求 `torch<2.0`，与当前本地模型环境的 PyTorch 2.x 不兼容，因此必须使用隔离环境：

```bash
cd /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo
bash scripts/install_molscribe_env.sh
```

图片根目录和历史 `source_md` 路径映射需要按实际挂载位置填写。默认 `keyword` 模式较安全；确认希望尝试所有化学块图片时才使用 `--candidate-mode all`：

```bash
.venv-molscribe/bin/python scripts/enrich_molecular_structures.py \
  data/china_textbook_k9_chunks/chunks.jsonl \
  data/china_textbook_k9_chunks/chunks.molscribe.jsonl \
  --image-root /实际的图片存储根目录 \
  --path-map /data/wyd-workspace/education=/home/dockerContainerSave/data/wyd-workspace/education \
  --device cuda:0
```

输出包括增强后的 JSONL、结构明细 JSONL、预测缓存和统计报告。增强后的文本可以按原有流程重新写入 Chroma；原始 `chunks.jsonl` 不会被原地覆盖。若只想处理人工确认的图片，可准备每行一个图片文件名的 allowlist，并使用 `--candidate-mode allowlist --allowlist images.txt`。

MolScribe 结果只作为机器生成证据保留：SMILES 经 RDKit 解析失败、原子数不足或置信度低于 `0.80` 时会被丢弃。低置信度图片仍保留在原 `metadata.images` 中，便于人工复核。

## 环境要求

- Python 3.10+
- 本地存在 bge-m3 模型和 Chroma 向量库
- 本地存在 DeepSeek-V4 GGUF、CUDA 版 llama.cpp 和现有 8-GPU 启动脚本
- 8×RTX 4080 或能够容纳当前权重与 KV cache 的等效环境
- 使用 GPU embedding 时需要可用的 CUDA/PyTorch 环境

安装示例：

```bash
cd /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo
pip install -e ".[dev,ui]"
pip install -r llm-test/requirements.txt
pip install jieba rank-bm25
```

当前一键启动不需要 DeepSeek API key。启动器会只在 Teacher 子进程中强制使用本地 `8080/v1` 服务，覆盖 `.env` 中可能残留的付费 API 地址、模型名和密钥，避免意外回退。

联网检索是可选能力，继续复用 `llm-test/config.py` 中的原项目 Tavily 配置。正式部署时应通过安全的服务端配置提供凭据，不要把真实密钥写进 README、前端或公开仓库。

## 一键启动当前原型

```bash
cd /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo
python3 scripts/run_llm_led_teacher.py
```

脚本会：

1. 在 `8080` 启动本地 DeepSeek-V4，或复用模型别名匹配的健康服务；
2. 等待 `/health` 和 `/v1/models` 均验证通过；
3. 在 `8002` 启动或复用 Chroma Server；
4. 在应用进程中预热 bge-m3；
5. 在 `8502` 启动 LLM-led Teacher 页面；
6. 退出时关闭由本脚本启动的模型、Chroma 和网页子进程。

启动成功后只需访问 `http://localhost:8502`。默认向量库路径为 `data/china_textbook_k9_vectordb`。

首次打开页面会先看到“登录 / 注册”窗口。新用户在注册页输入用户名和至少 8 位密码，
系统会自动创建不可重复的内部学生 ID 并立即登录；之后使用同一用户名和密码登录即可。
本地账号默认保存在 `data/runtime/users.sqlite3`，密码只保存加盐哈希，不保存明文。
该功能是本地演示级账号系统，不包含短信验证码、找回密码或公网生产级安全防护。

每轮成功回答会自动把完整对话保存到
`data/runtime/conversations.sqlite3`。重新登录后自动打开最近一次对话，也可以在侧边栏
“历史对话”中切换或新建；查询始终同时校验内部学生 ID 和对话 ID，不同账号不能读取彼此记录。
每个新回合还会随 episode 持久化请求分类、TeachingContract、真实执行过的公开 Agent/Tool 事件、边界审计、可选 verifier 结果和公式规范化标记；历史详情页可重新展示这些 trace，但不会保存或展示 prompt、私有 tool details 或模型思维链。

模型默认使用 `131072` 上下文、单并发槽、K=`q8_0`、V=`q4_0`，并把 batch/ubatch 都限制为 `128`。这套配置已在当前 8×16 GiB GPU 环境完成真实长提示词回归；旧的 `262144 / parallel=-1 / batch=512` 配置会在首轮计算时使 GPU4 OOM。首次加载可能需要数分钟，启动器最多等待 900 秒，并在开放网页前执行一次最小真实推理来验证 CUDA 计算可用。模型端口被未知或不健康服务占用、模型提前退出或健康检查超时时，整套启动直接失败，不会调用 DeepSeek 付费 API。

## 端口与对外访问

| 端口 | 服务 | 是否需要公开 |
|---:|---|---|
| `8502` | 当前 LLM-led Teacher 网页 | 是，用户只访问这个端口 |
| `8002` | Chroma 内部 HTTP API | 否，只供服务器上的应用访问 |
| `8080` | 本地 llama.cpp OpenAI-compatible API | 否，只供服务器上的 Teacher 访问 |
| `8501` | 旧结构化 Agent Streamlit | 否，除非专门展示 baseline |
| `8000` | 旧结构化 Agent FastAPI | 否，除非有独立调用方 |

通过 VS Code 端口转发或反向代理发布时，只需公开 `8502`。浏览器不会直接访问 `8002` 或 `8080`；网页服务在服务器内部调用 Chroma 和本地模型。正式对外使用时应采用稳定域名、HTTPS、反向代理和访问控制，开发隧道只适合临时演示。

可覆盖的一键启动参数：

```bash
python3 scripts/run_llm_led_teacher.py \
  --app-port 8502 \
  --rag-port 8002 \
  --model-port 8080 \
  --model-startup-timeout 900 \
  --llm-request-timeout 300 \
  --db-path data/china_textbook_k9_vectordb
```

关键默认配置：

| 配置 | 默认值 | 用途 |
|---|---|---|
| 模型别名 | `deepseek-v4-flash` | 由 `/v1/models` 健康检查验证 |
| 模型地址 | `http://127.0.0.1:8080/v1` | Agent 分类/决策、教师生成、语义边界审计和修正请求 |
| `DEEPSEEK_TIMEOUT_SECONDS` | `300` | 单次本地模型请求超时 |
| `TEACHER_RAG_DEVICE` | `cpu` | 512K 本地模型占满 GPU 时避免 bge-m3 OOM；显式环境变量优先 |
| `TEACHER_RAG_CPU_THREADS` | `4` | RAG 相关 CPU 线程数 |
| `CHROMA_SERVER` | `127.0.0.1:8002` | 原检索器访问的 Chroma Server |

需要查看全部覆盖项时运行 `python3 scripts/run_llm_led_teacher.py --help`。只有确认某张 GPU 留有足够模型权重和推理峰值之外的余量时，才应显式设置 `TEACHER_RAG_DEVICE=cuda:N`。

## 回答速度与流式展示

一次普通回合通常包括请求分类、若干有界 Agent 决策、TeachingContract 提交、教师生成和语义边界审计；触发安全修正时还会多一次模型调用。页面会显示分类、工具、生成、审计和修正等真实阶段状态。

回答生成期间，展开的状态框会实时追加实际发生的 Agent 决策、Tool 结果及各后台阶段，并在阶段结束时显示耗时；完成后还会追加教学契约、RAG 和审计摘要。最终文本中的明确 LaTeX 分隔符和高置信公式块会转换成 Streamlit/MathJax 可渲染的 `$...$` 或 `$$...$$`，围栏代码块保持原样；规范化后的文本会进入显示和后续对话历史。

最终回答采用安全流式展示：初稿不会直接发给学生，必须先完成审计和必要修正，随后安全版本才逐字符显示。它改善等待体验，但不会缩短上游模型和检索本身的耗时。

## 验证

所有自动化测试都使用 fake client/mock LLM，不调用真实付费 API：

```bash
python -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
python -m pytest tests/prototypes/test_llm_led_streaming.py -q
python -m pytest
python scripts/run_eval.py
```

重点保护的教学不变式：

- 概念解释契约不能应用 active problem 或披露最终答案；
- 只有学生明确索要完整解答时才允许 `full_solution` 和最终答案；
- 旧 `turn_guidance` 由 TeachingContract 派生，不能反向放宽契约；
- 初稿越过目标、应用或披露边界时触发语义修正；
- 修正后最多推进一个有意义步骤，且最多一个结尾问题；
- 缺少适用 verifier 时不会阻断回答，也不会伪造“已验证”证据；
- 公式输出规范化为 MathJax 兼容 Markdown，并持久化规范化结果；
- 历史记录保存真实公开 Agent trace、TeachingContract 和边界审计；
- RAG 不改变 TeachingContract 或 active problem；
- 九学科新题可以自动切换，追问继承当前学科和学段；
- 非 K9 请求以及未先查本地的联网动作会被工具门禁拒绝。

## 旧结构化 Agent baseline

旧版用于研究 Agent workflow、student memory、math_checker、evaluation 和 Pedagogical Graph RAG，仍可单独运行：

```bash
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
streamlit run app/ui/streamlit_app.py --server.port 8501
```

当前 LLM-led Prototype 刻意绕过 TeachingPlanner、掌握度控制和 Pedagogical Graph RAG，不要把两个入口混为同一实现。

## 本地 DeepSeek-V4 运行参数

当前网页默认使用 `DeepSeek-V4-Flash-GGUF-IQ3_XXS`。一键脚本复用现有 8-GPU 模型启动脚本，并自动传入已在当前 8 张 RTX 4080 上验证的 256K 总上下文、4 个共享 KV 槽位与混合量化 KV cache 配置：

```bash
python3 scripts/run_llm_led_teacher.py
```

当前配置下 DeepSeek 分片已占用全部 8 张 GPU；GPU7 再加载 bge-m3 后实测只剩约 52 MiB，并在检索峰值触发过 `OutOfMemoryError`。因此一键启动默认让 RAG embedding 使用 CPU。增加并发、改变 tensor split、扩大上下文或重新启用 GPU RAG 前，需要重新验证显存与延迟。

## 项目结构

```text
app/
  agent/                         旧结构化 Agent、策略与 math_checker
  api/                           旧 baseline FastAPI
  llm/deepseek_client.py         DeepSeek OpenAI-compatible 统一客户端
prototypes/llm_led_teacher/
  logic.py                       LLM-led 编排、规范化与审计
  orchestrator.py                九学科强制分类与六步 Agent Loop
  teaching_policy.py             从原始请求收紧 TeachingContract 权限
  boundary_audit.py              教学范围/披露边界的语义审计与保守复查
  math_markdown.py               MathJax 兼容 Markdown 公式规范化
  orchestration_tools.py         本地/联网/Skill 工具及确定性门禁
  orchestration_models.py        可持久化分类、动作和公开 trace 契约
  rag_hook.py                    纯本地和纯联网 RAG 适配层
  streaming.py                   审计后逐字符展示
  streamlit_app.py               当前 8502 页面
scripts/
  run_llm_led_teacher.py         本地模型 + Chroma + 当前网页一键启动
  run_warmed_streamlit.py        RAG 预热后启动 Streamlit
tests/prototypes/                Prototype 离线回归测试
llm-test/                        原混合 RAG、联网搜索和试卷评测
data/china_textbook_k9_vectordb/ 本地 Chroma 向量库
```

## 开发约束

- 先阅读 [`AGENTS.md`](AGENTS.md)、[`CLAUDE.md`](CLAUDE.md)、[`CODEX_HANDOFF.md`](CODEX_HANDOFF.md) 和 [`prototypes/llm_led_teacher/README.md`](prototypes/llm_led_teacher/README.md)。
- 修改行为必须先写失败测试，所有测试使用 fake client/mock LLM。
- verifier 支持域内的学生步骤对错由确定性 checker/verifier 裁决，不由 LLM 自行宣判；支持域外不得伪称已验证。
- Agent、RAG 和兼容 `turn_guidance` 都不能覆盖本地策略收紧后的 TeachingContract 或原题上下文。
- 不通过无限加长 system prompt 代替可测试的节奏不变式。
- `.env`、API key 和其他凭据不得提交或打印。

更详细的设计与交接信息见 [`CODEX_HANDOFF.md`](CODEX_HANDOFF.md)。
