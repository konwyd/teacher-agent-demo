# LLM-led Teacher：中文 K9 多学科教师原型

这是一个面向中文小学、初中九学科的研究型教学 Agent。当前主入口是运行在 `8502` 的 **LLM-led Teacher Prototype**：DeepSeek 负责教学表达，有界 Agent Loop 负责学科识别、Skill 选择和工具调用，本地策略、确定性 verifier 与语义边界审计共同约束回答范围。

项目的目标不是让模型无条件一次性给出答案，而是让每轮回复服从明确的教学契约：概念问题只解释概念，普通求解只推进允许的步骤，只有学生明确索要完整解答时才开放最终答案。

> 当前默认演示是 `8502`。旧结构化 Teacher Agent 仍保留为研究 baseline（Streamlit `8501`、FastAPI `8000`），但不是同一条运行链路。

## 快速开始

### 环境要求

- Python 3.10+
- 本地 bge-m3 模型与 Chroma 向量库
- CUDA 版 llama.cpp、本地 DeepSeek-V4 Flash GGUF 和对应模型启动脚本
- 当前默认参数按 8×RTX 4080（每卡 16 GiB）环境设计；其他硬件需要重新验证显存、上下文和并发参数

安装项目依赖：

```bash
cd /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo
pip install -e ".[dev,ui]"
pip install -r llm-test/requirements.txt
pip install jieba rank-bm25
```

### 一键启动本地版本

```bash
cd /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo
python3 scripts/run_llm_led_teacher.py
```

启动器会依次：

1. 在 `8080` 启动本地 DeepSeek-V4，或复用模型别名匹配的健康服务；
2. 验证 `/health`、`/v1/models` 和最小真实推理；
3. 在 `8002` 启动或复用 Chroma Server；
4. 预热 bge-m3 检索模型；
5. 在 `8502` 启动 Streamlit 页面；
6. 退出时关闭由本次启动器创建的子进程。

启动成功后访问：<http://localhost:8502>

查看全部启动参数：

```bash
python3 scripts/run_llm_led_teacher.py --help
```

常用覆盖示例：

```bash
python3 scripts/run_llm_led_teacher.py \
  --model-port 8080 \
  --rag-port 8002 \
  --app-port 8502 \
  --ctx-size 524288 \
  --parallel 4 \
  --unified-kv \
  --continuous-batching \
  --batch-size 128 \
  --ubatch-size 128 \
  --db-path data/china_textbook_k9_vectordb
```

不要直接沿用旧 README 中的 `131072`、`262144` 或 `parallel=1` 配置；当前启动器默认值以 `python3 scripts/run_llm_led_teacher.py --help` 和脚本内常量为准。

## 当前能力

- 九学科：数学、语文、英语、物理、化学、生物、历史、地理、道法
- 有界 Agent Loop：强制分类学科、学段、新题/追问，并按需选择规范 Skill 与工具
- 教学契约：`concept_explanation`、`one_step`、`guided_solution`、`full_solution`、`brief_redirect`
- 回答边界：本地策略根据学生原始请求收紧题目应用权和答案披露权
- 确定性验证：当前 verifier 支持 Fraction 一元一次方程；超出支持域时明确记录不可验证，不伪造证据
- 安全展示：初稿完成边界审计和必要修正后，才逐字符显示给学生
- 混合检索：bge-m3 + Chroma + BM25 + RRF，可选且受门禁约束的 Tavily 联网
- 登录与会话：本地账号、会话历史、跨账号隔离、Agent trace 和教学契约持久化
- 学习连续性：同一账号的跨会话承接、结构化学生档案和主题记忆；记忆始终是软参考
- 长对话：完整历史保存在 SQLite，工作上下文接近窗口时对旧轮次做有界语义压缩
- 本地/API 双后端：同一教学工作流可切换本地 llama.cpp 或托管 DeepSeek API

## 单轮处理流程

```text
学生消息
  ↓
恢复会话、KV 状态与有界学习记忆；必要时压缩旧工作上下文
  ↓
强制分类学科、学段、是否新题
  ↓
有界 Agent Loop 按需加载 Skill、检索本地知识，必要时受控联网
  ↓
提交 TeachingContract：目标、模式、允许/禁止内容、题目应用与答案披露权限
  ↓
本地策略按学生原始请求收紧契约；可选 verifier 检查支持域内的数学步骤
  ↓
DeepSeek 生成教师初稿
  ↓
语义边界审计；越界时按公开修正指令重写并执行高置信字面复查
  ↓
规范化 MathJax Markdown，保存对话、契约、公开 trace、审计与记忆回写任务
  ↓
只展示最终安全回答
```

语义边界审计检查的是回答是否服从教学目标、题目应用权限、答案披露权限和步骤范围，**不是通用事实核验器，也不是数学正确性的完整证明**。

## 教学边界

- `concept_explanation` 只讲通用概念，不代入当前题，也不披露当前题答案。
- 普通求解只推进契约允许的范围；学生明确要求“完整步骤和答案”时才允许 `full_solution`。
- 旧兼容字段 `single_move` / `complete_response` 从 `TeachingContract` 派生，不能反向放宽权限。
- verifier 支持域内的步骤对错由确定性 checker 裁决；支持域外记录 `not_applicable` / `unable_to_verify`。
- RAG、Skill、历史对话与学生记忆都是不可信软参考，不能覆盖当前消息、active problem、教学契约或确定性验证。
- 非 K9 请求不访问本地教材库或联网，只做简短回应并引导回学习内容。

## 模型后端配置

复制示例配置并只在本地填写真实凭据：

```bash
cp .env.example .env
```

可以一次配置两个 profile，再通过 `TEACHER_LLM_BACKEND` 切换：

```dotenv
TEACHER_LLM_BACKEND=local

TEACHER_LOCAL_LLM_API_KEY=local
TEACHER_LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1
TEACHER_LOCAL_LLM_MODEL=deepseek-v4-flash
TEACHER_LOCAL_LLM_CONTEXT_WINDOW_TOKENS=524288

TEACHER_API_LLM_API_KEY=
TEACHER_API_LLM_BASE_URL=https://api.deepseek.com
TEACHER_API_LLM_MODEL=deepseek-chat
TEACHER_API_LLM_CONTEXT_WINDOW_TOKENS=131072
```

- `TEACHER_LLM_BACKEND=local`：使用本地 profile。
- `TEACHER_LLM_BACKEND=api`：使用托管 API profile，会产生外部请求和费用。
- 留空：沿用兼容的 `DEEPSEEK_*` 配置。

一键启动器负责启动本地模型并设置本地 `DEEPSEEK_*` 地址。若 `.env` 中显式设置了 `TEACHER_LLM_BACKEND=api`，profile 选择仍会优先，因此启动本地演示前请设为 `local` 或留空，避免误用付费 API。

对比两个后端时可使用固定题集工具：

```bash
python3 scripts/run_backend_comparison.py --backend both --repeats 3
```

结果默认写入 `logs/backend_comparison/`；该目录不应提交。

## 512K unified-KV、并发与上下文

当前本地启动器默认配置：

| 配置 | 默认值 | 说明 |
|---|---:|---|
| 总上下文 | `524288` | 单会话逻辑上限与共享物理 KV 池大小 |
| 并发槽 | `4` | 四个逻辑 slot 共享 unified KV，不是每槽固定分到 131K |
| unified KV | 开启 | 活跃序列竞争同一个物理 KV 池 |
| continuous batching | 开启 | 支持多个请求并行推进 |
| RAM prompt cache | `65536 MiB` | llama.cpp 空闲 slot 缓存 |
| K/V cache 类型 | `q8_0` / `q4_0` | 当前显存折中 |
| batch / ubatch | `128 / 128` | 当前验证配置 |
| SSD KV 预算 | `100 GiB` | 80 GiB 开始清理，降至 70 GiB |

四个请求可以并发解码，但它们的活跃 KV 总量仍必须装入共享的 524,288-token 物理池。`parallel=4` 不代表四个会话都能同时占满 512K；高压场景可能排队或由 llama-server 拒绝。

缓存分为四层：

1. GPU unified KV：最快，模型服务退出后消失；
2. 64 GiB RAM prompt cache：保存空闲 slot，机器或服务重启后消失；
3. `data/runtime/kv-states/`：按用户和会话保存兼容性校验后的 SSD 快照；
4. `data/runtime/conversations.sqlite3`：完整对话的持久化事实来源。

KV 快照丢失、过期或不兼容只影响恢复速度，不应丢失对话；系统会从 SQLite 文本重建。可运行基础并发验证：

```bash
python3 scripts/validate_unified_kv.py --base-url http://127.0.0.1:8080/v1 --rounds 2
```

完整历史始终保存在 SQLite。模型工作上下文接近窗口时，系统将较旧的完整轮次合并为有界语义摘要，保留最近完整轮次和当前消息。默认触发比例为 `0.80`，目标比例为 `0.55`，最少保留最近 `12` 轮；摘要失败时使用有界降级记录，不能阻断当前教学。

## 登录、会话与学习记忆

首次进入 `8502` 会显示登录/注册页面。用户名不能重复，密码至少 8 位；本地账号保存在 `data/runtime/users.sqlite3`，只保存加盐哈希。

完整对话保存在 `data/runtime/conversations.sqlite3`。查询同时校验内部学生 ID 和会话 ID，不同账号不能读取彼此记录。历史详情可展示公开 Agent/Tool 事件、TeachingContract、审计、verifier、上下文压缩与 KV 状态，但不保存或展示模型思维链、私有 prompt 或私有 tool details。

学习记忆包含两类：

- 跨会话承接：仅在“上次、继续、这个”等明确承接表达出现时，检索同一账号的相关历史。
- 结构化学生记忆：保存称呼、讲解偏好、主题进度、困难和建议下一步，并通过异步 outbox 回写。

记忆有歧义时应先澄清；记忆故障不能阻断回答。任何历史文字都按不可信数据处理，不能作为系统指令执行。

## RAG 与联网边界

```text
学生题目
  ├─ bge-m3：在应用进程生成查询向量，默认使用 CPU
  ├─ Chroma Server：查询本地 HNSW 索引
  ├─ BM25：关键词召回
  ├─ RRF：融合语义与关键词排序
  └─ Agent 确认存在具体证据缺口时：门禁允许后至多调用一次 Tavily
```

- 浏览器不下载向量库或 embedding 模型，所有检索都在服务端完成。
- Chroma 当前主要使用 CPU；bge-m3 可指定 GPU，但默认 `TEACHER_RAG_DEVICE=cpu`，避免与 8-GPU 模型争抢显存。
- 本地工具返回 raw vector、BM25、RRF 与融合分，供 Agent 判断证据质量；不会因低分自动联网。
- 联网只允许 K9 请求，且本轮必须已成功执行本地检索并给出具体证据缺口；每轮最多一次。
- 检索内容不能改变分类、TeachingContract 或 active problem，与当前题无关时应忽略。

系统使用两个逻辑隔离的知识库：

- 教材题库库：保留原有教材、题目和标准知识检索逻辑；
- 优质问答库：独立 Chroma collection，只存夜间任务从历史对话中提炼的可复用问答。

请求分类器独立输出“是否参考优质问答”开关；没有打开该开关时，即使优质问答库刚发布新版本也不会执行向量检索。打开后，历史回答只作为生成参考，不能直接返回、不能覆盖当前题、教学契约或确定性验证。正命中和空结果缓存都绑定优质问答库版本，新版本发布后才重新检索。

该功能默认关闭。随一键启动器启用在线检索和 UTC 0–2 点夜间任务：

```bash
python3 scripts/run_llm_led_teacher.py --quality-qa-nightly
```

也可以手动执行一次增量更新：

```bash
python3 scripts/update_quality_qa.py
```

任务按会话 `updated_at` 水位扫描，只把结构化质量筛选、隐私与提示注入检查通过的内容入库；相同来源会幂等替换，删除已失效条目，并执行内容指纹和语义去重。模型使用过优质问答参考的回答不会再次回流。只有实际内容变化才提升库版本。

## 端口

| 端口 | 服务 | 是否需要公开 |
|---:|---|---|
| `8502` | 当前 LLM-led Teacher 网页 | 是，用户入口 |
| `8002` | Chroma 内部 HTTP API | 否 |
| `8080` | 本地 llama.cpp OpenAI-compatible API | 否 |
| `8501` | 旧结构化 Agent Streamlit | 仅展示 baseline 时 |
| `8000` | 旧结构化 Agent FastAPI | 仅独立调用 baseline 时 |

通过 VS Code 端口转发或反向代理发布时通常只公开 `8502`。正式对外使用应增加 HTTPS、稳定域名、反向代理和访问控制；当前账号系统是本地研究演示，不包含短信验证、找回密码或公网生产级安全防护。

## 测试与评测

默认自动化测试使用 fake client/mock LLM，不调用真实付费 API：

```bash
python3 -m pytest
python3 scripts/run_eval.py
```

Prototype 聚焦回归：

```bash
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
python3 -m pytest tests/prototypes/test_llm_led_streaming.py -q
python3 -m pytest tests/prototypes/test_context_compaction.py -q
python3 -m pytest tests/test_kv_cache.py -q
```

真实模型烟测会产生较高算力成本：

```bash
python3 scripts/run_llm_led_agent_smoke.py --live
```

主要回归不变式：

- 概念解释不能代入 active problem 或披露最终答案；
- 只有明确索要完整解答时才允许 `full_solution`；
- verifier 支持域内的结论不能被 LLM 推翻，支持域外不能伪称已验证；
- Skill、RAG、旧 guidance 和记忆都不能放宽 TeachingContract；
- 边界修正后仍服从步骤范围和结尾提问约束；
- 九学科新题可切换，追问继承当前学科与学段；
- 非 K9 请求和未先查本地的联网动作会被门禁拒绝；
- KV、压缩、记忆或缓存故障不会造成对话丢失或阻断正常教学。

## 旧结构化 baseline

```bash
python3 -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
streamlit run app/ui/streamlit_app.py --server.port 8501
```

旧 baseline 用于研究 LangGraph workflow、mastery memory、math_checker、evaluation 与 Pedagogical Graph RAG。当前 LLM-led Prototype 不经过相同的 TeachingPlanner 流程，不要把两个入口视为同一实现。

## 可选：化学分子结构图增强

MolScribe 1.1.1 要求 `torch<2.0`，必须使用隔离环境，不能安装进当前 PyTorch 2.x 项目环境：

```bash
bash scripts/install_molscribe_env.sh

.venv-molscribe/bin/python scripts/enrich_molecular_structures.py \
  data/china_textbook_k9_chunks/chunks.jsonl \
  data/china_textbook_k9_chunks/chunks.molscribe.jsonl \
  --image-root /实际的图片存储根目录 \
  --device cuda:0
```

该工具只增强命中结构式相关关键词的化学图片，不覆盖原始 OCR 或 `metadata.images`。RDKit 解析失败、原子数不足或置信度低于 `0.80` 的预测会被丢弃。

## 项目结构

```text
app/
  agent/                         旧结构化 Agent、策略、checker/verifier
  api/                           旧 baseline FastAPI
  auth/                          本地账号
  llm/                           DeepSeek 客户端与 KV 持久化
  memory/                        会话归档、跨会话与结构化学生记忆
prototypes/llm_led_teacher/
  logic.py                       当前回合编排、生成、审计与持久化
  orchestrator.py                九学科分类和有界 Agent Loop
  teaching_policy.py             TeachingContract 权限收紧
  boundary_audit.py              教学范围与披露边界审计
  orchestration_tools.py         Skill、本地/联网工具与门禁
  context_compaction.py          长对话工作上下文压缩
  qa_cache.py                    独立的版本化优质问答参考库
  streamlit_app.py               当前 8502 页面
scripts/
  run_llm_led_teacher.py         本地模型 + Chroma + 页面一键启动
  run_backend_comparison.py      本地/API 固定题集对比
  validate_unified_kv.py         unified-KV 并发验证
  run_eval.py                    离线教学评测
tests/                           离线单元、集成与行为回归
llm-test/                        混合 RAG、联网搜索和试卷评测
data/china_textbook_k9_vectordb/ 本地 Chroma 向量库
data/runtime/                    账号、会话、KV 等运行时数据（不提交）
```

## 开发约束

- 开始修改前阅读 [`AGENTS.md`](AGENTS.md)、[`CLAUDE.md`](CLAUDE.md)、[`CURRENT_WORK_HANDOFF.md`](CURRENT_WORK_HANDOFF.md) 和 [`prototypes/llm_led_teacher/README.md`](prototypes/llm_led_teacher/README.md)。
- 修改行为时先补失败测试；所有默认测试必须保持离线。
- 不绕过确定性 checker/verifier，不让记忆或 RAG 成为硬流程控制。
- 不通过无限加长 prompt 代替可测试的教学不变式。
- 不提交 `.env`、API key、`logs/`、SQLite、KV 快照或其他运行时数据。
- 合入前至少执行 `python3 -m pytest`、`python3 scripts/run_eval.py` 和 `git diff --check`。

最新阶段状态优先查看 [`CURRENT_WORK_HANDOFF.md`](CURRENT_WORK_HANDOFF.md)、[`progress.md`](progress.md) 和实际代码；较早的 [`CODEX_HANDOFF.md`](CODEX_HANDOFF.md) 与 [`HANDOFF.md`](HANDOFF.md) 仅作为历史背景。
