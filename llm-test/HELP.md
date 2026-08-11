# 中考真题 LLM 测评系统 — 完整说明文档

## 目录

- [1. 系统概述](#1-系统概述)
- [2. 系统理论逻辑](#2-系统理论逻辑)
- [3. 系统架构](#3-系统架构)
- [4. 搭建流程](#4-搭建流程)
- [5. 使用方法](#5-使用方法)
- [6. 配置说明](#6-配置说明)
- [7. 输出结果](#7-输出结果)
- [8. 进阶指南](#8-进阶指南)

---

## 1. 系统概述

本系统用 LLM 自动完成中考真题的**答题**和**评分**，支持两种测评场景：

**场景一：全卷测评**（`evaluator.py`）

用一份完整的中考试卷，让"教师模型"逐题作答，"裁判模型"按评分标准打分。支持挂载 RAG 教材检索、对比有/无 RAG 的分数差异。覆盖数学、语文、英语、物理、化学、历史、道德与法治共 7 科。

**场景二：较难题批量测评**（`hard_question_eval.py`）

从题库 JSONL 中加载高频错题/较难题（如初中数学 56 道），批量推理、逐题评分。额外计算 RAG 命中率——判断检索到的教材/联网资料是否真的覆盖了题目答案。

**核心能力**

- 双模型协作：教师模型答题 + 裁判模型评分，角色分离保证公正性
- RAG 知识增强：本地向量库（ChromaDB + bge-m3）检索教材和题库片段，注入 prompt 辅助答题
- 智能联网回退：本地检索不足时自动触发 Tavily 联网搜索，搜索结果经分块+向量重排序后与本地结果合并，按相似度统一排序取 top-k
- 双模式对比：一键对比无 RAG vs 有 RAG 的得分，量化知识增强效果
- 本地模型支持：可选本地 DeepSeek-R1-Distill-Qwen-7B 作为教师模型

---

## 2. 系统理论逻辑

### 2.1 教师-裁判双角色分离

系统将"答题"和"评分"解耦为两个独立 LLM 调用：

| 角色 | 模型 | 任务 | 配置 |
|------|------|------|------|
| 教师 | deepseek-chat（或本地 R1-7B） | 阅读试卷/题目，给出答案 | temperature=0.3 |
| 裁判 | deepseek-v4-flash | 对比教师答案与参考答案，按标准打分 | temperature=0.0 |

评分使用 temperature=0 保证一致性，裁判始终运行在云端 API（不依赖本地），确保评分标准统一。

### 2.2 RAG 知识增强层

RAG 层为教师模型提供"开卷"能力。教师收到题目的同时会收到从知识库检索到的参考资料，作为答题辅助。

**本地检索（ChromaDB + bge-m3）**

系统连接 ChromaDB 向量数据库，包含两个集合：

- `k9_textbook`：K9 教材全文（按学科、学段、年级、章节组织）
- `k9_exercises`：K9 题库（按学科、题型、难度、知识点组织）

检索流程：题目文本 → 去除 LaTeX 公式 → 提取关键词 → 构造检索 query → bge-m3 向量化 → ChromaDB HNSW 近似最近邻搜索 → 返回 top-k 片段。

metadata 过滤能力：支持按 `subject`（学科）、`stage`（学段：小学/初中）条件过滤，减少无关检索噪音。

**联网回退（Tavily + bge-m3 重排序）**

当本地检索的命中质量不足时，自动触发联网搜索。完整流程如下：

```
Tavily API 搜索（advanced 模式，含 raw_content）
    → 排除低质量域名（scribd, doc88, baidu wenku 等）
    → 提取 AI 摘要 + 网页全文
    → 长网页按 300 字/块 切分（50 字重叠）
    → 所有文本块用 bge-m3 向量化
    → 与查询向量做余弦相似度计算
    → 取 top-3 最相关片段
```

**auto 模式触发与合并逻辑（核心）**

这是系统最关键的决策逻辑，决定了什么时候联网、本地和联网结果如何组合：

1. 先执行本地 ChromaDB 检索（教材 topk 条 + 题库 3 条）
2. 统计本地结果中相似度 ≥ 阈值（默认 0.65）的命中数
3. 若命中数 < topk → 触发联网搜索
4. 将与联网结果附加 `source="online"` 标记并计算 `sim=score`（余弦相似度）
5. 本地全部结果 + 联网全部结果 → 按 `sim` 降序统一排序 → 取 top-k
6. 联网高分结果可以挤掉本地低分结果；如果联网结果全部高于本地，本地会被全部替换

关键设计原则：

- **阈值仅用于判断是否联网**，不作为最终知识过滤条件。合并后以相似度排序为准
- **只搜一次，合并一次，不死循环**：无论结果好坏，一次联网搜索后即进入排序阶段
- **跨来源统一评分体系**：本地相似度（1-chroma_distance）和联网相似度（bge-m3 余弦点积）都在 0~1 范围内可比

**三种联网模式**

| 模式 | 命令参数 | 行为 |
|------|----------|------|
| off | 不加 `--online-rag` | 仅本地 ChromaDB 检索 |
| auto | `--online-rag auto` | 本地命中不足时联网，合并排序取 top-k |
| online | `--online-rag online` | 跳过本地检索，直接全程联网 |

### 2.3 题库同题命中判断

本地题库检索时额外计算一个"同题命中"标记（`is_exact_match`）：题库 top-1 相似度 ≥ 0.7 即判定为"在题库中找到了几乎一样的题目"。在 `hard_question_eval.py` 中，同题命中的题目会额外关注教师是否参考了题库给出的正确答案和解析。

### 2.4 RAG 命中率计算（仅较难题测评）

较难题测评使用双层判断机制评估 RAG 是否真的"命中了"题目答案：

- **字面命中（Literal Match）**：去 LaTeX/空白后，答案字符串直接出现在检索上下文中
- **语义命中（Semantic Match）**：用 bge-m3 计算「题目片段+答案」与「每个检索片段」的余弦相似度，最高 ≥ 0.65 即判定为命中

任一判断通过即为命中，两种命中数分别统计。

---

## 3. 系统架构

```
llm-test/
├── config.py              # 配置文件（API密钥、模型参数、RAG配置）
├── llm_client.py          # LLM 客户端封装（OpenAI接口 + llama-cpp本地模型）
├── evaluator.py           # 全卷测评主脚本（7科中考真题）
├── hard_question_eval.py  # 较难题批量测评脚本（JSONL题库）
├── rag_retriever.py       # RAG检索核心（ChromaDB教材+题库检索、auto联网决策）
├── online_rag.py          # 联网搜索模块（Tavily搜索→分块→bge-m3重排序）
├── question_rag.py        # 单题联网RAG推理（交互式/命令行）
├── chat.py                # DeepSeek-R1-7B 本地交互对话
├── split_exam.py          # 原始试卷拆分工具（题目/答案分离）
├── requirements.txt       # Python依赖
│
├── testmd/
│   ├── paper/             # 7套中考真题试卷（.md）
│   ├── key/               # 7套参考答案（.md）
│   └── json/              # 较难题JSONL数据
│
├── markdown/              # 原始MinerU转换的中考试卷
├── model/                 # 本地GGUF模型文件
└── results/               # 测评结果输出（.json + .md）
```

**模块依赖关系**

```
config.py  ← 所有模块的配置来源
    │
llm_client.py  ← 教师模型、裁判模型客户端
    │           ← LocalLLMClient（llama-cpp本地模型适配器）
    │
rag_retriever.py  ← ChromaDB 本地检索 + auto联网决策
    │                  ↓ 触发联网时调用
    │              online_rag.py  ← Tavily搜索 + 分块 + 重排序
    │
evaluator.py / hard_question_eval.py  ← 测评编排层
question_rag.py  ← 单题推理（独立使用）
chat.py  ← 本地对话（独立使用）
split_exam.py  ← 数据预处理（独立使用）
```

---

## 4. 搭建流程

### 4.1 环境准备

```bash
# 创建 conda 环境（推荐）
conda create -n rag python=3.11 -y
conda activate rag

# 安装 PyTorch（GPU 版本，根据 CUDA 版本选择）
# 以 CUDA 12.4 为例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖
cd /data/education/llm-test
pip install -r requirements.txt
```

依赖清单（`requirements.txt`）：

```
openai>=1.0.0              # OpenAI 兼容 API 客户端
tqdm>=4.64.0               # 进度条
chromadb>=0.5              # 向量数据库
sentence-transformers>=2.7 # bge-m3 嵌入模型
tavily-python>=0.3         # 联网搜索 SDK
llama-cpp-python           # 本地模型推理（可选，仅 --local 模式需要）
```

### 4.2 配置 API 密钥

编辑 `config.py`，填入教师模型和裁判模型的 API 信息：

```python
# 教师模型（负责答题）
TEACHER_MODEL = {
    "base_url": "https://api.deepseek.com",
    "api_key": "sk-xxxxxxxx",
    "model_name": "deepseek-chat",
    "temperature": 0.3,
    "max_tokens": 32768,
}

# 裁判模型（负责评分）
JUDGE_MODEL = {
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "sk-xxxxxxxx",
    "model_name": "deepseek-v4-flash",
    "temperature": 0.0,
    "max_tokens": 16384,
}
```

联网搜索需要配置 Tavily API key（在 [tavily.com](https://app.tavily.com) 免费注册获取）：

```python
ONLINE_RAG = {
    "mode": "auto",
    "local_threshold": 0.65,
    "tavily_api_key": "${TAVILY_API_KEY}",
    "search_topk": 5,
    "max_context_chars": 2400,
}
```

### 4.3 准备向量数据库

系统依赖外部的 ChromaDB Server + bge-m3 向量库。向量库与 `teacher-agent-demo` 项目共用，默认地址：

- Chroma Server：`127.0.0.1:8002`（可通过环境变量 `CHROMA_SERVER` 覆盖）
- 向量库路径：`/data/wyd-workspace/education/data/china_textbook_k9_vectordb`
- 嵌入模型：`/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3`

确保 Chroma Server 已启动且两个集合（`k9_textbook`、`k9_exercises`）存在。

### 4.4 准备试卷数据

试卷来自 MinerU 转换的 2026 年河南省中考真题 markdown 文件。原始文件位于 `markdown/` 目录。

运行拆分脚本将试卷分离为题面和答案：

```bash
python split_exam.py
```

输出：
- `testmd/paper/2026XX试卷.md` — 7 科试卷题面
- `testmd/key/2026XX试卷答案.md` — 7 科参考答案

### 4.5 准备本地模型（可选）

如果使用 `--local` 模式（本地 DeepSeek-R1-7B 答题），需要下载 GGUF 格式的模型文件到 `model/` 目录：

```bash
# 推荐 Q4_K_L 量化版本（4.74 GB）
# 从 HuggingFace 下载:
# https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF
```

配置文件中的本地模型路径：

```python
LOCAL_MODEL = {
    "model_path": "/data/education/llm-test/model/DeepSeek-R1-Distill-Qwen-7B-Q4_K_L.gguf",
    "n_ctx": 16384,
    "n_threads": 32,
}
```

---

## 5. 使用方法

### 5.1 全卷测评（evaluator.py）

**基本用法**

```bash
# 测评全部 7 科（无 RAG）
python evaluator.py

# 只测评数学
python evaluator.py 数学

# 挂载 RAG 教材检索
python evaluator.py --rag

# 数学 + RAG
python evaluator.py 数学 --rag

# 使用本地模型答题
python evaluator.py --local
python evaluator.py 数学 --local --rag
```

**联网 RAG**

```bash
# auto 模式：本地命中不足时自动联网
python evaluator.py 数学 --rag --online-rag auto

# online 模式：全程联网搜索
python evaluator.py 数学 --rag --online-rag online
```

**对比模式**

```bash
# 无 RAG vs 有 RAG 对比测评
python evaluator.py --compare

# 对比 + 本地模型 + 联网回退
python evaluator.py --compare --local --online-rag auto
```

### 5.2 较难题批量测评（hard_question_eval.py）

```bash
# 无 RAG，全部 56 题
python hard_question_eval.py

# 挂载 RAG
python hard_question_eval.py --rag

# auto 联网模式
python hard_question_eval.py --rag --online-rag auto

# 强制联网
python hard_question_eval.py --rag --online-rag online

# 本地模型
python hard_question_eval.py --local --rag

# 对比模式
python hard_question_eval.py --compare

# 查看详细 RAG 检索过程
python hard_question_eval.py --rag --online-rag auto --verbose

# 只跑前 5 题（调试用）
python hard_question_eval.py --rag --limit 5
```

### 5.3 单题推理（question_rag.py）

用于快速测试单道题目的联网 RAG 效果：

```bash
# 基础用法
python question_rag.py "已知x=2是方程x²-mx=6的一个根，则m的值为" -s 数学

# 对比模式：同时输出无 RAG 和联网 RAG 的答案
python question_rag.py "简述光合作用的过程" -s 生物 --compare

# 使用本地模型
python question_rag.py "勾股定理的应用" -s 数学 --local

# 交互模式
python question_rag.py -s 数学
# 然后粘贴题目，输入 END 结束
```

### 5.4 本地对话（chat.py）

直接与本地 DeepSeek-R1-7B 进行多轮交互对话：

```bash
python chat.py
```

命令快捷键：
- `/think` — 切换显示/隐藏模型的思考过程（`<think>` 块）
- `/clear` — 清空对话历史
- `quit` / `exit` — 退出

---

## 6. 配置说明

### 6.1 模型配置（config.py）

**TEACHER_MODEL** — 教师模型

| 参数 | 说明 | 默认值 |
|------|------|--------|
| base_url | API 地址 | `https://api.deepseek.com` |
| api_key | API 密钥 | 需填写 |
| model_name | 模型名称 | `deepseek-chat` |
| temperature | 随机性（答题适当降低） | 0.3 |
| max_tokens | 最大输出 tokens | 32768 |

**JUDGE_MODEL** — 裁判模型

| 参数 | 说明 | 默认值 |
|------|------|--------|
| base_url | API 地址 | `https://api.deepseek.com/v1` |
| api_key | API 密钥 | 需填写 |
| model_name | 模型名称 | `deepseek-v4-flash` |
| temperature | 评分严格一致 | 0.0 |
| max_tokens | 最大输出 tokens | 16384 |

### 6.2 RAG 配置

**RAG_CONFIG** — 本地教材检索

| 参数 | 说明 | 默认值 |
|------|------|--------|
| enabled | 是否启用 | True |
| topk | 教材检索条数 | 5 |
| max_context_chars | 注入 prompt 的最大字符数 | 2400 |
| chroma_server | Chroma Server 地址 | `127.0.0.1:8002` |
| collections | 检索集合 | `["k9_textbook", "k9_exercises"]` |

**ONLINE_RAG** — 联网搜索

| 参数 | 说明 | 默认值 |
|------|------|--------|
| mode | 模式：off/auto/online | auto |
| local_threshold | 触发联网的相似度阈值 | 0.65 |
| tavily_api_key | Tavily API key | 需填写 |
| search_topk | 联网搜索返回条数 | 5 |
| max_context_chars | 联网上下文最大字符数 | 2400 |

### 6.3 命令行参数对照

| 参数 | 作用 | 适用脚本 |
|------|------|----------|
| `--rag` | 启用 RAG 教材检索 | evaluator / hard_question_eval |
| `--online-rag MODE` | 联网模式（auto/online） | 同上 |
| `--compare` | 对比模式（无RAG vs 有RAG） | 同上 |
| `--local` | 使用本地模型答题 | 同上 |
| `--limit N` | 限制题目数（调试用） | hard_question_eval |
| `--verbose` | 打印完整 RAG 检索过程 | hard_question_eval |
| `--help` / `-h` | 查看帮助 | 全部 |

---

## 7. 输出结果

每次测评生成两个文件，保存在 `results/` 或 `results/hard_questions/` 目录下：

### 7.1 JSON 文件（结构化数据）

包含完整的原始数据：
- 教师答案原文
- 裁判评分原文
- RAG 检索引用列表
- 得分、总分、均分
- 测评时间戳

### 7.2 Markdown 文件（人类可读）

全卷测评报告包含：
- 测评时间和 RAG 模式
- 教材参考来源列表
- 教师答案（完整作答）
- 裁判评分（评分明细 + 总分统计 + 综合评价）

较难题测评报告包含：
- 每道题的题干、参考答案、RAG 参考来源
- 教师答案和裁判评分理由
- RAG 命中率统计（字面命中 + 语义命中）
- `<details>` 折叠块展示传给模型的实际检索内容
- `<details>` 折叠块展示裁判的完整评分原文

### 7.3 控制台输出

测评运行中实时显示：
- 进度条（当前科目/题号、已完成数、预计剩余时间）
- 每题得分和评分理由摘要
- RAG 检索命中信息
- 最终汇总：按科目的得分表、对比表（差值）

---

## 8. 进阶指南

### 8.1 调整联网触发阈值

阈值越高，本地结果越"挑剔"，越容易触发联网：

```python
# config.py
ONLINE_RAG["local_threshold"] = 0.65  # 默认。0.7=更严格，0.5=更宽松
```

### 8.2 调整 RAG 上下文长度

增大 `max_context_chars` 可以注入更多资料，但注意模型上下文窗口限制：

```python
RAG_CONFIG["max_context_chars"] = 3600    # 默认 2400
ONLINE_RAG["max_context_chars"] = 3600    # 默认 2400
```

### 8.3 更换教师/裁判模型

系统通过 OpenAI 兼容接口调用，支持任何兼容的 API：

```python
# 使用 GPT-4o
TEACHER_MODEL = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxxxx",
    "model_name": "gpt-4o",
    ...
}

# 使用任何 OpenAI 兼容服务（如 vLLM、Ollama 等）
TEACHER_MODEL = {
    "base_url": "http://localhost:8000/v1",
    "api_key": "not-needed",
    "model_name": "qwen2.5-72b",
    ...
}
```

### 8.4 添加新科目

1. 将原始试卷放入 `markdown/` 目录（MinerU 转换的 md 格式）
2. 运行 `python split_exam.py` 生成题面和答案文件
3. 在 `config.py` 的 `SUBJECT_MAP` 中添加科目名映射
4. 确保 `testmd/paper/` 中有对应的试卷文件，`testmd/key/` 中有对应的答案文件

### 8.5 接入自己的向量库

如果需要使用自建的 ChromaDB 向量库：

```bash
# 设置环境变量指定 Chroma Server 地址
export CHROMA_SERVER=192.168.1.100:8002
```

或在 `rag_retriever.py` 中修改默认值：

```python
DEFAULT_VECTORDB_PATH = "/your/custom/vectordb/path"
DEFAULT_COLLECTION = "your_collection_name"
```

### 8.6 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| RAG 不可用 | Chroma Server 未启动 | 启动 Chroma Server：`chroma run --path /path/to/vectordb --port 8002` |
| 联网不可用 | Tavily API key 未配置 | 编辑 config.py 填入 key |
| 模型返回空 | max_tokens 不够 | 增大 max_tokens 或缩短试卷 |
| 评分截断 | 裁判输出超长 | 增大 JUDGE_MODEL 的 max_tokens |
| 本地模型 OOM | 显存/内存不足 | 换用 Q4_K_L 量化版本，减小 n_ctx |
| 网络超时 | API 限流 | LLMClient 内置 3 次指数退避重试 |
