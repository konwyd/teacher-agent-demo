# Teacher Agent RAG 知识库部署与对接手册

> 本文档自包含，读者按顺序操作即可完成 RAG 知识库的挂载与对接。
> 适用版本：chromadb 1.5.9 / sentence-transformers 5.6.0 / bge-m3
> 最后更新：2026-07-15

---

## 目录

1. [RAG 架构总览](#一rag-架构总览)
2. [知识库集合详解](#二知识库集合详解)
   - [2.1 k9_textbook — K9 全科教材](#21-k9_textbook--k9-全科教材)
   - [2.2 k9_exercises — 题库（M3KE）](#22-k9_exercises--题库m3ke)
   - [2.3 curriculum_std — 课程标准](#23-curriculum_std--课程标准)
   - [2.4 teacher_cert — 教师资格证](#24-teacher_cert--教师资格证)
3. [环境前置条件](#三环境前置条件)
4. [部署 Chroma Server](#四部署-chroma-server)
5. [应用层挂载配置](#五应用层挂载配置)
6. [检索逻辑详解](#六检索逻辑详解)
7. [挂载验证（分步）](#七挂载验证分步)
8. [使用方式](#八使用方式)
9. [故障排查](#九故障排查)
10. [附录](#十附录)

---

## 一、RAG 架构总览

Teacher Agent 的 RAG 系统由 **4 个知识库集合** 构成，统一存储在同一个 ChromaDB 向量库中，通过 Chroma Server 对外提供检索服务。**已支持多模态图文混合检索**（文本向量匹配 + 图片路径返回）。

```
┌─────────────────────────────────────────────────────────────┐
│                    Teacher Agent 应用                        │
│  app/agent/workflow.py → retrieve_node()                    │
│                                                              │
│  检索链路（每次 /turn 调用触发）:                              │
│    1. MultimodalRAG → k9_textbook（教材原文+图片）top-5          │
│    2. KBManager    → k9_exercises（题库）top-3                │
│    3. KBManager    → curriculum_std（课标）top-2              │
│    4. KBManager    → teacher_cert（教资）top-2                │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (127.0.0.1:8002)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  ChromaDB Server（PID 长驻）                  │
│  chroma run --path <vectordb> --port 8002 --host 127.0.0.1  │
└────────────────────────┬────────────────────────────────────┘
                         │ 读写本地文件
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        向量库目录 (1.1 GB)                                    │
│  /data/wyd-workspace/education/data/china_textbook_k9_vectordb│
│  ├── chroma.sqlite3          (元数据 + 向量)                  │
│  └── {segment-uuid}/         (HNSW 索引段)                    │
│                                                              │
│  集合一览：                                                    │
│    k9_textbook     62,772 条  K9 全科教材（含图片55.9%）        │
│    k9_exercises    20,122 条  题库（M3KE 公开基准）            │
│    curriculum_std      18 条  课程标准                        │
│    teacher_cert        50 条  教师资格证                      │
│    ──────────────────────────                                │
│    合计             82,962 条                                │
│                                                              │
│  多模态能力：                                                  │
│    📷 教材图片 53,957 张，关联 21,443 个 chunk (55.9%)        │
│    检索时文本匹配 → 返回文字 + 图片 URL → 前端渲染             │
└─────────────────────────────────────────────────────────────┘
```

**嵌入模型**：BAAI/bge-m3（1024 维，多语言中文强），所有集合共享同一套嵌入函数，进程级单例仅加载一次（约 2.2 GB 显存/内存）。

**检索方式**：cosine 相似度（向量归一化后的内积），HNSW 近似最近邻索引。

---

## 二、知识库集合详解

### 2.1 k9_textbook — K9 全科教材

#### 基本信息

| 项目 | 内容 |
|------|------|
| **集合名** | `k9_textbook` |
| **记录数** | 62,772 条 |
| **数据来源** | 人教版 K9（小学+初中）全套教材 PDF，共 130 本 |
| **来源地址** | 本地 PDF 文件：`/data/wyd-workspace/education/data/china_textbook_k9/` |
| **数据性质** | 教材原文片段（非题目） |
| **检索代码** | `app/retrieval/textbook_rag.py` — `TextbookRAG` 类 |
| **默认 top-k** | 5 |
| **学科过滤** | 默认 `subject=数学`（`TEACHER_RAG_SUBJECT_FILTER`） |

#### 内容说明

覆盖 19 个学科，小学 26,244 条 + 初中 36,528 条：

| 学科 | chunks | 学科 | chunks |
|------|--------|------|--------|
| 科学 | 12,012 | 美术 | 1,419 |
| 英语 | 8,613 | 生物学 | 1,375 |
| 数学 | 6,470 | 历史 | 1,318 |
| 体育与健康 | 4,519 | 物理 | 1,049 |
| 音乐 | 4,340 | 化学 | 991 |
| 语文 | 4,175 | 地理 | 719 |
| 日语 | 3,506 | 人文地理 | 458 |
| 俄语 | 3,356 | 地理图册 | 428 |
| 语文·书法练习指导 | 2,704 | — | — |
| 道德与法治 | 2,690 | — | — |
| 艺术 | 2,630 | — | — |

#### 处理流程

```
130 本 PDF
  │ ① MinerU 解析（主流学科）
  │ ② PyMuPDF 解析（补缺学科：语文/历史/科学/道法/艺术）
  ▼
Markdown（520 文件）
  │ ③ 分块：按 ## 标题切章节，单节 > 400 字再滑窗切分（重叠 80 字）
  ▼
chunks.jsonl（47 MB, 36,058 块）+ missing_subjects_chunks.jsonl（36 MB）
  │ ④ 向量化入库：bge-m3 编码 → ChromaDB
  ▼
k9_textbook 集合（62,772 条）
```

#### 每条 chunk 结构

```json
{
  "text": "解一元一次方程的一般步骤包括：去分母、去括号、移项...",
  "metadata": {
    "stage": "初中",            "subject": "数学",
    "version": "人教版",         "grade": "七年级",
    "semester": "上册",          "chapter": "归纳",
    "chapter_idx": 5,
    "stem": "义务教育教科书·数学七年级上册",
    "source_pdf": "...",         "images": "fig1.png"
  }
}
```

#### 检索时的格式化

`format_context(hits)` 输出示例：
```
[教材参考1] 初中数学 七年级上册 《归纳》
解一元一次方程的一般步骤包括：去分母、去括号、移项、合并同类项、系数化为1...
```

`format_refs(hits)` 输出示例：
```
["数学·七年级上册《归纳》(相似度0.74)", ...]
```

---

### 2.2 k9_exercises — 题库（M3KE）

#### 基本信息

| 项目 | 内容 |
|------|------|
| **集合名** | `k9_exercises` |
| **记录数** | 20,122 条 |
| **数据来源** | **M3KE**：天津大学 NLP 实验室 + 华为诺亚方舟实验室，CC BY-NC-SA 4.0 |
| **来源地址** | GitHub: `https://github.com/tjunlp-lab/M3KE`<br>HuggingFace: `https://huggingface.co/datasets/TJUNLP/M3KE` |
| **论文** | arXiv:2305.10263（2023） |
| **数据性质** | 全部为**四选一单选题**，覆盖 71 个学科，小学到大学 |
| **检索代码** | `app/retrieval/kb_manager.py` — `KBManager.retrieve_exercises()` |
| **默认 top-k** | 3 |
| **学科过滤** | 默认 `subject=Math`（检索时按学科名过滤） |

#### 内容说明

| 学段 | 题目数 | 示例学科 |
|------|:---:|------|
| 小学 | 681 | Chinese (语文), Math (数学) |
| 初中 | 2,167 | Math, Physics, Chemistry, Biology, Geography, Chinese, History, Politics |
| 高中 | 1,959 | Math, Physics, Chemistry, Biology, Geography, Chinese, History, Politics |
| 大学 | 12,366 | 高等数学、线性代数、概率论、操作系统、计算机网络、数据结构、内科学、外科学、药理学、免疫学、法理学、刑法学、经济学、管理学 等 |
| 其他 | 2,949 | 公务员考试、计算机等级考试、中医学、古代汉语、电影、音乐、舞蹈、美术、宗教 等 |

#### 数据格式（原始 M3KE）

```json
{
  "id": 0,
  "question": "下列等式中，不是整式的是?",
  "A": "x- 1/2y",
  "B": "3/7x",
  "C": "1/(x-1)",
  "D": "0",
  "answer": "C"
}
```

#### 入库后的 chunk 结构

```json
{
  "text": "【题目】下列等式中，不是整式的是?\nA. x- 1/2y\nB. 3/7x\nC. 1/(x-1)\nD. 0\n【答案】C",
  "metadata": {
    "doc_type": "exercise",
    "subject": "Math",
    "ques_type": "单选题",
    "difficulty": "",
    "knowledges": "",
    "source": "M3KE",
    "source_file": "Math-Natural Sciences-Junior high school",
    "question_id": 0,
    "has_answer": true
  }
}
```

#### 检索时的格式化

`kb.format_context(hits)` 输出示例：
```
[题库1] 题库·Math《》
【题目】下列等式中，不是整式的是?
A. x- 1/2y
B. 3/7x
C. 1/(x-1)
D. 0
【答案】C
```

`kb.format_refs(hits)` 输出示例：
```
["题库·Math(相似度0.58)", ...]
```

#### 入库脚本

`/tmp/ingest_m3ke.py` — 从 GitHub raw 下载 `data/M3KE.zip`，解析 test 集全部 20,122 题，从 dev 集补充答案（355 条答案模板），批量写入 `k9_exercises` 集合。

---

### 2.3 curriculum_std — 课程标准

#### 基本信息

| 项目 | 内容 |
|------|------|
| **集合名** | `curriculum_std` |
| **记录数** | 18 条 |
| **数据来源** | 《义务教育数学课程标准（2022年版）》 |
| **来源地址** | 本地 Markdown：`/data/wyd-workspace/education/data/teacher_kb_source/课标_义务教育数学2022.md` |
| **数据性质** | 课程标准的章节片段 |
| **检索代码** | `app/retrieval/kb_manager.py` — `KBManager.retrieve(collection="curriculum_std")` |
| **默认 top-k** | 2 |

#### 内容覆盖

| 章节 | 条数 |
|------|:--:|
| 课程性质与基本理念 | 2 |
| 核心素养的内涵 | 1 |
| 核心素养的具体表现 | 2 |
| 课程目标 | 2 |
| 课程内容——数与代数 | 2 |
| 课程内容——图形与几何 | 2 |
| 课程内容——统计与概率 | 1 |
| 课程内容——综合与实践 | 1 |
| 学业质量标准 | 1 |
| 教学建议 | 2 |
| 评价建议 | 1 |
| 前言/概述 | 1 |

#### 数据格式

```json
{
  "text": "## 课程目标\n义务教育数学课程总目标：通过义务教育阶段的数学学习，学生能获得适应社会生活和进一步发展所必需的数学基础知识、基本技能、基本思想、基本活动经验...",
  "metadata": {
    "doc_type": "curriculum_std",
    "chapter": "课程目标",
    "chapter_idx": 1,
    "source_file": "课标_义务教育数学2022.md",
    "source_type": "课程标准",
    "subject": "数学",
    "stage": "义务教育"
  }
}
```

#### 入库脚本

`scripts/build_kb_from_md.py` — 读取 MD 文件，按 `##` 标题切分，> 500 字再切（重叠 100 字），写入 `curriculum_std` 集合。

---

### 2.4 teacher_cert — 教师资格证

#### 基本信息

| 项目 | 内容 |
|------|------|
| **集合名** | `teacher_cert` |
| **记录数** | 50 条 |
| **数据来源** | 教师资格证考试大纲 |
| **来源地址** | 本地 Markdown（3 个文件）：<br>`/data/wyd-workspace/education/data/teacher_kb_source/教资_综合素质.md`<br>`/data/wyd-workspace/education/data/teacher_kb_source/教资_教育知识_中学.md`<br>`/data/wyd-workspace/education/data/teacher_kb_source/教资_教育知识_小学.md` |
| **数据性质** | 教资考试知识点片段 |
| **检索代码** | `app/retrieval/kb_manager.py` — `KBManager.retrieve(collection="teacher_cert")` |
| **默认 top-k** | 2 |

#### 内容覆盖

| 来源文件 | 对应科目 | 条数 |
|----------|----------|:--:|
| 教资_综合素质.md | 综合素质（中小学通用） | 13 |
| 教资_教育知识_中学.md | 中学教育知识与能力 | 23 |
| 教资_教育知识_小学.md | 小学教育教学知识与能力 | 14 |

**综合素质** 覆盖：职业理念（教育观/学生观/教师观）、教育法律法规、教师职业道德规范、文化素养、基本能力。

**中学教育知识与能力** 覆盖：教育本质属性、教育学发展、中学课程、中学教学、中学生学习心理、发展心理、心理辅导、德育、班级管理。

**小学教育教学知识与能力** 覆盖：教育基础、学生指导、班级管理、教学设计、教学实施、教学评价。

#### 入库脚本

`scripts/build_kb_from_md.py` — 同课标，写入 `teacher_cert` 集合。

---

### 2.5 已注册但未建库的集合

代码 `app/retrieval/kb_manager.py:88-93` 中注册了 `teacher_guide`（教参/教师教学用书），当前向量库中**没有该集合**，待补充数据源后入库。

---

## 三、环境前置条件

### 3.1 系统与硬件

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux |
| Python | 3.10+ |
| 内存 | ≥ 8 GB（bge-m3 模型加载约 2.2 GB） |
| 磁盘 | ≥ 10 GB（向量库 1.1 GB + 模型 4.3 GB） |

### 3.2 Python 依赖

```bash
pip install "chromadb==1.5.9" "sentence-transformers>=2.7"

cd /data/education/teacher-agent-demo
pip install -e ".[dev,ui,rag]"
```

验证：

```bash
python3 -c "import chromadb; print('chromadb', chromadb.__version__)"
python3 -c "import sentence_transformers; print('st', sentence_transformers.__version__)"
# 预期：chromadb 1.5.9  /  st 5.6.0
```

### 3.3 嵌入模型文件

路径：`/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3`（约 4.3 GB）

验证：

```bash
ls -d /home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3
du -sh /home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3
```

如缺失，通过 ModelScope 下载：

```bash
pip install modelscope
python3 -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-m3', cache_dir='/home/lizhaoyuan/.cache/modelscope')
"
```

### 3.4 关键目录

```
/data/wyd-workspace/education/data/
├── china_textbook_k9/              # 源 PDF（仅建库时需要，运行时不需要）
├── china_textbook_k9_parsed/       # MinerU 解析的 MD（中间产物）
├── china_textbook_k9_chunks/       # 分块后的 JSONL（中间产物）
├── china_textbook_k9_vectordb/     # ★ 向量库（运行时必须）
│   ├── chroma.sqlite3
│   └── {segment-uuid}/
└── teacher_kb_source/              # 课标/教资源 MD（建库时需要）
    ├── 教资_教育知识_中学.md
    ├── 教资_教育知识_小学.md
    ├── 教资_综合素质.md
    └── 课标_义务教育数学2022.md
```

> **对接方只需保证向量库目录和 bge-m3 模型可用即可**，其余目录是建库中间产物，运行时不需要。

---

## 四、部署 Chroma Server

RAG 采用 **Server 模式**（非嵌入式），chroma server 作为独立进程长驻运行。

### 4.1 启动命令

```bash
chroma run \
  --path /data/wyd-workspace/education/data/china_textbook_k9_vectordb \
  --port 8002 \
  --host 127.0.0.1
```

参数说明：
- `--path`：向量库目录（`chroma.sqlite3` 所在目录）
- `--port`：监听端口，默认 8000，这里用 8002 避免与 FastAPI 冲突
- `--host`：`127.0.0.1` 仅本机访问；如需远程改为 `0.0.0.0`

### 4.2 后台运行（推荐 nohup）

```bash
nohup chroma run \
  --path /data/wyd-workspace/education/data/china_textbook_k9_vectordb \
  --port 8002 \
  --host 127.0.0.1 \
  > /tmp/logs/chroma_server.log 2>&1 &

echo $! > /tmp/chroma_server.pid
```

### 4.3 验证 Server 已启动

```bash
# 健康检查
curl -s http://127.0.0.1:8002/api/v2/pre-flight-checks
# 预期：{"max_batch_size":5461,"supports_base64_encoding":true}

# 列出集合
python3 -c "
import chromadb
c = chromadb.HttpClient(host='127.0.0.1', port=8002)
for col in c.list_collections():
    print(col.name, col.count())
"
# 预期输出：
#   k9_exercises            20122
#   curriculum_std             18
#   k9_textbook             62772
#   teacher_cert               50
```

### 4.4 停止 Server

```bash
kill $(cat /tmp/chroma_server.pid)
# 或
pkill -f "chroma run"
```

### 4.5 开机自启（systemd）

```ini
[Unit]
Description=ChromaDB Server for Teacher Agent RAG
After=network.target

[Service]
Type=simple
User=lizhaoyuan
ExecStart=/home/lizhaoyuan/miniconda3/bin/chroma run \
  --path /data/wyd-workspace/education/data/china_textbook_k9_vectordb \
  --port 8002 --host 127.0.0.1
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 五、应用层挂载配置

### 5.1 配置文件

编辑 `/data/education/teacher-agent-demo/.env`：

```bash
# ===== Textbook RAG 配置 =====
TEACHER_RAG_ENABLED=1
TEACHER_RAG_VECTORDB_PATH=/data/wyd-workspace/education/data/china_textbook_k9_vectordb
TEACHER_RAG_EMBED_MODEL=/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3
TEACHER_RAG_COLLECTION=k9_textbook
TEACHER_RAG_TOPK=5
TEACHER_RAG_SUBJECT_FILTER=数学

# ===== ChromaDB Server 模式 =====
CHROMA_SERVER=127.0.0.1:8002
```

### 5.2 配置项详解

| 环境变量 | 说明 | 默认值 | 必填 |
|----------|------|--------|------|
| `TEACHER_RAG_ENABLED` | RAG 全局开关（1/0） | `1` | 是 |
| `TEACHER_RAG_VECTORDB_PATH` | 向量库目录路径 | 见上 | 是 |
| `TEACHER_RAG_EMBED_MODEL` | bge-m3 模型路径 | 见上 | 是 |
| `TEACHER_RAG_COLLECTION` | 教材集合名 | `k9_textbook` | 是 |
| `TEACHER_RAG_TOPK` | 教材检索返回条数 | `5` | 否 |
| `TEACHER_RAG_SUBJECT_FILTER` | 学科过滤 | `数学` | 否 |
| `CHROMA_SERVER` | Chroma Server 地址 | `127.0.0.1:8002` | 是 |

> `CHROMA_SERVER` 是 Server 模式的开关。设置后应用走 HttpClient；不设置则走 PersistentClient（嵌入式，不推荐）。

### 5.3 应用如何读取配置

```
.env → load_dotenv()
  │
  ├── app/retrieval/textbook_rag.py    → 读 TEACHER_RAG_* 配置教材检索
  ├── app/retrieval/kb_manager.py      → 读 TEACHER_RAG_* 配置题库/课标/教资检索
  └── app/retrieval/chroma_compat.py   → 读 CHROMA_SERVER 决定 HttpClient/PersistentClient
```

### 5.4 开关控制（三层）

| 层级 | 控制方式 | 优先级 |
|------|----------|--------|
| 第一层 | API 请求参数 `use_rag`（true/false/null） | 最高 |
| 第二层 | 环境变量 `TEACHER_RAG_ENABLED` | 默认 |
| 第三层 | Streamlit UI 复选框 | 同 API 层 |

- `use_rag=true`：强制启用，绕过环境变量（仍需 server 可达）
- `use_rag=false`：完全跳过 RAG
- `use_rag=null`（不传）：跟随 `TEACHER_RAG_ENABLED`

---

## 六、检索逻辑详解

### 6.1 检索入口

Agent 六节点状态机的第 2 步 `retrieve_node`（`app/agent/workflow.py:351`）统一触发全部 RAG 检索。

### 6.2 完整检索流程

```
学生消息到达 retrieve_node
    │
    ├── Step 1: Graph RAG（概念图检索）
    │   概念图 → 精确匹配知识点、前置知识、练习题、常见错因
    │
    ├── Step 2: 多模态教材检索（MultimodalRAG）
    │   query = state.message + state.original_problem + concept.name
    │   例: "老师 2x+3=7 怎么解 2x+3=7 一元一次方程"
    │   → bge-m3 编码 query → ChromaDB k9_textbook cosine top-5
    │   → 默认过滤 subject=数学
    │   → 从 metadata.images 解析关联图片 URL
    │   → format_context → "[教材参考1] 初中数学七年级上册《归纳》\n...[关联图片: xxx.jpg]"
    │   → format_refs   → "数学·七年级上册《归纳》(相似度0.74) 📷×3"
    │   → format_image_urls → 图片URL列表（前端渲染）
    │   → 注入 RetrievalResult.textbook_context / textbook_refs / textbook_image_urls
    │
    ├── Step 3: 题库检索（KBManager → k9_exercises）
    │   ex_query = concept.name + original_problem
    │   例: "一元一次方程 2x+3=7"
    │   → k9_exercises 集合 cosine top-3（默认过滤 subject=Math）
    │   → format_context → "[题库1] 题库·Math\n【题目】..."
    │   → format_refs   → "题库·Math(相似度0.58)"
    │   → 注入 RetrievalResult.exercise_context / exercise_refs
    │
    ├── Step 4: 课标检索（KBManager → curriculum_std）
    │   → curriculum_std 集合 cosine top-2
    │   → 注入 RetrievalResult.curriculum_context / curriculum_refs
    │
    └── Step 5: 教资检索（KBManager → teacher_cert）
        → teacher_cert 集合 cosine top-2
        → 注入 RetrievalResult.cert_context / cert_refs
```

### 6.3 检索结果如何使用

| 检索结果 | 注入位置 | 说明 |
|----------|----------|------|
| `textbook_context` | DeepSeek system prompt（`_build_teach_context`） | 要求讲解时优先依据教材原文 |
| `exercise_context` | DeepSeek system prompt（仅 `show_worked_example` / `give_practice_question` 时） | 参考同类题解题思路 |
| `curriculum_context` | 已检索但暂未注入 prompt | 保留扩展接口 |
| `cert_context` | 已检索但暂未注入 prompt | 保留扩展接口 |
| `textbook_refs` | API 返回 + UI 展示 | 简短引用，含相似度 |
| `exercise_refs` | API 返回 + UI 展示 | 同上 |
| `curriculum_refs` | API 返回 + UI 展示 | 同上 |
| `cert_refs` | API 返回 + UI 展示 | 同上 |

### 6.4 API 返回示例

```json
{
  "textbook_refs": ["数学·七年级下册《复习巩固》(相似度0.69)", ...],
  "exercise_refs": ["题库·Math(相似度0.58)", ...],
  "curriculum_refs": ["课程标准·义务教育数学《课程目标》(相似度0.47)", ...],
  "cert_refs": ["教资·《基本能力——信息处理与逻辑思维》(相似度0.42)", ...]
}
```

### 6.5 优雅降级

任何环节失败都不影响 agent 主流程：

```
chromadb 未安装？        → is_rag_available()=False → 跳过
向量库路径不存在？       → 同上
bge-m3 加载失败？       → get_textbook_rag() 返回 None → 跳过
检索查询出错？          → retrieve() 内部 try/except → 返回空列表
TEACHER_RAG_ENABLED=0？ → 跳过
某个集合不存在？         → _get_collection() 返回 None → 仅该集合跳过
```

### 6.6 进程级单例

- `get_textbook_rag()` 用 `@lru_cache`，bge-m3 模型只加载一次
- `KBManager` 复用 textbook_rag 的 emb_fn，避免重复加载 2.2GB 模型
- 强制模式 `get_textbook_rag_forced()` / `get_kb_manager_forced()` 用独立缓存，绕过 env 开关

---

## 七、挂载验证（分步）

### Step 1：验证 Chroma Server 可达

```bash
curl -s http://127.0.0.1:8002/api/v2/pre-flight-checks
# 预期：{"max_batch_size":5461,"supports_base64_encoding":true}
```

### Step 2：验证 4 个集合并统计记录数

```bash
python3 -c "
import chromadb
client = chromadb.HttpClient(host='127.0.0.1', port=8002)
for col in client.list_collections():
    print(col.name, col.count())
"
# 预期：
#   k9_exercises            20122
#   curriculum_std             18
#   k9_textbook             62772
#   teacher_cert               50
```

### Step 3：验证应用层可连接

```bash
cd /data/education/teacher-agent-demo
python3 -c "
from app.retrieval.textbook_rag import get_textbook_rag, is_rag_available
from app.retrieval.kb_manager import get_kb_manager
print('is_rag_available:', is_rag_available())
rag = get_textbook_rag()
print('TextbookRAG:', 'OK' if rag else 'FAIL')
kb = get_kb_manager()
print('KBManager:', 'OK' if kb else 'FAIL')
if kb:
    for key in ('k9_exercises', 'curriculum_std', 'teacher_cert'):
        coll = kb._get_collection(key)
        print(f'  {key}: {coll.count() if coll else \"N/A\"}')
"
```

### Step 4：验证检索可执行（4 个集合）

```bash
cd /data/education/teacher-agent-demo
python3 -c "
from app.retrieval.textbook_rag import get_textbook_rag
from app.retrieval.kb_manager import get_kb_manager

rag = get_textbook_rag()
kb = get_kb_manager()

if rag:
    hits = rag.retrieve('解一元一次方程 移项 变号')
    print(f'教材检索: {len(hits)} 条')
    for h in hits[:2]:
        m = h['meta']
        print(f'  {m.get(\"grade\",\"\")}{m.get(\"semester\",\"\")} 《{m.get(\"chapter\",\"\")}》 sim={1-h[\"distance\"]:.2f}')

if kb:
    for coll_name in ('k9_exercises', 'curriculum_std', 'teacher_cert'):
        hits = kb.retrieve('一元一次方程', coll_name, topk=2)
        print(f'{coll_name} 检索: {len(hits)} 条')
        for h in hits:
            m = h['meta']
            print(f'  {h.get(\"collection\",\"\")} sim={1-h[\"distance\"]:.2f}')
"
```

---

## 八、使用方式

### 8.1 启动应用

```bash
cd /data/education/teacher-agent-demo

# 终端 1：后端 API（端口 8000）
python -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# 终端 2：前端 UI（端口 8501）
streamlit run app/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### 8.2 通过 API 调用

```bash
# 跟随默认
curl -X POST http://localhost:8000/turn \
  -H "Content-Type: application/json" \
  -d '{"student_id":"demo","message":"老师 2x+3=7 怎么解"}'

# 强制启用 RAG
curl -X POST http://localhost:8000/turn \
  -H "Content-Type: application/json" \
  -d '{"student_id":"demo","message":"老师 2x+3=7 怎么解","use_rag":true}'

# 查看 RAG 状态
curl http://localhost:8000/rag/status
```

### 8.3 通过代码调用

```python
from app.agent.workflow import run_turn

state = run_turn("student1", "老师 3x+4=7 怎么解", use_rag=True)

# 查看 4 类引用
print(state.retrieval.textbook_refs)    # 教材
print(state.retrieval.exercise_refs)    # 题库
print(state.retrieval.curriculum_refs)  # 课标
print(state.retrieval.cert_refs)        # 教资
```

### 8.4 直接调用 RAG（脱离 agent）

```python
from app.retrieval.textbook_rag import get_textbook_rag
from app.retrieval.kb_manager import get_kb_manager

rag = get_textbook_rag()
kb = get_kb_manager()

# 教材检索
hits = rag.retrieve("解一元一次方程 移项 变号")
for h in hits:
    print(f"{h['meta']['grade']}{h['meta']['semester']} sim={1-h['distance']:.2f}")

# 题库检索
ex_hits = kb.retrieve_exercises("一元一次方程", topk=3)
for h in ex_hits:
    print(h['text'][:80])
```

### 8.5 Streamlit UI 使用

1. 打开 `http://10.250.30.117:8501`
2. 侧边栏「教材 RAG 检索」复选框控制开关
3. 展开「诊断与检索详情」查看教材/题库/课标/教资四类引用
4. 侧边栏「🗑 新建对话」按钮清空历史

---

## 九、故障排查

### 9.1 Chroma Server 无法启动

| 现象 | 排查 | 解决 |
|------|------|------|
| `Address already in use` | 端口被占 | `lsof -i:8002` → `kill` |
| `No such file or directory` | 路径错误 | 确认 `--path` 指向 `chroma.sqlite3` 所在目录 |
| 启动后立即退出 | 查看日志 | `cat /tmp/logs/chroma_server.log` |

### 9.2 应用连接 Server 失败

```bash
# 检查 .env
grep CHROMA_SERVER /data/education/teacher-agent-demo/.env
# 应为：CHROMA_SERVER=127.0.0.1:8002

# 检查 server 是否可达
curl http://127.0.0.1:8002/api/v2/pre-flight-checks
```

### 9.3 RAG 检索返回空

```bash
# 1. RAG 是否启用
grep TEACHER_RAG_ENABLED /data/education/teacher-agent-demo/.env

# 2. 依赖
python3 -c "import chromadb, sentence_transformers; print('OK')"

# 3. 向量库路径
ls /data/wyd-workspace/education/data/china_textbook_k9_vectordb/chroma.sqlite3

# 4. 模型路径
ls /home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3

# 5. 应用可用性
cd /data/education/teacher-agent-demo
python3 -c "from app.retrieval.textbook_rag import is_rag_available; print(is_rag_available())"
```

### 9.4 首次检索很慢

bge-m3 模型首次加载约 2.2 GB，耗时 10-30 秒。后续复用进程级单例。预热：

```python
from app.retrieval.textbook_rag import get_textbook_rag
get_textbook_rag()
```

### 9.5 学科过滤导致检索不到

默认 `TEACHER_RAG_SUBJECT_FILTER=数学`，只检索数学教材。改：

```bash
TEACHER_RAG_SUBJECT_FILTER=物理   # 改为物理
TEACHER_RAG_SUBJECT_FILTER=       # 不过滤
```

题库检索中 `KBManager.retrieve_exercises` 默认过滤 `subject="数学"`，修改 `workflow.py:391` 的调用参数即可。

---

## 十、附录

### 10.1 关键文件清单

| 文件 | 作用 |
|------|------|
| `app/retrieval/textbook_rag.py` | 教材 RAG 检索器（k9_textbook） |
| `app/retrieval/kb_manager.py` | 多集合管理器（k9_exercises / curriculum_std / teacher_cert） |
| `app/retrieval/chroma_compat.py` | ChromaDB 兼容层 + client 工厂 |
| `app/retrieval/graph_retriever.py` | Graph RAG（概念图，独立于向量 RAG） |
| `app/agent/workflow.py` | Agent 工作流，`retrieve_node` 触发全部检索 |
| `app/agent/state.py` | RetrievalResult 承载 4 类检索字段 |
| `app/api/main.py` | FastAPI 接口，`/turn` 和 `/rag/status` |
| `app/ui/streamlit_app.py` | Streamlit 前端 |
| `.env` | 运行时配置 |
| `scripts/build_vector_db.py` | 教材 chunks → ChromaDB |
| `scripts/build_textbook_from_pdf.py` | PDF → 分块 + 入库（补缺学科） |
| `scripts/build_kb_from_md.py` | Markdown → ChromaDB（课标/教资） |
| `/tmp/ingest_m3ke.py` | M3KE 数据集 → ChromaDB（题库） |

### 10.2 配置参数速查表

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `TEACHER_RAG_ENABLED` | RAG 全局开关 | `1` |
| `TEACHER_RAG_VECTORDB_PATH` | 向量库目录 | `/data/wyd-workspace/education/data/china_textbook_k9_vectordb` |
| `TEACHER_RAG_EMBED_MODEL` | bge-m3 模型路径 | `/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3` |
| `TEACHER_RAG_COLLECTION` | 教材集合名 | `k9_textbook` |
| `TEACHER_RAG_TOPK` | 教材检索 top-k | `5` |
| `TEACHER_RAG_SUBJECT_FILTER` | 学科过滤 | `数学` |
| `CHROMA_SERVER` | Chroma Server 地址 | `127.0.0.1:8002` |

### 10.3 重建向量库命令

```bash
# 停 chroma server
pkill -f "chroma run"

# 备份旧库
mv /data/wyd-workspace/education/data/china_textbook_k9_vectordb \
   /data/wyd-workspace/education/data/china_textbook_k9_vectordb_bak_$(date +%Y%m%d)

# 1. 教材入库
cd /data/wyd-workspace/education
python scripts/build_vector_db.py

# 2. 补缺学科教材
cd /data/education/teacher-agent-demo
python scripts/build_textbook_from_pdf.py

# 3. 题库入库（M3KE）
python /tmp/ingest_m3ke.py

# 4. 课标入库
python scripts/build_kb_from_md.py \
  --source-dir /data/wyd-workspace/education/data/teacher_kb_source \
  --pattern '课标_*.md' --collection curriculum_std --doc-type curriculum_std

# 5. 教资入库
python scripts/build_kb_from_md.py \
  --source-dir /data/wyd-workspace/education/data/teacher_kb_source \
  --pattern '教资_*.md' --collection teacher_cert --doc-type teacher_cert

# 重启 chroma server
chroma run --path /data/wyd-workspace/education/data/china_textbook_k9_vectordb \
  --port 8002 --host 127.0.0.1 &
```

### 10.4 对接检查清单

对接方完成以下全部项即视为对接成功：

- [ ] chromadb 1.5.9 已安装
- [ ] sentence-transformers 已安装
- [ ] bge-m3 模型文件存在（4.3 GB）
- [ ] 向量库目录存在且 `chroma.sqlite3` 可访问
- [ ] `.env` 配置完整（含 `CHROMA_SERVER=127.0.0.1:8002`）
- [ ] Chroma Server 已启动并可通过 curl 验证
- [ ] 4 个集合均存在且记录数正确
- [ ] `is_rag_available()` 返回 True
- [ ] `get_textbook_rag()` 返回非 None
- [ ] `get_kb_manager()` 返回非 None
- [ ] 教材检索返回 ≥ 1 条
- [ ] 题库检索返回 ≥ 1 条
- [ ] 课标检索返回 ≥ 1 条
- [ ] 教资检索返回 ≥ 1 条
- [ ] `/turn` 接口返回含 `textbook_refs`、`exercise_refs`、`curriculum_refs`、`cert_refs` 的响应

### 10.5 数据来源汇总

| 集合 | 数据名称 | 来源机构 | 来源地址 | 许可证 |
|------|----------|----------|----------|--------|
| k9_textbook | 人教版 K9 教材 | 人民教育出版社 | 本地 PDF | 教育用途 |
| k9_exercises | M3KE | 天津大学 NLP + 华为 | `https://github.com/tjunlp-lab/M3KE`<br>`https://huggingface.co/datasets/TJUNLP/M3KE` | CC BY-NC-SA 4.0 |
| curriculum_std | 义务教育数学课程标准（2022） | 教育部 | 本地 MD | 公开文件 |
| teacher_cert | 教师资格证考试大纲 | 教育部考试中心 | 本地 MD | 公开文件 |

---

*本文档由 CodeBuddy 基于当前运行状态生成，配置与数据均已现场复核。*
*最后更新：2026-07-14 07:43*
