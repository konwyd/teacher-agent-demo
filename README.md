# Teacher Agent — RAG 知识增强型 AI 教师

> 一个基于 DeepSeek 的中文 K9 数学教师 Agent，搭载 **28.8 万条多模态 RAG 知识库**（教材原文、题库、课程标准、教师资格证），支持图文混合检索。

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 内存 ≥ 16 GB（bge-m3 模型 + ChromaDB 共需约 6 GB）
- 磁盘 ≥ 10 GB

### 1. 安装依赖

```bash
cd teacher-agent-demo
pip install -e ".[dev,ui,rag]"
pip install chromadb==1.5.9 sentence-transformers>=2.7
```

### 2. 下载嵌入模型

```bash
pip install modelscope
python3 -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-m3', cache_dir='/home/YOUR_USER/.cache/modelscope')
"
```

### 3. 配置

编辑 `teacher-agent-demo/.env`：

```bash
# DeepSeek API（必需）
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# RAG 知识库
TEACHER_RAG_ENABLED=1
TEACHER_RAG_VECTORDB_PATH=/data/education/data/china_textbook_k9_vectordb
TEACHER_RAG_EMBED_MODEL=/home/YOUR_USER/.cache/modelscope/BAAI/bge-m3
TEACHER_RAG_COLLECTION=k9_textbook
TEACHER_RAG_TOPK=5
TEACHER_RAG_SUBJECT_FILTER=数学

# Chroma Server（先启动）
CHROMA_SERVER=127.0.0.1:8002

# 多模态图片访问
TEACHER_IMAGE_BASE_URL=http://YOUR_IP:8000/images
```

### 4. 启动服务

```bash
# 终端 1：Chroma Server（向量检索）
chroma run --path data/china_textbook_k9_vectordb --port 8002 --host 127.0.0.1 &

# 终端 2：FastAPI 后端
cd teacher-agent-demo
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &

# 终端 3：Streamlit 前端
streamlit run teacher-agent-demo/app/ui/streamlit_app.py --server.port 8501 &
```

打开浏览器访问 `http://YOUR_IP:8501` 即可开始教学对话。

---

## 📊 RAG 知识库总览

| 集合 | 记录数 | 内容 | 来源 |
|------|:---:|------|------|
| `k9_textbook` | 62,772 | K9 全科教材（19 学科），55.9% 含图片 | 130 本人教版 PDF |
| `k9_exercises` | 225,760 | 全题型题库（选择+填空+解答+应用） | M3KE + TAL-SCQ5K + APE210K + 中考真题 |
| `curriculum_std` | 18 | 义务教育数学课程标准（2022） | 课标 MD |
| `teacher_cert` | 50 | 教师资格证考试资料 | 教资 MD |
| **合计** | **288,600** | — | — |

### 题库明细

| 数据源 | 题目数 | 题型 | 来源机构 |
|--------|:---:|------|------|
| M3KE | 20,122 | 选择题（71 学科） | 天津大学 NLP + 华为 |
| TAL-SCQ5K | 5,000 | 选择题（含解题步骤） | 好未来（学而思）MIT |
| APE210K | 200,488 | 应用题 | 开源 |
| 河南中考真题 | 150 | 全题型 | 官方真题 |

### 多模态能力

- **图文混合检索**：文本向量匹配 → 从 metadata 提取关联图片 → 返回图片 URL
- **图片规模**：53,957 张教材插图
- **实现**：`app/retrieval/multimodal_rag.py`

---

## 🧪 评测系统

`llm-test/` 目录包含完整的 A/B 对比评测框架：

```bash
cd llm-test

# 配置 config.py 中的 API key

# 纯模型（无 RAG）
python evaluator.py 数学

# 挂载 RAG（开卷考试）
python evaluator.py --rag 数学

# 全科对比（无 RAG vs 有 RAG）
python evaluator.py --compare-all

# 导出中考题 JSON（供测试同学使用）
# 已导出至 results/zhongkao_export/
```

---

## 📁 项目结构

```
education/
├── README.md                       # 本文档
├── teacher-agent-demo/             # 主项目
│   ├── app/
│   │   ├── agent/                  # Agent 工作流（六节点状态机）
│   │   │   ├── workflow.py         # 诊断→检索→策略→教学→记忆→日志
│   │   │   └── math_checker.py     # 数学步骤检查器
│   │   ├── retrieval/
│   │   │   ├── textbook_rag.py     # 教材 RAG 检索器
│   │   │   ├── multimodal_rag.py   # 多模态图文混合检索
│   │   │   ├── kb_manager.py       # 多集合管理器（题库/课标/教资）
│   │   │   └── chroma_compat.py    # ChromaDB 兼容层
│   │   ├── api/main.py             # FastAPI 接口
│   │   ├── ui/streamlit_app.py     # Streamlit 前端
│   │   └── llm/deepseek_client.py  # DeepSeek API 客户端
│   ├── docs/                       # 文档
│   │   └── RAG_DEPLOYMENT_GUIDE.md # RAG 部署对接手册
│   ├── scripts/                    # 建库脚本
│   ├── .env                        # 配置（需填入 API key）
│   └── pyproject.toml
├── llm-test/                       # 评测系统
│   ├── evaluator.py                # 主评测逻辑
│   ├── rag_retriever.py            # RAG 检索器
│   ├── config.py                   # 配置
│   ├── testmd/paper/               # 中考题 MD
│   ├── testmd/key/                 # 答案 MD
│   └── results/zhongkao_export/    # 导出 JSON 题库
└── data/                           # 数据文件
    ├── china_textbook_k9_vectordb/ # ★ 向量库（2.2 GB，运行时必须）
    ├── china_textbook_k9_chunks/   # 教材分块（83 MB，重建时需要）
    ├── k9_exercises_source/        # 题库源数据（88 MB）
    └── teacher_kb_source/          # 课标/教资源 MD（72 KB）
```

---

## 🔍 RAG 检索原理

```
学生提问 "2x+3=7 怎么解"
    ↓
bge-m3 编码为 1024 维向量
    ↓
ChromaDB HNSW 近似最近邻搜索（cosine 相似度）
    ↓
返回 top-5 教材 chunk（原文 + 图片 URL）+ top-3 题库 + top-2 课标 + top-2 教资
    ↓
注入 DeepSeek prompt → 生成教师回复
```

- **嵌入模型**：BAAI/bge-m3（1024 维，中文语义检索强）
- **相似度**：cosine（向量归一化后内积）
- **索引**：HNSW 近似最近邻

---

## 🔧 故障排查

| 问题 | 解决 |
|------|------|
| Chroma Server 无法启动 | 检查端口占用：`lsof -i:8002` |
| RAG 检索返回空 | 检查 `.env` 中 `TEACHER_RAG_ENABLED=1` 且 Chroma Server 可达 |
| 首次检索很慢 | bge-m3 首次加载 10-30 秒，后续复用 |
| 图片无法显示 | 检查 `TEACHER_IMAGE_BASE_URL` 配置 |

---

## 📄 许可证

本项目代码基于 MIT 许可。知识库数据来源：

| 数据 | 许可证 |
|------|--------|
| M3KE | CC BY-NC-SA 4.0 |
| TAL-SCQ5K | MIT |
| APE210K | 开源 |
| 人教版教材 | 教育用途 |
| 课标/教资 | 公开文件 |

---

## 🙏 致谢

- [M3KE](https://github.com/tjunlp-lab/M3KE) — 天津大学 NLP + 华为
- [TAL-SCQ5K](https://huggingface.co/datasets/math-eval/TAL-SCQ5K) — 好未来
- [APE210K](https://github.com/Chenny0808/ape210k) — 开源数学应用题
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — 嵌入模型
- [ChromaDB](https://www.trychroma.com/) — 向量数据库
