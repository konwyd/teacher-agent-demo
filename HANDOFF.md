# Teacher Agent Demo — 项目交接文档

> 适用于下一位接手本项目的开发者/AI 会话。最后更新：2026-07-20

---

## 0. 一句话总结

这是一个 **LLM 主导的初中数学教学 Agent 演示项目**。当前活跃分支是 `codex/llm-led-teaching-prototype`（位于 GitHub `Kuiyun24z/teacher-agent-demo`），本地工作目录为 `/home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo`。**最新对话**已把"LM 主导 Teacher Prototype"前端 + 本地/联网 RAG 检索都跑起来，唯一待办是重启 streamlit 让 Pydantic 重新加载 `PrototypeTurnRecord` 的 rag 字段。

---

## 1. 仓库结构

```
teacher-agent-demo/
├── app/                              # ★ 新分支 agent 主体（LangGraph + 通用教学管线）
│   ├── agent/
│   │   ├── workflow.py               # execute_turn / run_turn 入口
│   │   ├── state.py                  # Pydantic 状态模型（含 rag 字段）
│   │   ├── math_checker.py           # 一元一次方程步骤检查器（纯规则）
│   │   ├── teaching_actions.py       # 9 个 TeachingAction 枚举
│   │   ├── teaching_policy.py        # 教学动作选择
│   │   ├── teaching_planner.py       # LLM 预算管理
│   │   ├── teaching_harness.py       # 通用教学管线
│   │   ├── response_generation.py    # ResponseGuard 守卫
│   │   ├── safety_gate.py            # 安全约束
│   │   ├── optional_verifier.py      # 验证器协议
│   │   ├── linear_equation_adapter.py # 遗留方程适配器
│   │   ├── legacy_action_compat.py   # 旧动作枚举兼容层
│   │   └── ...（共 22 个文件）
│   ├── llm/deepseek_client.py        # DeepSeek API 客户端（唯一 LLM 入口）
│   ├── retrieval/
│   │   ├── graph_retriever.py        # Graph RAG mock（JSON 概念图）
│   │   ├── context_provider.py       # 图上下文适配器
│   │   └── pedagogical_retriever.py  # 教学 RAG 检索器
│   ├── memory/                       # 三层记忆（StudentProfile / EpisodeState / LearningEvent）
│   ├── evaluation/                   # evaluator + 混合评测
│   ├── api/main.py                   # FastAPI（端点 /health, /turn, /conversations/new 等）
│   └── ui/streamlit_app.py           # 旧版 Streamlit（已被 prototype 替代）
│
├── prototypes/llm_led_teacher/       # ★★ LLM 主导 prototype 页面（当前活跃 UI）
│   ├── streamlit_app.py              # Streamlit 入口，标题 "🧪 LLM 主导 Teacher Prototype"
│   ├── logic.py                      # 教学循环（verification-only orchestration）
│   ├── rag_hook.py                   # ★★ RAG 检索钩子（最近新增）
│   ├── README.md
│   └── NOTES.md
│
├── llm-test/                         # ⚠️ 不要动！RAG 检索器 + 评测系统
│   ├── rag_retriever.py              # ★ 核心：retrieve_for_exam() 管道
│   ├── online_rag.py                 # 联网 RAG（Tavily）
│   ├── config.py                     # 全局配置（API key、Chroma、Tavily）
│   ├── evaluator.py                  # 全卷评测系统
│   ├── hard_question_eval.py
│   ├── question_rag.py
│   ├── split_exam.py
│   ├── ingest_zhongkao_v4.py
│   ├── chat.py
│   ├── llm_client.py
│   ├── markdown/                     # 7 科中考真题 MD
│   ├── testmd/                       # 拆分后的题+答
│   └── results/                      # 评测结果
│
├── data/                             # 数据目录
│   ├── china_textbook_k9_vectordb/   # ChromaDB 向量库（~2.3 GB，62772 条教材）
│   ├── china_textbook_k9_chunks/     # 教材分块
│   ├── k9_exercises_source/          # 题库源数据
│   ├── teacher_kb_source/            # 课标+教资
│   ├── sample/                       # ★ 新增：Graph RAG mock 数据
│   │   ├── concepts.json
│   │   ├── exercises.json
│   │   ├── misconceptions.json
│   │   └── pedagogical_graph.json
│   ├── eval/                         # ★ 新增：评测用例（jsonl）
│   └── runtime/                      # 学生档案/事件（JSON）
│
├── tests/                            # ★ 全部 237 个测试通过
├── scripts/                          # run_eval.py 等
├── pyproject.toml                    # 项目配置
├── .env                              # ★ DeepSeek API key
└── logs/                             # 运行日志
```

---

## 2. 当前服务状态

| 服务 | 端口 | 状态 | 启动方式 |
|------|------|------|----------|
| ChromaDB Server | **8002** | ✅ 运行中（heartbeat 正常） | `chroma run --path data/china_textbook_k9_vectordb --port 8002 --host 127.0.0.1` |
| DeepSeek LLM | 远程 | ✅ 已配置（API key 在 `.env`） | — |
| Streamlit (Prototype) | **8501** | ⚠️ 需重启 | 见下文命令 |
| FastAPI（旧） | 8000/8001 | ❌ 已停 | — |

**Streamlit 启动命令**（必须设 PYTHONPATH，因为 prototype 依赖 `llm-test`）：

```bash
cd /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo
# 杀掉旧进程
pkill -9 -f streamlit; sleep 2
# 启动 prototype
PYTHONPATH="$(pwd):$(pwd)/llm-test:$PYTHONPATH" nohup streamlit run \
  prototypes/llm_led_teacher/streamlit_app.py \
  --server.port 8501 --server.headless true \
  --server.address 0.0.0.0 --browser.gatherUsageStats false \
  > /tmp/streamlit.log 2>&1 &
```

**环境变量 / .env 配置**（已写入 `.env`）：

```bash
DEEPSEEK_API_KEY=<your-key>
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

Tavily 密钥通过 `TAVILY_API_KEY` 环境变量配置；Chroma 地址仍由 `llm-test/config.py` 配置。

---

## 3. 最新一次对话做了什么

### 3.1 拉取新分支的 agent 代码
从 `https://github.com/Kuiyun24z/teacher-agent-demo/tree/codex/llm-led-teaching-prototype` 下载了完整的 `app/` 模块（22 个文件）、`tests/`（237 个测试全绿）、`data/sample/`、`data/eval/`、`scripts/`。

### 3.2 把 RAG 挂到 prototype
**核心修改**：

1. **新建** `prototypes/llm_led_teacher/rag_hook.py`：
   - 把 `llm-test/` 加到 `sys.path`
   - 暴露 `retrieve_context(problem, online_mode="auto"/"off")` → 返回 `RagContext`
   - 内部调用 `rag_retriever.retrieve_for_exam()`

2. **修改** `prototypes/llm_led_teacher/logic.py`：
   - `PrototypeTurnRecord` 增加 4 个字段：`rag_source, rag_text, rag_refs, rag_exact_match`
   - `_SYSTEM_PROMPT` 增加 RAG 知识库使用说明
   - `_teacher_payload()` 支持注入 `textbook_context, references, rag_source`
   - `run_turn()` 中：
     - **新题** → `online_mode="auto"`（本地+联网）
     - **追问** → `online_mode="off"`（仅本地）

3. **修改** `prototypes/llm_led_teacher/streamlit_app.py`：
   - 侧边栏显示 RAG 状态
   - 展开面板显示检索到的教材内容 + 来源引用

### 3.3 修复环境问题
- `sentence_transformers` 加载失败：原因是 `torch 2.13.0` 和 `torchvision 0.18.1` 版本不兼容
- 解决：卸载 `torchaudio` 后 `sentence_transformers` 可正常加载 bge-m3

### 3.4 当前问题
- Streamlit 显示 `AttributeError: 'PrototypeTurnRecord' object has no attribute 'rag_text'`
- **原因**：streamlit 进程加载的是旧版 `logic.py`（`.pyc` 缓存或 fork 前的版本），但磁盘上的 logic.py 已经是新版本
- **解决**：杀掉 streamlit，按上述命令重启即可

---

## 4. 关键架构与设计原则

### 4.1 教学 agent 流程（prototype）
```
用户消息 → Observer (LLM JSON)
       → 新题检测
       → RAG 检索（auto: 本地+联网；follow-up: 仅本地）
       → 验证（math_checker）
       → Teacher LLM (DeepSeek) 生成回复
       → Claim 审计（提取所有方程逐一验证）
       → 警告检测 → 必要时 repair
       → 写入 PrototypeTurnRecord
```

### 4.2 铁律（来自 CODEX_HANDOFF.md）
- 任何 LLM 调用 ≤ 3 次/turn（planner 预算）
- `math_checker` 判定数学正误，LLM **不得** 改判
- 每轮最多 1 个问题
- 不提前泄露答案
- retrieved examples 不能污染原题主线

### 4.3 RAG 检索管道
```
query → _strip_latex → _extract_keywords
     → TextbookRAG.hybrid_retrieve (BM25 + bge-m3 + RRF)
     → ExerciseRAG.hybrid_retrieve
     → is_exact_match? (top1 sim ≥ 0.7)
     → online_mode:
         "off"   → 只用本地
         "auto"  → 本地不满足阈值时触发 Tavily
         "online"→ 跳过本地
     → RRF 融合 → 过滤 sim<0.44 → top-10
     → format_context + format_refs
```

### 4.4 数据流
- **ChromaDB**: 62772 条教材 + 225760 条题库
- **Graph RAG mock**: `data/sample/*.json` 中的概念图
- **学生记忆**: `data/runtime/student_profiles.json`, `episode_states.json`, `learning_events.jsonl`

---

## 5. 已知坑 / TODO

| 优先级 | 问题 | 解决方向 |
|--------|------|----------|
| 🔴 P0 | Streamlit 加载了旧 PrototypeTurnRecord（缺 rag 字段） | 重启 streamlit |
| 🟡 P1 | `transformers`/`torchvision`/`torchaudio` 三方版本冲突 | 重装配套版本或用 Docker |
| 🟡 P1 | `_SYSTEM_PROMPT` 是为"无 RAG"设计的，已加 RAG 说明但未充分调试 | 多轮测试 + 调 prompt |
| 🟢 P2 | FastAPI 后端在 8001 已停，未来可能需要同时跑 API + UI | 启动 `uvicorn app.api.main:app --port 8001` |
| 🟢 P2 | Tavily API key 硬编码在 `llm-test/config.py` | 改为环境变量 |
| 🟢 P3 | UI 上"自动重载"有时不生效 | 改用 `--server.runOnSave true` |

---

## 6. 常见命令速查

```bash
# 跑测试
cd /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo
python3 -m pytest tests/ -q --ignore=tests/test_deepseek_integration.py --ignore=tests/test_api.py

# 单测 RAG 检索
PYTHONPATH="$(pwd)/llm-test:$PYTHONPATH" python3 -c "
from rag_retriever import retrieve_for_exam
ctx, refs, exact = retrieve_for_exam('3x+4=7 怎么解', subject='数学', online_mode='auto')
print('text len:', len(ctx), 'refs:', len(refs), 'exact:', exact)
"

# 跑 prototype（用户界面）
PYTHONPATH="$(pwd):$(pwd)/llm-test:$PYTHONPATH" streamlit run prototypes/llm_led_teacher/streamlit_app.py --server.port 8501

# 跑 FastAPI
python3 -m uvicorn app.api.main:app --port 8001

# 跑评测（不接 UI）
cd llm-test && python3 evaluator.py --compare-all
```

---

## 7. 关键文件位置速查

| 用途 | 路径 |
|------|------|
| Prototype 入口 | `prototypes/llm_led_teacher/streamlit_app.py` |
| Prototype 教学逻辑 | `prototypes/llm_led_teacher/logic.py` |
| RAG 钩子（最近新增） | `prototypes/llm_led_teacher/rag_hook.py` |
| 核心 RAG 检索器（不修改） | `llm-test/rag_retriever.py` |
| 联网 RAG（不修改） | `llm-test/online_rag.py` |
| 全局配置（DeepSeek/Chroma/Tavily） | `llm-test/config.py` |
| DeepSeek 客户端 | `app/llm/deepseek_client.py` |
| 数学验证器 | `app/agent/math_checker.py` |
| Agent 工作流 | `app/agent/workflow.py` |
| Chroma 向量库 | `data/china_textbook_k9_vectordb/` |
| Graph RAG mock 数据 | `data/sample/` |
| API key | `.env` |
| 流日志 | `logs/turn_logs.jsonl` |
| 评测日志 | `llm-test/results/` |

---

## 8. 启动顺序（最小可用）

1. **ChromaDB Server**（如未运行）：
   ```bash
   chroma run --path /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo/data/china_textbook_k9_vectordb --port 8002 --host 127.0.0.1 &
   ```

2. **Streamlit Prototype**：
   ```bash
   cd /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo
   pkill -9 -f streamlit; sleep 2
   PYTHONPATH="$(pwd):$(pwd)/llm-test:$PYTHONPATH" streamlit run \
     prototypes/llm_led_teacher/streamlit_app.py --server.port 8501
   ```

3. 浏览器打开 `http://localhost:8501` → 输入 `3x+4=9 这个方程式怎么解` → 测试

---

## 9. 故障排查

- **Streamlit 不更新代码**：杀掉重启，删除 `__pycache__`
- **RAG 检索为空**：`curl http://127.0.0.1:8002/api/v2/heartbeat` 确认 Chroma 在跑
- **`sentence_transformers` 报错**：检查 `transformers/torch/torchvision/torchaudio` 版本一致
- **DeepSeek 调用失败**：检查 `.env` 中的 `DEEPSEEK_API_KEY`
- **Tavily 联网失败**：检查 `llm-test/config.py` 的 `tavily_api_key`

---

**最后重要提示**：

1. ⚠️ **不要删 `llm-test/`**，那是 RAG 检索的核心
2. ⚠️ **不要删 `data/china_textbook_k9_vectordb/`**，2.3 GB 向量库
3. ✅ `app/` 是新分支的代码，可以自由修改
4. ✅ `prototypes/llm_led_teacher/` 是当前 UI，可以自由修改
5. 🔄 重启 streamlit 之前务必 `pkill -9 -f streamlit`
