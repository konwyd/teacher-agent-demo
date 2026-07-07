# Teacher Agent — 中文初中数学辅导 Agent（Research Demo）

Graph-grounded Adaptive Tutor Agent 研究原型：诊断学生卡点 → 检索知识（mock Graph RAG）→
确定性教学策略选动作 → DeepSeek 生成引导（带合规后处理）→ 更新学生掌握度 → 研究日志。

- Runtime LLM：**DeepSeek API**（OpenAI-compatible；无 key 时全链路 mock 可离线跑通）
- MVP 范围：一元一次方程（等式性质/移项/合并同类项/去括号/分数系数）
- 核心特性：原题主线锚定、math_checker 精确步骤判定（支持跳步/多步计划/直接答案）、
  掌握度自适应教学深浅、golden cases 回归评测

## 快速开始

```bash
conda create -n teacher-agent python=3.11 -y && conda activate teacher-agent
pip install -e ".[dev,ui]"
cp .env.example .env        # 填入 DEEPSEEK_API_KEY（可留空，走 mock）

python -m pytest            # 全离线测试
python scripts/run_eval.py  # golden cases 评测（mock）

# demo：两个终端
python -m uvicorn app.api.main:app --reload
streamlit run app/ui/streamlit_app.py     # http://localhost:8501
```

试一试（同一 student_id 连续输入）：
`老师 3x+4=7 怎么解？` → `为什么+4移过去变成-4` → `先都减四，然后同时除以3`

## 开发说明

- **交接文档**：架构、设计原则、已修 bug、路线图见 [`CODEX_HANDOFF.md`](CODEX_HANDOFF.md)。
- **coding agent 工作规则**：见 [`AGENTS.md`](AGENTS.md)（不变式清单 + 修改 checklist）。
- **目录**：`app/api` 路由 · `app/agent` 工作流/schema/math_checker · `app/llm` DeepSeek 唯一出口 ·
  `app/retrieval` Graph RAG（mock）· `app/memory` 对话状态+掌握度 · `app/evaluation` 评测 ·
  `data/sample` 知识数据 · `data/eval` golden cases · `tests` 全离线测试。
- **铁律**：任何改动后 `python -m pytest` 和 `python scripts/run_eval.py` 必须全绿；
  学生步骤对错由 math_checker 裁决而非 LLM；每轮一个问题；不提前泄答案；不无故引入新题；
  `.env` 永不入库。
- **真实 LLM 评测**：`USE_REAL_LLM_FOR_EVAL=true python scripts/run_eval.py`
  （PowerShell：`$env:USE_REAL_LLM_FOR_EVAL="true"`），结果在 `logs/eval_results.jsonl`。
- **清空学生记忆**：删除 `data/runtime/student_profiles.json`；对话状态在进程内存，重启后端即重置。
