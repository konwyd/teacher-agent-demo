# Teacher Agent — LLM 主导的中文初中数学教师智能体

这是一个中文初中数学 Teacher Agent research demo。项目关注的是教学质量：让模型根据学生的表达选择合适的教学动作、讲解深度和措辞，而不是仅仅提高解题答案的正确率。

## 当前架构

运行时 LLM 是 **DeepSeek**。DeepSeek 负责选择教学动作、讲解深度与具体措辞；确定性外壳负责保持当前原题的连续性、验证已支持的数学内容、有界地调用可选检索、审计回复，以及最多进行一次修复调用。

`TeachingPlanner`、`StrategyRegistry` 与由掌握度驱动的硬流程控制均不属于当前 canonical loop。`app/agent` 被保留为 legacy/research baseline，供结构化工作流与研究模块继续使用。

Teacher Agent 可以通过 DeepSeek 教授一般的初中数学。确定性数学验证目前对已支持的一元一次方程最强；对于未支持的主题，系统会继续教学，但不会假装已由 checker 验证。

## 教学与安全边界

- `app/agent/math_checker.py` 是已支持数学验证的确定性权威；LLM 不推翻其判定。
- 原题在首轮抽取后持续保留，普通讲解始终围绕原题；检索到的例题只在明确的例题或练习动作中使用。
- 记忆与 RAG 是软约束，只影响讲解深浅、提示、复习与练习选择，不强制学生按固定步骤作答。
- 禁止提交 `.env` 或任何 API key；真实对话所需的 key 只从环境变量读取。

## 快速开始

在 PowerShell 中执行：

```powershell
conda create -n teacher-agent python=3.11 -y
conda activate teacher-agent
pip install -e ".[dev,ui]"
Copy-Item .env.example .env
streamlit run prototypes/llm_led_teacher/streamlit_app.py
```

真实对话需要在 `.env` 中配置 `DEEPSEEK_API_KEY`；绝不要将该文件或 key 提交到仓库。

## 可选 RAG

RAG 是可选的教学决策辅助，不是启动教学的前提。需要启用时安装额外依赖，并选择服务端或本地持久化路径之一：

```powershell
pip install -e ".[dev,ui,rag]"
$env:TEACHER_RAG_ENABLED = "true"
$env:CHROMA_SERVER = "localhost:8002"
```

也可以用 `TEACHER_RAG_VECTORDB_PATH` 指定本地持久化向量库路径。完整环境变量和运行说明见 [prototypes/llm_led_teacher/README.md](prototypes/llm_led_teacher/README.md)。检索被禁用、为空或失败时，都不会阻断教学。

## 验证

以下离线回归门应在修改后通过：

```powershell
python -m pytest
python scripts/run_eval.py
```

## 仓库结构

- `prototypes/llm_led_teacher/`：canonical autonomous loop 与 UI。
- `app/llm/`：唯一的 DeepSeek 边界。
- `app/retrieval/`：可复用的可选语义检索。
- `app/agent/math_checker.py`：确定性数学权威。
- `app/agent/`、`app/api/`、`app/ui/`、`app/memory/`：保留的结构化工作流 baseline 与共享研究模块。
- `app/evaluation/`、`data/eval/`、`scripts/run_eval.py`、`tests/`：回归门。

## 当前状态与已知限制

canonical autonomous loop 与 UI 已位于 `prototypes/llm_led_teacher/`。当前仍待完成多问题输入处理与 `single_move` 强制执行；设计与后续工作见 [single-move turn contract](docs/superpowers/specs/2026-08-10-single-move-turn-contract-design.md)。

## 开发文档

- [CODEX_HANDOFF.md](CODEX_HANDOFF.md)：项目交接、关键不变式与历史背景。
- [AGENTS.md](AGENTS.md)：开发规则与修改 checklist。
- [prototypes/llm_led_teacher/README.md](prototypes/llm_led_teacher/README.md)：canonical loop 的详细运行与 RAG 配置。
