# AGENTS.md — 给 coding agent（Codex 等）的工作规则

本仓库是中文初中数学 Teacher Agent research demo。**runtime LLM 是 DeepSeek API，你只是开发助手。**
接手前先读 `CODEX_HANDOFF.md`（完整交接文档）和 `CLAUDE.md`（原始项目规范）。

## 必须遵守

1. **不要重构整个项目。** 目录分层（api/agent/llm/retrieval/memory/evaluation）是刻意设计，做增量修改。
2. **不要删除或弱化 evaluation。** `data/eval/` 的 golden cases 是回归门；`app/evaluation/` 和 `scripts/run_eval.py` 保护现有教学能力。
3. **不要绕过 math_checker。** 学生步骤/多步计划/最终答案的对错判定必须走 `app/agent/math_checker.py`（Fraction 精确计算），不允许交给 LLM 自由判断，也不允许在 teach 阶段推翻 checker 结论。
4. **不要把 memory 变成硬流程控制。** 掌握度只影响讲解深浅/类比/复习/变式题；不得用它拒绝学生跳步、强制一步一答，checker 判定分支必须始终排在 mastery 分支之前。
5. **不要把普通教材 RAG 当成核心创新。** 本项目重点是 teacher quality，不是 answer accuracy；RAG 的价值在教学决策信息，而不是让 DeepSeek “会做题”。
6. **不要让 retrieved examples 污染 original_problem。** 检索到的例题只能在 `show_worked_example` / `give_practice_question` 等明确动作中使用；普通讲解必须围绕原题，不能无条件把新题塞进上下文。
7. **Pedagogical RAG 只能作为教学决策辅助。** 它可以提供前置知识、常见误区、诊断问题、分层提示、变式题、掌握度阈值和策略建议，但不能替代 workflow 决策，不能推翻 math_checker。
8. **memory 和 RAG 都是软约束。** 它们只能影响讲解深浅、提示层级、复习和变式题选择，不能强制学生按脚本回答，也不能阻止学生跳步、直接给答案或问为什么。
9. **不要提交 .env 或任何 API key。** key 只从环境变量读取，永不打印、永不硬编码；`logs/`、`data/runtime/` 也在 .gitignore 中。
10. **所有 LLM 调用必须经过 `app/llm/deepseek_client.py`**，结构化输出用 JSON mode + Pydantic 校验。
11. **修改后必须运行并通过：**
   ```
   python -m pytest
   python scripts/run_eval.py
   ```
   两者默认全离线（mock LLM），必须全绿。
12. **如果测试失败，优先修测试暴露的真实问题**（grounding 被破坏、泄答案、多问题、误判步骤等），不要通过放宽断言/删测试来"变绿"。放宽评测期望需要有明确的教学合理性理由（先在 PR/说明中写清楚）。

## 教学行为不变式（改任何 prompt/策略/模板前先对照）

- original_problem 首轮抽取，跨轮不覆盖；回复始终围绕它。
- 非 show_worked_example/give_practice_question 的回复中，出现的方程必须与原题同解。
- 每轮恰好一个问句，放结尾。
- 学生未解出且未索要时，不出现最终答案。
- 学生可单步/多步/直接答案/问为什么，均需被正确认可或纠正。
- show_worked_example 只有三种触发：学生要例子、同知识点连续两次答错、说完全不懂。

## 修改 checklist

- [ ] 改了 workflow/prompt/模板 → 跑 pytest + run_eval
- [ ] 改了 math_checker → 补充/更新 `tests/test_math_checker.py`、`tests/test_flexible_steps.py`
- [ ] 新增教学能力 → 在 `data/eval/golden_conversations.jsonl` 加 golden case + evaluator 行为检查
- [ ] 改了数据 schema → 同步 `app/agent/state.py`、API response、UI 详情区、TurnLog
- [ ] 想验证真实 LLM 行为 → `USE_REAL_LLM_FOR_EVAL=true python scripts/run_eval.py`（花钱，谨慎）
