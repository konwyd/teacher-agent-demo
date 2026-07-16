# AGENTS.md — 给 coding agent（Codex 等）的工作规则

本仓库是中文初中数学 Teacher Agent research demo。**runtime LLM 是 DeepSeek API，你只是开发助手。**
接手前先读 `CODEX_HANDOFF.md`（完整交接文档）和 `CLAUDE.md`（原始项目规范）。

## 必须遵守

1. **不要重构整个项目。** 目录分层（api/agent/llm/retrieval/memory/evaluation）是刻意设计，做增量修改。
2. **不要删除或弱化 evaluation。** `data/eval/` 的 golden cases 是回归门；`app/evaluation/` 和 `scripts/run_eval.py` 保护现有教学能力。
3. **不要绕过 math_checker。** 学生步骤/多步计划/最终答案的对错判定必须走 `app/agent/math_checker.py`（Fraction 精确计算），不允许交给 LLM 自由判断，也不允许在 teach 阶段推翻 checker 结论。
4. **不要把 memory 变成硬流程控制。** 掌握度只影响讲解深浅/类比/复习/变式题；不得用它拒绝学生跳步、强制一步一答，checker 判定分支必须始终排在 mastery 分支之前。
5. **不要提交 .env 或任何 API key。** key 只从环境变量读取，永不打印、永不硬编码；`logs/`、`data/runtime/` 也在 .gitignore 中。
6. **所有 LLM 调用必须经过 `app/llm/deepseek_client.py`**，结构化输出用 JSON mode + Pydantic 校验。
7. **修改后必须运行并通过：**
   ```
   python -m pytest
   python scripts/run_eval.py
   ```
   两者默认全离线（mock LLM），必须全绿。
8. **如果测试失败，优先修测试暴露的真实问题**（grounding 被破坏、泄答案、多问题、误判步骤等），不要通过放宽断言/删测试来"变绿"。放宽评测期望需要有明确的教学合理性理由（先在 PR/说明中写清楚）。

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
