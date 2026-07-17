# LLM 主导原型非阻断验证设计

## 目标

即使当前数学问题没有确定性 verifier，DeepSeek 仍必须能够完成自然教学并把回答展示给学生。
`UNVERIFIED` 只表示 Agent 没有程序证据，不表示回答错误，也不得成为阻断条件。

本设计取代原型中“主回答必须填写 `LLMTeachingTurn` 复杂 JSON”的运行方式，并使
`2026-07-17-llm-led-schema-retry-design.md` 只保留为历史实验记录。

## 主流程

1. Agent稳定保存原题、真实对话历史和已有 deterministic verification。
2. DeepSeek通过 `DeepSeekClient.chat()` 只生成面向学生的自然语言回复。
3. Agent调用 `math_checker.find_equations(response)` 提取可解析的一元一次方程。
4. 可解析方程走现有 `LinearEquationVerifier`；无法解析的文字结论不猜测、不阻断。
5. 只有确定性冲突、受支持题型的明显提前泄露、空回复等硬错误才阻断或触发一次纯文本修正。
6. 没有 verifier 时记录 `unverified`，原回复正常进入会话和 UI。

## 数据边界

LLM 主调用不再生成以下研究元数据：

- `action_label`
- `student_assessment`
- `disclosure_level` / `disclosure_reason`
- `task_shape` / `response_scope`
- `evidence_turn_ids`
- `confidence` / `rationale`
- `mathematical_claims`

回合记录只保留教师文本、学生步骤验证、自动提取的方程、方程审计、修正原因和剩余警告。
后续若需要研究标签，应由独立的非阻断 observer 产生；observer 失败只能写 `unknown`。

## 保留的不变式

- original problem 跨轮稳定。
- 一元一次方程继续使用现有精确 checker。
- 多步方程普通求助默认每轮只推进一个有意义步骤。
- 学生明确索要完整解答时允许完整回答。
- 学生明确不要答案时，对 checker 能识别的最终形式继续审计。
- 每轮最多一个必要问题。
- 不调用真实 API 运行自动化测试。

## 验收

- 相反数、几何等无 verifier 问题可返回自然语言并标记 `unverified`。
- 主教学调用不使用 `chat_json`，不可能再因研究枚举填错而失败。
- 教师回答中的可解析方程由 `find_equations` 自动进入 checker。
- 错误方程可触发一次纯文本修正。
- 完整 pytest 与离线 `run_eval` 通过。
