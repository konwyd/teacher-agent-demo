# LLM 主导原型结构化输出重试设计

## 问题

DeepSeek 可能生成合法 JSON，但把一个枚举值放入错误字段。例如把只属于
`task_shape` 的 `direct_fact` 填进 `disclosure_reason`。Pydantic 会正确拒绝
该对象，但当前原型直接把校验错误显示给学生。

## 决策

在 LLM 主导原型内部增加一次结构校验修复重试：

- 仅捕获 `pydantic.ValidationError`，不吞掉网络、超时或配置错误。
- 把原始回合上下文、Pydantic 错误位置、错误输入和允许值反馈给 DeepSeek。
- 重试仍使用 `LLMTeachingTurn` JSON Schema，由 Pydantic 再次校验。
- 最多重试一次；第二次失败时保留原有显式报错，避免无限调用。
- 初次教学生成与审计后的教学修复共用同一结构重试边界。
- 不修改全局 `DeepSeekClient`，不影响旧 Teacher Agent 或离线评测调用次数。

## 验收

- 首次返回 `disclosure_reason="direct_fact"` 时，原型发起第二次调用并得到合法回合。
- 第二次调用明确收到字段错误和合法字段含义。
- `DeepSeekError` 等非结构错误不重试。
- 不调用真实 API 的目标测试、完整 pytest 和 `scripts/run_eval.py` 全部通过。
