# Skill 编排接口修复设计

## 背景与问题

当前 Agent 提示词要求“同一参数不要重复调用”，但运行时只对本地检索做了重复签名检查，没有约束 `load_skill`。同时，Agent 决策输入没有包含可用 Skill 的名称和说明。模型因而会猜测“数学教学技能”等不存在的名称，并在四轮预算内重复调用，造成无效 Token 消耗和通用安全兜底回答。

本次只修复 Skill 编排接口，不扩充学科 Skill，不调整 KV、模型上下文、登录、缓存统计或 UI 布局。

## 目标

- Agent 每轮都能看到当前运行时的正式 Skill 目录。
- `load_skill` 只能加载目录中存在的规范名称。
- 相同的无效 Skill 请求不能重复消耗迭代预算。
- 非法 Skill 请求能够确定性降级到与回答模式匹配的现有 Skill。
- 保持教学契约、数学检查器和现有知识检索边界不变。

## 非目标

- 不增加“一元一次方程教学”等学科专用 Skill。
- 不重构整个 Agent Loop 或替换当前模型提供者。
- 不改变教材库索引、联网搜索条件或答案审计策略。
- 不解决本次记录中独立存在的边界审计 `ValidationError`；该问题另开修复轮。

## 方案

### 1. 显式 Skill 目录

编排器在调用 Action Provider 时加入 `available_skills`，每项包含规范 `name` 和简短 `description`。Agent 系统提示同步说明 `load_skill.name` 必须逐字取自该目录，不得创建学科名、自然语言别名或新名称。

目录由 `OrchestrationRuntime.skill_catalog` 生成，避免提示词和运行时各维护一份名单。

### 2. 确定性 Skill 选择

新增一个无副作用的选择函数，根据已经校验的分类结果返回现有 Skill：

| 条件 | 规范 Skill |
|---|---|
| 明确授权完整解答 | `complete_solution` |
| 概念解释 | `concept_explanation` |
| 有具体错误诊断信号 | `error_diagnosis` |
| 其余引导式请求 | `socratic_guidance` |

当模型提交不存在的 Skill 时，编排器不再次询问模型猜名称，而是加载上述确定性选择结果，并在 trace 中记录原请求失败及自动降级结果。降级不改变教学契约，也不扩大答案披露权限。

### 3. 运行时硬性去重

`OrchestrationRuntime` 记录已经尝试的 Skill 名称。工具注册表在执行 `load_skill` 前检查：

- 已成功加载同名 Skill：返回幂等成功，不重复追加。
- 已失败的同名 Skill：返回 `duplicate skill request`，不再次查找。
- 新的非法名称：记录为已失败并返回 `unknown skill`。

编排器对第一次 `unknown skill` 立即执行确定性降级，因此正常链路不会进入第二次非法调用。硬性去重作为防御层，覆盖自定义 Provider 和未来调用方。

### 4. 可观察性

Agent trace 增加但不暴露内部推理：

- 非法请求的 Skill 名称；
- 自动选择的规范 Skill；
- 降级原因 `unknown_skill_fallback`。

公开进度只展示一次“教学方法已自动匹配”，不再连续显示多条加载失败。现有 Token 统计逻辑保持不变。

## 数据流

1. 分类器产生已校验的学科、学段和回答模式。
2. 编排器构造运行时 Skill 目录并将其作为 `available_skills` 传给 Action Provider。
3. Provider 返回 `load_skill` 决策。
4. 工具注册表校验名称和重复状态。
5. 若名称合法则正常加载；若名称非法，编排器立即加载确定性选择的规范 Skill。
6. 后续 Agent 观察中包含规范 Skill 结果，循环继续或生成教学契约。

## 错误处理

- Skill 目录为空时不尝试自动加载，保留安全教学契约并以明确 termination reason 收口。
- 确定性选择结果不在目录中时不猜测其他名称，使用现有安全契约收口。
- 自动降级最多执行一次，不增加 Agent 决策调用次数。
- Tool 名非法、检索失败和 Provider JSON 错误沿用现有处理，不在本次范围内改变。

## 测试策略

先写失败测试，再实现生产代码：

1. Action Provider 每轮输入包含完整、准确的 `available_skills`。
2. 模型请求不存在的“数学教学技能”时，只产生一次模型决策，随后自动加载 `socratic_guidance`。
3. 明确完整解答请求的非法 Skill 自动降级到 `complete_solution`，且教学契约仍由原始授权约束。
4. 重复的非法 Skill 名称在工具层返回 `duplicate skill request`。
5. 重复加载已成功 Skill 为幂等成功，不产生重复 `loaded_skills`。
6. 现有 Agent Loop、快速路由、公开进度和教学边界测试全部通过。
7. 运行仓库要求的完整 `python -m pytest` 与 `python scripts/run_eval.py`。

## 验收标准

- “老师，2x + 3 = 9 怎么解？”不再出现四次 `unknown skill`。
- 即使模型生成非法 Skill 名，也最多发生一次无效请求，并自动使用合法教学 Skill。
- 相同场景的 Agent 决策调用次数不因 Skill 名错误增长到迭代上限。
- 不改变最终答案披露权限，不绕过数学检查器，不破坏现有 UI、登录、缓存和上下文统计功能。
