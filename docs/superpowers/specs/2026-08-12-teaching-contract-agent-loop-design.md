# 8502 教学契约 Agent Loop 与教材公式展示设计

日期：2026-08-12

## 目标

修复 8502 当前三个相互关联的问题：

1. 教师回复中的 LaTeX 定界符不稳定，导致 `\\frac`、`\\sqrt` 等源码直接显示；
2. `complete_response` 同时表示“完整解释概念”和“完整求解当前题”，导致学生只问求根公式时，教师擅自把 active problem 算到最终答案；
3. Agent 与 Tools 虽在运行，但它们被嵌套在旧的 `retrieving` 阶段中，页面刷新后又只显示固定阶段摘要，用户无法确认实际编排动作。

改造后，Agent 先提交可验证的教学契约，再按需调用 Skill 和 Tools。教师回答必须遵守契约规定的披露边界，并经过通用语义边界检查。数学公式以教材排版形式展示。页面持久显示真实 Agent trace，同时把确定性 verifier 降级为可选的窄域工具。

## 根因

### 公式显示

当前教师自由输出 Markdown。模型有时把展示公式写成 `[ x = \\frac{...} ]`，而不是 `$$...$$` 或 `\\[...\\]`。Streamlit 因而把它当作普通文本。逐字符输出还会在公式尚未闭合时反复渲染，进一步放大显示不稳定。

### 答案越界

`TurnGuidance.pacing=complete_response` 只描述回复完整度，没有描述允许披露的内容范围。系统提示又把概念解释、原因说明和完整解答放在同一个枚举值里。现有 `_collect_warnings()` 仅在一元一次方程 verifier 能识别最终形式时阻止答案泄露；二次方程和其他学科无法覆盖。

### Agent 不可见

`run_turn()` 仍先调用独立 Observer，再把整个 Agent Loop 放进名为 `retrieving` 的固定阶段。运行时 Agent 事件会短暂写入 status，但 `st.rerun()` 后历史回合只重建 Observer、RAG、审计三条固定摘要，没有渲染 `turn.agent_trace`。

## 总体方案

采用“教学契约 + 通用语义边界检查”，替代“不断扩展题型 verifier”。

```text
学生消息 + active problem + 最近历史
              │
              ▼
      Agent 分类并规划本轮教学契约
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
  load_skill  local RAG  guarded web
      └───────┼────────┘
              ▼
        submit_teaching_contract
              │
              ▼
          教师生成初稿
              │
              ▼
       通用语义边界检查
        │通过          │违规
        ▼              ▼
   公式规范化       按违规项修正一次
        └──────────────┘
              ▼
       教材公式排版 + 持久 Agent trace
```

## 教学契约

新增 `TeachingContract`，由 Agent 在 finish 时提交。它不是私有思维链，而是可持久化、可展示、可检查的教学决策。

字段：

- `response_mode`：
  - `concept_explanation`：完整解释当前概念，但不求解 active problem；
  - `one_step`：只推进一个产生新结果的步骤；
  - `guided_solution`：可以给方法和局部结果，但不披露最终答案；
  - `full_solution`：允许完整求解，仅在学生明确要求完整答案时合法；
  - `brief_redirect`：非 K9 简短回答并引导回学习内容。
- `requested_target`：本轮实际要回答的对象，例如“求根公式的定义”，不能自动扩展成整个 active problem。
- `allowed_content`：最多五条简短、公开的内容边界。
- `forbidden_content`：最多五条明确禁止项。
- `may_apply_to_active_problem`：是否允许把概念代入当前题。
- `may_reveal_final_answer`：是否允许披露最终答案。
- `reason`：公开、简短的契约理由。

硬约束：

- `full_solution` 才能设置 `may_reveal_final_answer=true`；
- `brief_redirect` 不能调用本地或联网工具；
- `concept_explanation` 默认 `may_apply_to_active_problem=false`、`may_reveal_final_answer=false`；
- 学生明确说“不要答案”时，任何模式都强制 `may_reveal_final_answer=false`；
- 普通“怎么解”“我不会”不得自动生成 `full_solution`。

求根公式场景的契约示例：

```json
{
  "response_mode": "concept_explanation",
  "requested_target": "解释一元二次方程求根公式及符号含义",
  "allowed_content": ["给出通用形式", "解释 a、b、c 和判别式"],
  "forbidden_content": ["代入当前方程", "计算当前方程的根"],
  "may_apply_to_active_problem": false,
  "may_reveal_final_answer": false,
  "reason": "学生询问公式本身，没有要求解当前方程"
}
```

## Observer 合并

删除独立的 Observer LLM 调用。请求分类、是否新题、学科、学段、回复模式和披露边界统一由 Agent 的首次结构化规划负责。

仍保留少量确定性规范化：

- 明确完整解答关键词才允许 `full_solution`；
- 明确不要答案时强制禁止最终答案；
- 旧历史缺少教学契约时使用安全默认值；
- Agent 输出非法或互相矛盾时允许修正一次，再降级为 `guided_solution` 或 `brief_redirect`。

这消除 Observer 与 Agent 对 `is_new_problem`、节奏和答案范围的重复判断，并把省下的一次本地模型调用用于语义边界检查，使正常回合调用次数基本不增加。

## 通用语义边界检查

新增结构化 `BoundaryAudit`，检查目标是教学边界，而不是对所有学科事实做通用正确性证明。

输入仅包含：

- 用户当前消息；
- active problem；
- `TeachingContract`；
- 教师初稿。

输出：

- `passed`；
- `violations`，枚举为 `off_target`、`applied_without_permission`、`final_answer_leak`、`too_many_steps`、`ignored_no_answer`、`unsupported_claim`；
- 每项公开的简短证据摘要；
- `repair_instructions`。

边界检查使用同一个本地 DeepSeek JSON 客户端，不调用外部模型。它不展示逐字思维链。发现违规时沿用现有“一次修正”机制，并把契约、违规类型和修正要求传给修正请求。修正后再做一次低成本确定性边界复核；若仍明显包含契约禁止内容，则返回受限安全回答，不展示违规初稿。

## verifier 的新定位

现有 `LinearEquationVerifier` 和 `math_checker` 保留，但只作为可选能力：

- 支持的题型可以核验具体等式变形和确定性数学断言；
- 不支持时明确显示 `not_applicable`，不再把“unverified”包装成全局审计失败；
- 不再承担通用答案披露边界；
- 未来某学科有高价值确定性工具时，通过 Tool Registry 按需增加，而不是扩展一个全科大审计器。

## 教材公式格式

最终展示统一使用 Streamlit/MathJax 可渲染的教材公式：

- 行内公式使用 `$...$`；
- 独立公式使用 `$$...$$`；
- 支持上下分数线、根号、上下标、正负号和关系符号；
- 不把公式改成斜杠纯文本；
- 不改变代码块、普通方括号和引用文本。

新增纯函数 `normalize_math_markdown(text)`，只修复高置信度的数学定界符问题：

- `\\[...\\]` 统一为 `$$...$$`；
- 单独成段、内部含明确 LaTeX 命令的 `[ ... ]` 转为 `$$...$$`；
- `\\(...\\)` 统一为 `$...$`；
- 普通中文方括号不转换。

规范化在回答通过边界检查后、写入 `PrototypeTurnRecord` 前执行，因此当前回合和历史回放使用同一份可渲染文本。当前回答不再逐字符切断公式；按“普通文本块 + 完整公式块”流式展示，公式块闭合后一次提交。

## 页面与可观测性

页面主进度改为两层：

1. `Agent Loop`：分类、加载 Skill、本地检索、联网门禁、提交教学契约；
2. `Answer Guardrails`：生成、语义边界检查、必要修正、公式规范化。

历史回合的“验证详情”新增：

- 学科和学段；
- 教学契约；
- 持久化 Agent trace，每条显示 step、动作/Tool、成功状态、耗时和公开摘要；
- BoundaryAudit 结果和修正项；
- 可选 verifier 结果。

移除固定的“识别请求、检索参考、审计回答”三条伪摘要。旧的阶段耗时可以保留为性能数据，但标题改为真实组件名称，不再让用户误以为固定阶段就是编排器。

## 数据兼容

`PrototypeTurnRecord` 新增以下带默认值字段：

- `teaching_contract: TeachingContract | None = None`；
- `boundary_audit: BoundaryAudit | None = None`；
- `formula_normalized: bool = False`。

旧会话缺少这些字段时正常加载。登录、用户隔离、conversation ID、pending message、上下文计量器、Token 和 KV cache 字段不改名、不迁移数据库。

## 错误处理

- Agent 规划失败：一次结构化修正后使用安全教学契约；
- Tool 失败：作为 observation 返回，不终止 Streamlit；
- 边界检查失败或超时：采用契约驱动的保守修正，不能直接放行初稿；
- 修正仍越界：生成不含具体答案的受限回复；
- 公式规范化失败：保留原文本并记录诊断，不中断回答；
- verifier 不支持：显示为可选工具不适用，不阻断其他学科教学。

## 测试与验收

必须先写失败测试，覆盖：

1. “求根公式是什么”生成 `concept_explanation` 契约，禁止代入当前二次方程和披露根；
2. “请完整解这道题”才允许 `full_solution`；
3. “不要告诉我答案”覆盖所有其他模型建议；
4. 初稿给出当前二次方程的根时，BoundaryAudit 触发修正；
5. 概念解释完整但未解 active problem 时通过；
6. 普通方括号保持不变，LaTeX 方括号定界符转换成教材公式；
7. 分数、根号、上下标在 Streamlit Markdown 中使用闭合的 `$`/`$$`；
8. 当前回合与历史回放使用同一规范化文本；
9. Agent Tool trace 在运行中和刷新后的详情中都可见；
10. 页面不再用固定三条摘要冒充 Agent 编排；
11. 旧会话、登录、pending message、上下文计量器、Token/KV cache 测试继续通过；
12. 全量 pytest 和 38 条离线教学评测通过。

真实 8502 验收至少包含：

- 在已有 `3x^2+2x-11=0` active problem 中追问“求根公式是什么”，只显示教材排版的通用公式和符号解释，不代入、不计算根；
- 明确说“请用求根公式完整解这道题”时允许完整求解；
- 验证详情中能看到实际 `load_skill`、`search_local_knowledge`、`finish/submit_teaching_contract` trace；
- 数学、物理和 off-topic 各完成一轮，8502 WebSocket 与 8080/8002 服务保持健康。

## 非目标

- 不实现覆盖全部数学题型和九学科的确定性正确性证明；
- 不引入外部审计模型或新付费 API；
- 不把模型私有思维链展示给用户；
- 不移除现有登录、历史、上下文计量器、Token 或 KV cache 功能；
- 不修改或提交 Chroma 数据文件。
