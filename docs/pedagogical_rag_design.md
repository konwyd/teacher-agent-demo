# Pedagogical Graph RAG Design

## 1. 项目定位

本项目不是为了证明“给 DeepSeek 接一个 K9 教材 RAG 后，模型更会做题”。DeepSeek 已经掌握大部分初中数学基础知识，单纯把教材段落、例题解析或答案检索出来，并不能构成本项目的核心创新。

本项目要做的是中文初中数学 Pedagogical Agent：让 LLM 在 Agent workflow、student memory、math_checker、evaluation 和后续 Pedagogical Graph RAG 的约束下，更像一个循序渐进的好老师。研究重点不是 answer accuracy，而是 teacher quality：是否能诊断学生卡点，是否能分层提示，是否守住原题主线，是否不过早泄答案，是否能根据学生掌握度调节讲解方式。

## 2. 为什么普通 K9 Textbook RAG 不是核心

普通 Textbook RAG 的典型价值是补充事实知识、教材表述和例题材料。但在本项目里，它不是核心原因有三点：

1. DeepSeek 对一元一次方程、等式性质、移项、去括号等 K9 知识通常已经会做，普通检索不会显著改变基础求解能力。
2. 教学失败往往不是“不知道答案”，而是“不知道此刻该怎么教”：该先诊断还是提示，该不该给例题，该问哪一个问题，该如何处理学生跳步。
3. 普通教材片段如果直接进入 prompt，容易污染 `original_problem`，引入无关方程、提前暴露最终答案，或让老师偏离当前学生的真实卡点。

因此，教材文本 RAG 可以作为后续补充材料来源，但不能被当成本项目的主要创新点，也不能替代 workflow、math_checker、memory 和 evaluation。

## 3. 为什么 Pedagogical Graph RAG 更有价值

Pedagogical Graph RAG 的核心不是“检索答案”，而是检索教学决策信息。它应该告诉 Agent：

- 当前概念依赖哪些前置知识。
- 学生的错误可能对应哪些常见误区。
- 下一步最小诊断问题是什么。
- 可以给哪一级提示，提示到什么程度仍不泄最终答案。
- 什么情况下给 worked example，什么情况下回到原题。
- 学生掌握度处于低、中、高时，教学策略如何变化。
- 下一道变式题应该如何控制难度递进。

这类信息更适合作为图谱，因为教学决策依赖关系明确：概念有前置，误区连接到诊断问题，提示层级连接到教学动作，练习题连接到难度和技能，掌握度阈值连接到策略。

## 4. Workflow、Memory、Math Checker、RAG 的分工

### Agent workflow

workflow 是教学状态机，负责把每轮对话拆成稳定步骤：诊断输入、检索证据、选择教学动作、生成回复、更新记忆、写日志。它决定“此刻走哪条教学分支”，不能被 RAG 输出直接替代。

### math_checker

math_checker 是数学步骤、学生多步计划和最终答案的确定性裁判。只要学生给出可判定的数学步骤，checker 结论必须优先于 LLM、memory 和 RAG。RAG 可以解释“为什么这种错法常见”，但不能推翻 checker 对对错的判断。

### student memory

memory 保存学生在概念和技能上的掌握度，用来调节讲解深浅、复习频率、类比方式和变式题难度。它是软约束，不能强制学生按脚本一步一答，不能拒绝学生跳步，也不能覆盖 checker 分支。

### Pedagogical RAG

Pedagogical RAG 提供教学决策辅助，包括前置知识、常见误区、诊断问题、分层提示、例题、练习、掌握度阈值和策略建议。它也是软约束：workflow 可以参考它选择动作和生成素材，但不能让 retrieved examples 污染 `original_problem`，不能无条件把例题塞进普通提示，也不能把检索结果当成学生当前题目的主线。

## 5. Pedagogical Graph Schema

### Concept

表示一个教学概念或微概念。

建议字段：

- `concept_id`: 稳定 ID，例如 `linear_equation`
- `name`: 中文名称
- `grade`: 年级
- `description`: 简要定义或教学说明
- `prerequisites`: 前置概念 ID 列表
- `skills`: 相关技能 ID 列表
- `common_misconceptions`: 常见误区 ID 列表
- `diagnostic_questions`: 诊断问题 ID 列表
- `hint_levels`: 分层提示 ID 列表
- `recommended_exercises`: 推荐练习 ID 列表
- `mastery_based_strategy`: 不同掌握度区间下的策略配置

### Prerequisite

表示概念之间的前置关系。可以在 mock JSON 中作为 `prerequisites` 字段，也可以在图数据库中建边。

建议边：

- `(Concept)-[:REQUIRES]->(Concept)`
- 边属性：`importance`, `reason`

### Misconception

表示学生常见错误及其诊断线索。

建议字段：

- `misconception_id`
- `name`
- `description`
- `example_wrong_step`
- `diagnostic_signal`
- `repair_strategy`

建议边：

- `(Misconception)-[:ABOUT]->(Concept)`
- `(Misconception)-[:CHECKED_BY]->(DiagnosticQuestion)`

### DiagnosticQuestion

表示用来判断学生是否理解某个前置点或误区的问题。

建议字段：

- `question_id`
- `prompt`
- `target_concept_id`
- `checks`
- `expected_signal`
- `avoid_revealing_answer`

建议边：

- `(Concept)-[:HAS_DIAGNOSTIC]->(DiagnosticQuestion)`
- `(DiagnosticQuestion)-[:DIAGNOSES]->(Misconception)`

### HintLevel

表示逐层提示，控制从轻提示到更具体提示的递进。

建议字段：

- `hint_id`
- `level`: 1/2/3
- `prompt_template`
- `reveals_answer`: 默认 `false`
- `next_if_failed`

建议边：

- `(Concept)-[:HAS_HINT]->(HintLevel)`
- `(HintLevel)-[:NEXT_HINT]->(HintLevel)`

### WorkedExample

表示例题。只能在 `show_worked_example` 动作下进入回复上下文。

建议字段：

- `example_id`
- `stem`
- `solution_steps`
- `teaching_focus`
- `concept_ids`
- `difficulty`
- `return_to_original_prompt`

建议边：

- `(WorkedExample)-[:ILLUSTRATES]->(Concept)`
- `(WorkedExample)-[:ADDRESSES]->(Misconception)`

### Exercise

表示练习题或变式题。

建议字段：

- `exercise_id`
- `stem`
- `answer`
- `solution_steps`
- `concept_ids`
- `difficulty`
- `variation_from`
- `common_misconceptions`

建议边：

- `(Exercise)-[:PRACTICES]->(Skill)`
- `(Exercise)-[:TARGETS]->(Concept)`

### Skill

表示可练习、可评估的操作能力。

建议字段：

- `skill_id`
- `name`
- `observable_behaviors`
- `related_concepts`

建议边：

- `(Concept)-[:BUILDS_SKILL]->(Skill)`

### MasteryThreshold

表示掌握度阈值和教学动作之间的映射。

建议字段：

- `threshold_id`
- `concept_id`
- `low`: 例如 `0.0-0.3`
- `medium`: 例如 `0.3-0.7`
- `high`: 例如 `0.7-1.0`
- `policy`

建议边：

- `(Concept)-[:USES_THRESHOLD]->(MasteryThreshold)`
- `(MasteryThreshold)-[:SELECTS]->(TeachingStrategy)`

### TeachingStrategy

表示教学策略建议。

建议字段：

- `strategy_id`
- `name`
- `when_to_use`
- `preferred_actions`
- `avoid`
- `teacher_tone`

建议边：

- `(TeachingStrategy)-[:USES_HINT]->(HintLevel)`
- `(TeachingStrategy)-[:USES_DIAGNOSTIC]->(DiagnosticQuestion)`
- `(TeachingStrategy)-[:RECOMMENDS_EXERCISE]->(Exercise)`

## 6. Mock JSON 到 Neo4j / Qdrant 的迁移路径

当前阶段只保留本地 mock JSON，不接 Neo4j、不接 Qdrant、不接真实教材 PDF。迁移可以分四步做：

1. **Mock JSON 固化 schema**：先用 `data/sample/pedagogical_graph.json` 表达概念、前置、误区、诊断问题、提示层级、练习和掌握度策略，确保 evaluator 能保护教学行为。
2. **Retriever 接口稳定化**：保持 `app/retrieval/graph_retriever.py` 的调用语义稳定，只让它返回教学决策证据，不让它直接覆盖 workflow 决策。
3. **Neo4j 图谱迁移**：当 mock schema 稳定后，将 `Concept`、`Misconception`、`DiagnosticQuestion`、`HintLevel`、`Exercise`、`Skill`、`MasteryThreshold`、`TeachingStrategy` 映射为节点，将 prerequisites、diagnoses、has_hint、recommends_exercise 等映射为边。先做只读 retriever，不做自动写图。
4. **Qdrant 向量补充**：Qdrant 只存长文本材料，如教材解释、例题讲解、误区描述的扩展文本。图谱负责教学路径和结构关系，向量库负责补充表述材料。检索结果进入 teach prompt 前必须经过 workflow 动作过滤，尤其是 worked examples 只能在 `show_worked_example` 或 `give_practice_question` 场景使用。

## 7. 安全边界

- Pedagogical RAG 只能作为教学决策辅助，不能替代 math_checker。
- memory 和 RAG 都是软约束，不能强制学生按脚本回答。
- retrieved examples 不得污染 `original_problem`。
- 普通教材 Textbook RAG 不是核心创新，不能把项目目标改写成“让模型更会做题”。
- 新增 RAG 能力必须继续通过 `python -m pytest` 和 `python scripts/run_eval.py`。
