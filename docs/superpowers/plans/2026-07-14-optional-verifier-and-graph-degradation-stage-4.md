# 阶段四：可选数学验证与 GraphRAG 降级执行计划

**目标：** 在不改变一元一次方程兼容教学行为的前提下，明确数学验证和图检索的可选边界；无图模式不能读取本地 JSON 图数据，默认通用 Harness 不因验证器或检索器缺席而阻塞。

**范围依据：** `docs/topic-agnostic-teaching-harness-audit.md` 的阶段四设计，以及用户在 2026-07-14 对阶段四的直接实施授权。

## 保持不变的边界

- `app/agent/math_checker.py` 继续是既有一元一次方程步骤、计划和最终答案判定的唯一事实来源；不移动、不弱化其 Fraction 精确计算。
- `app/agent/linear_equation_adapter.py` 继续保留一元一次方程的概念识别、教学流程和兼容 API；通用 Harness 不新增按知识点分支。
- `app/retrieval/graph_retriever.py` 与 `app/retrieval/pedagogical_retriever.py` 的现有本地图谱查询能力和测试保持可用。
- Golden cases、严格评测断言、API 输出字段和已有三层记忆行为不放宽。

## 实施步骤

1. 在 `app/agent/optional_verifier.py` 定义 `VerificationRequest`、`OptionalVerifier` 协议、`NoopVerifier` 和 `LinearEquationVerifier`。前者一律返回 `not_applicable`；后者仅在题目可抽取为一元一次方程时调用现有 `math_checker.check_step`，并把结果映射为中性的 `VerificationResult`。
2. 在 `app/retrieval/context_provider.py` 定义 `GraphContextProvider` 协议、`EmptyGraphContextProvider` 与 `LocalGraphContextProvider`。空提供者不导入或实例化任何 JSON retriever；本地提供者延迟调用既有 `get_retriever` 和 `get_local_pedagogical_graph_retriever`。
3. 修改通用 `teaching_harness`：默认使用空验证器和空图上下文提供者，将其结果写入现有中性 `TeachingState`；显式注入时才调用实现。验证状态为 `not_applicable` 不得改变既有教学策略。
4. 修改线性方程兼容适配器：所有本地图谱访问经提供者获得；图禁用时返回空检索/空教学上下文和 `disabled` 后端，保留 equation checker、记忆和确定性练习回退。适配器的线性方程验证投影使用新的验证结果映射，不让通用层直接导入 `math_checker`。
5. 修改 `mode_runner._graph_setting`，同时控制显式 `USE_GRAPH_CONTEXT` 开关；`agent_no_graph` 选择空提供者，`agent_with_graph` 选择本地提供者。保留 `USE_PEDAGOGICAL_RAG` 的兼容环境变量含义。
6. 先新增单元和集成测试：验证器的适用/不可适用状态、空图提供者不触发 reader、通用 Harness 的无依赖降级、以及 `agent_no_graph` 在 monkeypatch 图 reader 为失败时仍可完成一轮教学。随后运行全量离线 pytest 与 mock eval。

## 验收命令

```powershell
python -m pytest tests/test_optional_verifier.py tests/test_graph_context_provider.py tests/test_eval_modes.py tests/test_topic_agnostic_workflow.py
python -m pytest
python scripts/run_eval.py
```

## 非目标

- 不新增任何数学知识包、题库、图节点、领域专用分支或真实模型调用。
- 不替换现有 `math_checker`，不把它推广成其它数学领域求解器。
- 不修改 prompt、现有 golden case 内容或评分标准，不修复与本阶段无关的问题。
- 不引入 Neo4j、Qdrant 或把 RAG 变为默认教学的必要前置条件。
