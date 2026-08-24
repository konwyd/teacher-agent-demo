# Repository Subsystem Ownership

本文件定义多开发者 / 多 Codex sessions 并行开发时的子系统责任边界。

Owner 当前可先填写为 `[TBD]`；在负责人确定前，路径、protected interfaces 和 mandatory co-review 规则仍然有效。

Ownership 表示维护、架构责任和最终审核责任，不允许 owner 绕过 `AGENTS.md` 中的仓库级不变式。


## 1. Governance Model

- 每个文件或接口只有一个 primary owner。
- shared interface 使用“single owner + mandatory co-review”，不采用模糊的共同 ownership。
- 目录位置只是 ownership 的证据之一；实际归属同时由依赖方向、数据流、接口消费者和架构责任决定。
- integration file 即使包含多个子系统的数据，也保留单一 owner；跨域 contract 通过 mandatory co-review 治理。
- 修改跨子系统 contract 前，作者必须列出：
  - producer
  - consumers
  - schema compatibility
  - persistence compatibility
  - evaluation impact
  - migration/default behavior
  - rollback behavior
- 8502 是当前唯一主运行时。
- `prototypes/llm_led_teacher/**` 中属于 orchestration/runtime 的代码视为正式 Agent Runtime，不因目录名为 `prototypes` 而视为实验性实现。
- 8501/8000 是冻结的 legacy baseline，只接受 regression fixes 和 critical fixes；新功能不承担 legacy feature compatibility 义务。


## 2. Active Session Scope

本仓库可能由多个开发者通过不同 Codex conversations 同时开发，并且这些 conversations 可能来自同一个 Codex 账号。

Codex 不得根据：
- 当前账号
- conversation history
- git author
- workspace 名称

推断当前开发者身份或当前 session 所代表的 subsystem。

每个开发任务开始时，应明确当前 session 的工作范围：

Active subsystem:
[SUBSYSTEM]

Acting owner:
[OWNER]

Task:
[SHORT DESCRIPTION]

如果当前 active subsystem 无法从当前任务中明确判断，Codex 必须在修改代码前向开发者确认。

一个 session 默认只允许：

- 修改其 active subsystem 的 primary paths；
- 修改已经明确批准的 shared interfaces；
- 修改为完成当前任务所必需且得到明确批准的跨 subsystem 文件。

一个 session 不得：

- 因发现其他 subsystem 的问题而顺手修改；
- 清理看似“无关”但可能属于其他开发者的 dirty changes；
- 将其他 subsystem 的问题自动纳入当前 task scope；
- 根据目录名自行扩展 ownership。

如果发现跨 subsystem blocker，应先报告：
- blocker 所在 subsystem
- 相关文件
- 当前任务为何依赖它
- 建议由谁处理

未经明确批准不得跨 ownership 修改。


## 3. Runtime Generations

### Primary Runtime

当前正式运行时：

- `prototypes/llm_led_teacher/**` 中的 8502 runtime components

8502 是当前产品与研究开发主线。

### Frozen Legacy Baseline

以下链路属于 8501/8000 legacy baseline：

- `app/agent/workflow.py`
- `app/api/**` 中仅服务旧 baseline 的链路
- `app/ui/**` 中仅服务旧 baseline 的链路
- 其他明确属于旧 LangGraph / 8000 / 8501 的代码

Legacy baseline：

- 不接受新 feature development；
- 只接受 regression fixes；
- 只接受 critical fixes；
- 不要求为 8502 新功能实现 feature compatibility；
- 但 shared interface 修改仍不得导致无意的 import failure、startup failure 或基础测试破坏。

Legacy compatibility protection 不等于继续维护 legacy feature parity。


# 4. Agent Runtime

Owner:
3322

Primary paths:

- `prototypes/llm_led_teacher/logic.py`
- `prototypes/llm_led_teacher/orchestrator.py`
- `prototypes/llm_led_teacher/fast_plan.py`
- `prototypes/llm_led_teacher/fast_routing.py`
- `prototypes/llm_led_teacher/teaching_policy.py`
- `prototypes/llm_led_teacher/boundary_audit.py`
- `prototypes/llm_led_teacher/orchestration_models.py`
- `app/agent/**`

Responsibilities:

- 8502 turn orchestration
- runtime routing
- bounded Agent Loop
- active problem continuity
- TeachingContract 生成与执行
- 教学策略
- disclosure behavior
- safety boundary
- response generation
- repair
- deterministic verifier integration
- public trace / turn record 构造
- frozen legacy baseline 的 regression / critical maintenance

Protected interfaces:

- `app/agent/math_checker.py`
- `app/agent/optional_verifier.py`
- `app/agent/state.py`
- `TeachingContract`
- `RequestClassification`
- `AgentOrchestrationResult`
- `PrototypeEpisode`
- `PrototypeTurnRecord`
- Agent-to-Memory writeback payload
- Agent-to-Retrieval request/result contracts

Mandatory co-review:

- Memory System：
  - memory resolution
  - writeback payload
  - episode/student evidence contracts

- Retrieval / GraphRAG：
  - tool permission
  - retrieval decision
  - evidence contracts
  - RagContext contracts

- Evaluation Framework：
  - observable teaching behavior
  - checker behavior
  - safety boundary changes

- LLM Infrastructure：
  - client behavior
  - usage contract
  - KV behavior
  - structured-output contracts

- Product：
  - persisted turn schemas
  - UI-visible turn schemas
  - public trace

Rules:

- `math_checker` 在适用范围内保持数学正确性权威。
- Memory 和 RAG 只能作为软约束或辅助证据。
- Memory 和 RAG 不得覆盖 verifier 或 TeachingContract。
- Agent Runtime 不得直接操作 learner-memory 数据库表。
- Agent Runtime 不得自行实现 Memory admission/lifecycle 逻辑。
- 不持久化或展示 private reasoning。
- 不为新功能维护 8501/8000 feature compatibility。


## New Files Under `prototypes/llm_led_teacher/**`

新文件不得仅因为位于 `prototypes/llm_led_teacher/**` 就自动归 Agent Runtime。

新文件 ownership 根据架构职责确定：

- orchestration / runtime / teaching policy → Agent Runtime
- UI / product interaction → Product
- retrieval adapter / runtime retrieval → Retrieval / GraphRAG
- student memory / experience memory semantic lifecycle → Memory System
- model gateway / inference infrastructure → LLM Infrastructure
- evaluation-only logic → 对应 Evaluation owner

如果无法明确判断，应在实现前确定 ownership。


# 5. Memory System

Owner:
Haoxi

Memory System 包含两个逻辑上严格隔离的 memory domain：

1. Student Memory
2. Agent Experience Memory


## 5.1 Student Memory

Primary paths:

- `app/memory/**`
- `tests/behavior/test_structured_student_memory.py`
- `tests/behavior/test_memory_v2_baseline.py`
- Memory-specific tests and fixtures

Responsibilities:

- structured learner-memory storage
- memory resolution
- async writeback
- global profile
- topic memory
- conversation / episode persistence
- candidate extraction contracts
- provenance
- temporal scope
- lifecycle
- admission
- matching
- evidence ledger
- shadow telemetry
- user/conversation scoping
- privacy
- persistence reliability

Protected interfaces:

- `app/memory/contracts.py`
- `app/memory/schemas.py`
- `StudentMemoryService` public methods and models
- `StudentMemoryResolution.prompt_payload()`
- `LocalConversationStore` public archive API
- authoritative baseline enqueue/outbox/write path
- Memory V2 shadow record schemas
- Memory V2 ledger schemas

Mandatory co-review:

- Agent Runtime：
  - resolution invocation
  - writeback invocation
  - payload semantics
  - episode fields

- Evaluation Framework：
  - public evaluation contracts
  - shared metric semantics

- Product：
  - identity
  - conversation ownership
  - archive behavior

- Data Pipeline：
  - archive read contract

- Retrieval：
  - student memory / curated knowledge / experience source boundaries

Rules:

- Memory V2 ownership属于 Memory System。
- Memory V2 必须保持 shadow，直到新的架构决策明确批准 authoritative behavior。
- shadow admission/controller 不得授权、阻止或改变 authoritative baseline writes。
- Memory failure 必须 fail open，不得阻断正常教学。
- Memory 不得覆盖 checker。
- Memory 不得覆盖 TeachingContract。
- Memory 不得覆盖当前学生消息。
- curated knowledge 不得隐式写入 student memory。
- Agent Experience Memory 不得隐式写入 Student Memory。


## 5.2 Memory Admission Evaluation

Memory V2 专项 evaluation 属于 Memory System，而不是通用 Evaluation Framework。

Primary paths:

- `app/evaluation/memory_admission/**`
- `scripts/run_memory_admission_eval.py`
- `scripts/run_memory_admission_robustness.py`
- Memory admission development datasets
- Memory admission holdout datasets
- Memory admission robustness datasets
- Memory admission manifests
- Memory admission probes / trajectories / reports

Responsibilities:

- Memory admission policy benchmark
- development / holdout evaluation
- robustness evaluation
- adversarial probes
- long-trajectory evaluation
- admission-specific metrics
- admission-specific manifests
- policy freeze / dataset seal verification
- memory-specific experimental reporting

Protected interfaces:

- Memory admission development dataset
- Memory admission holdout dataset
- robustness dataset
- frozen policy manifests
- dataset label seals
- admission-specific metric definitions
- benchmark decision labels

Mandatory co-review:

- Evaluation Framework：
  - 如果修改公共 evaluator framework
  - 如果修改通用 metric semantics
  - 如果修改跨 subsystem evaluation contract

- Agent Runtime：
  - 如果 benchmark 依赖 Agent-produced runtime payload contract

Rules:

- Memory owner 可以维护 Memory admission benchmark。
- sealed holdout / frozen labels 不得仅为了提高结果而修改。
- benchmark 数据变更必须记录原因。
- policy changes 与 benchmark-label changes 必须分离治理。
- 不得通过修改 dataset expectation 掩盖 policy failure。


## 5.3 Agent Experience Memory / Quality-QA

Quality-QA 在语义上属于 Agent Experience Memory。

它不是 Student Memory。

Semantic owner:
Memory System

Current relevant paths may include:

- `app/data_pipeline/quality_qa_pipeline.py`
- Quality-QA candidate / experience schemas
- Quality-QA admission / versioning / lifecycle logic
- Quality-QA experience-store logic

Responsibilities:

- agent teaching experience 的语义定义
- experience candidate lifecycle
- experience admission
- deduplication
- versioning
- provenance
- quality gates
- experience-memory storage contract
- experience-memory lifecycle

Rules:

- Quality-QA 不得写入 Student Memory。
- Quality-QA 不得与 student profile/topic memory 混合。
- Student Memory 与 Agent Experience Memory 必须保持 source-class 隔离。
- Experience evidence 不得被当作某个学生的个人事实。
- 将来即使代码物理迁入 `app/memory/**`，也应保持 Student Memory / Agent Experience Memory 的逻辑隔离。

Runtime retrieval ownership 仍属于 Retrieval / GraphRAG。

通用离线文件加工基础设施仍可属于 Data Pipeline。


# 6. Evaluation Framework

Owner:
3322

Primary paths:

- `app/evaluation/**`，但不包括 `app/evaluation/memory_admission/**`
- `data/eval/**`
- `scripts/run_eval.py`
- general evaluation-specific tests and fixtures

Responsibilities:

- golden conversation replay
- deterministic teaching-behavior checks
- grounding checks
- disclosure checks
- checker behavior evaluation
- RAG contamination checks
- generic teaching quality regression
- shared evaluation framework
- shared evaluator contracts
- shared metric infrastructure
- baseline comparison
- general reporting

Protected interfaces:

- general golden datasets
- general holdout datasets
- evaluator case/result schemas
- shared metric definitions
- pass/fail semantics
- checker-backed correctness expectations

Mandatory co-review:

- Agent Runtime：
  - teaching behavior expectation changes
  - safety contract expectation changes

- Memory：
  - shared evaluation contract changes affecting Memory

- Retrieval：
  - grounding/evidence/contamination expectations

- LLM Infrastructure：
  - mock/live evaluation isolation

Rules:

- Production runtime 不得导入 evaluation runtime。
- golden/holdout datasets 必须保持受保护状态。
- 不得只为了测试变绿而：
  - 删除 case
  - 放宽断言
  - 改 metric
  - 改 expected behavior
- 数据或期望变更必须有明确教学/架构理由。
- Memory admission 专项 benchmark 的 semantic ownership 属于 Memory System。


# 7. Retrieval / GraphRAG

Owner:
[TBD]

Primary paths:

- `app/retrieval/**`
- `prototypes/llm_led_teacher/rag_hook.py`
- `prototypes/llm_led_teacher/retrieval_policy.py`
- `prototypes/llm_led_teacher/orchestration_tools.py`
- `prototypes/llm_led_teacher/qa_cache.py`
- Quality-QA runtime retrieval adapters

Responsibilities:

- graph retrieval
- vector retrieval
- web retrieval
- retrieval provider contracts
- retrieval routing
- thresholds
- diagnostics
- retrieval provenance
- guarded knowledge tools
- teaching evidence delivery
- Quality-QA runtime querying
- curated knowledge / student memory / agent experience source separation

Protected interfaces:

- `GraphContextProvider`
- `GraphContextRequest`
- `RetrievalEvidence`
- `RagContext`
- retrieval decision schemas
- ToolResult retrieval payload
- source identifiers
- Quality-QA runtime query/result contracts

Mandatory co-review:

- Agent Runtime：
  - tool permission
  - evidence use
  - TeachingContract integration

- Memory：
  - source-class separation
  - Quality-QA experience schema
  - student-memory boundaries

- Evaluation：
  - grounding
  - contamination assertions

- Data Pipeline：
  - index format
  - offline corpus lifecycle

Rules:

- Retrieval 不裁决学生答案正确性。
- Retrieval 不覆盖 TeachingContract。
- Retrieval 不覆盖 workflow policy。
- retrieved examples 必须保持 action-gated。
- curated knowledge 不是 Student Memory。
- Quality-QA 是 Agent Experience Memory，不是 Student Memory。
- Retrieval 负责运行时查询，不拥有 Quality-QA 的 semantic lifecycle。


# 8. LLM Infrastructure

Owner:
[TBD]

Primary paths:

- `app/llm/**`

Responsibilities:

- 唯一 OpenAI-compatible model gateway
- backend/settings
- timeouts
- retries
- structured JSON/Pydantic calls
- usage metadata
- slot routing
- KV-cache infrastructure
- API key safety
- private reasoning safety
- telemetry boundary

Protected interfaces:

- `DeepSeekClient.chat()`
- `DeepSeekClient.chat_json()`
- `DeepSeekSettings`
- `llama_slot()`
- usage metadata schemas
- KV metadata schemas

Mandatory co-review:

受 client behavior 影响的：

- Agent Runtime
- Evaluation
- Memory
- Data Pipeline
- Product

Rules:

- 其他模块不得直接实例化 OpenAI/Anthropic client。
- API keys 不得进入日志。
- private reasoning 不得进入日志、持久化或 UI。
- mock evaluation 与 live configuration 必须保持隔离。


# 9. Product / API / UI / Auth

Owner:
3322

Primary paths:

- `app/api/**`
- `app/ui/**`
- `app/auth/**`
- `prototypes/llm_led_teacher/streamlit_app.py`
- `prototypes/llm_led_teacher/conversation_ui.py`

Responsibilities:

- user identity
- authentication
- conversation selection
- API request/response contracts
- UI rendering
- public trace
- runtime status display
- runtime record 到用户可见状态的映射

Protected interfaces:

- API request/response models
- `user_id` ownership rules
- `conversation_id` ownership rules
- persisted episode schema/version
- public trace redaction
- UI-visible diagnostics

Mandatory co-review:

- Agent Runtime：
  - turn schemas
  - public trace

- Memory：
  - archive operations
  - identity
  - conversation ownership

- LLM Infrastructure：
  - runtime availability/configuration

- Evaluation：
  - API/UI behavior contracts

Rules:

- Product/API/UI/Auth 保持统一 ownership。
- 不拆分为多个 owner，除非未来规模确实需要。
- 8502 是当前产品入口。
- 8501/8000 UI/API 只接受 regression/critical fixes。


# 10. Data Pipeline

Owner:
[TBD]

Data Pipeline 仅负责通用离线数据加工基础设施。

它不是 Quality-QA 的 semantic owner。

Primary paths:

- `app/data_pipeline/**` 中的通用 processing/enrichment infrastructure
- 通用 offline indexing support
- pipeline-specific tests and fixtures

Responsibilities:

- archive/corpus 的通用离线读取
- data transformation infrastructure
- enrichment infrastructure
- index input generation infrastructure
- pipeline idempotency
- lineage tracking
- batch processing
- contamination controls

Protected interfaces:

- generic archive read contracts
- generic pipeline lineage schemas
- generic index-input schemas

Mandatory co-review:

- Memory：
  - Student Memory archive compatibility
  - Quality-QA / Agent Experience Memory semantic contracts

- Retrieval：
  - runtime index format

- LLM Infrastructure：
  - model-assisted enrichment

- Evaluation：
  - dataset contamination
  - evaluation-data boundaries

Rules:

- Data Pipeline 不拥有 Quality-QA 的语义定义、admission、lifecycle 或 experience-memory contract。
- 如果 Data Pipeline 代码生成 Quality-QA candidate，其 semantic contract 仍由 Memory System 维护。
- Data Pipeline 不得直接写入 Student Memory。


# 11. Shared Interface Ownership Matrix

| Shared interface | Single owner | Mandatory co-review |
|---|---|---|
| `app/agent/state.py` | Agent Runtime | Memory, Retrieval, Evaluation, Product（按受影响字段） |
| `app/agent/math_checker.py` | Agent Runtime | Evaluation |
| `app/agent/optional_verifier.py` | Agent Runtime | Evaluation |
| `app/memory/schemas.py` | Memory System | Agent Runtime, Evaluation, Product（按消费者） |
| `app/memory/contracts.py` | Memory System | Agent Runtime, Evaluation |
| `StudentMemoryService` public API | Memory System | Agent Runtime, Product |
| `LocalConversationStore` public API | Memory System | Product, Agent Runtime, Data Pipeline |
| Memory V2 admission contracts | Memory System | Agent Runtime, Evaluation（按受影响 contract） |
| Memory admission dev/holdout/robustness datasets | Memory System | Evaluation Framework |
| Quality-QA experience contracts | Memory System | Retrieval, Data Pipeline |
| Quality-QA runtime query/result contracts | Retrieval / GraphRAG | Memory, Agent Runtime |
| `app/retrieval/context_provider.py` | Retrieval / GraphRAG | Agent Runtime |
| Retrieval evidence models | Retrieval / GraphRAG | Agent Runtime, Evaluation |
| `app/llm/deepseek_client.py` | LLM Infrastructure | 所有受影响调用方 |
| `prototypes/llm_led_teacher/orchestration_models.py` | Agent Runtime | Memory, Retrieval, Product（按受影响 model） |
| `PrototypeEpisode` / `PrototypeTurnRecord` | Agent Runtime | Memory, Product, Evaluation |
| Agent-to-Memory writeback payload | Agent Runtime | Memory System |
| API request/response schemas | Product | Agent Runtime, Memory, Retrieval（按字段） |
| General golden/holdout datasets | Evaluation Framework | 对应被评测 subsystem |


# 12. Shared Contract Change Protocol

修改任何 shared interface 时，变更说明必须包含：

1. Owner
2. Producer
3. Consumers
4. Current behavior
5. Proposed behavior
6. Schema compatibility
7. Persistence compatibility
8. Evaluation impact
9. Migration/default behavior
10. Rollback behavior

以下均属于 contract change：

- 字段删除
- 字段改名
- 字段类型变化
- 默认值变化
- serialization 变化
- lifecycle 语义变化
- source/provenance 语义变化
- public method behavior 变化
- persisted schema 变化

字段新增应优先：

- 保持向后兼容
- 提供安全默认值
- 不破坏旧记录读取

Shared contract change 必须获得 matrix 所列受影响 subsystem 的 mandatory co-review。


# 13. Legacy Baseline Policy

8000 / 8501 及旧 LangGraph workflow 已冻结：

- 不接受新 feature development。
- 只接受 regression fixes。
- 只接受 critical fixes。
- 8502 新功能不需要实现 legacy feature compatibility。
- 修改 shared interface 时仍需确认不会造成意外的：
  - import failure
  - startup failure
  - basic test failure
- 这种检查属于基础 regression protection，不等于 feature compatibility。
- 如果修复只适用于 legacy，变更说明必须明确标注 `legacy-only`。
- 不得因为 legacy 代码存在而阻止合理的 8502 架构演进。


# 14. Ownership Changes

Ownership 不由 Codex 自动决定。

以下情况必须由开发者明确确认：

- 新 subsystem 出现
- subsystem 合并
- subsystem 拆分
- primary owner 变化
- shared interface owner 变化
- Quality-QA 物理迁移
- Evaluation ownership 调整
- legacy runtime 被正式删除
- Memory V2 从 shadow 升级为 authoritative

Codex 可以：

- 发现新的潜在 subsystem
- 分析 dependency boundary
- 提议 ownership change

Codex 不可以：

- 自行改变 owner
- 自行改变 subsystem boundary
- 因移动文件而自动改变 semantic ownership
- 因新增目录而自动创建新的 ownership domain