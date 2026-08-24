# Repository Architecture Audit

本次审计为只读操作。审计期间未修改源码、测试、现有文档、`AGENTS.md` 或其他文件。

## 1. Current Architecture Summary

### 1.1 现有硬约束

从 `AGENTS.md`、`CODEX_HANDOFF.md` 和 `CLAUDE.md` 可以归纳出：

- 现有 `api/agent/llm/retrieval/memory/evaluation` 分层是刻意设计的，不应整体重构。
- `math_checker` 是学生步骤、计划和答案正确性判定的最终权威；LLM、Memory、RAG 和教学策略都不能推翻它。
- Memory 是软个性化信号，只能影响讲解深度、提示层级、复习和变式题，不能控制学生是否能跳步或直接回答。
- Evaluation 是回归门，不能删除、弱化或通过放宽断言规避真实问题。
- Pedagogical RAG 只能提供教学决策证据，不能替代 Agent workflow 或数学判定。
- Retrieved examples 只能在例题/练习动作中出现，不能污染 `original_problem`。
- 所有模型调用必须经过 `app/llm/deepseek_client.py`。
- 结构化 LLM 输出应使用 JSON mode，并通过 Pydantic 校验。

### 1.2 实际存在两代运行链路

目录结构不能直接代表当前运行责任。

当前主链路是 8502 LLM-led Teacher：

```text
prototypes/llm_led_teacher/logic.py
→ MultisubjectAgentLoop
→ TeachingContract
→ Retrieval/Tools
→ Teacher generation
→ verifier/boundary audit
→ Memory writeback
→ conversation archive/UI
```

旧链路仍作为 baseline 保留：

```text
app/agent/workflow.py
→ diagnose
→ retrieve
→ teaching_policy
→ teach
→ memory_update
→ log
```

重要结论：

- `app/agent/**` 不是当前 8502 Agent Loop 的完整所在地。
- `app/api/**` 和 `app/ui/**` 主要服务旧 baseline。
- 当前 8502 UI、Agent orchestration 和一部分 Retrieval 位于 `prototypes/llm_led_teacher/**`。
- `app/memory/**` 同时服务当前 8502 和旧 baseline，内部也存在两代 memory model。

### 1.3 主要数据流

#### Teaching interaction flow

```text
8502 Streamlit UI
→ load conversation archive
→ PrototypeEpisode
→ load global memory profile
→ context compaction
→ classify scope/subject/problem continuity
→ optional student-memory resolution
→ fast route or bounded Agent Loop
→ guarded Retrieval/Skill tools
→ TeachingContract
→ deterministic verifier where applicable
→ teacher response generation
→ semantic boundary audit
→ math claim audit/repair
→ PrototypeTurnRecord
→ memory writeback
→ episode/archive persistence
→ UI display
```

关键边界：

- Agent → Memory：`StudentMemoryService.resolve()` 与 `enqueue_turn()`
- Agent → Retrieval：`RagContext`、retrieval decision、guarded tools
- Agent → Verifier：`VerificationRequest` / `VerificationResult`
- Agent → Generation：`TeachingContract` 和 bounded teacher payload
- Runtime → Product：`PrototypeTurnRecord` / `PrototypeEpisode`

#### Memory flow

```text
student turn
→ runtime constructs bounded writeback payload
→ build typed MemoryCandidate
→ freeze existing-memory and shadow-ledger snapshots
→ authoritative baseline enqueue
→ async outbox
→ extract profile/topic memory
→ upsert SQLite learner memory
→ shadow admission evaluation
→ shadow ledger telemetry
```

当前 Memory V2 admission 仍是 shadow/non-authoritative；权威写入仍由 `learner_memory.py` 的 baseline `enqueue_turn()` 路径决定。

#### Evaluation flow

```text
runtime or isolated policy component
→ evaluator
→ golden/development/holdout cases
→ deterministic behavior checks
→ metrics/report
```

分成两类：

- Teacher behavior evaluation：Agent、checker、grounding、disclosure、RAG 行为。
- Memory admission evaluation：直接消费 Memory contracts、controller、normalizer 和 frozen datasets。

## 2. Proposed Subsystems

| Subsystem | Primary Paths | Responsibility |
|---|---|---|
| Agent Runtime | `prototypes/llm_led_teacher/{logic,orchestrator,fast_plan,fast_routing,teaching_policy,boundary_audit,...}.py`; `app/agent/**` | 回合编排、教学契约、策略、验证、响应生成、安全边界、旧 baseline workflow |
| Memory System | `app/memory/**` | 学生档案、主题记忆、会话/episode 状态、异步写回、Memory V2 admission、provenance 和 ledger |
| Evaluation | `app/evaluation/**`, `data/eval/**`, evaluation scripts | Golden replay、行为检查、Memory admission benchmark、robustness 和指标 |
| Retrieval / GraphRAG | `app/retrieval/**`; `prototypes/llm_led_teacher/{rag_hook,retrieval_policy,orchestration_tools,qa_cache}.py` | 教学证据检索、GraphRAG、向量/网络检索、检索决策与结果契约 |
| LLM Infrastructure | `app/llm/**` | 唯一模型客户端、backend 配置、JSON/Pydantic 调用、usage、KV cache |
| Product / API / UI / Auth | `app/api/**`, `app/ui/**`, `app/auth/**`, `prototypes/llm_led_teacher/{streamlit_app,conversation_ui}.py` | 用户入口、认证、API contract、页面状态、会话选择和显示 |
| Data / Experience Pipeline | `app/data_pipeline/**` | conversation archive 到 Quality-QA/经验库的数据处理与离线 enrichment |

### Dependency direction proposal

建议维持以下方向：

```text
Product
  ↓
Agent Runtime
  ├─→ Memory public services/contracts
  ├─→ Retrieval provider/contracts
  ├─→ LLM gateway
  └─→ deterministic verifier

Evaluation
  ├─→ public runtime entrypoints
  └─→ public Memory policy contracts

Data Pipeline
  ├─→ archive read API
  ├─→ LLM gateway
  └─→ dedicated experiential store
```

应避免：

- Memory 导入 Agent orchestration/policy。
- Retrieval 决定教学动作或数学正确性。
- Evaluation runtime 反向被生产代码导入。
- Product 直接操作 Memory SQLite schema。
- 任意模块绕过 `deepseek_client.py`。
- Agent 直接写 learner-memory 数据库表。

## 3. Shared Interfaces

| File | Reason Shared | Owners Involved |
|---|---|---|
| `app/agent/state.py` | 同时定义 Agent state、verification、Graph context、legacy projection、memory fields 和 log contract；被 Agent、Retrieval、Evaluation、Prototype 和测试广泛导入 | Agent, Memory, Retrieval, Evaluation, Product |
| `app/memory/schemas.py` | `EpisodeState`、`LearningEvent`、`StudentEvidenceRecord` 同时被 Memory store、Agent runtime、student model 和 prototypes 消费 | Memory, Agent, Evaluation |
| `app/memory/contracts.py` | Memory V2 canonical schema；Runtime 产出 shadow record，Memory policy 消费，Evaluation 直接构造和验证 | Memory, Agent, Evaluation |
| `app/agent/math_checker.py` | 当前 runtime、baseline、simulator 和 evaluator 共用的确定性裁决边界 | Agent, Evaluation |
| `app/agent/optional_verifier.py` | 将通用 verifier contract 映射到 `math_checker`，被两代 runtime 使用 | Agent, Evaluation |
| `prototypes/llm_led_teacher/orchestration_models.py` | TeachingContract、classification、tool result、trace 和 memory resolution 汇聚在同一 contract 文件 | Agent, Memory, Retrieval, Product |
| `prototypes/llm_led_teacher/logic.py` | `PrototypeEpisode`/`PrototypeTurnRecord` 是 Agent 运行状态、UI 持久化、Memory writeback 和审计元数据的集成点 | Agent, Memory, Product, Evaluation |
| `app/memory/learner_memory.py` | 实现属于 Memory，但 `StudentMemoryService`、identity、resolution 和 prompt payload 是跨域公共 API | Memory, Agent, Product |
| `app/memory/conversation_archive.py` | 存储属于 Memory，但 episode JSON 由 Agent 定义，并被 UI 和 Quality-QA pipeline 使用 | Memory, Agent, Product, Data Pipeline |
| `app/retrieval/context_provider.py` | Retrieval request/provider 协议，但返回类型定义在 `app.agent.state`；当前形成双向类型耦合 | Retrieval, Agent |
| `app/retrieval/graph_retriever.py` | Retrieval evidence 被 Agent state、日志和 Evaluation 直接序列化 | Retrieval, Agent, Evaluation |
| `app/llm/deepseek_client.py` | 所有 LLM 消费者依赖的唯一 gateway | LLM Infrastructure, Agent, Evaluation, Data Pipeline |
| `app/api/main.py` | API response 展开 Agent、Memory、Retrieval 和 log 字段，修改会影响多个域 | Product, Agent, Memory, Retrieval |

建议治理方式：

- 文件可以有单一维护 owner，但表中 contract 变更必须获得涉及 subsystem 的协调确认。
- “共享接口”不应理解为所有团队均可无协调修改。
- 对 public model 使用兼容性规则：字段新增优先带默认值；重命名、删除、语义改变属于 breaking change。

## 4. Ownership Proposal

以下为拟议 `OWNERS.md` 草案，不代表已分配真实人员。

```markdown
# Subsystem Ownership

Ownership governs architectural responsibility and review coordination.
It does not grant permission to bypass repository-wide invariants.

## Memory System

Owner:
[TBD]

Primary paths:
- app/memory/**
- tests/behavior/test_structured_student_memory.py
- tests/behavior/test_memory_v2_baseline.py
- tests/test_memory_*.py

Responsibilities:
- Structured learner-memory storage and retrieval
- Conversation and episode persistence
- Student profile and topic-memory semantics
- Memory candidate, provenance and lifecycle contracts
- Admission policy, matching, evidence ledger and shadow telemetry
- Async outbox and writeback reliability
- Memory isolation, privacy and user scoping

Protected interfaces:
- app/memory/contracts.py
- app/memory/schemas.py
- app/memory/learner_memory.py::StudentMemoryService
- app/memory/conversation_archive.py::LocalConversationStore
- StudentMemoryResolution.prompt_payload()
- Runtime memory writeback payload contract

Requires coordination with:
- Agent Runtime for read/write invocation and payload semantics
- Evaluation for admission datasets and metrics
- Product for archive identity and conversation ownership
- Data Pipeline for archive readers
- Retrieval when defining curated knowledge versus student memory

Rules:
- Memory remains a soft teaching signal
- Memory may not override checker or TeachingContract authority
- Shadow admission may not authorize or block baseline writes unless explicitly promoted
- Curated knowledge and Quality-QA experience may not enter student memory implicitly

## Agent Runtime

Owner:
[TBD]

Primary paths:
- prototypes/llm_led_teacher/logic.py
- prototypes/llm_led_teacher/orchestrator.py
- prototypes/llm_led_teacher/fast_plan.py
- prototypes/llm_led_teacher/fast_routing.py
- prototypes/llm_led_teacher/teaching_policy.py
- prototypes/llm_led_teacher/boundary_audit.py
- prototypes/llm_led_teacher/orchestration_models.py
- app/agent/**

Responsibilities:
- Turn orchestration and runtime routing
- Active-problem continuity
- TeachingContract creation and enforcement
- Teaching policy, safety, disclosure and repair
- Deterministic verifier integration
- Teacher response generation
- Runtime trace and turn-record construction
- Legacy baseline compatibility

Protected interfaces:
- app/agent/math_checker.py
- app/agent/optional_verifier.py
- app/agent/state.py
- TeachingContract
- RequestClassification
- PrototypeEpisode and PrototypeTurnRecord
- Agent-to-Memory writeback payload
- Agent-to-Retrieval request/result contracts

Requires coordination with:
- Memory for resolution and writeback contract changes
- Retrieval for tool and evidence contracts
- LLM Infrastructure for client behavior
- Evaluation for observable behavior changes
- Product for persisted/UI-visible turn schemas

Rules:
- math_checker remains authoritative where applicable
- Memory and RAG are advisory
- Retrieved examples cannot silently replace the active problem
- Runtime must not persist private reasoning
- Unsupported verifier cases must remain not-applicable/unable-to-verify

## Evaluation

Owner:
[TBD]

Primary paths:
- app/evaluation/**
- data/eval/**
- scripts/run_eval.py
- scripts/run_memory_admission_eval.py
- scripts/run_memory_admission_robustness.py
- evaluation-specific fixtures and tests

Responsibilities:
- Golden conversation replay
- Grounding, disclosure and teaching-behavior checks
- Memory admission development/holdout benchmarks
- Robustness probes and metrics
- Baseline comparison and regression reporting

Protected interfaces:
- Golden case schemas
- Sealed holdout datasets
- Metric definitions
- Evaluation result schemas
- Checker-backed correctness expectations

Requires coordination with:
- Agent Runtime for behavior-contract changes
- Memory for admission-contract changes
- Retrieval for evidence/grounding assertions
- LLM Infrastructure for mock versus live mode

Rules:
- Production code must not import evaluation runtime
- Golden/holdout expectations cannot be weakened solely to make tests pass
- Changes to protected metrics require documented rationale

## Retrieval / GraphRAG

Owner:
[TBD]

Primary paths:
- app/retrieval/**
- prototypes/llm_led_teacher/rag_hook.py
- prototypes/llm_led_teacher/retrieval_policy.py
- prototypes/llm_led_teacher/orchestration_tools.py
- prototypes/llm_led_teacher/qa_cache.py

Responsibilities:
- Retrieval provider interfaces
- Graph and vector retrieval
- Retrieval routing and diagnostics
- Evidence provenance and source identifiers
- Curated knowledge and Quality-QA retrieval separation

Protected interfaces:
- GraphContextProvider
- GraphContextRequest
- RetrievalEvidence
- RagContext
- ToolResult retrieval payloads

Requires coordination with:
- Agent Runtime for tool permissions and evidence use
- Memory for source-class separation
- Evaluation for grounding and contamination checks
- Data Pipeline for corpus/index lifecycle

Rules:
- Retrieval cannot determine student correctness
- Retrieval cannot override TeachingContract or workflow policy
- New examples must remain action-gated
- Curated knowledge is not student-specific memory

## LLM Infrastructure

Owner:
[TBD]

Primary paths:
- app/llm/**

Responsibilities:
- OpenAI-compatible client gateway
- Backend selection and settings
- Structured JSON/Pydantic calls
- Timeouts, retries and safe metadata
- KV-cache integration

Protected interfaces:
- DeepSeekClient.chat()
- DeepSeekClient.chat_json()
- DeepSeekSettings
- llama_slot()
- usage metadata schema

Requires coordination with:
- Agent Runtime
- Evaluation
- Data Pipeline
- Product launcher/configuration

Rules:
- No other module may instantiate OpenAI/Anthropic clients directly
- Keys and private reasoning must never be logged
- Mock evaluation behavior must remain isolated from live configuration

## Product / API / UI / Auth

Owner:
[TBD]

Primary paths:
- app/api/**
- app/ui/**
- app/auth/**
- prototypes/llm_led_teacher/streamlit_app.py
- prototypes/llm_led_teacher/conversation_ui.py

Responsibilities:
- User identity and authentication
- API request/response contracts
- Session and conversation selection
- UI rendering and public trace display
- Mapping runtime records into user-visible state

Protected interfaces:
- TurnRequest and TurnResponse
- user_id/conversation_id ownership rules
- persisted episode schema version
- public trace redaction rules

Requires coordination with:
- Agent Runtime for turn schemas
- Memory for archive operations and identity
- LLM Infrastructure for runtime availability
- Evaluation for UI/API smoke contracts

## Data / Experience Pipeline

Owner:
[TBD]

Primary paths:
- app/data_pipeline/**

Responsibilities:
- Offline archive processing
- Quality-QA candidate generation
- Corpus enrichment and indexing inputs
- Pipeline versioning and idempotency

Protected interfaces:
- ConversationRecord read contract
- Quality-QA candidate/index schema

Requires coordination with:
- Memory for archive read compatibility
- Retrieval for index format
- LLM Infrastructure for model-assisted enrichment
- Evaluation for dataset contamination controls
```

## 5. Conflict Risks

| Area | Risk | Reason | Recommended ownership |
|---|---:|---|---|
| `app/agent/state.py` | High | 同时包含 Agent、Memory、Retrieval、logging 和 legacy compatibility 字段；任一字段语义变化影响多个消费者 | Shared interface；Agent Runtime 维护，跨域审批 |
| `prototypes/llm_led_teacher/logic.py` | High | 同时构造 turn record、Memory payload、admission shadow 和 episode 状态 | Agent Runtime 单一 owner；Memory payload 区域强制协调 |
| `app/memory/contracts.py` | High | Memory policy、runtime telemetry、evaluation datasets 共用 | Shared interface；Memory owner 维护 |
| `app/memory/schemas.py` | High | `EpisodeState` 和 student evidence 被 Agent 直接读取、写入 | Shared interface；Memory owner 维护，Agent co-review |
| `app/agent/student_model.py` | Medium | 代码位于 Agent，但直接改变 `EpisodeState.student_evidence` 的 memory representation | Agent Runtime 单一 owner；Memory schema 变更需协调 |
| `app/memory/learner_memory.py` | Medium–High | Memory 内部实现，但 public resolution/writeback API 被当前 Agent Loop 紧密依赖 | Memory 单一 owner；公开方法为 protected interface |
| `app/memory/conversation_archive.py` | Medium | Memory 管存储，Agent 定义 episode JSON，UI 和 pipeline 读写 | Memory 单一 owner；episode payload 为共享 contract |
| `conversation_state.py` / `episode_state.py` | Medium | 主要属于旧 baseline，但 workflow 直接依赖；容易误改后影响 8000/8501 | Memory owner；明确标注 legacy scope |
| `app/retrieval/context_provider.py` | High | Retrieval 导入 `agent.state.GraphContext`，而 `agent.state` 又导入 Retrieval evidence，边界存在双向类型耦合 | Shared interface；Retrieval owner 维护，Agent co-review |
| `prototypes/llm_led_teacher/orchestration_models.py` | High | Agent result 同时嵌入 Memory resolution、Retrieval decision、tool trace 和 teaching contract | Shared interface；Agent owner 维护 |
| `app/agent/math_checker.py` | High impact | 修改会影响 runtime、simulator、evaluation 和教学安全 | Agent Runtime 单一 owner；protected authority |
| `app/api/main.py` | Medium | 手工展开多个 subsystem 的字段，schema drift 风险高 | Product 单一 owner；各 subsystem 对字段语义 co-review |
| Quality-QA pipeline/cache | Medium | 属于 agent experience/retrieval，不应被当作 student memory 或 curated truth | Data Pipeline 与 Retrieval 明确分工 |

### 最可能发生 Memory/Agent 冲突的位置

优先级从高到低：

1. `prototypes/llm_led_teacher/logic.py` 中的 `memory_writeback_payload`
2. `app/memory/contracts.py`
3. `app/memory/schemas.py`
4. `app/agent/state.py`
5. `app/memory/learner_memory.py::StudentMemoryService`
6. `app/agent/student_model.py`
7. `app/memory/conversation_archive.py`
8. `app/retrieval/context_provider.py`

## 6. AGENTS.md Improvement Suggestions

以下均为建议，不直接修改 `AGENTS.md`。

### Add section: Runtime Generations

```markdown
## Runtime generations

- 8502 LLM-led Teacher under prototypes/llm_led_teacher is the current product path.
- app/agent/workflow.py and app/api/app/ui represent the retained 8501/8000 baseline.
- Before changing a shared module, identify which runtime generation consumes it.
- Do not assume app/agent/** is the complete current Agent Loop.
```

### Add section: Subsystem Ownership

```markdown
## Subsystem ownership

- Each change must identify its primary subsystem.
- Changes outside the primary subsystem require coordination with that subsystem owner.
- Directory location is evidence, not proof, of ownership.
- Integration files retain a single owner even when they contain shared interfaces.
```

### Add section: Shared Interfaces

```markdown
## Shared interfaces

Protected shared interfaces include:
- app/agent/state.py
- app/memory/schemas.py
- app/memory/contracts.py
- app/agent/math_checker.py
- app/agent/optional_verifier.py
- app/retrieval/context_provider.py
- prototypes/llm_led_teacher/orchestration_models.py
- PrototypeEpisode / PrototypeTurnRecord
- StudentMemoryService public methods
- API request/response models

Changing field names, meanings, defaults or serialization requires impact review
across all consumers and corresponding compatibility tests.
```

### Add section: Memory–Agent Boundary

```markdown
## Memory–Agent boundary

Agent Runtime may:
- request bounded memory resolution
- consume prompt_payload()
- enqueue a bounded post-turn writeback event

Agent Runtime must not:
- write Memory database tables directly
- decide Memory lifecycle/admission internally
- treat memory as mathematical evidence
- expose raw archived dialogue as trusted prompt content

Memory may not:
- override TeachingContract
- override deterministic verification
- block normal teaching on read/write failure
- infer student facts from curated or agent-experience data
```

### Add section: Contract Change Protocol

建议要求每次 shared-interface 修改说明：

- producer
- consumers
- schema compatibility
- persistence compatibility
- evaluation impact
- migration/default behavior
- rollback behavior

### Add section: Protected Files

建议将以下文件标记为 protected：

- `app/agent/math_checker.py`
- `app/memory/contracts.py`
- `app/memory/schemas.py`
- `app/agent/state.py`
- `app/llm/deepseek_client.py`
- `prototypes/llm_led_teacher/orchestration_models.py`
- `prototypes/llm_led_teacher/logic.py` 的 persisted record/writeback contract
- Evaluation golden/holdout datasets

## 7. Questions Requiring Human Decision

1. Ownership 范围是否以当前 8502 为主，还是要求每个 owner 同时负责旧 8501/8000 baseline？

2. `prototypes/llm_led_teacher/**` 是否仍视为 prototype，还是事实上已经是正式 Agent Runtime？这会影响 OWNERS 命名和 protected paths。

3. `app/agent/state.py` 应继续作为综合共享 schema，还是未来拆分为 Agent、Verification、Retrieval 和 API contracts？当前建议只治理，不立即重构。

4. `EpisodeState` 和 `StudentEvidenceRecord` 的语义 owner 是 Memory，还是 Agent Runtime？建议 Memory owner 维护 schema、Agent owner 维护证据生成策略。

5. Memory V2 admission 何时允许从 shadow 变成 authoritative？在明确决策前，应继续禁止它影响 baseline 写入。

6. Quality-QA 应归 Retrieval 还是 Data/Experience Pipeline？建议 pipeline 负责生成与版本，Retrieval 负责查询和运行时使用。

7. Shared interface 是采用“单一 owner + mandatory co-review”，还是独立 Architecture/Contracts owner？当前规模更适合前者。

8. Evaluation holdout 数据是否需要更严格的修改权限和变更记录？建议与普通测试 fixture 分开治理。

9. Product/API/UI 是统一 subsystem，还是 8502 UI、baseline API、auth 分开？当前建议先统一，避免 owner 过度碎片化。

10. 是否将“修改当前 8502 runtime 时必须检查 baseline compatibility”设为长期规则，还是正式声明 baseline 冻结、仅修回归？这会显著影响多 Agent 并行开发成本。
