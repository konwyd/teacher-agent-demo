# PROGRESS.md — Current Project State

本文件记录仓库当前开发状态。

它是当前状态快照，不是 changelog，也不是 architecture decision log。

重要架构决策记录在 `DECISIONS.md`。
Subsystem ownership 和 shared-interface governance 记录在 `OWNERSHIP.md`。
Coding agent 工作规则记录在 `AGENTS.md`。

更新原则：

- 每个 subsystem owner 只更新自己负责的 section。
- 不要重写其他 subsystem 的进度。
- 已完成且仍影响当前开发的信息可以保留。
- 已失去当前意义的历史过程应删除或压缩。
- architecture decision 不要复制到这里，应写入 `DECISIONS.md`。
- task materially changes project state 时，在结束任务前更新对应 section。


## 1. Project Status

Primary runtime:
- 8502 LLM-led Teacher

Runtime location:
- `prototypes/llm_led_teacher/**`

Legacy:
- 8501 / 8000 frozen baseline
- no new feature development
- regression / critical fixes only

Development model:
- multiple developers
- multiple Codex sessions
- subsystem ownership defined in `OWNERSHIP.md`
- independent parallel work should prefer separate branches/worktrees


## 2. Memory System

Owner:
Haoxi

Status:
ACTIVE DEVELOPMENT

Current phase:
Memory V2 — CandidateExtractorV2 comparative spike / Route B Tasks 1–6 implemented; development-only real-model tuning pending

Active branch:
`spike/candidate-extractor-v2-embedding-hybrid`

Last updated:
2026-08-24

### Completed

- Memory V2 Phase 1.5 typed candidate, provenance, temporal-scope, and lifecycle contracts
- Phase 2A deterministic shadow admission controller, matching, evidence ledger, and type-specific policy
- development / sealed-holdout benchmark freeze, Phase 2A.5 adversarial robustness, and long-trajectory evaluation
- CandidateExtractorV2 comparative governance checkpoint and isolated Route A/B/C worktrees
- Route B deterministic qualifier parser with clause-local negation, condition, temporal, correction, uncertainty, and hard-negative handling
- Route B local-only BGE-M3 semantic-family ranker with lazy loading, normalized batching, prototype caching, validation, and operational counters
- Route B utterance and frozen-upstream candidate extraction for the eight supported memory families, including bounded multi-candidate merge and malformed-input diagnostics
- Route B read-only development/Layer B benchmark adapter and CLI with deterministic replay and operational reporting
- Route B optional dependency, model metadata, reproducible procedure, safety controls, and provisional evidence documentation

### Current findings

- Admission Policy V1 is not currently the primary bottleneck; candidate extraction quality remains a major comparative question.
- Route B keeps embeddings advisory: deterministic parsers own fields, source kind, qualifiers, and temporal scope.
- Route B uses local `BAAI/bge-m3` (MIT, 1024 dimensions) through `sentence-transformers==5.6.0`, defaults to CPU, and does not download weights implicitly.
- The Route B runner accepts only `development` and unchanged `layer_b`; Route B sealed-holdout runs remain zero.
- Route B is isolated from the authoritative runtime and baseline writer. Memory V2 admission remains shadow-only and baseline writing remains authoritative.
- Causal duplicate observations can still incorrectly increase evidence maturity; the extractor spike intentionally does not solve causal-root identity.

### Current blockers

1. Route B quality and operational values are not yet measured with the real local BGE-M3 backend. Tasks 1–6 provide implementation and offline unit/regression evidence only.
2. Causal observation identity remains a separate future contract problem outside the CandidateExtractorV2 spike.
3. Overall route comparison and any controlled sealed-holdout evaluation belong to the coordinator; this branch cannot declare a winner.

### Next

1. Prepare the first real local BGE-M3 run using development for tuning evidence and unchanged Layer B for regression evidence.
2. Replace provisional quality/resource placeholders in `docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md` with fresh Task 7 measurements.
3. Preserve generic prototypes and generalized parser rules; do not add case-ID or copied-utterance branches.
4. Run and tune Route B on the frozen development split only; rerun unchanged Layer B after generalized changes.
5. Record all frozen metrics, qualifier/contamination slices, latency, throughput, RAM/VRAM, encode counts, dependency versions, replay, and line count.
6. Run full repository tests, offline evaluation, Memory admission robustness, manifest verification, and protected-scope audits.
7. Hand the clean isolated branch to the coordinator without changing AdmissionPolicyV1, shared contracts, frozen evaluator/data/labels, existing Layer B labels, runtime policy, or causal-dedup semantics.


## 3. Agent Runtime / Agent Loop

Owner:
3322

Status:
ACTIVE DEVELOPMENT — UNCOMMITTED IMPLEMENTATION / OFFLINE-VERIFIED

Current phase:
8502 fast-route orchestration optimization integration and live acceptance

### Completed

- The committed 8502 baseline has a bounded, on-demand Agent Loop with trusted `TeachingContract` enforcement, deterministic math-verifier integration, guarded Skill/Tool execution, public trace, semantic boundary audit/repair, and safe fallback behavior.
- Canonical Skill selection, successful/failed call idempotency, classification-bound retrieval filters, and verifier-gated diagnostic capability are present in the committed baseline.
- Active-problem continuity, strict fenced-JSON handling, deterministic single-question normalization, verified-incorrect response correction, and placeholder-reply smoke gates are present in the committed baseline.
- Structured student-memory resolution/clarification/writeback is integrated into the 8502 turn path as a fail-open auxiliary signal; Memory V2 admission remains outside Agent Runtime authority.
- The current uncommitted fast-route optimization implementation has offline regression evidence: 120 targeted Runtime tests passed, the full suite passed 863/863, and offline evaluation passed 38/38 on 2026-08-21.

### Current findings

- The current worktree implements four observable routes: `deterministic_fast_path`, `model_classifier_direct_path`, `dynamic_agent_path`, and `memory_clarification_reply`.
- Ordinary K9 requests can use a deterministic plan after either a fast-rule hit or model classification; explicit dynamic-evidence requests alone enter the bounded dynamic Agent path, whose current default budget is two action-model iterations.
- Local/Quality-QA retrieval is now demand-driven, and `retrieval_decision` is derived from the actual plan/tool trace rather than left at its default value.
- Low-risk concept generations can skip the LLM semantic auditor only after deterministic hard-boundary checking; higher-risk continuation/disclosure/verifier cases retain semantic audit and conditional repair.
- The optimization exists only as uncommitted changes in the main worktree at HEAD `d7ed183`; it is not part of the tracked branch tip.
- No 8502 process is currently running and `127.0.0.1:8502/_stcore/health` refuses connections. Deployment and real-model behavior are therefore UNKNOWN / NEEDS CONFIRMATION.
- No conflict was found between the inspected implementation and the current Agent Runtime decisions in `DECISIONS.md`.

### Current work

- Review and integrate the uncommitted fast-route orchestration optimization across routing, deterministic planning, bounded dynamic tools, risk-based audit, persisted route metadata, Streamlit diagnostics, and smoke-test reporting.
- Preserve backward compatibility for the new defaulted `PrototypeTurnRecord` and `AgentOrchestrationResult` metadata fields.

### Current blockers

- Live 8502 acceptance is blocked because the 8502 runtime is not running; real-model call-budget, latency, retrieval, disclosure, and cross-subject smoke results remain unverified.
- Mandatory co-review completion for the persisted/UI-visible turn metadata changes is UNKNOWN / NEEDS CONFIRMATION (Product and Evaluation are affected consumers under `OWNERSHIP.md`).
- The current worktree contains unrelated governance, Chroma/runtime-data, Memory report, and other dirty changes; Runtime integration must use a scoped commit and must not absorb or discard those changes.

### Next

1. Review the scoped Runtime diff and confirm required Product/Evaluation co-review for persisted/UI-visible metadata.
2. Commit or otherwise checkpoint only the approved Agent Runtime implementation, tests, and directly related documentation.
3. Start/reload 8502 from the intended branch/worktree without changing model/KV deployment parameters.
4. Run the real-model cross-subject smoke matrix and record route, Skill/Tool trace, call count, token use, latency, repair/audit behavior, disclosure, and retrieval boundaries.
5. Reconcile measured live results with the call-budget targets before declaring the optimization deployed or complete.

Branch / worktree:
- Main worktree: `/home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo`
- Branch: `codex/llm-led-teaching-agent`
- Local HEAD: `d7ed183f83c162b9dcbc78c5e7c992e758bf1497`; this dirty local checkout is intentionally preserved and is not the CandidateExtractorV2 spike baseline
- Runtime-related linked worktrees: `.worktrees/skill-orchestration-interface-fix` at `8000972`; `.worktrees/teaching-contract-8502` at `b073eaf`
- Working state: dirty; the fast-route optimization and its tests/docs are uncommitted in the main worktree

Last materially updated:
2026-08-21


## 4. Retrieval / GraphRAG

Owner:
[TBD]

Status:
[ACTIVE / STABLE / PAUSED]

### Current state

- Existing retrieval infrastructure is present.
- Pedagogical retrieval remains advisory to Agent Runtime.
- Curated knowledge, Student Memory and Agent Experience Memory remain separate source classes.

### Current work

[TBD]

### Blockers

[TBD]

### Next

[TBD]


## 5. Evaluation Framework

Owner:
[TBD]

Status:
STABLE / SHARED INFRASTRUCTURE

### Current state

- General teacher behavior evaluation remains a regression gate.
- Golden/holdout expectations are protected.
- Memory Admission evaluation is owned by Memory System as a subsystem-specific benchmark.
- Shared evaluation infrastructure remains under Evaluation Framework governance.

### Current work

[TBD]

### Blockers

[TBD]

### Next

[TBD]


## 6. Agent Experience Memory / Quality-QA

Semantic owner:
Memory System

Runtime retrieval:
Retrieval / GraphRAG

Status:
EXISTING / FUTURE CONSOLIDATION

### Current state

- Quality-QA represents agent teaching experience.
- It is not Student Memory.
- It must not be implicitly written into student profile/topic memory.
- Current physical location may remain outside `app/memory/**`.
- Semantic ownership does not depend on current directory location.

### Current work

No structural migration required during current Memory Admission work.

### Next

Revisit physical consolidation only when it provides a concrete architectural benefit.


## 7. Product / API / UI / Auth

Owner:
[TBD]

Status:
STABLE

Primary product path:
8502

### Current work

[TBD]

### Blockers

[TBD]

### Next

[TBD]


## 8. LLM Infrastructure

Owner:
[TBD]

Status:
STABLE

### Current state

- Production LLM calls use the existing DeepSeek client infrastructure.
- Current Memory work should not change model deployment / KV configuration unless explicitly scoped.

### Current work

[TBD]

### Blockers

[TBD]

### Next

[TBD]


## 9. Active Parallel Work

This section only lists work that is currently active.

### Memory

Subsystem:
Memory System

Owner:
Haoxi

Task:
CandidateExtractorV2 / Memory V2 continuation

Branch / worktree:
[UPDATE WHEN ACTIVE WORKTREE IS CREATED]

Status:
BLOCKED / PREPARING

Blocked by:
governance harness absent from the frozen benchmark parent; Route A/B/C implementation has not started


### Agent Runtime

Subsystem:
Agent Runtime

Owner:
[OWNER]

Task:
[CURRENT TASK]

Branch / worktree:
[BRANCH / WORKTREE]

Status:
[STATUS]

Blocked by:
[BLOCKER / NONE]


## 10. Current Cross-Subsystem Coordination

Only list active coordination needs.

### Memory ↔ Agent Runtime

Relevant shared boundaries include:
- Agent-to-Memory writeback payload
- runtime-produced evidence/provenance
- causal observation identity
- persisted episode/turn contracts

Current rule:
Do not change these contracts from one subsystem without following `OWNERSHIP.md` mandatory co-review.


## 11. Immediate Repository Priorities

Current recommended order:

1. Finish collaboration harness.
2. Restore a clean worktree-compatible research baseline.
3. Continue Memory CandidateExtractorV2 comparative spike.
4. Continue Agent Runtime work independently in its own ownership/worktree.
5. Coordinate only when either line reaches a shared contract boundary.


## 12. Update Template

When a task materially changes a subsystem, update only that subsystem.

Use:

Current phase:
...

Completed:
- ...

Current findings:
- ...

Current work:
- ...

Blockers:
- ...

Next:
1. ...
2. ...

Branch / worktree:
...

Last materially updated:
YYYY-MM-DD
