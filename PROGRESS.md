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
Memory V2 — Phase 2A.6 preparation / CandidateExtractorV2 work

### Completed

- Memory system architecture audit
- Memory V2 Phase 1.5 contract foundation
- typed MemoryCandidate / provenance / temporal scope / lifecycle contracts
- shadow record infrastructure
- Memory V2 Phase 2A deterministic shadow admission controller
- relation matching and admission feature extraction
- isolated ShadowEvidenceLedger
- type-specific deterministic admission policy
- development / sealed holdout benchmark
- Phase 2A.5 adversarial robustness evaluation
- long-trajectory admission evaluation
- policy-level robustness analysis
- subsystem ownership / shared-interface audit
- multi-agent collaboration governance established through `OWNERSHIP.md`

### Current findings

- Admission Policy V1 is not currently the primary bottleneck.
- Candidate extraction quality is a major remaining bottleneck.
- Causal duplicate observations can incorrectly increase evidence maturity.
- Same real student observation must eventually be distinguishable from repeated tool/retry events.
- Candidate extraction must preserve qualifiers such as:
  - negation
  - temporal scope
  - condition
  - correction
  - polarity
  - scope
- Memory V2 admission remains shadow-only.
- Baseline writer remains authoritative.

### Current work

CandidateExtractorV2 comparative spike preparation.

Planned experimental routes:

A. deterministic extractor

B. embedding + deterministic qualifier parser

C. small specialized local extraction model

All routes should:
- use the same frozen benchmark
- start from the same checkpoint
- remain isolated
- not modify AdmissionPolicyV1
- not modify matcher/ledger semantics
- not solve causal-root identity inside the extractor spike

### Current blockers

1. Worktree baseline compatibility bootstrap

Known issues:
- frozen robustness manifest currently mixes historical provenance with runtime verification semantics
- launcher/test path assumptions are not worktree-portable

Required bootstrap should remain infrastructure-only:
- preserve historical freeze provenance
- verify current integrity through hashes/seals
- make launcher/test path resolution worktree-safe
- do not change Memory policy behavior

2. Causal evidence identity

Need a future contract capable of distinguishing approximately:

conversation
→ turn
→ causal observation
→ runtime/tool/retry event

Same causal observation must not independently increase evidence maturity through duplicate runtime events.

3. Candidate extraction quality

Current extraction robustness is materially weaker than admission-policy robustness.

### Next

1. Finish collaboration harness setup.
2. Resolve worktree compatibility bootstrap.
3. Create a clean shared checkpoint.
4. Rebuild CandidateExtractorV2 A/B/C worktrees from the same SHA.
5. Run common baseline verification.
6. Execute comparative extractor spikes.
7. Compare quality, qualifier preservation, latency and operational complexity.
8. Separately design causal-root / causal-observation identity contract.
9. Keep Memory V2 shadow until explicit architecture promotion decision.


## 3. Agent Runtime / Agent Loop

Owner:
3322

Status:
ACTIVE DEVELOPMENT — UNCOMMITTED CONTINUOUS REASONING TRACE / OFFLINE-VERIFIED, LIVE SMOKE BLOCKED

Current phase:
8502 continuous Agent Reasoning Trace integration and live acceptance

### Completed

- The committed 8502 baseline has a bounded, on-demand Agent Loop with trusted `TeachingContract` enforcement, deterministic math-verifier integration, guarded Skill/Tool execution, public trace, semantic boundary audit/repair, and safe fallback behavior.
- Canonical Skill selection, successful/failed call idempotency, classification-bound retrieval filters, and verifier-gated diagnostic capability are present in the committed baseline.
- Active-problem continuity, strict fenced-JSON handling, deterministic single-question normalization, verified-incorrect response correction, and placeholder-reply smoke gates are present in the committed baseline.
- Structured student-memory resolution/clarification/writeback is integrated into the 8502 turn path as a fail-open auxiliary signal; Memory V2 admission remains outside Agent Runtime authority.
- The uncommitted fast-route optimization remains present in this worktree.
- A dynamic, structured `reasoning_trace` has been added as a deterministic projection of actual 8502 runtime facts. It is persisted per turn, streamed through the existing event bridge, rendered as an always-visible continuous public summary, and keeps legacy public events/sections for rollback and old-history fallback.
- Trace evidence is source-attributed to validated fields such as `RequestClassification.subject`, `RetrievalDecision.mode`, `RagContext.hits`, `TeachingContract.response_mode`, `VerificationResult.status`, and `BoundaryAudit.passed`; it does not add an LLM call or expose model `reasoning_content`.
- Current offline evidence on 2026-08-21: focused Reasoning Trace suites passed 27 public-trace, 21 streaming, 40 turn-contract, and 14 UI tests; fresh full pytest passed 876/876; fresh offline evaluation passed 38/38.

### Current findings

- The current worktree implements four observable routes: `deterministic_fast_path`, `model_classifier_direct_path`, `dynamic_agent_path`, and `memory_clarification_reply`.
- Ordinary K9 requests can use a deterministic plan after either a fast-rule hit or model classification; explicit dynamic-evidence requests alone enter the bounded dynamic Agent path, whose current default budget is two action-model iterations.
- Local/Quality-QA retrieval is now demand-driven, and `retrieval_decision` is derived from the actual plan/tool trace rather than left at its default value.
- Low-risk concept generations can skip the LLM semantic auditor only after deterministic hard-boundary checking; higher-risk continuation/disclosure/verifier cases retain semantic audit and conditional repair.
- Reasoning Trace stages are not a fixed four-stage template. Memory, Tool, verifier, repair, and final-recheck stages appear only after their corresponding runtime facts occur; repeated facts update the original stage sequence.
- The main UI now uses `🧠 AI教学决策过程`, renders escaped natural-language summaries as a continuous non-collapsible flow, supports progressive snapshots, and retains history through `PrototypeTurnRecord.reasoning_trace`. Source fields and internal enums remain in structured records/diagnostics and are not shown in the main reading flow.
- Legacy `sections` fallback is read-only, uses decision-oriented titles, and suppresses additional free-form details that could contain an old retrieval reason.
- The optimization exists only as uncommitted changes in the main worktree at HEAD `d7ed183`; it is not part of the tracked branch tip.
- A fresh browser-level real-model smoke of the continuous trace is still NEEDS CONFIRMATION. The latest attempt reached the local model service but failed with CUDA OOM on device 0; the 8080/8002/8502 processes were then stopped, so current UI behavior is verified by Streamlit AppTest rather than a completed manual browser run.
- No conflict was found between the inspected implementation and the current Agent Runtime decisions in `DECISIONS.md`.

### Current work

- Review and integrate the uncommitted fast-route and continuous Reasoning Trace changes without absorbing unrelated worktree modifications.
- After the local model service is restored without changing unapproved deployment parameters, perform a browser-level 8502 smoke for at least a simple no-RAG concept request, a real tool/retrieval request, a verifier-applicable continuation, and a repair path.

### Current blockers

- Mandatory co-review completion for the persisted/UI-visible turn metadata changes is UNKNOWN / NEEDS CONFIRMATION (Product and Evaluation are affected consumers under `OWNERSHIP.md`).
- Browser-level real-model acceptance is blocked by the observed CUDA OOM on device 0 and stopped local 8080/8002/8502 services. Resolution belongs to the model/runtime deployment path and is not authorized in this UI correction.
- The current worktree contains unrelated governance, Chroma/runtime-data, Memory report, and other dirty changes; Runtime integration must use a scoped commit and must not absorb or discard those changes.
- Ordinary `git status` / `git diff` currently under-report tracked worktree changes because of index stat-cache behavior; `git diff-files --raw` confirms the dirty files. Index refresh or mutation has not been performed because this is a shared checkout.

### Next

1. Complete Product/Evaluation co-review for the persisted/UI-visible `reasoning_trace` contract.
2. Restore the approved local runtime configuration, then run the browser-level 8502 scenario smoke and compare each public summary with the actual diagnostics for the same turn.
3. Commit or otherwise checkpoint only the approved Agent Runtime/Product implementation, tests, and directly related Reasoning Trace documents.
4. Run the real-model cross-subject smoke matrix and record route, Skill/Tool trace, call count, token use, latency, repair/audit behavior, disclosure, and retrieval boundaries.

Branch / worktree:
- Main worktree: `/home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo`
- Branch: `codex/llm-led-teaching-agent` tracking `origin/codex/llm-led-teaching-agent` (behind 1 at inspection time)
- HEAD: `d7ed183f83c162b9dcbc78c5e7c992e758bf1497`
- Runtime-related linked worktrees: `.worktrees/skill-orchestration-interface-fix` at `8000972`; `.worktrees/teaching-contract-8502` at `b073eaf`
- Working state: dirty; the fast-route optimization, dynamic Reasoning Trace, tests, and docs are uncommitted in the main worktree, alongside unrelated changes owned by other work

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
worktree compatibility bootstrap


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
