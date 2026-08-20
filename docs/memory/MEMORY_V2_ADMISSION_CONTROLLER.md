# Memory v2 Phase 2A — Shadow Admission Controller

## 1. Status and scope

Phase 2A is implemented as shadow-only telemetry. The existing 8502 learner-memory writer remains authoritative. This phase does not implement authoritative admission, MemCon, reward, legacy migration, consolidation, cleanup, or forgetting.

## 2. Objective

The controller proposes `REJECT`, `PENDING`, `ADMIT`, or `SUPERSEDE` for typed `MemoryCandidate` values using deterministic, explainable rules.

## 3. Runtime boundary

`prepare_shadow_admission()` runs before the unchanged baseline enqueue. `finalize_shadow_admission()` runs afterward and cannot change, cancel, or create a baseline write.

## 4. Frozen pre-write invariant

Every decision consumes an immutable `FrozenAdmissionContext` captured before enqueue. Finalization has no live `StudentMemoryService` argument and cannot re-read state written by the current turn.

## 5. Existing-memory semantics

Pre-write `GlobalStudentProfile` and `TopicMemory` snapshots are used only for relation/match. `ExistingMemorySnapshot.evidence_eligible` is structurally fixed to false.

## 6. Evidence-source invariant

Maturity comes only from candidate provenance and prior independent shadow-ledger evidence. Baseline TopicMemory never increases support, contradiction, source diversity, or maturity.

## 7. Candidate normalization

Structured type-specific fields produce bounded semantic keys and normalized payloads. Unknown identity remains unset instead of using an LLM or embeddings.

## 8. Relation matching

The matcher returns `NEW`, `DUPLICATE`, `SUPPORTS`, `REFINES`, `CONTRADICTS`, `SUPERSEDES`, or unset. A ledger item supports only the same normalized value; the same key with a different value contradicts.

## 9. Supersession safety

`SUPERSEDE` requires one unambiguous committed match, an explicit typed correction, and a user-stated source. Behavioral evidence and inference cannot supersede explicit facts or preferences.

## 10. Explainable features

Eight bounded features are recorded: future utility, factual confidence, novelty, stability, temporal validity, evidence maturity, contradiction strength, and content-type prior. Each includes a public source code, not private reasoning.

## 11. Policy order

Policy v1 applies source/target hard gates, relation rules, type-specific rules, then evidence thresholds and conservative fallback. It is not a global importance score.

## 12. PENDING semantics

`PENDING` means evidence has not matured or a relation is unresolved. It is a shadow semantic state, not a queue that writes to real student memory.

## 13. Mistakes and mastery

One verified error remains pending evidence, and one correct answer does not imply mastery. Independent verified observations across turns/conversations may cross the fixed maturity threshold.

## 14. Source separation

Curated educational knowledge, student-specific memory, and agent/quality-QA experience remain separate. Curated and agent-experience candidates are rejected by student admission policy.

## 15. Shadow Evidence Ledger

The isolated SQLite ledger defaults to `data/runtime/memory_admission_shadow.sqlite3`. Unique provenance fingerprints make retries idempotent. It does not import or write the learner-memory store.

## 16. Decision and append order

The candidate is evaluated against the frozen ledger snapshot first. Its provenance is appended only after the decision, preventing self-promotion during the current turn.

## 17. Failure behavior

Preparation, matching, controller, telemetry, and ledger failures are fail-open. Baseline enqueue remains unchanged and errors are bounded in shadow records where possible.

## 18. Divergence semantics

Baseline `ENQUEUE` diverges from `REJECT/PENDING`; baseline `SKIP` diverges from `ADMIT/SUPERSEDE`. Divergence is observational: enqueue is not semantic admission.

## 19. Runtime analytics

Reducers report decisions by type/source/scope, divergence, controller errors, and latency. Human-review JSONL contains only bounded candidate summaries, provenance references, features, evidence counts, decisions, baseline action, and errors.

## 20. Benchmark construction

The fixed benchmark has 48 hand-labeled cases: 30 development and 18 sealed holdout, with disjoint IDs, 17 sequential rows, and all 25 required safety/relation/temporal/source categories. No full conversation text is stored.

## 21. Development results

Development: 30/30 accuracy, macro-F1 1.0, ADMIT precision 1.0 (11 predicted admits), sequential promotion recall 1.0 (5/5), contradiction-pending rate 1.0 (1/1). All six unsafe rates were 0. Baseline enqueue rate was 100%, proposal-accept rate 40%, and divergence 60% (18/30).

## 22. Sealed holdout results

The holdout was run once after development: 18/18 accuracy, macro-F1 1.0, ADMIT precision 1.0 (7 predicted admits), sequential promotion recall 1.0 (4/4), contradiction-pending rate 1.0 (1/1). All measured unsafe rates were 0; curated holdout support was zero and is reported explicitly. Baseline enqueue rate was 94.44%, proposal-accept rate 44.44%, and divergence 61.11% (11/18).

## 23. Limitations and Phase 2B blockers

The benchmark is compact and policy-aligned, so perfect classification is not evidence of production generalization. Candidate extraction remains lexical and legacy memory lacks claim-level provenance. Before Phase 2B, humans must review false-admit risk, divergence cases, correction semantics, ledger privacy/retention, operational rollback, writer-boundary authorization, and broader adversarial/live shadow data. `READY_FOR_PHASE_2B = REVIEW_REQUIRED`.
