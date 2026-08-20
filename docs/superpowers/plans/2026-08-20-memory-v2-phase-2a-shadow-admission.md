# Memory v2 Phase 2A Shadow Admission Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a deterministic, persistent, shadow-only admission controller with frozen pre-write context, evidence accumulation, benchmark evaluation, analytics, and complete safety verification.

**Architecture:** The runtime prepares candidates and freezes committed-memory and ledger snapshots before the unchanged baseline enqueue call. After baseline returns, a pure controller evaluates only the frozen context; a separate ledger records independent candidate provenance after the decision, and bounded telemetry records proposal/baseline divergence. Matching, features, policy, persistence, evaluation, and integration remain separate modules.

**Tech Stack:** Python 3.10, Pydantic v2 models, SQLite, pytest, JSONL evaluation artifacts.

**Spec:** `docs/superpowers/specs/2026-08-20-memory-v2-phase-2a-shadow-admission-design.md`

## Global Constraints

- WIP is exactly one task at a time.
- Phase 2A decisions never block, authorize, or mutate real memory writes.
- Every decision consumes a snapshot frozen before baseline enqueue.
- Baseline TopicMemory is match-only and never contributes evidence maturity.
- Ledger maturity uses only unique candidate provenance and prior shadow evidence.
- No new LLM, embedding, vector database, MemCon, reward, migration, cleanup, forgetting, or authoritative admission.
- Existing 0.68/0.85/0.75, top-k, confidence merge, and compaction thresholds remain unchanged.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Backward-Compatible Phase 2A Contracts and Configuration

**Files:**
- Modify: `app/memory/contracts.py`
- Create: `app/memory/admission_config.py`
- Test: `tests/test_memory_admission_contracts.py`

**Interfaces:**
- Produces: `AdmissionFeatureValue`, `AdmissionFeatures`, `EvidenceSummary`, `ExistingMemorySnapshot`, `FrozenAdmissionContext`, extended `AdmissionReasonCode`, and backward-compatible optional `MemoryAdmissionShadowRecord` fields.
- Produces: `AdmissionPolicyConfig` with `policy_name`, `policy_version`, source priors, temporal/stability priors, and maturity/type thresholds.

- [ ] **Step 1: Write failing contract/config tests**

  Add literal assertions that all feature values reject values outside `[0,1]`, Phase 1.5 shadow JSON still validates with defaults, snapshot records are immutable/match-only, reason families serialize, and every threshold/prior is available from one config object with policy version `teacher-memory-admission-v1`.

- [ ] **Step 2: Verify RED**

  Run: `python3 -m pytest tests/test_memory_admission_contracts.py -q`

  Expected: collection/import failure for the new contract/config names.

- [ ] **Step 3: Implement minimal models/config**

  Extend only the canonical contracts. Keep all new shadow fields defaulted:

  ```python
  class AdmissionFeatureValue(BaseModel):
      value: float = Field(ge=0.0, le=1.0)
      source: str = Field(max_length=120)

  class AdmissionFeatures(BaseModel):
      future_utility: AdmissionFeatureValue
      factual_confidence: AdmissionFeatureValue
      novelty: AdmissionFeatureValue
      stability: AdmissionFeatureValue
      temporal_validity: AdmissionFeatureValue
      evidence_maturity: AdmissionFeatureValue
      contradiction_strength: AdmissionFeatureValue
      content_type_prior: AdmissionFeatureValue
  ```

  `ExistingMemorySnapshot` contains a stable ID, target/type, semantic key,
  bounded payload, source kind when known, lifecycle, and `evidence_eligible=False`.
  `FrozenAdmissionContext` contains tuples of existing snapshots and ledger cluster
  snapshots plus capture timestamp/identity. Add config descriptions as Pydantic
  field descriptions.

- [ ] **Step 4: Verify GREEN and existing contracts**

  Run: `python3 -m pytest tests/test_memory_admission_contracts.py tests/test_memory_v2_contracts.py -q`

---

### Task 2: Candidate Normalization and Deterministic Relation Matcher

**Files:**
- Create: `app/memory/admission_normalization.py`
- Create: `app/memory/admission_matcher.py`
- Test: `tests/test_memory_admission_matcher.py`

**Interfaces:**
- Produces: `NormalizedCandidate(semantic_key, normalized_payload, explicit_correction)`.
- Produces: `RelationMatch(relation, matched_memory_id, confidence, reason_code)`.
- Consumes: `MemoryCandidate`, `ExistingMemorySnapshot`, `AdmissionPolicyConfig`.

- [ ] **Step 1: Write failing table-driven relation tests**

  Use hand-authored candidate/snapshot literals for exact duplicate, independent
  support, structured refinement, contradiction, explicit correction/supersession,
  ambiguous contradiction, near-duplicate preference specialization, and unrelated
  values. Assert ambiguous cases return `relation is None`.

- [ ] **Step 2: Verify RED**

  Run: `python3 -m pytest tests/test_memory_admission_matcher.py -q`

- [ ] **Step 3: Implement normalizer**

  Normalize only structured fields known to each current memory type. Produce
  semantic keys from target/type/subject/profile-or-topic key. Detect explicit
  correction only from typed payload flags (`explicit_correction=True` or
  `correction_of`) rather than guessing from free text.

- [ ] **Step 4: Implement matcher**

  Compare immutable snapshots without store access. Require exactly one
  unambiguous existing match plus user-stated correction for `SUPERSEDES`.
  Treat same semantic key/value as duplicate when it is already active, and as
  support only when comparing prior ledger evidence rather than committed memory.

- [ ] **Step 5: Verify GREEN**

  Run: `python3 -m pytest tests/test_memory_admission_matcher.py -q`

---

### Task 3: Isolated Shadow Evidence Ledger

**Files:**
- Create: `app/memory/admission_ledger.py`
- Test: `tests/test_memory_admission_ledger.py`

**Interfaces:**
- Produces: `ShadowEvidenceLedger(path)` with `snapshot(student_key, semantic_keys)`, `append(student_key, normalized_candidate, decision)`, `inspect(student_key)`, and `reset(student_key)`.
- Consumes/produces immutable `EvidenceClusterSnapshot` contracts.

- [ ] **Step 1: Write failing real-SQLite tests**

  With `tmp_path`, assert first evidence creates a cluster, distinct provenance
  increases support/turn/conversation counts, duplicate provenance is idempotent,
  explicit contradictions are separate, data reloads across instances, reset is
  scoped, and the learner-memory store cannot list ledger records.

- [ ] **Step 2: Verify RED**

  Run: `python3 -m pytest tests/test_memory_admission_ledger.py -q`

- [ ] **Step 3: Implement SQLite schema/repository**

  Store one bounded JSON cluster per `(student_key, semantic_key)` with a unique
  provenance fingerprint table or equivalent uniqueness constraint. Transactions
  update cluster metadata only after inserting a new provenance fingerprint.
  Never import or instantiate `LearnerMemoryStore`.

- [ ] **Step 4: Verify GREEN and mutation cases**

  Run: `python3 -m pytest tests/test_memory_admission_ledger.py -q`

---

### Task 4: Admission Feature Extractor

**Files:**
- Create: `app/memory/admission_features.py`
- Test: `tests/test_memory_admission_features.py`

**Interfaces:**
- Produces: `AdmissionFeatureExtractor(config).extract(candidate, normalized, relation_match, frozen_context) -> AdmissionFeatures`.
- Consumes only frozen context; no repository/store dependencies.

- [ ] **Step 1: Write failing feature tests**

  Cover future utility by type, all six provenance priors, persistent preference
  stability, session task instability, unknown temporal fallback, single verified
  evidence maturity, repeated distinct evidence maturity, duplicate provenance,
  source diversity, contradiction strength, novelty for every relation, and proof
  that adding committed TopicMemory without ledger evidence does not increase
  maturity.

- [ ] **Step 2: Verify RED**

  Run: `python3 -m pytest tests/test_memory_admission_features.py -q`

- [ ] **Step 3: Implement feature extraction**

  Use only config values and bounded deterministic formulas. Evidence maturity is
  computed from candidate provenance plus frozen ledger supports, distinct turns,
  conversations, source kinds, verified count, and contradictions. Existing
  memory snapshots are excluded from the evidence inputs by type and validation.

- [ ] **Step 4: Verify GREEN**

  Run: `python3 -m pytest tests/test_memory_admission_features.py -q`

---

### Task 5: Type-Aware Policy and Pure Controller

**Files:**
- Create: `app/memory/admission_policy.py`
- Create: `app/memory/admission_controller.py`
- Test: `tests/test_memory_admission_controller.py`

**Interfaces:**
- Produces: `AdmissionPolicyV1.decide(candidate, features, match) -> AdmissionDecision`.
- Produces: `AdmissionEvaluation(candidate, normalized, match, features, evidence_summary, decision, latency_seconds)`.
- Produces: `MemoryAdmissionController.evaluate(candidate, frozen_context) -> AdmissionEvaluation`.

- [ ] **Step 1: Write failing decision tests**

  Assert stable explicit preference/profile/long goal admits; task state rejects;
  explicit short-lived difficulty is pending; one verified error is pending;
  mature repeated verified evidence may admit; unsupported inference never admits;
  active duplicate rejects; unambiguous explicit correction supersedes; unresolved
  contradiction stays pending; curated/quality-QA targets do not enter student
  policy; one correct progress event does not imply mastery.

- [ ] **Step 2: Verify RED**

  Run: `python3 -m pytest tests/test_memory_admission_controller.py -q`

- [ ] **Step 3: Implement hard gates and type policies**

  Return structured reason codes and `policy_name/version` on every branch. Do not
  compute a global weighted score. Set lifecycle only as a proposed semantic state.

- [ ] **Step 4: Implement controller orchestration**

  Keep controller pure and dependency-inject normalizer/matcher/extractor/policy.
  Measure deterministic wall latency; never call an LLM or repository.

- [ ] **Step 5: Verify GREEN**

  Run: `python3 -m pytest tests/test_memory_admission_controller.py -q`

---

### Task 6: Two-Stage Shadow Facade and 8502 Safety Integration

**Files:**
- Modify: `app/memory/foundation.py`
- Modify: `prototypes/llm_led_teacher/logic.py`
- Test: `tests/test_memory_admission_shadow_integration.py`
- Test: `tests/prototypes/test_llm_led_turn_contract.py`

**Interfaces:**
- Produces: `PreparedAdmissionBatch` containing candidates and frozen contexts.
- Produces: `prepare_shadow_admission(...)` and `finalize_shadow_admission(prepared, baseline_job_id, ledger)`.
- Preserves: existing `observe_baseline_memory_action(...)` compatibility wrapper.

- [ ] **Step 1: Write failing integration/safety tests**

  Use a real temporary learner-memory service and ledger. Prove snapshot capture
  occurs before enqueue; force baseline outbox completion before finalize and show
  no false duplicate/maturity; controller reject still enqueues; controller admit
  cannot force a baseline skip; controller/ledger/matcher failures preserve the
  same job/profile/topic/archive/retrieval state; flags off reproduce Phase 1.5;
  and client LLM call counts are unchanged.

- [ ] **Step 2: Verify RED**

  Run: `python3 -m pytest tests/test_memory_admission_shadow_integration.py -q`

- [ ] **Step 3: Implement prepare/finalize facade**

  Prepare reads learner-memory and ledger once before enqueue and freezes Pydantic
  copies. Finalize never accepts a live learner-memory store and evaluates only the
  prepared object. Append candidate provenance after decision. Record features,
  match, evidence, decision, divergence, latency, and bounded errors.

- [ ] **Step 4: Integrate at `_run_turn_impl()`**

  Construct the unchanged baseline payload once. In shadow mode call prepare,
  execute the original enqueue exactly once with the identical payload, then call
  finalize. Every shadow block is fail-open; baseline status remains authoritative.

- [ ] **Step 5: Verify GREEN and baseline parity**

  Run: `python3 -m pytest tests/test_memory_admission_shadow_integration.py tests/test_memory_v2_foundation.py tests/behavior/test_memory_v2_baseline.py tests/prototypes/test_llm_led_turn_contract.py -q`

---

### Task 7: Benchmark Dataset, Evaluator, Metrics, Analytics, and Review Export

**Files:**
- Create: `app/evaluation/memory_admission/__init__.py`
- Create: `app/evaluation/memory_admission/models.py`
- Create: `app/evaluation/memory_admission/evaluator.py`
- Create: `app/evaluation/memory_admission/analytics.py`
- Create: `app/evaluation/memory_admission/data/development.jsonl`
- Create: `app/evaluation/memory_admission/data/holdout.jsonl`
- Create: `scripts/run_memory_admission_eval.py`
- Test: `tests/test_memory_admission_evaluation.py`

**Interfaces:**
- Produces: `MemoryAdmissionEvaluator.run(split) -> MemoryAdmissionEvaluationReport`.
- Produces: required classification, safety, evidence, baseline, and divergence metrics with support counts.
- Produces: `export_shadow_review(records, path, max_records)` bounded JSONL.

- [ ] **Step 1: Write failing dataset/evaluator tests**

  Assert at least 48 unique cases, at least 12 sequential cases, disjoint fixed
  dev/holdout IDs, all 25 categories, required labels/reason families, no long raw
  conversations, literal confusion-matrix metric examples, all safety formulas,
  sequential promotion formulas, analytics dimensions, and bounded review fields.

- [ ] **Step 2: Verify RED**

  Run: `python3 -m pytest tests/test_memory_admission_evaluation.py -q`

- [ ] **Step 3: Implement models/evaluator/metrics**

  Evaluate each sequence against an isolated in-memory/temporary ledger state.
  Compute metrics from literal labels/predictions. Define zero-denominator
  precision/recall as `0.0` with support exposed, except when a safety violation
  numerator/denominator are both zero, where the unsafe rate is `0.0`.

- [ ] **Step 4: Author development and sealed holdout datasets**

  Include compact, non-duplicative cases for every required category. Use at least
  30 development and 18 holdout cases, with at least 7 and 5 sequential cases
  respectively. Do not modify policy after observing holdout.

- [ ] **Step 5: Implement analytics/export/CLI and verify GREEN**

  Run: `python3 -m pytest tests/test_memory_admission_evaluation.py -q`

  Run: `python3 scripts/run_memory_admission_eval.py --split development`

  Then run holdout once and preserve its output artifact.

---

### Task 8: Documentation, Harness Decisions, and Final Completion Audit

**Files:**
- Create: `docs/memory/MEMORY_V2_ADMISSION_CONTROLLER.md`
- Modify: `DECISIONS.md`
- Modify: `PROGRESS.md` if it exists at implementation time
- Modify: `.env.example`

**Interfaces:**
- Documents all 23 required controller sections, benchmark and holdout results,
  safety/divergence findings, limitations, and explicit Phase 2B blockers.

- [ ] **Step 1: Document implementation and decisions A-H**

  Add ledger path/review export configuration and keep all defaults off. Record
  that PENDING is semantic evidence accumulation, admission is multi-signal and
  type-specific, mistakes/inference are conservative, ledger is isolated, writes
  remain baseline-controlled, ADMIT precision dominates aggregate accuracy, and
  no LLM judge exists.

- [ ] **Step 2: Run focused Phase 2A tests**

  Run: `python3 -m pytest tests/test_memory_admission_contracts.py tests/test_memory_admission_matcher.py tests/test_memory_admission_ledger.py tests/test_memory_admission_features.py tests/test_memory_admission_controller.py tests/test_memory_admission_shadow_integration.py tests/test_memory_admission_evaluation.py -q`

- [ ] **Step 3: Run existing focused regressions**

  Run the structured memory, baseline, compaction, cross-session, quality-QA,
  GraphRAG, pedagogical RAG, and turn-contract suites named in the Phase 2A brief.

- [ ] **Step 4: Run full and evaluation gates**

  Run: `python3 -m pytest -q`

  Run: `python3 scripts/run_eval.py`

  Run: `python3 scripts/run_memory_admission_eval.py --split development`

  Run: `python3 scripts/run_memory_admission_eval.py --split holdout`

- [ ] **Step 5: Run syntax/diff and requirement audit**

  Compile every new/modified Python module, run `git diff --check`, inspect the
  final diff and worktree, and map each numbered Phase 2A requirement/test/gate to
  direct code, dataset, or command evidence. Stop without implementing Phase 2B.
