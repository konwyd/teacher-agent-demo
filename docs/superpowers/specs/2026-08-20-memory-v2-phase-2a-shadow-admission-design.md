# Memory v2 Phase 2A Shadow Admission Controller Design

## Status

Approved on 2026-08-20. This design covers Phase 2A only. It stops before
authoritative admission, MemCon, reward optimization, legacy migration, cleanup,
forgetting, or any change to existing learner-memory heuristics.

## Purpose

Build a deterministic, explainable Shadow Admission Controller that evaluates
the canonical Phase 1.5 `MemoryCandidate` contract and proposes `REJECT`,
`PENDING`, `ADMIT`, or `SUPERSEDE`. The proposal is telemetry only. Existing
`StudentMemoryService.enqueue_turn()` and its writer remain authoritative.

## Non-Negotiable Snapshot Invariants

1. Every admission evaluation uses a frozen pre-write snapshot captured before
   the baseline enqueue call. It contains committed student-memory matches and
   ledger state as they existed before the current turn's baseline write.
2. Evaluation may execute after enqueue, but it must consume only that frozen
   snapshot. It may not re-read learner memory or the ledger before deciding.
3. Baseline `TopicMemory` is relation/match input only. Its presence never counts
   as supporting or contradicting evidence and never raises evidence maturity.
4. Evidence maturity derives only from candidate provenance and independently
   recorded shadow evidence, including verified observations.
5. The current candidate is evaluated against the frozen ledger snapshot and is
   appended to the ledger only after the decision is formed.

These rules prevent the current baseline write from creating a false duplicate,
false relation, or synthetic evidence promotion.

## Architecture

```text
build MemoryCandidate(s)
        │
        ├─ capture committed-memory snapshot ─┐
        └─ capture ledger snapshot ───────────┤  before baseline enqueue
                                             ▼
baseline enqueue/write (unchanged)     FrozenAdmissionContext
        │                                    │
        │                                    ▼
        │                           CandidateNormalizer
        │                                    │
        │                           MemoryRelationMatcher
        │                                    │
        │                           AdmissionFeatureExtractor
        │                                    │
        │                           AdmissionPolicyV1
        │                                    │
        │                           AdmissionDecision
        │                                    │
        └──────────────────────────► shadow record + ledger append
```

The controller has no learner-memory writer dependency. Runtime integration owns
snapshot capture, baseline execution, controller invocation, telemetry, and
fail-open handling.

## Component Boundaries

### Canonical contracts

`app/memory/contracts.py` remains the canonical schema module. Phase 2A adds
backward-compatible, defaulted models/fields for admission features, evidence
summaries, frozen existing-memory views, divergence, latency, and controller
errors. Existing Phase 1.5 JSON remains valid. The schema version is advanced in
the new records while old records remain readable.

### Admission policy configuration

`AdmissionPolicyConfig` contains every source prior and decision threshold. It
has a fixed policy name/version and field descriptions explaining rationale.
No thresholds are distributed through feature or policy functions, and Phase 2A
does not tune configuration online.

### Candidate normalizer

The normalizer creates a bounded semantic identity from typed payload fields. It
normalizes casing, whitespace, punctuation, stable list ordering, subject/topic,
profile keys, and known preference values. It does not use embeddings or an LLM.
Unknown or insufficiently structured content produces a conservative identity and
may result in an unset relation.

### Existing-memory adapter

The adapter converts committed `GlobalStudentProfile` and `TopicMemory` values to
immutable `ExistingMemorySnapshot` records before enqueue. It performs no writes,
does not expose a live store, and marks these records as match-only rather than
evidence. Quality-QA and curated GraphRAG content are not adapted as student
memory.

### Relation matcher

`MemoryRelationMatcher` compares normalized structured identity and values in this
order: exact identity/value, same identity with same-source independent evidence,
strict structured refinement, explicit contradiction, and explicit correction.
It returns `NEW`, `DUPLICATE`, `SUPPORTS`, `REFINES`, `CONTRADICTS`,
`SUPERSEDES`, or unset. `SUPERSEDES` requires an unambiguous existing item and an
explicit user correction; inferred or behavioral evidence cannot supersede an
explicit user preference/fact.

### Admission features

`AdmissionFeatures` contains eight bounded `[0, 1]` signals:

- `future_utility`: likelihood the information changes future teaching behavior.
- `factual_confidence`: provenance-aware reliability, using candidate confidence
  only as one input.
- `novelty`: new semantic information after relation handling.
- `stability`: likelihood the claim remains valid across future interactions.
- `temporal_validity`: suitability for persistent semantic use.
- `evidence_maturity`: maturity of independent supporting evidence.
- `contradiction_strength`: strength and independence of contrary evidence.
- `content_type_prior`: bounded type-specific persistence prior.

Every value records a short public rationale/source code, never hidden reasoning.
Missing information uses explicit conservative defaults from configuration.

### Shadow Evidence Ledger

The ledger is an isolated SQLite repository, defaulting to
`data/runtime/memory_admission_shadow.sqlite3`. A cluster stores its deterministic
semantic key, memory type, bounded supporting/contradicting provenance references,
support counts, distinct turns/conversations, first/last seen timestamps, current
maturity, and last proposed decision. Provenance uniqueness prevents retry or
duplicate-event inflation.

The repository supports snapshot, append, inspect, reset-by-student/test scope,
and reload. It is never called by learner-memory retrieval and never writes
`student_topic_memories`, profiles, archives, GraphRAG, or quality-QA.

### Admission policy v1

Policy order is fixed:

1. Target/source separation and hard safety gates.
2. Relation handling.
3. Memory-type-specific rules.
4. Feature and evidence-maturity thresholds.
5. Conservative fallback.

It is not a global importance score.

Hard gates:

- curated knowledge or agent experience cannot be admitted as student memory;
- unsupported model inference cannot directly become a durable student belief;
- `TURN_ONLY`, `SESSION`, and `TASK_STATE` cannot receive persistent `ADMIT`;
- an active exact duplicate cannot receive another `ADMIT`;
- ambiguous contradiction cannot yield `SUPERSEDE`.

Type policies cover current 8502-relevant types: preference, profile fact, task
state, difficulty, behavior evidence, topic progress, and learning goal. A single
verified error is `PENDING` evidence, not a misconception belief. Repeated,
independent verified evidence may promote the evidence candidate according to the
configured maturity threshold. One correct answer never means mastery.

### Controller

`MemoryAdmissionController.evaluate(candidate, frozen_context)` orchestrates the
pure normalizer, matcher, feature extractor, and policy. It returns a decision plus
the public evaluation artifacts required for telemetry. It does not write SQLite,
invoke the teacher agent, call an LLM, or mutate context.

### Shadow facade and runtime integration

The Phase 1.5 observer becomes a shadow facade with two explicit stages:

1. `prepare(...)` builds candidates and freezes committed-memory/ledger snapshots
   before baseline enqueue.
2. `finalize(prepared, baseline_result)` evaluates candidates, records divergence,
   appends independent evidence to the ledger after decision, and emits bounded
   shadow records.

The baseline call and payload remain unchanged. A controller, matcher, snapshot,
or ledger exception never blocks or changes baseline behavior. Errors are recorded
when possible. The `enabled` flags remain non-authoritative in Phase 2A; only
shadow-mode flags activate evaluation.

## Divergence Semantics

`divergence` compares operational baseline intent with semantic proposal:

- baseline `ENQUEUE` vs controller `REJECT` or `PENDING` is divergent;
- baseline `SKIP` vs controller `ADMIT` or `SUPERSEDE` is divergent;
- baseline `ENQUEUE` vs controller `ADMIT` or `SUPERSEDE` is aligned;
- baseline `SKIP` vs controller `REJECT` or `PENDING` is aligned.

This comparison never grants authority to either mapping. `ENQUEUE` is not a
semantic `ADMIT` label.

## Failure Semantics

All shadow stages are fail-open. Timeout is enforced by bounded deterministic work
and runtime latency observation rather than an extra thread or LLM. Invalid
features, matcher errors, controller errors, and ledger errors produce bounded
error codes on telemetry when possible. They never cancel or retry baseline
enqueue and never change the turn response.

## Benchmark

`app/evaluation/memory_admission/` owns a versioned evaluator and JSONL datasets.
The initial dataset contains at least 48 human-labeled cases, at least 12 of them
sequential. Development and holdout case IDs are fixed in the dataset before
policy evaluation. Holdout results are reported once without post-hoc tuning.

Every case includes expected action, relation where applicable, memory type,
source kind, temporal scope, reason family, compact rationale, pre-write existing
memory state, and event sequence. Categories cover all 25 required positive,
negative, contradiction, leakage, temporal, duplicate, and weak-source cases.

Metrics include overall accuracy, macro F1, per-action precision/recall, ADMIT
precision, all required contamination/safety rates, sequential promotion and
contradiction metrics, baseline write/proposal/divergence rates, and bounded
divergence examples. Zero-denominator metrics use a documented deterministic
convention and expose support counts.

## Runtime Analytics and Human Review

An analytics reducer consumes shadow records and reports candidate/decision
distribution, decisions by type/source/scope, divergence, maturity at admit,
inference/temporary admits, duplicate/contradiction counts, and latency. A JSONL
review exporter writes only candidate summaries, provenance references, features,
relation, evidence counts, decision/reasons, baseline action, divergence, and
errors. It never copies full conversations.

## Testing Strategy

TDD covers each component independently and then the complete seam:

- feature ranges, source priors, fallbacks, and maturity;
- all relation outcomes and conservative unset behavior;
- ledger accumulation, provenance idempotency, contradictions, reset, reload, and
  retrieval isolation;
- all type policies and hard gates;
- frozen pre-write snapshot behavior, including an asynchronous baseline write;
- proof that baseline enqueue/skip, TopicMemory, profile, archive, retrieval, and
  LLM call count remain unchanged under every controller outcome and failure;
- benchmark schema, split integrity, category coverage, metrics, and holdout run;
- bounded review export and analytics.

Existing memory, compaction, quality-QA, GraphRAG, full pytest, compile, evaluation,
and diff checks remain mandatory completion gates.

## Known Limitations

Deterministic structured matching cannot reliably resolve every paraphrase. It
returns an unset relation rather than guessing. Runtime candidate extraction still
inherits the limited Phase 1.5 lexical adapter; benchmark cases exercise the
canonical candidate interface directly. No claim-level provenance exists on legacy
TopicMemory, so it can be matched but cannot become evidence. There is still no
reliable online reward.

## Phase 2B Review Blockers

Authority requires a separate human review of holdout safety, shadow divergence,
false-admit examples, operational retention/privacy for ledger data, correction
semantics, rollback, and authorization at the writer boundary. Phase 2A does not
create active memory, block writes, supersede real memory, alter confidence, tune
retrieval, migrate legacy records, implement MemCon, or optimize reward.
