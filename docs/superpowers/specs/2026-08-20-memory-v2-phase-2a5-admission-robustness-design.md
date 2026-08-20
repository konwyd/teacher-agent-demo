# Memory v2 Phase 2A.5 Admission Robustness Evaluation Design

## Status

Approved on 2026-08-20. This phase attacks the frozen `AdmissionPolicyV1`; it does not repair or tune it. Layer C is optional and never blocks completion of Layers A/B.

## Immutable boundaries

The evaluator must not modify `AdmissionPolicyV1`, `AdmissionPolicyConfig`, thresholds, priors, `MemoryRelationMatcher`, `ShadowEvidenceLedger`, or existing contracts. It must not implement authoritative admission, MemCon, online reward, or `AdmissionPolicyV2`. Failures remain recorded evidence.

The existing 30-case development set and 18-case sealed holdout remain byte-for-byte unchanged and are not used to tune robustness expectations.

## Frozen policy manifest

`policy_v1_manifest.json` records:

- policy name/version and a canonical JSON dump of the complete config;
- SHA-256 for the relevant policy, config, matcher, normalizer, feature, and controller files;
- repository git HEAD;
- each relevant repository-relative path and whether that path was dirty at freeze time;
- robustness dataset seal SHA-256 and a canonical digest of every case ID, expected decision, expected relation, labels, and trajectory expectations.

Verification compares only listed files and their recorded bytes/config. Unrelated dirty files do not fail the manifest. A relevant-file hash, HEAD, config, or dataset-seal mismatch fails before evaluation. The dataset is sealed before its first formal run and expectations are never changed in response to PolicyV1 output.

## Evaluation layers

### Layer A: policy robustness

Layer A directly constructs `MemoryCandidate`, `FrozenAdmissionContext`, and real temporary `ShadowEvidenceLedger` histories. It invokes `MemoryAdmissionController` only. It never imports the teacher loop, GraphRAG, quality-QA, or `StudentMemoryService`, and never calls an LLM.

The new fixed attack set covers paraphrases and deceptive neighbors; negation/correction; conditional and temporal preferences; source mutation; misconception trajectories A-E; same-session/cross-session evidence; stale beliefs; noisy content; capability-independent preferences; duplicates, contradictions, refinements, and supersession; and metamorphic relations.

### Layer B: extraction robustness

Layer B feeds natural-language utterances plus frozen classification, TeachingContract, and verifier fixtures into the existing `build_memory_candidates()` adapter. It records expected candidate type/source/scope/payload separately from downstream admission expectations. Extraction, normalization, relation, feature, ledger, and policy failures have separate attribution.

### Layer C: bounded live smoke

At most five trajectories, bounded turns and configured timeouts, may run against the current local 32B path. It reports wall latency, model/tool/duplicate call counts, candidates, decisions, and extraction failures. It is not included in Layer A metrics. When health/cost makes it impractical, status is `DEFERRED`; this does not block Phase 2A.5.

## Causal duplicate attack

The attack varies event/tool/verifier references while retaining one `(conversation_id, turn_id)` causal observation. The existing ledger is exercised unchanged. If support count or maturity inflates, attribution is `CAUSAL_DEDUP_GAP`, surfaced as `ROBUSTNESS_GAP_CAUSAL_DEDUP` and a Phase 2B blocker. It is never counted as a PolicyV1 failure and is not repaired in this phase.

## Long-trajectory simulator

Deterministic 20-, 50-, and 100-turn scenarios use the real controller and a temporary SQLite ledger. They mix preferences, temporary goals, progress, mistakes, corrections, contradictions, duplicates, and session changes. Reports include decision timelines, ledger and pending-cluster sizes, promotion points, pending survival, promotion latency, contradictions, stale pending items, premature admits, missed promotion, and contradiction recovery.

## Metrics

The report includes robustness accuracy/macro-F1, ADMIT precision/recall, relation accuracy, paraphrase consistency, temporal discrimination, conditional preference and correction accuracy, premature admit, missed promotion, stale belief, duplicate-evidence inflation, causal-duplicate inflation, inference admit, temporary-state admit, and incorrect supersede rates. Every rate exposes numerator/support and uses zero when support is zero.

Long-trajectory metrics include maximum pending ledger size, survival distribution, independent-evidence promotion latency, and contradiction recovery behavior.

## Failure attribution

Every mismatch has exactly one primary class:

- `CANDIDATE_EXTRACTION_FAILURE`
- `NORMALIZATION_FAILURE`
- `RELATION_MATCH_FAILURE`
- `FEATURE_FAILURE`
- `EVIDENCE_LEDGER_FAILURE`
- `POLICY_FAILURE`
- `CAUSAL_DEDUP_GAP`
- `LIVE_AGENT_LOOP_FAILURE`
- `LIVE_MODEL_FAILURE`

Contract limitations are not relabeled as policy failures. Reports preserve case-level expected/actual values and compact rationale.

## Outputs

The CLI writes a machine-readable JSON report and `docs/memory/MEMORY_V2_ADMISSION_ROBUSTNESS.md` summarizes the frozen manifest, datasets, attacks, metrics, long trajectories, failures, live smoke, limitations, Phase 2B blockers, and non-implemented PolicyV2 recommendations.

## Completion gates

Completion requires successful manifest verification, sealed dataset verification, Layer A/B execution, all three long trajectories, required metrics and attribution, focused tests, existing Phase 2A tests, full pytest, existing evaluation, compile/diff checks, and proof that frozen policy files did not change. Final status remains `READY_FOR_PHASE_2B = REVIEW_REQUIRED`.
