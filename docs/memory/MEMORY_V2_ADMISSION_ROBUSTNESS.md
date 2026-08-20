# Memory v2 Phase 2A.5 — Admission Robustness

## PolicyV1 frozen manifest

`POLICY_V1_FROZEN = YES`  
Git HEAD: `8b87de3e8ee42e495e85d5a241c84548545f75e7`  
Dataset label seal: `7c819226dd88418e59519809c01b431cbab60c0628a1759f32774c81354c8a8a`

- `app/memory/admission_policy.py` — SHA-256 `770a1cbf09af08d2b7da34a51060556ffe5ee6362082b6bd1cac005df4fff124`, dirty=`true`
- `app/memory/admission_config.py` — SHA-256 `de4ff9b891074e1c51508076014bb0e3b251e8ba67389aee83dcd610ae08c792`, dirty=`true`
- `app/memory/admission_matcher.py` — SHA-256 `4aad4c08446dd9b2e940290217c203a851cb37a2ab06f08393d0800474f41d4c`, dirty=`true`
- `app/memory/admission_normalization.py` — SHA-256 `54a44703e67502ba5e3d0cec81f417468c18cc7caec2ecfa4685856980f8bfa8`, dirty=`true`
- `app/memory/admission_features.py` — SHA-256 `598ad5f299677af04cc59bc6a42ad280b00dd60889813636640ce15ec82fb134`, dirty=`true`
- `app/memory/admission_controller.py` — SHA-256 `6de33456fd74508076a0a9726469b9866fa37b6446d6a0a61659265bc3bb944a`, dirty=`true`
- `app/memory/admission_ledger.py` — SHA-256 `3289a8341e4368124ca675a6b152071766d086e56b533189d8259e37832f67d0`, dirty=`true`
- `app/memory/contracts.py` — SHA-256 `b40dfaa1c93ed8967c84534703df5446893b5e0819000a5cf34281d72e7a038a`, dirty=`true`

The manifest verifies only listed paths; unrelated dirty worktree state is ignored.

## Evaluation layers

- Layer A: 62 direct candidate/context/ledger attacks; no LLM, agent loop, GraphRAG, quality-QA, or StudentMemoryService writes.
- Layer B: 32 frozen-upstream extraction attacks; accuracy 0.4375 (14/32).
- Layer C: `DEFERRED` — local quantized 32B live inference excluded by bounded-cost rule; Layers A/B and long trajectories are complete. It is excluded from PolicyV1 metrics.

## Dataset construction and attack taxonomy

The new sealed set is disjoint from the frozen 30 development and 18 holdout cases. It covers paraphrase/deceptive-neighbor, negation/correction, conditional and temporal preference, source mutation, misconception trajectories A–E, same/cross-session evidence, stale beliefs, noise, capability-independent preferences, relations, and metamorphic mutations.

## Layer A robustness results

- `admit_precision`: 0.8438 (27/32)
- `admit_recall`: 1.0000 (27/27)
- `causal_duplicate_inflation_rate`: 1.0000 (5/5)
- `conditional_preference_accuracy`: 1.0000 (6/6)
- `correction_accuracy`: 1.0000 (6/6)
- `duplicate_evidence_inflation_rate`: 0.0000 (0/2)
- `incorrect_supersede_rate`: 0.0000 (0/62)
- `missed_promotion_rate`: 0.0000 (0/27)
- `model_inference_admit_rate`: 0.0000 (0/1)
- `paraphrase_consistency`: 1.0000 (3/3)
- `premature_admit_rate`: 0.1429 (5/35)
- `relation_accuracy`: 1.0000 (62/62)
- `robustness_accuracy`: 0.9194 (57/62)
- `robustness_macro_f1`: 0.9468 (57/62)
- `stale_belief_rate`: 0.0000 (0/5)
- `temporal_discrimination_accuracy`: 1.0000 (9/9)
- `temporary_state_admit_rate`: 0.0000 (0/9)

## Layer B extraction results

Extraction failures and normalization failures are reported separately from Policy failures. Layer B accuracy: 0.4375 (14/32).

## Long-trajectory results

- 20 turns: final ledger=5, max pending=3, promotions=[10, 15, 20], latency=[2, 3, 4], contradictions=6, stale pending=2, recovery=admit_observed_after_contradiction
- 50 turns: final ledger=5, max pending=3, promotions=[10, 15, 20, 25, 30, 35, 40, 45, 50], latency=[2, 3, 4, 5, 6, 7, 8, 9, 10], contradictions=17, stale pending=2, recovery=admit_observed_after_contradiction
- 100 turns: final ledger=5, max pending=3, promotions=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], latency=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], contradictions=34, stale pending=2, recovery=admit_observed_after_contradiction

## Named misconception trajectories A–E

- A: decisions=['pending', 'admit', 'admit'], relations=['new', 'supports', 'supports'], keys=['student:behavior_evidence:数学:sign_error', 'student:behavior_evidence:数学:sign_error', 'student:behavior_evidence:数学:sign_error'], support=3, causal_turns=3, events=['verified:1', 'verified:2', 'verified:3'], gap=none
- B: decisions=['pending', 'pending', 'pending', 'pending'], relations=['new', 'contradicts', 'contradicts', 'contradicts'], keys=['student:topic_progress:数学:fractions', 'student:topic_progress:数学:fractions', 'student:topic_progress:数学:fractions', 'student:topic_progress:数学:fractions'], support=1, causal_turns=4, events=['verified:1', 'verified:2', 'verified:3', 'verified:4'], gap=none
- C: decisions=['pending', 'pending', 'pending'], relations=['new', 'contradicts', 'contradicts'], keys=['student:topic_progress:数学:fractions', 'student:topic_progress:数学:fractions', 'student:topic_progress:数学:fractions'], support=1, causal_turns=3, events=['verified-error', 'student-correction', 'verified-correct'], gap=none
- D: decisions=['pending', 'pending', 'pending'], relations=['new', 'new', 'new'], keys=['student:behavior_evidence:数学:sign_error', 'student:behavior_evidence:数学:arithmetic_error', 'student:behavior_evidence:数学:copying_error'], support=3, causal_turns=3, events=['verified:sign_error', 'verified:arithmetic_error', 'verified:copying_error'], gap=none
- E: decisions=['pending', 'admit', 'admit'], relations=['new', 'supports', 'supports'], keys=['student:behavior_evidence:数学:sign_error', 'student:behavior_evidence:数学:sign_error', 'student:behavior_evidence:数学:sign_error'], support=3, causal_turns=1, events=['tool-call', 'retry', 'outbox-retry'], gap=ROBUSTNESS_GAP_CAUSAL_DEDUP

Pending survival distributions and decision timelines are present in the machine-readable report.

## Causal duplicate findings

`CAUSAL_DUPLICATE_INFLATION_RATE = 1.0000 (5/5)`. Any observed inflation is classified as `CAUSAL_DEDUP_GAP`, not `POLICY_FAILURE`; PolicyV1, ledger, and contracts remain unchanged.

## Failure attribution

- `CANDIDATE_EXTRACTION_FAILURE`: 15 — rob-b-002, rob-b-003, rob-b-004, rob-b-005, rob-b-006, rob-b-008, rob-b-009, rob-b-010, rob-b-011, rob-b-014, rob-b-015, rob-b-018, rob-b-019, rob-b-030, rob-b-032
- `NORMALIZATION_FAILURE`: 3 — rob-b-012, rob-b-013, rob-b-016
- `RELATION_MATCH_FAILURE`: 0
- `FEATURE_FAILURE`: 0
- `EVIDENCE_LEDGER_FAILURE`: 0
- `POLICY_FAILURE`: 0
- `CAUSAL_DEDUP_GAP`: 5 — rob-a-047, rob-a-048, rob-a-049, rob-a-050, rob-a-051
- `LIVE_AGENT_LOOP_FAILURE`: 0
- `LIVE_MODEL_FAILURE`: 0

## Live-smoke findings

Status: `DEFERRED`. LIVE_SMOKE_CONFOUNDED_BY_AGENT_LOOP = UNKNOWN; deferred before model invocation. Wall latency is `None` and no live calls are included in Layer A/B scores.

## Existing regression results

- Full pytest: `816 passed in 71.95s`
- Existing offline evaluation: `38/38 passed (offline mock LLM)`

## Known limitations

This is a deterministic adversarial set, not a production prevalence estimate. Layer B evaluates the current lexical adapter with frozen upstream fixtures. Legacy memory lacks causal-root and claim-level provenance.

## Phase 2B blockers

- ROBUSTNESS_GAP_CAUSAL_DEDUP: contracts lack a causal-root identity
- Human review of every false ADMIT and missed promotion
- Production shadow divergence and privacy/retention review
- Stale-belief and correction lifecycle semantics

## PolicyV2 recommendations (not implemented)

- Add an explicit causal-root observation identity before evidence accumulation.
- Preserve conditional and temporal qualifiers in a future extraction contract.
- Define stale-belief recovery and retraction semantics with a new sealed holdout.
- Any behavior change must be named `AdmissionPolicyV2`; no V1 tuning is permitted.

## Stop boundary

`POLICY_V1_MODIFIED = NO`  
`AUTHORITATIVE_ADMISSION_IMPLEMENTED = NO`  
`MEMCON_IMPLEMENTED = NO`  
`ONLINE_REWARD_IMPLEMENTED = NO`  
`PHASE_2A_5_COMPLETE = YES`  
`READY_FOR_PHASE_2B = REVIEW_REQUIRED`
