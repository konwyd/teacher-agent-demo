# Memory v2 — Phase 1.5 Contract Foundation

Status: contract-only foundation for the authoritative 8502 runtime. No admission
policy or Memory Controller is implemented in this phase.

## 1. Scope and non-goals

Phase 1.5 standardizes candidate, provenance, temporal, relation, lifecycle,
decision, shadow-observation, and outcome-observation data. It does **not** change
the existing learner-memory read/write behavior, thresholds, confidences, archive,
GraphRAG, quality-QA, or stored records. `memory_admission.enabled` and
`memory_controller.enabled` are intentionally non-authoritative.

## 2. Architectural boundary

The source-of-truth path remains `logic.py` → `StudentMemoryService.enqueue_turn()`
→ outbox → `_extract_and_write()` → `upsert_topic_memory()`. The Phase 1.5 adapter
observes the exact enqueue outcome and adds bounded shadow metadata to
`PrototypeTurnRecord`. It cannot authorize, reject, supersede, mutate, or retrieve
memory. Conversation archive persistence and context compaction are independent.

## 3. Target memory classes

- `student`: student-specific semantic facts, preferences, evidence, goals, and state.
- `agent_experience`: reusable teaching experience such as quality-QA.
- `curated_knowledge`: textbook, exercise, and GraphRAG knowledge.

Archive/log and KV/retrieval caches are operational stores, not these semantic
classes. Curated knowledge is invalid as a student-memory candidate.

## 4. Memory type taxonomy

Canonical types are `profile_fact`, `preference`, `topic_progress`, `difficulty`,
`behavior_evidence`, `learning_goal`, `learning_episode`, `task_state`,
`teaching_experience`, and `knowledge_reference`. The taxonomy is deliberately
small and extensible; operational queues and cache entries are excluded.

## 5. Candidate schema

`MemoryCandidate` contains schema/candidate ID, target class, memory type, bounded
structured payload, source kind, one or more provenance references, confidence,
temporal scope, and creation time. Candidate creation is descriptive only and is
never permission to persist. Payloads are capped at 4096 encoded bytes and shadow
payloads do not copy the raw `student_message`.

## 6. Provenance and source typing

Sources are `user_stated`, `observed_behavior`, `verified_result`,
`model_inference`, `external_tool`, and `curated_knowledge`. Each candidate must
have provenance with its declared source kind. Thus inference cannot masquerade
as a user assertion or verified result. References identify conversation, turn,
event, tool, or verifier without storing private reasoning.

## 7. Temporal scope

`turn_only` expires with the turn; `session` is useful only in the active episode;
`short_lived` may support near-term continuity but needs later revalidation;
`persistent` represents an explicitly durable claim; `unknown` prevents accidental
durability when validity is unclear. Temporal scope is metadata in Phase 1.5, not
an enforcement policy.

## 8. Candidate relations

Future comparison may produce `NEW`, `DUPLICATE`, `SUPPORTS`, `REFINES`,
`CONTRADICTS`, or `SUPERSEDES`. Phase 1.5 performs no matching and therefore emits
no relation at runtime.

## 9. Lifecycle semantics

- `PENDING`: captured but not admitted for active use.
- `ACTIVE`: admitted and eligible for policy-controlled access.
- `SUPERSEDED`: retained for lineage but replaced by a newer active item.
- `RETRACTED`: retained as an invalidated belief/evidence item and not trusted.
- `DELETED`: logically or physically unavailable under retention/privacy rules.

Supersession preserves lineage; retraction states that the belief is invalid;
deletion is a retention/privacy operation. None are applied to legacy records now.

## 10. Admission decision contract

`AdmissionDecision` can represent `REJECT`, `PENDING`, `ADMIT`, or `SUPERSEDE`,
with public reason codes, matched ID, relation, policy identity/version, resulting
confidence, lifecycle status, and timestamp. This is a future policy output type.
The Phase 1.5 runtime always leaves `proposed_decision` unset.

## 11. Confidence semantics

Confidence is evidence-dependent, not a monotonic score or truth probability.
Future contradictory evidence may lower it; explicit correction may supersede a
previous high-confidence item. Phase 1.5 preserves all existing fixed confidence
values and max-ratchet behavior as characterized baseline behavior.

## 12. Shadow record contract

`MemoryAdmissionShadowRecord` stores the candidate, observed baseline action
(`enqueue`/`skip`), skip reason, unset proposed decision, empty match/relation,
policy=`none`, observer/schema versions, conversation/turn, and timestamp. It is
comparison telemetry only. Observer failure is fail-open and cannot block teaching
or memory writeback.

## 13. Feature flags

The dotted design names map to environment variables:

| Design flag | Environment variable | Phase 1.5 behavior |
|---|---|---|
| `memory_admission.enabled` | `TEACHER_MEMORY_ADMISSION_ENABLED` | parsed, non-authoritative |
| `memory_admission.shadow_mode` | `TEACHER_MEMORY_ADMISSION_SHADOW_MODE` | emit shadow records |
| `memory_controller.enabled` | `TEACHER_MEMORY_CONTROLLER_ENABLED` | parsed, non-authoritative |
| `memory_controller.shadow_mode` | `TEACHER_MEMORY_CONTROLLER_SHADOW_MODE` | emit shadow records |

All default off. Setting an `enabled` flag alone cannot alter a read or write.

## 14. Current candidate adapter

The observer maps existing bounded enqueue inputs to typed candidates for explicit
grade/name, known teaching-preference phrases, deterministic verification results,
explicit difficulty phrases, and requested task target. IDs are deterministic per
conversation/turn/type/payload. This adapter intentionally duplicates a small
portion of current extraction logic; consolidating extraction is deferred because
refactoring the baseline writer would violate the no-behavior-change constraint.

## 15. Feedback observation contract

`MemoryOutcomeObservation` can record verifier status, tool successes/failures,
retrieval hits/top similarity, token counts, latency, teacher-turn count, and
explicit correction. It contains no `reward`. These are measurements for later
offline study, not evidence of causal utility.

> No reliable online reward signal currently exists.

## 16. Existing-record policy

All pre-Phase-1.5 records are `legacy/unverified` for future policy purposes until
an explicit migration validates and labels them. This phase performs no migration,
cleanup, lifecycle conversion, re-indexing, confidence recalculation, or deletion.

## 17. Knowledge and experience separation

Textbook/exercise RAG and GraphRAG remain curated knowledge and are not routed into
student-memory admission. Quality-QA remains agent experience with its own pipeline
and collection. Tavily remains external turn knowledge. The foundation does not
unify these stores or feed retrieved text back as student memory.

## 18. Fixtures and tests

`tests/fixtures/memory_v2_candidate_cases.json` covers stable preference,
temporary task state, one verified error, model inference, explicit correction,
repeated verified evidence, curated GraphRAG knowledge, and quality-QA experience.
Contract tests validate serialization, boundaries, lifecycle/relations, unset
shadow proposals, and the absence of reward. Baseline characterization freezes
current enqueue/skip, extraction/merge, resolution, archive, and retrieval-feedback
separation behavior.

## 19. Observability and privacy

Shadow records are stored only as ordinary turn metadata when shadow mode is on.
They expose structured public reasons, never chain-of-thought. Payload and reference
sizes are bounded. Production telemetry storage, retention, sampling, and access
control remain deployment concerns and must be designed before enabling shadow mode
at scale.

## 20. Phase 2 boundary and readiness

Phase 2 may implement an Admission Controller against these types, compare its
proposal with the baseline in shadow mode, define matching/contradiction behavior,
and validate offline before authority. It must not become authoritative merely by
setting the Phase 1.5 flags. A Memory Controller, MemCon, reward optimization,
migration, cleanup, and cross-store unification remain separate later decisions.

Phase 1.5 is ready for Admission Controller work only after the repository's full
test suite and evaluation pass with the feature flags off.
