# CandidateExtractorV2 Route A — Deterministic Extractor Design

## Scope

Route A adds an offline deterministic extractor for the existing
`MemoryCandidate` contract. It is an isolated comparative implementation: it
does not replace `build_memory_candidates()`, enter the authoritative writer,
or change AdmissionPolicyV1, matching, ledger, causal identity, benchmark
labels, evaluator semantics, or shared contracts.

The extractor consumes the same bounded payload already used by the current
Memory V2 observer: student utterance, classification, teaching contract,
verification, observations, and model inferences. It may emit multiple
candidates, but candidate creation remains descriptive and non-authoritative.

## Architecture

The implementation is a staged deterministic parser:

1. Normalize text without discarding punctuation or qualifier-bearing words.
2. Reject privacy, quotation/meta, third-person, and instructional hard
   negatives before semantic extraction.
3. Match reusable lexical families for profile facts, teaching preferences,
   learning goals, difficulties, topic progress, and task state.
4. Parse negation, double negation, temporal bounds, conditions, scope,
   uncertainty, and correction/supersession qualifiers independently from the
   semantic family.
5. Convert frozen upstream verification, observations, and model inferences
   into their contract-defined candidate types.
6. Merge compatible fields and emit deterministic, stable-order candidates
   with deterministic IDs through the existing `MemoryCandidate` schema.

Lexicons describe semantic families and paraphrase classes. They must not
contain case IDs, complete benchmark utterances, or rules whose only purpose is
to reproduce one development label.

## Interface

`build_deterministic_memory_candidates(payload, *, conversation_id)` returns
`list[MemoryCandidate]`. Its input shape matches the current bounded extractor
adapter, so the frozen evaluator can call it through a route-local adapter
without importing evaluation code into production Memory code.

No exceptions from malformed optional upstream collections should escape for
ordinary mapping inputs. Invalid individual observation/inference records are
ignored; the shared Pydantic contract remains authoritative for emitted data.

## Qualifier semantics

- Negation and double negation are preserved as `negated` and `polarity` when
  the utterance still expresses a preference claim.
- A negated request that withdraws a technique without expressing a durable
  preference is a hard negative.
- Temporal language chooses `turn_only`, `session`, `short_lived`,
  `persistent`, or `unknown` and preserves benchmark-relevant temporal text.
- Conditional and scoped preferences retain `condition`, `scope`, `subject`,
  or `condition_map` as applicable.
- Explicit correction retains `explicit_correction` and, when identifiable,
  `correction_of`.
- Independent semantic claims can yield multiple candidates; fields belonging
  to the same profile claim remain one profile candidate.

## Benchmark and operations

A route-local runner evaluates only the development and unchanged Layer B
splits. It never loads the sealed holdout. It reports the frozen evaluator's
quality and safety metrics plus latency, peak RAM, VRAM=0, model calls=0,
offline capability, dependency footprint, and deterministic replay.

Development data is the only tuning source. Layer B is a regression gate, not
a source for case-specific memorization. Final Route A sealed-holdout execution
belongs to the coordinator.

## Verification

Implementation follows red-green-refactor cycles. Completion requires Route A
unit tests, development benchmark, Layer B regression, full pytest, offline
evaluation, Memory admission robustness, direct manifest verification, diff
hygiene, scope audit, and a clean committed worktree.
