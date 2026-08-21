# CandidateExtractorV2 Route B Design

## 1. Purpose

Route B is an isolated comparative implementation of CandidateExtractorV2 for
the Memory System. It uses a local embedding model to identify semantic memory
families and paraphrased candidate intent, then delegates every qualifier and
temporal decision to deterministic code.

The route consumes the frozen benchmark input shape—one student utterance plus
`FrozenUpstreamContext`—and returns the existing `MemoryCandidate` contract. It
does not call or infer labels from AdmissionPolicyV1.

## 2. Scope

Route B will:

- identify profile facts, teaching preferences, learning goals, difficulties,
  topic progress, task state, learning episodes, and behavior evidence;
- consume relevant frozen verification, observation, and model-inference
  context without changing its schema;
- preserve negation, double negation, conditions, temporal expressions,
  corrections, polarity, uncertainty, and contextual scope in candidate
  payloads;
- support multiple candidates and clause-local qualifiers;
- classify temporal scope as `turn_only`, `session`, `short_lived`,
  `persistent`, or `unknown` using deterministic rules;
- run fully offline with a local BGE-M3 model; and
- expose route-local operational measurements without changing the frozen
  evaluator.

Route B will not:

- modify or replace the current `build_memory_candidates()` baseline;
- change `MemoryCandidate`, benchmark schemas, evaluator behavior, manifest
  semantics, datasets, or labels;
- modify AdmissionPolicyV1, the matcher, the evidence ledger, the baseline
  writer, Memory V2 authority, Agent Runtime, or Agent-to-Memory contracts;
- solve causal observation identity or `CAUSAL_DEDUP_GAP`;
- call DeepSeek, a remote extraction API, an unrestricted LLM judge, Route A,
  or Route C; or
- inspect sealed-holdout labels or tune against sealed-holdout results.

## 3. Model and Dependency Choice

Route B uses the server's existing local `BAAI/bge-m3` snapshot through
`sentence-transformers`.

- Model identity: `BAAI/bge-m3`
- Model architecture: XLM-RoBERTa, 1024-dimensional dense embeddings
- Model license: MIT, as declared by the local model card
- Local snapshot: `/home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3`
- Snapshot disk footprint: approximately 4.3 GB because it contains both
  PyTorch and ONNX artifacts
- Active PyTorch weight file: approximately 2.27 GB
- Installed runtime versions at design time:
  `sentence-transformers==5.6.0`, `transformers==4.43.0`, `torch==2.4.0`
- Default execution device: CPU, to avoid competing with the local teacher
  runtime and to make peak VRAM zero

The project will declare a route-specific optional dependency group instead of
adding this research dependency to the base runtime. The route requires an
existing local model directory and must never download weights implicitly. A
missing or invalid model path produces a clear route-level configuration error;
it does not silently fall back to lexical Route A behavior.

## 4. Architecture

The implementation is a route-local package under
`app/memory/candidate_extractor_v2_route_b/` with four boundaries:

1. `embedding.py` owns local model loading, generic semantic-family prototypes,
   normalized dense encoding, and similarity ranking. It knows nothing about
   qualifiers or `TemporalScope`.
2. `qualifiers.py` performs deterministic clause segmentation, hard-negative
   gating, qualifier parsing, and temporal-scope resolution. It never consumes
   embedding similarity.
3. `extractor.py` maps ranked families and frozen upstream evidence to bounded
   field parsers, constructs existing `MemoryCandidate` values, and merges
   duplicate clause results without losing qualifiers.
4. `benchmark.py` adapts the extractor to the frozen evaluator for development
   and unchanged Layer B runs and records route-local operational data.

`__init__.py` exports only the Route B extractor, its configuration, and the
embedding protocol needed for deterministic tests. No production runtime path
imports Route B during this spike.

## 5. Data Flow

For each benchmark case:

1. Normalize Unicode and punctuation without deleting semantic words.
2. Segment the utterance into clauses while retaining conjunctions that bind a
   temporal boundary or correction.
3. Apply deterministic hard-negative guards for quoted/meta uses, third-person
   statements, instructional examples, and privacy prohibitions.
4. Parse qualifiers for every surviving clause and retain their source spans.
5. Embed qualifier-preserving clause text and rank it against generic semantic
   family prototypes.
6. Admit zero or more semantic families above development-tuned thresholds.
   Similarity proposes families only; it cannot change parsed qualifier values,
   candidate source kind, or temporal scope.
7. Run the family-specific deterministic field parser. A family without the
   required bounded fields produces no candidate even if similarity is high.
8. Add candidates derived from authoritative frozen upstream signals such as a
   verifier result, an observation, or an explicit model inference. Their
   `source_kind` is determined by the upstream signal, never by similarity.
9. Merge only candidates with the same semantic family and compatible payload.
   Clause-local qualifiers are retained; incompatible qualifiers remain
   separate candidates.
10. Construct `MemoryCandidate` values with deterministic IDs and bounded
    provenance using the existing contract.

## 6. Embedding Boundary

The embedding catalog contains short, generic descriptions of semantic intent,
not benchmark utterances or expected payloads. Each family uses multiple generic
prototypes to cover ordinary paraphrases. Development tuning may adjust family
thresholds and prototype wording, but it must not add one-off rules keyed to a
case ID or copy a development utterance.

The embedder interface accepts a batch of strings and returns normalized vectors.
The BGE-M3 implementation lazily loads one local model and caches prototype
vectors. A deterministic fake implementation supplies vectors in unit tests so
tests do not need 4.3 GB of model assets.

Multi-candidate extraction embeds clauses in one batch. The operational counter
reports encode invocations and encoded text count separately from LLM calls;
Route B always reports zero LLM/API calls.

## 7. Deterministic Qualifier Semantics

Qualifier parsing happens before semantic-family ranking and is authoritative.

- **Negation and polarity:** compute clause-local negation parity around the
  candidate predicate. Odd parity emits `negated=true` and
  `polarity=negative`; recognized double negation emits `negated=false` and
  `polarity=positive`.
- **Conditions:** preserve the bounded condition phrase introduced by markers
  such as “如果”, “只在”, and exception forms. Conditions do not lower or raise
  persistence by embedding score.
- **Temporal expressions:** recognize bounded turn expressions, active-session
  expressions, short-lived intervals, durable-from-now expressions, and
  uncertain future expressions. Preserve the surface expression in the payload.
- **Temporal scope:** deterministic precedence is turn-bounded, session-bounded,
  short-lived, explicit durable, hedged/ambiguous, then family default. A more
  persistent scope can never be selected because of higher similarity.
- **Corrections:** explicit correction markers set `explicit_correction=true`
  and a bounded `correction_of` semantic target. The new claim is emitted; the
  extractor does not perform lifecycle supersession.
- **Contextual scope:** general, subject-specific, and current-problem exception
  scopes are preserved in `scope` and/or `condition` fields.
- **Uncertainty:** hedges are preserved and force `TemporalScope.UNKNOWN` where
  the claim's validity is uncertain.

The parser returns a typed route-local value rather than mutating candidate
payloads opportunistically. Unit tests assert each qualifier independently and
in combinations.

## 8. Family-Specific Field Extraction

After semantic-family selection, deterministic parsers extract only bounded
fields supported by the current candidate contract:

- profile facts: grade and preferred name, including multi-field statements;
- preferences: canonical teaching preference plus parsed qualifiers;
- learning goals: subject, goal phrase, and deadline expression;
- difficulties: subject, topic, and student-stated difficulty;
- topic progress: subject, topic, and bounded progress status;
- task state: subject and active requested target;
- learning episodes: subject and bounded observed episode summary;
- behavior evidence: subject, verification status or bounded observed pattern.

Frozen upstream verification is the only source of `verified_result` candidates.
Explicit frozen model inferences remain `model_inference` with unknown temporal
scope. The embedding layer cannot promote either source kind.

## 9. Hard Negatives and Contamination Safety

Persistent candidates are suppressed when deterministic analysis identifies:

- a third-person referent rather than the student;
- a quoted phrase being defined or discussed;
- an instructional request that merely contains a preference-like word;
- a request not to store or remember a fact; or
- a family score without the mandatory bounded fields.

Ambiguous claims use `unknown` rather than `persistent`. Temporary expressions
must not inflate to persistent. False persistent candidates, hard-negative
behavior, negation loss, conditional corruption, correction loss, and temporal
inflation remain independently visible through the frozen evaluator.

## 10. Failure Behavior

- Missing local model assets: fail with a public configuration error before a
  benchmark run; do not download or substitute another model.
- Embedding shape or non-finite values: reject the batch with a route-local
  error; do not produce partially ranked families.
- No family above threshold: return no utterance-derived candidate, while still
  allowing valid frozen upstream candidates.
- Parser ambiguity: retain uncertainty and choose `unknown`; never infer
  persistence optimistically.
- Malformed optional upstream item: skip only that item and expose a bounded
  diagnostic count in the runner; do not change the frozen input contract.

Because Route B is not wired into the authoritative runtime, these failures
cannot block teaching or baseline memory writes.

## 11. Files and Ownership

Planned Route B files:

- `app/memory/candidate_extractor_v2_route_b/__init__.py`
- `app/memory/candidate_extractor_v2_route_b/embedding.py`
- `app/memory/candidate_extractor_v2_route_b/qualifiers.py`
- `app/memory/candidate_extractor_v2_route_b/extractor.py`
- `app/memory/candidate_extractor_v2_route_b/benchmark.py`
- `tests/test_candidate_extractor_v2_route_b.py`
- `scripts/run_candidate_extractor_route_b.py`
- `docs/memory/CANDIDATE_EXTRACTOR_V2_ROUTE_B.md`
- the Route B optional dependency entry in `pyproject.toml`
- only the Memory System section of `PROGRESS.md`

All implementation files are owned by the Memory System for this assigned
spike. Existing `app/evaluation/candidate_extractor/**`, admission code,
contracts, datasets, and Layer B files remain read-only. No shared-interface
change or mandatory cross-subsystem co-review is required by this design.

## 12. Test-Driven Implementation

Implementation follows red-green-refactor cycles:

1. qualifier parser tests for negation parity, conditions, temporal precedence,
   corrections, scope, uncertainty, and combined clauses;
2. hard-negative tests for third-person, meta, instructional, and privacy cases;
3. fake-embedding tests proving embeddings select families but cannot alter
   qualifiers or temporal scope;
4. multi-candidate and multi-field extraction tests;
5. frozen-upstream tests for verifier, observation, and model-inference sources;
6. local-model configuration and deterministic replay tests; and
7. route runner and operational-metadata tests.

No frozen test or evaluator assertion will be weakened or rewritten.

## 13. Experimental Procedure

Iterative tuning uses only `development.jsonl`. The route runner supports the
development split and unchanged Layer B by default. It does not expose a normal
sealed-holdout tuning mode. The coordinator can import the Route B extractor for
the single controlled final comparison.

The development loop records every threshold/prototype revision in the branch
diff and rejects case-ID-specific behavior. Existing Layer B is a regression
split, not a tuning label source.

Performance collected while other workers use the server is explicitly marked
provisional. CPU BGE-M3 measurements include cold model load separately from
warm per-case extraction.

## 14. Metrics and Reporting

The frozen evaluator supplies:

- detection precision, recall, and F1;
- exact candidate-set match;
- memory-type, source-kind, temporal-scope, and payload accuracy;
- qualifier preservation;
- false persistent hard negatives;
- negation loss, temporal inflation, conditional corruption, and correction
  loss; and
- normalization compatibility.

The runner derives label-slice summaries, including multi-candidate exact match,
only from frozen `case_results`; it does not alter evaluator semantics.

Operational reporting adds model identity, snapshot size, cold-load latency,
warm p50/p95 latency, throughput, process peak RAM, peak VRAM, encode calls,
zero remote/LLM calls, dependency footprint, offline capability, deterministic
replay, implementation line count, and known limitations.

## 15. Verification and Handoff

Before the Route B commit, run and record:

- Route B unit tests;
- CandidateExtractorV2 development benchmark;
- unchanged Layer B regression;
- `python -m pytest`;
- `python scripts/run_eval.py`;
- existing Memory admission robustness;
- benchmark-manifest verification; and
- diff checks proving protected files and labels are unchanged.

The final handoff reports the assigned branch, start SHA, final commit, clean
worktree status, exact commands/results, all frozen metrics, provisional resource
measurements, model/dependency details, limitations, and explicit `NO` values for
every protected-scope change. It reports zero Route B sealed-holdout runs and does
not declare an overall architecture winner.
