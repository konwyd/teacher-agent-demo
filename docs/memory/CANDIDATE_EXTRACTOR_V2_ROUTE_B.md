# CandidateExtractorV2 Route B

## Status and scope

Route B is an isolated Memory System research spike. Local BGE-M3 proposes one or more semantic families; deterministic code remains authoritative for bounded fields, qualifiers, source kind, and temporal scope. The route emits the existing `MemoryCandidate` contract and is not wired into the authoritative runtime or baseline writer.

As of 2026-08-24, Tasks 1–6 are implemented. Real-model development tuning and final verification remain pending. All quality and runtime measurements below are therefore provisional until Tasks 7–8 refresh this document.

- Branch: `spike/candidate-extractor-v2-embedding-hybrid`
- Route start: `e072586cfda38f88ee3d86dda615bee7da4c4cb9`
- Supported runner splits: `development`, `layer_b`
- Route B sealed-holdout runs: **0**
- Remote LLM/API calls: **0**

## Model and dependencies

| Item | Value |
|---|---|
| Model ID | `BAAI/bge-m3` |
| Architecture/output | XLM-RoBERTa; 1024-dimensional dense embedding |
| License | MIT, as declared by the local model card |
| Local-only path | `/home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3` |
| Snapshot footprint | approximately 4.3 GB (`du -sh`, 2026-08-24) |
| Active PyTorch weights | 2,271,145,830 bytes (approximately 2.27 GB) |
| Default device | CPU |
| Optional dependency | `sentence-transformers==5.6.0` |
| Observed transitive runtime | `transformers==4.43.0`, `torch==2.4.0` |

Install the research-only dependency without adding it to the base runtime:

```bash
python3 -m pip install -e '.[candidate-extractor-route-b]'
```

`BgeM3Embedder` requires the directory to exist, loads lazily with `local_files_only=True`, and never downloads weights implicitly. A missing path or invalid embedding batch raises `RouteBEmbeddingError`. CPU is the default to avoid competing with the teacher runtime; CUDA must be explicitly selected.

## Reproducible local procedure

Development-only run:

```bash
python3 scripts/run_candidate_extractor_route_b.py \
  --model-path /home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3 \
  --device cpu \
  --splits development \
  --output /tmp/candidate_extractor_route_b_development.json
```

Unchanged Layer B regression run:

```bash
python3 scripts/run_candidate_extractor_route_b.py \
  --model-path /home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3 \
  --device cpu \
  --splits layer_b \
  --output /tmp/candidate_extractor_route_b_layer_b.json
```

Combined development/Layer B evidence run:

```bash
python3 scripts/run_candidate_extractor_route_b.py \
  --model-path /home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3 \
  --device cpu \
  --splits development layer_b \
  --output /tmp/candidate_extractor_route_b.json
```

The CLI choices intentionally exclude sealed holdout. Iterative changes may use only development failures transformed into generalized tests; Layer B is a regression split, not a tuning-label source.

## Frozen evaluator metrics

The adapter preserves the frozen evaluator and derives label slices only from its `case_results`. Real BGE-M3 has not been run in Tasks 1–6, so no Route B quality value is asserted yet.

| Metric | Development | Existing Layer B |
|---|---:|---:|
| Detection precision / recall / F1 | Task 7 pending | Task 7 pending |
| Exact candidate-set accuracy | Task 7 pending | Task 7 pending |
| Memory-type accuracy | Task 7 pending | Task 7 pending |
| Source-kind accuracy | Task 7 pending | Task 7 pending |
| Temporal-scope accuracy | Task 7 pending | Task 7 pending |
| Payload accuracy | Task 7 pending | Task 7 pending |
| Normalization compatibility | Task 7 pending | Task 7 pending |
| Qualifier preservation | Task 7 pending | Task 7 pending |
| False-persistent hard negatives | Task 7 pending | Task 7 pending |
| Negation loss | Task 7 pending | Task 7 pending |
| Temporal inflation | Task 7 pending | Task 7 pending |
| Conditional corruption | Task 7 pending | Task 7 pending |
| Correction loss | Task 7 pending | Task 7 pending |
| Multi-candidate exact match | Task 7 pending | Task 7 pending |

## Qualifier safety

Embeddings select semantic families only. The deterministic parser runs first and exclusively owns qualifier values and temporal resolution.

| Safety dimension | Deterministic treatment | Frozen result |
|---|---|---:|
| Negation / double negation | Predicate-local parity and polarity | Task 7 pending |
| Temporal bounds | Turn, session, short-lived, durable, then ambiguity precedence | Task 7 pending |
| Conditions / contextual scope | Clause-local bounded condition and scope fields | Task 7 pending |
| Correction | Explicit marker and bounded `correction_of`; no lifecycle mutation | Task 7 pending |
| Uncertainty | Preserved; ambiguous persistence resolves to `unknown` | Task 7 pending |
| Cross-clause leakage | Merge only compatible same-family candidates | Task 7 pending |

## Contamination controls

| Risk | Control | Frozen result |
|---|---|---:|
| False persistent third-person fact | Predicate-local owner/recipient resolution | Task 7 pending |
| Quoted or meta-language fact | Hard-negative suppression | Task 7 pending |
| Instructional request inflation | Actor/action/object and recipient guards | Task 7 pending |
| Privacy prohibition | Deterministic do-not-store guard | Task 7 pending |
| Semantic score without fields | Family-specific mandatory-field parser rejects it | Task 7 pending |
| Temporary claim inflated by similarity | Embedding score cannot alter temporal scope | Task 7 pending |
| Malformed upstream item | Skip item and increment bounded diagnostic count | Task 7 pending |

## Provisional operational profile

| Measurement | Value |
|---|---:|
| Offline capability | Yes |
| Remote LLM/API calls | 0 |
| Snapshot / PyTorch weights | ~4.3 GB / 2,271,145,830 bytes |
| Cold-load-inclusive extraction | Task 7 pending |
| Warm p50 / p95 latency | Task 7 pending |
| Throughput | Task 7 pending |
| Process peak RAM | Task 7 pending |
| Peak VRAM on default CPU | 0 expected; Task 7 measurement pending |
| Encode calls / encoded texts | Task 7 pending |
| Malformed upstream items | Task 7 pending |
| Implementation line count | 1,540 nonblank, non-comment Python lines before Task 7 tuning |

The runner marks results provisional because other workers may share the server. `cold_load_inclusive_extract_ms` includes the first embedding-consuming extraction around lazy model load; it is not an isolated loader-only measurement.

## Repeatability

Unit tests use deterministic fake embeddings and do not load model assets. The benchmark runner performs a same-process replay and compares serialized frozen `case_results`; it reports separate primary, replay, and total encode counters. CPU/model-library behavior can vary across hosts, so exact timing is environmental even when candidate results replay identically.

## Implementation complexity and boundaries

The route consists of five local modules (`embedding`, `qualifiers`, `extractor`, `benchmark`, package exports), one local runner, and focused tests. The pre-tuning implementation contains 1,540 nonblank, non-comment Python lines as counted by the runner on 2026-08-24. No production runtime imports Route B, and ordinary package import does not eagerly load the frozen evaluation runtime or BGE-M3.

No shared contract, AdmissionPolicyV1 rule, matcher/ledger semantic, baseline writer, evaluator, benchmark data/label, existing Layer B label, policy, Agent Runtime path, or causal-dedup behavior is changed.

## Known limitations

- The Chinese parser is deliberately bounded; it is not general NER, semantic-role labeling, or discourse resolution.
- Generic embedding prototypes and thresholds are not yet development-tuned against the real local model.
- Quality, qualifier-safety, contamination, and resource tables require the Task 7 development/Layer B run.
- Local model presence and optional dependencies are operator prerequisites; missing assets fail closed for this isolated route.
- Malformed optional upstream items fail open per item, but diagnostics are aggregate rather than a persisted audit ledger.
- Route B does not solve causal observation identity or `CAUSAL_DEDUP_GAP`.
- The route remains shadow research evidence and cannot authorize or block authoritative baseline writes.
- No architecture winner can be declared before coordinator evaluation.
