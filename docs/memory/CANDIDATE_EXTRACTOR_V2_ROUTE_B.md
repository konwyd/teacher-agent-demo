# CandidateExtractorV2 Route B

## Status and scope

Route B is an isolated Memory System research spike. Local BGE-M3 proposes one or more semantic families; deterministic code remains authoritative for bounded fields, qualifiers, source kind, and temporal scope. The route emits the existing `MemoryCandidate` contract and is not wired into the authoritative runtime or baseline writer.

As of 2026-08-24, Tasks 1–8 are implemented and final repository, evaluation, robustness, manifest, and protected-scope verification is complete. The tables below report fresh real BGE-M3 CPU development evidence and unchanged Layer B regression evidence; resource measurements remain provisional because the server is shared.

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

The adapter preserves the frozen evaluator and derives aggregate slices only from its `case_results`. Tuning used development failures only. Layer B was rerun unchanged as regression evidence; its case results and labels were not used for tuning.

| Metric | Development | Existing Layer B |
|---|---:|---:|
| Detection precision / recall / F1 | 1.0000 / 1.0000 / 1.0000 | 0.9167 / 0.4231 / 0.5789 |
| Exact candidate-set accuracy | 1.0000 (32/32) | 0.28125 (9/32) |
| Memory-type accuracy | 1.0000 | 0.9091 |
| Source-kind accuracy | 1.0000 | 0.9091 |
| Temporal-scope accuracy | 1.0000 | 0.8182 |
| Payload accuracy | 1.0000 | 0.3636 |
| Normalization compatibility | 0.9310 (27/29) | 1.0000 |
| Qualifier preservation | 1.0000 (17/17) | N/A (0 support) |
| False-persistent hard negatives | 0 errors (0/4) | N/A |
| Negation loss | 0 errors (0/1) | N/A |
| Temporal inflation | 0 errors (0/29) | 0.0909 (1/11) |
| Conditional corruption | 0 errors (0/3) | N/A |
| Correction loss | 0 errors (0/2) | N/A |
| Multi-candidate exact match | 1.0000 (1/1) | N/A |

## Qualifier safety

Embeddings select semantic families only. The deterministic parser runs first and exclusively owns qualifier values and temporal resolution.

| Safety dimension | Deterministic treatment | Development | Existing Layer B |
|---|---|---:|---:|
| Negation / double negation | Predicate-local parity and polarity | 0 loss (0/1) | N/A |
| Temporal bounds | Turn, session, short-lived, durable, then ambiguity precedence | 0 inflation (0/29) | 1 inflation (1/11) |
| Conditions / contextual scope | Clause-local bounded condition and scope fields | 0 corruption (0/3) | N/A |
| Correction | Explicit marker and typed bounded `correction_of`; no lifecycle mutation | 0 loss (0/2) | N/A |
| Qualifier preservation | Deterministic qualifier payload | 1.0000 (17/17) | N/A (0 support) |
| Cross-clause leakage | Merge only compatible same-family candidates | No frozen failure | No dedicated support |

Layer B temporal-scope accuracy is 0.8182 and temporal inflation is 1/11. These are material regression signals, not noise to hide behind the perfect development result.

## Contamination controls

| Risk | Control | Development | Existing Layer B |
|---|---|---:|---:|
| False persistent third-person fact | Predicate-local owner/recipient resolution | 0 errors (0/4) | N/A |
| Quoted or meta-language fact | Hard-negative suppression | No frozen failure | No dedicated support |
| Instructional request inflation | Actor/action/object and recipient guards | No frozen failure | No dedicated support |
| Privacy prohibition | Deterministic do-not-store guard | No frozen failure | No dedicated support |
| Semantic score without fields | Family-specific mandatory-field parser rejects it | Precision 1.0000 | Precision 0.9167 |
| Temporary claim inflated by similarity | Embedding score cannot alter temporal scope | 0/29 | 1/11 |
| Malformed upstream item | Skip item and increment bounded diagnostic count | 0 | 0 |

Development safety slices are clean, but unchanged Layer B payload accuracy 0.3636, temporal-scope accuracy 0.8182, and temporal inflation 0.0909 show material generalization and contamination risk. Route B is not ready to be declared a winner.

## Provisional operational profile

| Measurement | Development | Existing Layer B |
|---|---:|---:|
| Offline capability | Yes | Yes |
| Remote LLM/API calls | 0 | 0 |
| Snapshot / PyTorch weights | ~4.3 GB / 2,271,145,830 bytes | same local snapshot |
| Cold-load-inclusive extraction | 5351.71 ms | 5567.95 ms |
| Warm p50 latency | 32.71 ms | 36.94 ms |
| Warm p95 latency | 44.79 ms | 54.52 ms |
| Throughput | 5.087 cases/s | 4.746 cases/s |
| Process peak RAM | 2158.40 MB | 2165.19 MB |
| Peak VRAM on default CPU | N/A | N/A |
| Primary encode calls / texts | 29 / 46 | 32 / 52 |
| Deterministic replay | Yes | Yes |
| Malformed upstream items | 0 | 0 |
| Implementation line count | 1,779 | 1,779 |

The runner marks resource results provisional because other workers may share the server. `cold_load_inclusive_extract_ms` includes the first embedding-consuming extraction around lazy model load; it is not an isolated loader-only measurement.

## Repeatability

Unit tests use deterministic fake embeddings and do not load model assets. The benchmark runner performs a same-process replay and compares serialized frozen `case_results`; it reports separate primary, replay, and total encode counters. CPU/model-library behavior can vary across hosts, so exact timing is environmental even when candidate results replay identically.

## Implementation complexity and boundaries

The route consists of five local modules (`embedding`, `qualifiers`, `extractor`, `benchmark`, package exports), one local runner, and focused tests. The post-tuning implementation contains 1,779 nonblank, non-comment Python lines as counted by the runner on 2026-08-24. No production runtime imports Route B, and ordinary package import does not eagerly load the frozen evaluation runtime or BGE-M3.

No shared contract, AdmissionPolicyV1 rule, matcher/ledger semantic, baseline writer, evaluator, benchmark data/label, existing Layer B label, policy, Agent Runtime path, or causal-dedup behavior is changed.

## Known limitations

- The Chinese parser is deliberately bounded; it is not general NER, semantic-role labeling, or discourse resolution.
- The frozen route-local contract distinguishes an explicit durable goal (`我想…前掌握…`, `deadline_expression`, persistent) from a deadline-first bounded directive (`…前掌握…`, `temporal_expression`, short-lived); near-paraphrases that change this modality remain a documented normalization boundary.
- Perfect development performance is provisional evidence on a small 32-case split and is not proof of generalization.
- Unchanged Layer B exact accuracy 0.28125, recall 0.4231, payload accuracy 0.3636, temporal-scope accuracy 0.8182, and temporal inflation 0.0909 show material generalization and contamination risk.
- No tuning used Layer B case results or labels; only aggregate unchanged regression metrics were recorded.
- Local model presence and optional dependencies are operator prerequisites; missing assets fail closed for this isolated route.
- Malformed optional upstream items fail open per item, but diagnostics are aggregate rather than a persisted audit ledger.
- Route B does not solve causal observation identity or `CAUSAL_DEDUP_GAP`.
- The route remains shadow research evidence and cannot authorize or block authoritative baseline writes.
- An earlier independent reviewer accidentally exposed sealed-holdout path content during a broad read-only search. The primary Route B worker did not receive or use labels/IDs, and Route B sealed-holdout executions remain 0, but a clean `SEALED_HOLDOUT_LABELS_INSPECTED=NO` attestation is no longer possible; coordinator governance review is required.
- No architecture winner can be declared before coordinator evaluation.
