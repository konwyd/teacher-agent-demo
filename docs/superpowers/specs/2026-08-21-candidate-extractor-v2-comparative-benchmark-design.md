# CandidateExtractorV2 Comparative Benchmark V1

This approved Phase 2A.6.1 checkpoint freezes implementation-neutral development and holdout labels before Routes A/B/C. Inputs are an utterance plus frozen upstream context; outputs are existing `MemoryCandidate` values. Admission policy is never invoked or scored.

Expected candidates freeze target class, type, payload, source, and temporal scope. Payload preserves negation, condition, scope, polarity, temporal expression, uncertainty, and correction semantics. IDs, timestamps, confidence, and provenance identifiers are not label targets. Unrepresentable semantics are explicitly tagged `CONTRACT_REPRESENTATION_GAP`.

The evaluator is order-independent and uses deterministic one-to-one maximum-score pairing. Detection, exact-set, field, qualifier, safety, normalization-compatibility, and nullable operational metrics remain separate. The unchanged Phase 2A.5 Layer B corpus is reported as a third split.

The baseline calls only the unchanged current extractor. The manifest separates historical source BASE provenance from runtime SHA-256 verification of datasets, evaluator, contract, and Layer B. No extractor route, PolicyV1, matcher, ledger, causal identity, authority, Agent Runtime, existing label, or admission dataset is changed.
