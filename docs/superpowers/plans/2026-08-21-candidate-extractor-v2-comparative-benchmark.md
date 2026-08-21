# CandidateExtractorV2 Comparative Benchmark Plan

1. Test and freeze schemas plus disjoint development/holdout labels.
2. Test and implement one order-independent evaluator with explicit safety metrics.
3. Test and add current-extractor and unchanged Layer B adapters.
4. Generate baseline results and a content-hash manifest rooted at BASE `d7ed183f83c162b9dcbc78c5e7c992e758bf1497`.
5. Run benchmark, full pytest, offline eval, and robustness gates; audit, commit, push, report, and stop before Routes A/B/C.
