from __future__ import annotations

from pathlib import Path

from app.evaluation.memory_admission.robustness.layer_b import LayerBEvaluator
from app.evaluation.memory_admission.robustness.models import FailureAttribution


DATA = Path("app/evaluation/memory_admission/robustness/data/layer_b.jsonl")


def test_layer_b_has_frozen_upstream_fixtures_and_attack_coverage() -> None:
    cases = LayerBEvaluator.load_cases(DATA)
    assert len(cases) >= 30
    assert len({case.case_id for case in cases}) == len(cases)
    labels = {label for case in cases for label in case.labels}
    assert {
        "paraphrase", "negation", "conditional_preference", "temporal",
        "noisy", "capability_independent", "verified_fixture",
    } <= labels
    assert all("classification" in case.upstream for case in cases)
    assert all("teaching_contract" in case.upstream for case in cases)
    assert all("verification" in case.upstream for case in cases)


def test_layer_b_uses_existing_adapter_without_agent_or_llm() -> None:
    report = LayerBEvaluator(DATA).run()
    assert len(report.records) >= 30
    assert report.llm_calls == 0
    assert report.agent_loop_calls == 0
    assert report.extraction_accuracy.support == len(report.records)
    assert report.failures == [
        row for row in report.records if row.failure_attribution is not None
    ]
    assert all(
        row.failure_attribution in {
            FailureAttribution.CANDIDATE_EXTRACTION_FAILURE,
            FailureAttribution.NORMALIZATION_FAILURE,
        }
        for row in report.failures
    )


def test_layer_b_preserves_extraction_failures_instead_of_relabeling_policy() -> None:
    report = LayerBEvaluator(DATA).run()
    visual_paraphrases = [
        row for row in report.records
        if "visual_paraphrase" in row.labels
    ]
    assert visual_paraphrases
    assert any(
        row.failure_attribution is FailureAttribution.CANDIDATE_EXTRACTION_FAILURE
        for row in visual_paraphrases
    )
    assert all(
        row.failure_attribution is not FailureAttribution.POLICY_FAILURE
        for row in report.records
    )
