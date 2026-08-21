import json
from pathlib import Path
from app.evaluation.candidate_extractor.baseline import DATA,current_extractor,load_cases,load_layer_b,run_baseline
from app.evaluation.candidate_extractor.evaluator import evaluate_cases
from app.evaluation.candidate_extractor.manifest import MANIFEST,verify_manifest
from app.evaluation.candidate_extractor.models import BenchmarkCase,ExpectedCandidate

def test_frozen_datasets_are_disjoint_and_cover_contract():
    dev=load_cases(DATA/"development.jsonl");hold=load_cases(DATA/"holdout.jsonl")
    assert len(dev)==32 and len(hold)==20
    assert {x.case_id for x in dev}.isdisjoint(x.case_id for x in hold)
    all_cases=dev+hold
    types={c.memory_type for x in all_cases for c in x.expected_candidates}
    assert {"profile_fact","preference","topic_progress","difficulty","behavior_evidence","learning_goal","learning_episode","task_state"}<=types
    sources={c.source_kind for x in all_cases for c in x.expected_candidates}
    assert {"user_stated","observed_behavior","verified_result","model_inference"}<=sources
    labels={label for x in all_cases for label in x.labels}
    assert {"HARD_NEGATIVE","NEGATION","DOUBLE_NEGATION","CONDITIONAL","CORRECTION","UNCERTAINTY","MULTI_CANDIDATE"}<=labels

def test_evaluator_is_order_independent_and_exposes_safety_denominators():
    case=BenchmarkCase(case_id="test-1",split="test",utterance="",labels=("MULTI_CANDIDATE",),rationale="synthetic",
        expected_candidates=(ExpectedCandidate(memory_type="profile_fact",payload={"grade":"三年级"},source_kind="user_stated",temporal_scope="persistent"),ExpectedCandidate(memory_type="preference",payload={"teaching_preferences":["示例讲解"]},source_kind="user_stated",temporal_scope="persistent")))
    actual=list(reversed(case.expected_candidates))
    report=evaluate_cases([case],lambda _:actual,split="test")
    assert report.exact_candidate_set_accuracy.value==1
    assert report.payload_accuracy.value==1
    assert report.temporal_inflation_rate.support==2

def test_current_baseline_is_offline_and_layer_b_is_unchanged_adapter():
    assert len(load_layer_b())==32
    report=run_baseline()
    assert set(report["reports"])=={"development","holdout","layer_b"}
    assert all(x["operational"]["additional_model_calls"]==0 for x in report["reports"].values())
    assert all(x["operational"]["deterministic_replay"] is True for x in report["reports"].values())

def test_frozen_manifest_verifies_content_not_current_head_or_dirty_state():
    manifest=json.loads(MANIFEST.read_text())
    assert manifest["historical_provenance"]["source_base_sha"]=="d7ed183f83c162b9dcbc78c5e7c992e758bf1497"
    assert manifest["historical_provenance"]["routes_implemented"] is False
    assert manifest["label_seal"]["sealed_before_first_formal_baseline"] is True
    assert verify_manifest()==[]
