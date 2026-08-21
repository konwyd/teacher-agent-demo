from __future__ import annotations
import json
from pathlib import Path
from app.evaluation.memory_admission.robustness.layer_b import LayerBEvaluator
from app.memory.foundation import build_memory_candidates
from .evaluator import evaluate_cases
from .models import BenchmarkCase, ExpectedCandidate, FrozenUpstreamContext, OperationalMetrics
ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).with_name("data")
LAYER_B = ROOT / "app/evaluation/memory_admission/robustness/data/layer_b.jsonl"
def load_cases(path: Path) -> list[BenchmarkCase]:
    return [BenchmarkCase.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def current_extractor(case: BenchmarkCase):
    ctx=case.frozen_upstream_context
    return build_memory_candidates({"turn_id":1,"student_message":case.utterance,"classification":ctx.classification,"teaching_contract":ctx.teaching_contract,"verification":ctx.verification},conversation_id=f"candidate-benchmark-{case.case_id}")
def load_layer_b() -> list[BenchmarkCase]:
    return [BenchmarkCase(case_id=f"layer-b-{c.case_id}",split="layer_b",utterance=c.utterance,frozen_upstream_context=FrozenUpstreamContext(**c.upstream),labels=c.labels,rationale=c.rationale,expected_candidates=tuple(ExpectedCandidate(**x.model_dump()) for x in c.expected_candidates)) for c in LayerBEvaluator.load_cases(LAYER_B)]
def run_baseline() -> dict:
    reports={}
    for split,cases in (("development",load_cases(DATA/"development.jsonl")),("holdout",load_cases(DATA/"holdout.jsonl")),("layer_b",load_layer_b())):
        op=OperationalMetrics(additional_model_calls=0,dependency_footprint=["python-stdlib","pydantic","current-memory-foundation"],offline_capability=True)
        report=evaluate_cases(cases,current_extractor,split=split,operational=op)
        replay=evaluate_cases(cases,current_extractor,split=split)
        report.operational.deterministic_replay=json.dumps(report.case_results,ensure_ascii=False,sort_keys=True)==json.dumps(replay.case_results,ensure_ascii=False,sort_keys=True)
        reports[split]=report.model_dump(mode="json")
    return {"benchmark_version":"candidate-extractor-v2-comparative-v1","extractor":"current_build_memory_candidates","reports":reports}
