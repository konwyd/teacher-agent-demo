"""Layer B: attack the existing lexical candidate extraction adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.memory_admission.robustness.metrics import RobustnessMetric
from app.evaluation.memory_admission.robustness.models import FailureAttribution
from app.memory.foundation import build_memory_candidates


class ExpectedCandidate(BaseModel):
    memory_type: str
    source_kind: str
    temporal_scope: str
    payload: dict[str, Any]


class LayerBCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    labels: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=240)
    utterance: str = Field(max_length=1200)
    upstream: dict[str, Any]
    expected_candidates: tuple[ExpectedCandidate, ...] = ()


class LayerBRecord(BaseModel):
    case_id: str
    labels: tuple[str, ...]
    expected_candidates: tuple[ExpectedCandidate, ...]
    actual_candidates: tuple[ExpectedCandidate, ...]
    failure_attribution: FailureAttribution | None = None


class LayerBReport(BaseModel):
    records: list[LayerBRecord]
    failures: list[LayerBRecord]
    extraction_accuracy: RobustnessMetric
    llm_calls: int = 0
    agent_loop_calls: int = 0


def _summary(candidate) -> ExpectedCandidate:
    return ExpectedCandidate(
        memory_type=candidate.memory_type.value,
        source_kind=candidate.source_kind.value,
        temporal_scope=candidate.temporal_scope.value,
        payload=candidate.payload,
    )


class LayerBEvaluator:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    @staticmethod
    def load_cases(path: Path) -> list[LayerBCase]:
        return [
            LayerBCase.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _record(case: LayerBCase) -> LayerBRecord:
        payload = {
            "turn_id": 1,
            "student_message": case.utterance,
            "classification": case.upstream["classification"],
            "teaching_contract": case.upstream["teaching_contract"],
            "verification": case.upstream["verification"],
        }
        actual = tuple(
            sorted(
                (_summary(item) for item in build_memory_candidates(
                    payload, conversation_id=f"layer-b-{case.case_id}"
                )),
                key=lambda item: (item.memory_type, item.source_kind),
            )
        )
        expected = tuple(sorted(
            case.expected_candidates,
            key=lambda item: (item.memory_type, item.source_kind),
        ))
        failure = None
        if len(actual) != len(expected) or [item.memory_type for item in actual] != [item.memory_type for item in expected]:
            failure = FailureAttribution.CANDIDATE_EXTRACTION_FAILURE
        elif actual != expected:
            failure = FailureAttribution.NORMALIZATION_FAILURE
        return LayerBRecord(
            case_id=case.case_id,
            labels=case.labels,
            expected_candidates=expected,
            actual_candidates=actual,
            failure_attribution=failure,
        )

    def run(self) -> LayerBReport:
        records = [self._record(case) for case in self.load_cases(self.data_path)]
        correct = sum(row.failure_attribution is None for row in records)
        return LayerBReport(
            records=records,
            failures=[row for row in records if row.failure_attribution is not None],
            extraction_accuracy=RobustnessMetric(
                value=correct / len(records) if records else 0.0,
                support=len(records),
                numerator=correct,
            ),
        )
