from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExpectedCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)
    target_memory_class: str = "student"
    memory_type: str
    payload: dict[str, Any]
    source_kind: str
    temporal_scope: str


class FrozenUpstreamContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    classification: dict[str, Any] = Field(default_factory=dict)
    teaching_contract: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    observations: tuple[dict[str, Any], ...] = ()
    model_inferences: tuple[dict[str, Any], ...] = ()


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(frozen=True)
    case_id: str = Field(min_length=4)
    split: str
    utterance: str
    frozen_upstream_context: FrozenUpstreamContext = Field(default_factory=FrozenUpstreamContext)
    labels: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    expected_candidates: tuple[ExpectedCandidate, ...] = ()
    representation_gaps: tuple[str, ...] = ()


class RateMetric(BaseModel):
    value: float | None
    support: int
    numerator: int | float


class OperationalMetrics(BaseModel):
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    throughput_cases_per_second: float | None = None
    peak_ram_mb: float | None = None
    peak_vram_mb: float | None = None
    additional_model_calls: int | None = None
    dependency_footprint: list[str] = Field(default_factory=list)
    offline_capability: bool | None = None
    deterministic_replay: bool | None = None


class BenchmarkReport(BaseModel):
    split: str
    case_count: int
    expected_candidate_count: int
    actual_candidate_count: int
    detection_precision: RateMetric
    detection_recall: RateMetric
    detection_f1: RateMetric
    exact_candidate_set_accuracy: RateMetric
    memory_type_accuracy: RateMetric
    source_kind_accuracy: RateMetric
    temporal_scope_accuracy: RateMetric
    payload_accuracy: RateMetric
    qualifier_preservation: RateMetric
    false_persistent_hard_negative_rate: RateMetric
    negation_loss_rate: RateMetric
    temporal_inflation_rate: RateMetric
    conditional_corruption_rate: RateMetric
    correction_loss_rate: RateMetric
    normalization_compatibility: RateMetric
    representation_gap_count: int
    operational: OperationalMetrics = Field(default_factory=OperationalMetrics)
    case_results: list[dict[str, Any]] = Field(default_factory=list)
