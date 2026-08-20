"""Deterministic evaluator for fixed development and sealed holdout cases."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.evaluation.memory_admission.models import (
    BenchmarkCase,
    EvaluationRecord,
    MemoryAdmissionEvaluationReport,
    MetricValue,
)
from app.memory.admission_controller import MemoryAdmissionController
from app.memory.admission_normalization import CandidateNormalizer
from app.memory.contracts import (
    AdmissionAction,
    BaselineMemoryAction,
    EvidenceClusterSnapshot,
    ExistingMemorySnapshot,
    FrozenAdmissionContext,
    MemoryCandidate,
    MemoryLifecycleStatus,
    MemorySourceKind,
    ProvenanceReference,
)


DATA_DIR = Path(__file__).with_name("data")
_ACTIONS = tuple(AdmissionAction)


def _ratio(numerator: int, denominator: int) -> MetricValue:
    return MetricValue(
        value=numerator / denominator if denominator else 0.0,
        support=denominator,
        numerator=numerator,
    )


def classification_metrics(
    labels: Iterable[AdmissionAction], predictions: Iterable[AdmissionAction]
) -> dict:
    gold, pred = list(labels), list(predictions)
    per_action = {}
    f1s = []
    for action in _ACTIONS:
        tp = sum(g is action and p is action for g, p in zip(gold, pred))
        fp = sum(g is not action and p is action for g, p in zip(gold, pred))
        fn = sum(g is action and p is not action for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_action[action.value] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": sum(g is action for g in gold),
            "predicted": sum(p is action for p in pred),
        }
    admit = per_action[AdmissionAction.ADMIT.value]
    return {
        "accuracy": {"value": sum(g is p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0, "support": len(gold)},
        "macro_f1": {"value": sum(f1s) / len(f1s), "support": len(gold)},
        "admit_precision": {"value": admit["precision"], "support": admit["predicted"]},
        "per_action": per_action,
    }


class MemoryAdmissionEvaluator:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        self.controller = MemoryAdmissionController()
        self.normalizer = CandidateNormalizer()

    def load(self, split: str) -> list[BenchmarkCase]:
        if split not in {"development", "holdout"}:
            raise ValueError("split must be development or holdout")
        path = self.data_dir / f"{split}.jsonl"
        return [BenchmarkCase.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]

    def _inputs(self, case: BenchmarkCase, index: int):
        now = datetime(2026, 8, 20, tzinfo=timezone.utc)
        provenance = ProvenanceReference(
            conversation_id=f"eval-{case.case_id}", turn_id=max(1, case.sequence_step + 1),
            source_kind=case.source_kind, event_reference=f"case:{case.case_id}",
        )
        candidate = MemoryCandidate(
            candidate_id=case.case_id, target_memory_class=case.target,
            memory_type=case.memory_type, payload=case.payload,
            source_kind=case.source_kind, provenance=[provenance],
            confidence=case.confidence, temporal_scope=case.temporal_scope,
            created_at=now,
        )
        normalized = self.normalizer.normalize(candidate)
        existing = []
        if case.existing_payload is not None:
            count = 2 if case.ambiguous_existing else 1
            existing = [ExistingMemorySnapshot(
                memory_id=f"existing-{case.case_id}-{item}",
                target_memory_class=case.target, memory_type=case.memory_type,
                semantic_key=normalized.semantic_key, payload=case.existing_payload,
                source_kind=case.existing_source, lifecycle_status=MemoryLifecycleStatus.ACTIVE,
            ) for item in range(count)]
        clusters = []
        if case.prior_supports or case.prior_contradictions:
            supports = tuple(ProvenanceReference(
                conversation_id=f"prior-{case.sequence_id or case.case_id}-{item}",
                turn_id=item + 1, source_kind=case.source_kind,
                event_reference=f"prior-support:{item}",
            ) for item in range(case.prior_supports))
            contradictions = tuple(ProvenanceReference(
                conversation_id=f"contrary-{case.case_id}-{item}", turn_id=item + 1,
                source_kind=MemorySourceKind.VERIFIED_RESULT,
                event_reference=f"prior-contradiction:{item}",
            ) for item in range(case.prior_contradictions))
            clusters = [EvidenceClusterSnapshot(
                cluster_id=f"cluster-{case.case_id}", semantic_key=normalized.semantic_key,
                normalized_payload=normalized.normalized_payload,
                memory_type=case.memory_type, supporting_provenance=supports,
                contradicting_provenance=contradictions, support_count=len(supports),
                contradiction_count=len(contradictions), distinct_turn_count=len(supports) + len(contradictions),
                distinct_conversation_count=len(supports) + len(contradictions),
                verified_support_count=sum(ref.source_kind is MemorySourceKind.VERIFIED_RESULT for ref in supports),
                source_kinds=tuple({ref.source_kind for ref in supports}), first_seen=now, last_seen=now,
            )]
        context = FrozenAdmissionContext(
            student_key=f"student-{case.sequence_id or case.case_id}",
            conversation_id=provenance.conversation_id, turn_id=provenance.turn_id,
            existing_memories=tuple(existing), evidence_clusters=tuple(clusters), captured_at=now,
        )
        return candidate, context

    def run(self, split: str) -> MemoryAdmissionEvaluationReport:
        cases = self.load(split)
        records = []
        for index, case in enumerate(cases):
            candidate, context = self._inputs(case, index)
            result = self.controller.evaluate(candidate, context)
            baseline_accepts = case.baseline_action is BaselineMemoryAction.ENQUEUE
            proposal_accepts = result.decision.decision in {AdmissionAction.ADMIT, AdmissionAction.SUPERSEDE}
            records.append(EvaluationRecord(
                case_id=case.case_id, category=case.category,
                expected_action=case.expected_action, predicted_action=result.decision.decision,
                expected_relation=case.expected_relation, predicted_relation=result.decision.relation,
                candidate=candidate.model_dump(mode="json", exclude={"provenance"}),
                provenance=[item.model_dump(mode="json") for item in candidate.provenance],
                features=result.features.model_dump(mode="json"), evidence=result.evidence_summary.model_dump(mode="json"),
                reasons=[item.value for item in result.decision.reason_codes], baseline_action=case.baseline_action,
                divergence=baseline_accepts != proposal_accepts, latency_seconds=result.latency_seconds,
            ))
        safety = self._safety(cases, records)
        sequential = self._sequential(cases, records)
        enqueued = sum(row.baseline_action is BaselineMemoryAction.ENQUEUE for row in records)
        divergences = sum(row.divergence for row in records)
        baseline = {
            "baseline_enqueue_rate": _ratio(enqueued, len(records)),
            "proposal_admit_rate": _ratio(sum(row.predicted_action in {AdmissionAction.ADMIT, AdmissionAction.SUPERSEDE} for row in records), len(records)),
            "divergence_rate": _ratio(divergences, len(records)),
        }
        return MemoryAdmissionEvaluationReport(
            split=split, cases=[case.case_id for case in cases], records=records,
            classification=classification_metrics([case.expected_action for case in cases], [row.predicted_action for row in records]),
            safety=safety, sequential=sequential, baseline=baseline,
            divergence_cases=[{"case_id": row.case_id, "baseline": row.baseline_action.value, "proposal": row.predicted_action.value, "category": row.category} for row in records if row.divergence][:12],
        )

    @staticmethod
    def _safety(cases, records):
        def rate(name, predicate, unsafe):
            pairs = [(case, row) for case, row in zip(cases, records) if predicate(case, row)]
            return name, _ratio(sum(unsafe(case, row) for case, row in pairs), len(pairs))
        specs = [
            rate("inference_admit_rate", lambda c, r: c.source_kind is MemorySourceKind.MODEL_INFERENCE, lambda c, r: r.predicted_action is AdmissionAction.ADMIT),
            rate("temporary_admit_rate", lambda c, r: c.temporal_scope.value in {"turn_only", "session"}, lambda c, r: r.predicted_action is AdmissionAction.ADMIT),
            rate("curated_to_student_admit_rate", lambda c, r: c.source_kind is MemorySourceKind.CURATED_KNOWLEDGE or c.target.value == "curated_knowledge", lambda c, r: r.predicted_action is AdmissionAction.ADMIT),
            rate("single_observation_admit_rate", lambda c, r: c.category in {"single_verified_error", "single_correct_result"}, lambda c, r: r.predicted_action is AdmissionAction.ADMIT),
            rate("contradiction_supersede_rate", lambda c, r: c.category == "contradiction", lambda c, r: r.predicted_action is AdmissionAction.SUPERSEDE),
            rate("duplicate_admit_rate", lambda c, r: c.category == "duplicate", lambda c, r: r.predicted_action is AdmissionAction.ADMIT),
        ]
        return dict(specs)

    @staticmethod
    def _sequential(cases, records):
        promotion = [(c, r) for c, r in zip(cases, records) if c.sequence_id and c.expected_action is AdmissionAction.ADMIT]
        contradiction = [(c, r) for c, r in zip(cases, records) if c.sequence_id and c.category == "contradiction"]
        return {
            "promotion_recall": _ratio(sum(r.predicted_action is AdmissionAction.ADMIT for c, r in promotion), len(promotion)),
            "contradiction_pending_rate": _ratio(sum(r.predicted_action is AdmissionAction.PENDING for c, r in contradiction), len(contradiction)),
        }
