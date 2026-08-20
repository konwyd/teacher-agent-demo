"""Layer A: direct deterministic attacks on frozen AdmissionPolicyV1."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.memory_admission.robustness.models import (
    FailureAttribution,
    LayerAReport,
    RobustnessCase,
    RobustnessRecord,
)
from app.memory.admission_controller import MemoryAdmissionController
from app.memory.admission_ledger import ShadowEvidenceLedger
from app.memory.admission_normalization import CandidateNormalizer
from app.memory.contracts import (
    AdmissionAction,
    AdmissionDecision,
    ExistingMemorySnapshot,
    FrozenAdmissionContext,
    MemoryCandidate,
    MemoryLifecycleStatus,
    ProvenanceReference,
)


NOW = datetime(2026, 8, 20, tzinfo=timezone.utc)


class LayerAEvaluator:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.controller = MemoryAdmissionController()
        self.normalizer = CandidateNormalizer()

    @staticmethod
    def load_cases(path: Path) -> list[RobustnessCase]:
        paths = sorted(path.glob("layer_a_*.jsonl")) if path.is_dir() else [path]
        return [
            RobustnessCase.model_validate_json(line)
            for source in paths
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _candidate(case: RobustnessCase) -> MemoryCandidate:
        references = []
        for index in range(case.candidate.provenance_count):
            turn_id = 42 if case.candidate.same_causal_turn else index + 1
            references.append(ProvenanceReference(
                conversation_id=f"current-{case.case_id}",
                turn_id=turn_id,
                source_kind=case.candidate.source_kind,
                event_reference=f"event:{index}",
                tool_reference=f"tool:{index}" if index else "",
                verifier_reference=f"verifier:{index}" if index else "",
            ))
        return MemoryCandidate(
            candidate_id=case.case_id,
            target_memory_class=case.candidate.target,
            memory_type=case.candidate.memory_type,
            payload=case.candidate.payload,
            source_kind=case.candidate.source_kind,
            provenance=references,
            confidence=case.candidate.confidence,
            temporal_scope=case.candidate.temporal_scope,
            created_at=NOW,
        )

    def _seed_ledger(
        self, ledger: ShadowEvidenceLedger, case: RobustnessCase, student_key: str
    ) -> None:
        for index, prior in enumerate(case.prior_evidence):
            candidate = MemoryCandidate(
                candidate_id=f"{case.case_id}-prior-{index}",
                target_memory_class=case.candidate.target,
                memory_type=case.candidate.memory_type,
                payload=prior.payload,
                source_kind=prior.source_kind,
                provenance=[ProvenanceReference(
                    conversation_id=prior.conversation_id,
                    turn_id=prior.turn_id,
                    source_kind=prior.source_kind,
                    event_reference=prior.event_reference,
                    tool_reference=prior.tool_reference,
                    verifier_reference=prior.verifier_reference,
                )],
                confidence=0.95,
                temporal_scope=case.candidate.temporal_scope,
                created_at=NOW,
            )
            normalized = self.normalizer.normalize(candidate)
            ledger.append(
                student_key=student_key,
                candidate=candidate,
                normalized=normalized,
                decision=AdmissionDecision(
                    candidate_id=candidate.candidate_id,
                    decision=AdmissionAction.PENDING,
                    policy_name="robustness-seed",
                    policy_version="fixture",
                ),
                evidence_maturity=0.0,
            )

    def _evaluate(
        self, case: RobustnessCase, root: Path
    ) -> RobustnessRecord:
        student_key = f"robust-{case.case_id}"
        ledger = ShadowEvidenceLedger(root / f"{case.case_id}.sqlite3")
        self._seed_ledger(ledger, case, student_key)
        candidate = self._candidate(case)
        normalized = self.normalizer.normalize(candidate)
        existing = ()
        if case.existing is not None:
            existing = tuple(ExistingMemorySnapshot(
                memory_id=f"existing-{case.case_id}-{index}",
                target_memory_class=case.candidate.target,
                memory_type=case.candidate.memory_type,
                semantic_key=normalized.semantic_key,
                payload=case.existing.payload,
                source_kind=case.existing.source_kind,
                lifecycle_status=MemoryLifecycleStatus.ACTIVE,
            ) for index in range(case.existing.duplicate_count))
        context = FrozenAdmissionContext(
            student_key=student_key,
            conversation_id=candidate.provenance[0].conversation_id,
            turn_id=candidate.provenance[0].turn_id,
            existing_memories=existing,
            evidence_clusters=ledger.snapshot(student_key),
            captured_at=NOW,
        )
        result = self.controller.evaluate(candidate, context)
        failure = None
        if case.expected_semantic_key and normalized.semantic_key != case.expected_semantic_key:
            failure = FailureAttribution.NORMALIZATION_FAILURE
        elif result.match.relation != case.expected_relation:
            failure = FailureAttribution.RELATION_MATCH_FAILURE
        elif result.decision.decision != case.expected_action:
            if "causal_duplicate" in case.labels:
                failure = FailureAttribution.CAUSAL_DEDUP_GAP
            else:
                failure = FailureAttribution.POLICY_FAILURE
        return RobustnessRecord(
            case_id=case.case_id,
            labels=case.labels,
            expected_action=case.expected_action,
            predicted_action=result.decision.decision,
            expected_relation=case.expected_relation,
            predicted_relation=result.match.relation,
            semantic_key=normalized.semantic_key,
            evidence_maturity=result.features.evidence_maturity.value,
            support_count=result.evidence_summary.support_count,
            distinct_turn_count=result.evidence_summary.distinct_turn_count,
            distinct_conversation_count=result.evidence_summary.distinct_conversation_count,
            failure_attribution=failure,
            reason_codes=tuple(item.value for item in result.decision.reason_codes),
            latency_seconds=result.latency_seconds,
        )

    def run(self, workspace: Path) -> LayerAReport:
        workspace.mkdir(parents=True, exist_ok=True)
        records = [self._evaluate(case, workspace) for case in self.load_cases(self.data_path)]
        return LayerAReport(
            records=records,
            failures=[row for row in records if row.failure_attribution is not None],
        )
