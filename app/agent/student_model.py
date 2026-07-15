"""Minimal evidence-ledger student model for the topic-neutral harness."""

from __future__ import annotations

from app.agent.state import MathAnalysis, VerificationResult, VerificationStatus
from app.agent.teaching_actions import TeachingAction, TeachingActionPlan
from app.memory.schemas import (
    EpisodeState,
    StudentEvidenceKind,
    StudentEvidenceRecord,
    StudentEvidenceSnapshot,
)


class StudentModel:
    """Record observed turn evidence without inferring a learner profile."""

    _MAX_RECORDS = 40

    def record_turn(
        self,
        episode: EpisodeState,
        *,
        verification: VerificationResult,
        action_plan: TeachingActionPlan,
        analysis: MathAnalysis,
        source_turn: int,
        scope_id: str = "",
    ) -> None:
        prior_action = next(
            (
                record
                for record in reversed(episode.student_evidence)
                if record.kind is StudentEvidenceKind.TEACHING_ACTION
                and record.scope_id == scope_id
            ),
            None,
        )

        verified_text = verification.evidence[0] if verification.evidence else verification.summary
        if verification.status is VerificationStatus.VERIFIED_CORRECT:
            episode.student_evidence.append(
                StudentEvidenceRecord(
                    source_turn=source_turn,
                    scope_id=scope_id,
                    kind=StudentEvidenceKind.CONFIRMED_CORRECT,
                    text=verified_text,
                    verifier_name=verification.verifier_name,
                )
            )
        elif verification.status is VerificationStatus.VERIFIED_INCORRECT:
            episode.student_evidence.append(
                StudentEvidenceRecord(
                    source_turn=source_turn,
                    scope_id=scope_id,
                    kind=StudentEvidenceKind.VERIFIED_ERROR,
                    text=verified_text,
                    verifier_name=verification.verifier_name,
                )
            )

        if analysis.likely_obstacle:
            episode.student_evidence.append(
                StudentEvidenceRecord(
                    source_turn=source_turn,
                    scope_id=scope_id,
                    kind=StudentEvidenceKind.UNRESOLVED_OBSTACLE,
                    text=analysis.likely_obstacle,
                    confidence=analysis.confidence,
                )
            )

        if (
            verification.status is VerificationStatus.VERIFIED_CORRECT
            and prior_action is not None
            and prior_action.teaching_action_code == TeachingAction.SCAFFOLD.value
        ):
            episode.student_evidence.append(
                StudentEvidenceRecord(
                    source_turn=source_turn,
                    scope_id=scope_id,
                    kind=StudentEvidenceKind.HINT_OUTCOME,
                    text=verified_text,
                    progressed_after_hint=True,
                )
            )

        episode.student_evidence.append(
            StudentEvidenceRecord(
                source_turn=source_turn,
                scope_id=scope_id,
                kind=StudentEvidenceKind.TEACHING_ACTION,
                text=action_plan.rationale or action_plan.action.value,
                confidence=action_plan.confidence,
                teaching_action_code=action_plan.action.value,
                scaffold_level=action_plan.scaffold_level,
            )
        )
        episode.student_evidence = episode.student_evidence[-self._MAX_RECORDS :]

    def snapshot(
        self,
        episode: EpisodeState,
        *,
        scope_id: str | None = None,
    ) -> StudentEvidenceSnapshot:
        records = [
            record
            for record in episode.student_evidence
            if scope_id is None or record.scope_id == scope_id
        ]
        confirmed_correct = [
            record
            for record in records
            if record.kind is StudentEvidenceKind.CONFIRMED_CORRECT
        ]
        verified_errors = [
            record
            for record in records
            if record.kind is StudentEvidenceKind.VERIFIED_ERROR
        ]
        unresolved_obstacles = [
            record
            for record in records
            if record.kind is StudentEvidenceKind.UNRESOLVED_OBSTACLE
        ]
        action_records = [
            record
            for record in records
            if record.kind is StudentEvidenceKind.TEACHING_ACTION
        ]
        latest_scaffold_index = next(
            (
                index
                for index in range(len(records) - 1, -1, -1)
                if records[index].kind is StudentEvidenceKind.TEACHING_ACTION
                and records[index].scaffold_level > 0
            ),
            None,
        )
        latest_scaffold = (
            records[latest_scaffold_index]
            if latest_scaffold_index is not None
            else None
        )
        records_after_latest_scaffold = (
            records[latest_scaffold_index + 1 :]
            if latest_scaffold_index is not None
            else []
        )
        latest_hint_outcome = next(
            (
                record
                for record in reversed(records_after_latest_scaffold)
                if record.kind is StudentEvidenceKind.HINT_OUTCOME
            ),
            None,
        )

        consecutive_verified_errors = 0
        for record in reversed(records):
            if record.kind is StudentEvidenceKind.VERIFIED_ERROR:
                consecutive_verified_errors += 1
            elif record.kind is StudentEvidenceKind.CONFIRMED_CORRECT:
                break

        return StudentEvidenceSnapshot(
            records=records,
            confirmed_correct=confirmed_correct,
            verified_errors=verified_errors,
            unresolved_obstacles=unresolved_obstacles,
            recent_action_codes=[record.teaching_action_code for record in action_records],
            latest_scaffold_level=(
                latest_scaffold.scaffold_level if latest_scaffold is not None else 0
            ),
            latest_verified_step=(
                confirmed_correct[-1].text if confirmed_correct else ""
            ),
            progress_after_latest_hint=(
                latest_hint_outcome.progressed_after_hint
                if latest_hint_outcome is not None
                else None
            ),
            consecutive_verified_errors=consecutive_verified_errors,
        )
