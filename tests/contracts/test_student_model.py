import pytest

from app.agent.student_model import StudentModel
from app.agent.teaching_actions import TeachingAction, TeachingActionPlan
from app.agent.state import MathAnalysis, VerificationResult, VerificationStatus
from app.memory.schemas import EpisodeState, StudentEvidenceKind


pytestmark = pytest.mark.safety_contract


def _episode() -> EpisodeState:
    return EpisodeState(conversation_id="evidence-conv", student_id="student")


def test_verified_error_is_stored_with_source_and_verifier() -> None:
    episode = _episode()
    StudentModel().record_turn(
        episode,
        verification=VerificationResult(
            status=VerificationStatus.VERIFIED_INCORRECT,
            evidence=["移项时符号没有改变"],
            verifier_name="linear_equation_fraction_checker",
        ),
        action_plan=TeachingActionPlan(action=TeachingAction.FOCUS),
        analysis=MathAnalysis(),
        source_turn=2,
        scope_id="linear-equation",
    )
    record = episode.student_evidence[0]
    assert record.kind is StudentEvidenceKind.VERIFIED_ERROR
    assert record.source_turn == 2
    assert record.verifier_name == "linear_equation_fraction_checker"


def test_snapshot_contains_evidence_not_personality_claims() -> None:
    snapshot = StudentModel().snapshot(_episode(), scope_id="linear-equation")
    dumped = snapshot.model_dump(mode="json")
    assert "preferred_teaching_style" not in dumped
    assert "ability_level" not in dumped
    assert "confirmed_correct" in dumped
    assert "verified_errors" in dumped


def test_correct_verification_after_scaffold_records_progress() -> None:
    episode = _episode()
    model = StudentModel()
    model.record_turn(
        episode,
        verification=VerificationResult(),
        action_plan=TeachingActionPlan(action=TeachingAction.SCAFFOLD, scaffold_level=1),
        analysis=MathAnalysis(),
        source_turn=1,
        scope_id="linear-equation",
    )
    model.record_turn(
        episode,
        verification=VerificationResult(
            status=VerificationStatus.VERIFIED_CORRECT,
            evidence=["2x=4"],
        ),
        action_plan=TeachingActionPlan(action=TeachingAction.CHECK),
        analysis=MathAnalysis(),
        source_turn=2,
        scope_id="linear-equation",
    )
    assert (
        model.snapshot(episode, scope_id="linear-equation").progress_after_latest_hint
        is True
    )
