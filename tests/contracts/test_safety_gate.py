import pytest

from app.agent.safety_gate import SafetyGate, SafetyInput
from app.agent.state import TeachingIntent, VerificationStatus
from app.agent.teaching_actions import DisclosureLevel, TeachingAction


pytestmark = pytest.mark.safety_contract


def test_missing_problem_forces_elicit_and_no_disclosure() -> None:
    result = SafetyGate().evaluate(SafetyInput(problem_context=""))
    assert result.allowed_actions == [TeachingAction.ELICIT]
    assert result.disclosure_ceiling is DisclosureLevel.NONE


def test_unrequested_candidate_never_receives_final_disclosure() -> None:
    result = SafetyGate().evaluate(
        SafetyInput(
            problem_context="3x+2y=5，6x-2y=3",
            candidate_available=True,
            candidate_confidence=0.99,
        )
    )
    assert result.disclosure_ceiling is DisclosureLevel.NEXT_STEP


def test_verified_error_limits_actions_without_forcing_one() -> None:
    result = SafetyGate().evaluate(
        SafetyInput(
            problem_context="2x+3=7",
            verification_status=VerificationStatus.VERIFIED_INCORRECT,
            has_verified_error=True,
        )
    )
    assert result.allowed_actions == [
        TeachingAction.FOCUS,
        TeachingAction.SCAFFOLD,
        TeachingAction.CORRECT,
        TeachingAction.EXPLAIN,
    ]


def test_model_requires_one_of_the_three_approved_triggers() -> None:
    ordinary = SafetyGate().evaluate(SafetyInput(problem_context="一个数学问题"))
    requested = SafetyGate().evaluate(
        SafetyInput(
            problem_context="一个数学问题",
            intent=TeachingIntent.REQUESTS_EXAMPLE,
        )
    )
    assert TeachingAction.MODEL not in ordinary.allowed_actions
    assert requested.worked_example_allowed is True
    assert TeachingAction.MODEL in requested.allowed_actions


@pytest.mark.parametrize("message", ["我完全不懂", "我一点都不会", "我完全不会做"])
def test_total_confusion_markers_are_narrow(message: str) -> None:
    context = SafetyInput.from_turn(problem_context="一个数学问题", message=message)
    assert context.expresses_total_confusion is True


@pytest.mark.parametrize(
    "message",
    [
        "我不是完全不懂，只是不知道怎么选方法",
        "我完全不懂？不是，我只是不知道怎么选方法",
        "题目中引用了“我完全不懂”",
        "我完全不懂这个术语，但题会做",
    ],
    ids=["negated", "negated_exact_phrase", "quoted", "qualified"],
)
def test_total_confusion_near_matches_do_not_unlock_model(message: str) -> None:
    context = SafetyInput.from_turn(problem_context="一个数学问题", message=message)
    result = SafetyGate().evaluate(context)

    assert context.expresses_total_confusion is False
    assert TeachingAction.MODEL not in result.allowed_actions


@pytest.mark.parametrize("message", ["  我完全不懂。 ", "我一点都不会！", "我完全不会做？"])
def test_total_confusion_allows_harmless_outer_formatting(message: str) -> None:
    context = SafetyInput.from_turn(problem_context="一个数学问题", message=message)
    assert context.expresses_total_confusion is True


def test_ordinary_method_uncertainty_does_not_unlock_model() -> None:
    context = SafetyInput.from_turn(
        problem_context="一个数学问题",
        message="我不知道怎么选择方法",
    )
    assert context.expresses_total_confusion is False
