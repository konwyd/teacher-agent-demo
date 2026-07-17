import pytest

from app.agent.state import VerificationStatus
from app.llm.deepseek_client import DeepSeekError
from prototypes.llm_led_teacher.logic import (
    DisclosureLevel,
    DisclosureReason,
    LLMTeachingTurn,
    PrototypeEpisode,
    ResponseScope,
    StudentAssessment,
    TaskShape,
    TeachingBehaviorLabel,
    run_turn,
)


class _FullSolutionThenSingleMoveClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return LLMTeachingTurn(
                response=(
                    "先两边加5得到3x=15，再两边除以3得到x=5，"
                    "代回原式也成立。"
                ),
                action_label=TeachingBehaviorLabel.EXPLAIN,
                student_assessment=StudentAssessment.NOT_AN_ATTEMPT,
                expects_student_response=False,
                disclosure_level=DisclosureLevel.FINAL,
                disclosure_reason=DisclosureReason.STUDENT_REQUESTED,
                task_shape="multi_step_problem",
                response_scope="full_solution",
                mathematical_claims=["3x=15", "x=5"],
            )
        return LLMTeachingTurn(
            response="先看常数项-5。你觉得两边同时做什么能消去它？",
            action_label=TeachingBehaviorLabel.SCAFFOLD,
            student_assessment=StudentAssessment.NOT_AN_ATTEMPT,
            expects_student_response=True,
            disclosure_level=DisclosureLevel.NONE,
            disclosure_reason=DisclosureReason.NONE,
            task_shape="multi_step_problem",
            response_scope="single_move",
            )


class _OneShotClient:
    def __init__(self, output: LLMTeachingTurn) -> None:
        self.output = output
        self.calls = 0

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.calls += 1
        return self.output


class _MislabelledFullSolutionClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return LLMTeachingTurn(
                response="先两边加5得到3x=15，再除以3，所以x=5。",
                action_label=TeachingBehaviorLabel.EXPLAIN,
                student_assessment=StudentAssessment.NOT_AN_ATTEMPT,
                expects_student_response=False,
                disclosure_level=DisclosureLevel.NONE,
                disclosure_reason=DisclosureReason.NONE,
                task_shape=TaskShape.MULTI_STEP_PROBLEM,
                response_scope=ResponseScope.SINGLE_MOVE,
                mathematical_claims=["3x=15", "x=5"],
            )
        return LLMTeachingTurn(
            response="先看常数项-5。你觉得两边同时做什么能消去它？",
            action_label=TeachingBehaviorLabel.SCAFFOLD,
            student_assessment=StudentAssessment.NOT_AN_ATTEMPT,
            expects_student_response=True,
            disclosure_level=DisclosureLevel.NONE,
            disclosure_reason=DisclosureReason.NONE,
            task_shape=TaskShape.MULTI_STEP_PROBLEM,
            response_scope=ResponseScope.SINGLE_MOVE,
        )


class _InvalidSchemaThenDirectFactClient:
    def __init__(self) -> None:
        self.calls = 0
        self.system_prompts: list[str] = []
        self.user_payloads: list[str] = []

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.calls += 1
        self.system_prompts.append(system)
        self.user_payloads.append(user)
        if self.calls == 1:
            return schema.model_validate(
                {
                    "response": "三角形的内角和是180度。",
                    "task_shape": "direct_fact",
                    "disclosure_reason": "direct_fact",
                }
            )
        return LLMTeachingTurn(
            response="三角形的内角和是180度。",
            action_label=TeachingBehaviorLabel.EXPLAIN,
            student_assessment=StudentAssessment.NOT_AN_ATTEMPT,
            expects_student_response=False,
            disclosure_level=DisclosureLevel.FINAL,
            disclosure_reason=DisclosureReason.STUDENT_REQUESTED,
            task_shape=TaskShape.DIRECT_FACT,
            response_scope=ResponseScope.SINGLE_MOVE,
            mathematical_claims=["三角形的内角和是180度"],
        )


class _TransportErrorClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.calls += 1
        raise DeepSeekError("simulated transport failure")


def test_how_to_solve_repairs_a_first_turn_full_solution() -> None:
    client = _FullSolutionThenSingleMoveClient()
    episode = PrototypeEpisode()

    record = run_turn(
        episode,
        "老师，3x-5=10怎么解？",
        client=client,
    )

    assert record.student_verification.status is VerificationStatus.UNABLE_TO_VERIFY
    assert client.calls == 2
    assert record.repair_reasons == [
        "多步题的普通求助不是完整解答请求；本轮应只推进一个有意义的步骤。"
    ]
    assert "x=5" not in record.teacher_output.response


def test_explicit_full_solution_request_is_allowed() -> None:
    client = _OneShotClient(LLMTeachingTurn(
        response="完整步骤是先两边加5得到3x=15，再除以3得到x=5。",
        action_label=TeachingBehaviorLabel.MODEL,
        student_assessment=StudentAssessment.NOT_AN_ATTEMPT,
        expects_student_response=False,
        disclosure_level=DisclosureLevel.FINAL,
        disclosure_reason=DisclosureReason.STUDENT_REQUESTED,
        task_shape=TaskShape.MULTI_STEP_PROBLEM,
        response_scope=ResponseScope.FULL_SOLUTION,
        mathematical_claims=["3x=15", "x=5"],
    ))

    record = run_turn(
        PrototypeEpisode(),
        "请直接告诉我答案和完整步骤：3x-5=10",
        client=client,
    )

    assert client.calls == 1
    assert record.repair_reasons == []
    assert "x=5" in record.teacher_output.response


def test_direct_fact_question_can_answer_immediately() -> None:
    client = _OneShotClient(LLMTeachingTurn(
        response="三角形的内角和是180度。",
        action_label=TeachingBehaviorLabel.EXPLAIN,
        student_assessment=StudentAssessment.NOT_AN_ATTEMPT,
        expects_student_response=False,
        disclosure_level=DisclosureLevel.FINAL,
        disclosure_reason=DisclosureReason.STUDENT_REQUESTED,
        task_shape=TaskShape.DIRECT_FACT,
        response_scope=ResponseScope.FULL_SOLUTION,
        mathematical_claims=["三角形内角和是180度"],
    ))

    record = run_turn(
        PrototypeEpisode(),
        "老师，三角形的内角和是多少？",
        client=client,
    )

    assert client.calls == 1
    assert record.repair_reasons == []
    assert "180" in record.teacher_output.response


def test_verified_final_claim_cannot_hide_behind_single_move_metadata() -> None:
    client = _MislabelledFullSolutionClient()

    record = run_turn(
        PrototypeEpisode(),
        "老师，3x-5=10怎么解？",
        client=client,
    )

    assert client.calls == 2
    assert record.repair_reasons == [
        "多步题的普通求助不是完整解答请求；本轮应只推进一个有意义的步骤。"
    ]
    assert "x=5" not in record.teacher_output.response


def test_schema_validation_error_is_retried_once_with_field_feedback() -> None:
    client = _InvalidSchemaThenDirectFactClient()

    record = run_turn(
        PrototypeEpisode(),
        "老师，三角形的内角和是多少？",
        client=client,
    )

    assert client.calls == 2
    assert record.teacher_output.task_shape is TaskShape.DIRECT_FACT
    assert record.teacher_output.disclosure_reason is DisclosureReason.STUDENT_REQUESTED
    assert "180" in record.teacher_output.response
    assert "disclosure_reason" in client.user_payloads[1]
    assert "direct_fact" in client.user_payloads[1]


def test_transport_error_is_not_retried_as_a_schema_error() -> None:
    client = _TransportErrorClient()

    with pytest.raises(DeepSeekError, match="simulated transport failure"):
        run_turn(
            PrototypeEpisode(),
            "老师，三角形的内角和是多少？",
            client=client,
        )

    assert client.calls == 1
