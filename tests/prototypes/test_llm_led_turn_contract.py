import pytest

from app.agent.state import VerificationStatus
from app.llm.deepseek_client import DeepSeekError
from prototypes.llm_led_teacher.logic import PrototypeEpisode, run_turn


class _TextClient:
    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.system_prompts: list[str] = []
        self.user_payloads: list[str] = []

    def chat(self, system, user, temperature=0.0, max_tokens=None):
        self.system_prompts.append(system)
        self.user_payloads.append(user)
        result = self.responses[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


def test_unsupported_opposite_number_response_is_returned_unverified() -> None:
    client = _TextClient("3的相反数是-3。")

    record = run_turn(
        PrototypeEpisode(),
        "老师，3的相反数是什么？",
        client=client,
    )

    assert client.calls == 1
    assert record.teacher_output.response == "3的相反数是-3。"
    assert record.evidence_level == "unverified"
    assert record.extracted_equations == []
    assert record.claim_audits == []
    assert record.repair_reasons == []


def test_direct_fact_question_can_answer_immediately_without_a_verifier() -> None:
    client = _TextClient("三角形的内角和是180度。")

    record = run_turn(
        PrototypeEpisode(),
        "老师，三角形的内角和是多少？",
        client=client,
    )

    assert client.calls == 1
    assert record.teacher_output.response == "三角形的内角和是180度。"
    assert record.evidence_level == "unverified"
    assert record.repair_reasons == []


def test_unverified_style_warning_does_not_block_the_teacher_response() -> None:
    response = "3的相反数是-3。你理解了吗？需要我再举个例子吗？"
    client = _TextClient(response)

    record = run_turn(
        PrototypeEpisode(),
        "老师，3的相反数是什么？",
        client=client,
    )

    assert client.calls == 1
    assert record.teacher_output.response == response
    assert record.repair_reasons == []
    assert record.remaining_warnings == ["教师回复包含多个问题。"]


def test_how_to_solve_repairs_a_first_turn_full_solution_from_text() -> None:
    client = _TextClient(
        "先两边加5得到3x=15，再两边除以3得到x=5。",
        "先看常数项-5。你觉得两边同时做什么能消去它？",
    )

    record = run_turn(
        PrototypeEpisode(),
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
    client = _TextClient("先两边加5得到3x=15，再除以3得到x=5。")

    record = run_turn(
        PrototypeEpisode(),
        "请直接告诉我答案和完整步骤：3x-5=10",
        client=client,
    )

    assert client.calls == 1
    assert record.repair_reasons == []
    assert "x=5" in record.teacher_output.response
    assert "x = 5" in record.extracted_equations


def test_verified_incorrect_teacher_equation_is_repaired_from_text() -> None:
    client = _TextClient(
        "两边加5以后得到3x=14。",
        "两边同时加5，右边应当计算10+5。你能先算出这个结果吗？",
    )

    record = run_turn(
        PrototypeEpisode(),
        "老师，3x-5=10怎么解？",
        client=client,
    )

    assert client.calls == 2
    assert record.repair_reasons == ["教师数学断言与 verifier 冲突：3x = 14"]
    assert "3x=14" not in record.teacher_output.response


def test_explicit_no_answer_request_repairs_a_detected_final_form() -> None:
    client = _TextClient(
        "两边加5得到3x=15，再除以3得到x=5。",
        "先观察-5这一项，想一想两边应同时做什么运算？",
    )

    record = run_turn(
        PrototypeEpisode(),
        "先别给答案，3x-5=10怎么解？",
        client=client,
    )

    assert client.calls == 2
    assert "学生明确不要答案，但回复出现了可验证的最终形式。" in record.repair_reasons
    assert "x=5" not in record.teacher_output.response


def test_transport_error_is_not_retried() -> None:
    client = _TextClient(DeepSeekError("simulated transport failure"))

    with pytest.raises(DeepSeekError, match="simulated transport failure"):
        run_turn(
            PrototypeEpisode(),
            "老师，三角形的内角和是多少？",
            client=client,
        )

    assert client.calls == 1
