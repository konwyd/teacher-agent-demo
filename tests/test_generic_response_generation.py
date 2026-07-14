"""Safety tests for guarded generic teaching responses."""

from app.agent.response_generation import generate_response
from app.agent.state import (
    GenericMathReasoning,
    TeachingActionKind,
    TeachingActionPlan,
    TeachingState,
)


class _ResponseClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def chat(self, system, user, temperature=0.2, max_tokens=None):
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.response


def _hint_state() -> TeachingState:
    return TeachingState(
        student_id="ratio-student",
        message="给我一个提示。",
        problem_context="男生和女生人数的比是3:2，总数30人。",
        action_plan=TeachingActionPlan(action=TeachingActionKind.GIVE_HINT),
        math_reasoning=GenericMathReasoning(
            reasoning_steps=["把3:2看成5个相等份。"],
            next_step_hint="先算总份数。",
            candidate_conclusion="男生18人，女生12人。",
            confidence=0.95,
        ),
    )


def test_response_omits_candidate_from_unapproved_context() -> None:
    client = _ResponseClient("先把3和2相加，得到总份数。你能先写出总份数吗？")

    response = generate_response(_hint_state(), client=client)

    assert "男生18人" not in client.calls[-1]["user"]
    assert "男生18人" not in response
    assert client.calls[-1]["max_tokens"] == 350
    assert response.count("？") == 1


def test_response_can_pass_candidate_only_after_explicit_authorization() -> None:
    client = _ResponseClient("总份数是5，所以每份是6。男生18人，女生12人。你能复述这个分配过程吗？")
    state = _hint_state()
    state.action_plan = TeachingActionPlan(
        action=TeachingActionKind.EXPLAIN_STEP,
        answer_disclosure_allowed=True,
    )
    state.answer_disclosure_allowed = True

    response = generate_response(state, client=client)

    assert "男生18人，女生12人。" in client.calls[-1]["user"]
    assert "男生18人，女生12人。" in response


def test_response_falls_back_when_model_asks_multiple_questions() -> None:
    client = _ResponseClient("先算总份数好吗？每份是多少呢？")

    response = generate_response(_hint_state(), client=client)

    assert response == "先把题目中的已知条件、所求内容和它们的关系分别写出来。你准备先整理哪一条信息？"
