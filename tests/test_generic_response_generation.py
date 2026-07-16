"""Safety tests for brief-only generic teaching responses."""

import json

import pytest

from app.agent.response_generation import ResponseGuard, generate_response
from app.agent.state import TeachingState
from app.agent.teaching_actions import (
    DisclosureLevel,
    SafetyConstraints,
    TeachingAction,
    TeachingActionPlan,
)
from app.agent.teaching_brief import TeachingBrief, TeachingBriefComposer


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


def _hint_brief(*, disclose_final: bool = False) -> TeachingBrief:
    return TeachingBriefComposer().compose(
        problem_context="男生和女生人数的比是3:2，总数30人。",
        current_student_request="给我一个提示。",
        verified_facts=[],
        student_evidence=["学生表示需要提示。"],
        action_plan=TeachingActionPlan(
            action=(TeachingAction.EXPLAIN if disclose_final else TeachingAction.SCAFFOLD),
            scaffold_level=(0 if disclose_final else 1),
        ),
        constraints=SafetyConstraints(
            allowed_actions=[
                TeachingAction.EXPLAIN if disclose_final else TeachingAction.SCAFFOLD
            ],
            disclosure_ceiling=(
                DisclosureLevel.FINAL
                if disclose_final
                else DisclosureLevel.NEXT_STEP
            ),
        ),
        next_step_candidates=["先算总份数。", "不要采用第二个候选。"],
        candidate_conclusion="男生18人，女生12人。",
    )


def test_response_omits_candidate_from_unapproved_context() -> None:
    client = _ResponseClient("先把3和2相加，得到总份数。你能先写出总份数吗？")

    response = generate_response(_hint_brief(), client=client)

    payload = json.loads(client.calls[-1]["user"])
    assert set(payload) == set(TeachingBrief.model_fields)
    assert "男生18人" not in payload["allowed_content"]
    assert "男生18人" not in response
    assert client.calls[-1]["max_tokens"] == 350
    assert response.count("？") == 1


def test_response_can_pass_candidate_only_after_explicit_authorization() -> None:
    client = _ResponseClient("总份数是5，所以每份是6。男生18人，女生12人。你能复述这个分配过程吗？")

    response = generate_response(_hint_brief(disclose_final=True), client=client)

    payload = json.loads(client.calls[-1]["user"])
    assert "男生18人，女生12人。" in payload["allowed_content"]
    assert "男生18人，女生12人。" in response


def test_response_falls_back_when_model_asks_multiple_questions() -> None:
    client = _ResponseClient("先算总份数好吗？每份是多少呢？")

    response = generate_response(_hint_brief(), client=client)

    assert response == "先把题目中的已知条件、所求内容和它们的关系分别写出来。你准备先整理哪一条信息？"


def test_response_without_an_approved_next_step_skips_the_client() -> None:
    client = _ResponseClient("不应调用。你能继续吗？")
    brief = TeachingBriefComposer().compose(
        problem_context="男生和女生人数的比是3:2，总数30人。",
        current_student_request="我不知道。",
        verified_facts=[],
        student_evidence=[],
        action_plan=TeachingActionPlan(
            action=TeachingAction.SCAFFOLD,
            scaffold_level=1,
        ),
        constraints=SafetyConstraints(
            allowed_actions=[TeachingAction.SCAFFOLD],
            disclosure_ceiling=DisclosureLevel.NEXT_STEP,
        ),
        next_step_candidates=[],
        candidate_conclusion="男生18人，女生12人。",
    )

    response = generate_response(brief, client=client)

    assert client.calls == []
    assert response == "先把题目中的已知条件、所求内容和它们的关系分别写出来。你准备先整理哪一条信息？"


def test_response_guard_rejects_normalized_forbidden_disclosure() -> None:
    brief = _hint_brief()

    assert not ResponseGuard.validate(
        "可以得到男生 18 人，女生 12 人。你能继续吗？",
        brief,
    )


def test_response_guard_uses_injected_grounding_for_non_example_action() -> None:
    brief = _hint_brief()

    assert not ResponseGuard.validate(
        "我们先看下一步。你能继续吗？",
        brief,
        grounding_guard=lambda response, active_brief: False,
    )


def test_response_generator_rejects_teaching_state_input() -> None:
    with pytest.raises(TypeError, match="TeachingBrief"):
        generate_response(
            TeachingState(
                student_id="boundary-student",
                message="给我提示",
                action_plan=TeachingActionPlan(
                    action=TeachingAction.SCAFFOLD,
                    scaffold_level=1,
                ),
            )
        )
