"""Tests for the topic-agnostic workflow facade and legacy adapter boundary."""

from app.agent.state import GenericMathReasoning, VerificationStatus
from app.agent.teaching_actions import TeachingAction
from app.agent.workflow import run_teaching_turn, run_turn


def test_non_equation_turn_uses_generic_harness_without_a_legacy_concept() -> None:
    state = run_teaching_turn(
        "ratio-student",
        "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
    )

    assert state.action_plan is not None
    assert state.action_plan.action is TeachingAction.SCAFFOLD
    assert state.graph_context.available is False
    assert state.verification.status is VerificationStatus.NOT_APPLICABLE
    assert state.current_phase == state.action_plan.action.value
    assert state.response.count("？") == 1


class _FakeReasoner:
    def analyze(self, message, problem_context, analysis):
        return GenericMathReasoning(
            reasoning_steps=["先把3:2看成5个相等份。"],
            next_step_hint="先求总份数。",
            candidate_conclusion="男生18人，女生12人。",
            confidence=0.95,
        )


class _FakeResponseClient:
    def chat(self, system, user, temperature=0.2, max_tokens=None):
        return "先把3和2相加得到总份数。你能先写出总份数吗？"


class _RecordingEquationSystemReasoner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def analyze(self, message, problem_context, analysis):
        self.calls.append((message, problem_context))
        return GenericMathReasoning(
            problem_interpretation="需要从两个方程中消去一个未知数。",
            reasoning_steps=["比较同一未知数在两个方程中的系数关系。"],
            next_step_hint="y 的系数互为相反数，可以把两式对应相加。",
            candidate_conclusion="x=8/9，y=7/6",
            confidence=0.95,
        )


class _SequentialEquationSystemResponseClient:
    def __init__(self) -> None:
        self._responses = [
            "先观察两式中 y 的系数互为相反数，可以考虑把两式相加来消去 y。你愿意先判断相加后哪个未知数会被消去吗？",
            "把两式左右两边分别对应相加，2y 和 -2y 会抵消，先得到只含 x 的式子。你能先写出相加后的等式吗？",
            "选择消元方法时比较同一未知数在两式中的系数；相反数适合直接相加消去。你能说明这题为什么先消去 y 更省步骤吗？",
        ]
        self.calls: list[str] = []

    def chat(self, system, user, temperature=0.2, max_tokens=None):
        self.calls.append(user)
        return self._responses[len(self.calls) - 1]


def test_non_equation_turn_uses_injected_reasoning_without_leaking_answer() -> None:
    state = run_teaching_turn(
        "ratio-reasoning-student",
        "男生和女生人数的比是3:2，总数30人，我不知道先想什么。",
        reasoner=_FakeReasoner(),
        response_client=_FakeResponseClient(),
    )

    assert state.action_plan is not None
    assert state.action_plan.action is TeachingAction.SCAFFOLD
    assert state.answer_disclosure_allowed is False
    assert "18" not in state.response
    assert state.math_reasoning.reasoning_steps
    assert state.trace[:2] == ["generic_analysis", "generic_math_reasoning"]


def test_equation_turn_still_uses_the_legacy_compatibility_adapter() -> None:
    state = run_turn("equation-student", "我不会解 2x+3=7")

    assert state.detected_concept_id == "c_linear"
    assert state.target_equation == "2x + 3 = 7"


def test_legacy_return_type_projects_a_non_equation_turn_without_c_linear() -> None:
    state = run_turn("geometry-student", "三角形的内角和为什么是180度？")

    assert state.detected_concept_id is None
    assert state.target_equation == ""
    assert state.response.count("？") == 1


def test_existing_equation_episode_keeps_using_adapter_for_a_follow_up_request() -> None:
    conversation_id = "equation-follow-up"
    run_turn("follow-up-student", "老师 2x+5=11 怎么解？", conversation_id=conversation_id)

    state = run_turn("follow-up-student", "你出一道题好吗？", conversation_id=conversation_id)

    assert state.detected_concept_id == "c_linear"
    assert state.generated_practice_problem


def test_equation_system_follow_ups_reuse_the_first_problem_context() -> None:
    messages = [
        "老师，3x+2y=5，6x-2y=3怎么解？",
        "具体要怎么做呢？",
        "我不知道要怎么选择方法",
    ]
    reasoner = _RecordingEquationSystemReasoner()
    response_client = _SequentialEquationSystemResponseClient()

    states = [
        run_teaching_turn(
            "equation-system-student",
            message,
            conversation_id="equation-system-three-turns",
            reasoner=reasoner,
            response_client=response_client,
            analysis_allow_llm=False,
        )
        for message in messages
    ]

    assert [state.problem_context for state in states] == [messages[0]] * 3
    assert [call[1] for call in reasoner.calls] == [messages[0]] * 3
    assert [
        state.context_resolution.is_reply_to_previous_teacher for state in states
    ] == [False, True, True]
    assert all(
        state.episode_state
        and state.episode_state.problem_context == messages[0]
        for state in states
    )
    assert all("我先不假设题型" not in state.response for state in states[1:])
    assert len({state.response for state in states[1:]}) == 2
    assert all(state.response.count("？") == 1 for state in states)
    assert all(
        "x=8/9" not in state.response and "y=7/6" not in state.response
        for state in states
    )
