from __future__ import annotations

from prototypes.llm_led_teacher.orchestration_models import (
    AgentDecision,
    RequestClassification,
    TeachingContract,
    TeachingResponseMode,
    ToolResult,
)
from prototypes.llm_led_teacher.orchestrator import (
    MultisubjectAgentLoop,
    RequestClassifier,
)


class _Classifier:
    def __init__(self, classification: RequestClassification) -> None:
        self.classification = classification

    def classify(self, *args, **kwargs) -> RequestClassification:
        return self.classification


class _Decisions:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def decide(self, payload) -> AgentDecision:
        self.calls += 1
        return self.decisions.pop(0)


class _Registry:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, name, arguments, runtime) -> ToolResult:
        self.calls.append((name, arguments))
        if name == "search_local_knowledge":
            runtime.local_search_succeeded = True
        return ToolResult(name=name, ok=True, data={"text": "证据", "hits": []})


def _k9(subject: str, *, new: bool) -> RequestClassification:
    return RequestClassification(
        scope="k9",
        subject=subject,
        stage="初中",
        is_new_problem=new,
        answer_mode="guided",
        reason="test route",
    )


def _guided_contract() -> TeachingContract:
    return TeachingContract(
        response_mode="guided_solution",
        requested_target="引导学生推进当前学习问题",
        allowed_content=["必要概念", "分步提示"],
        forbidden_content=["未经请求的最终答案"],
        may_apply_to_active_problem=True,
        may_reveal_final_answer=False,
        reason="student did not request a complete solution",
    )


def _finish(contract: TeachingContract | None = None) -> AgentDecision:
    return AgentDecision(
        kind="finish",
        reason="evidence is enough",
        plan="guided",
        contract=contract or _guided_contract(),
    )


def test_loop_reports_real_phase_before_each_agent_component_runs() -> None:
    timeline: list[str] = []

    class RecordingClassifier(_Classifier):
        def classify(self, *args, **kwargs) -> RequestClassification:
            timeline.append("classifier_call")
            return super().classify(*args, **kwargs)

    class RecordingDecisions(_Decisions):
        def decide(self, payload) -> AgentDecision:
            timeline.append("decision_call")
            return super().decide(payload)

    def record_progress(event: str | object) -> None:
        if isinstance(event, str):
            timeline.append(event)
        else:
            timeline.append(f"event:{event.event_type}")

    loop = MultisubjectAgentLoop(
        classifier=RecordingClassifier(_k9("数学", new=False)),
        action_provider=RecordingDecisions([_finish()]),
        registry=_Registry(),
    )

    loop.run(
        "3x+2=11",
        "给我一点提示",
        "数学",
        "初中",
        progress_callback=record_progress,
    )

    assert timeline[:5] == [
        "classification",
        "classifier_call",
        "event:classification",
        "planning",
        "decision_call",
    ]


def test_followup_inherits_math_subject_when_classifier_marks_same_problem() -> None:
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_Decisions([_finish()]),
        registry=_Registry(),
    )

    result = loop.run(
        problem_context="解方程 2x+3=7",
        student_message="为什么要减去3？",
        active_subject="数学",
        active_stage="初中",
    )

    assert result.classification.subject.value == "数学"
    assert result.classification.is_new_problem is False


def test_explicit_physics_problem_switches_subject() -> None:
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("物理", new=True)),
        action_provider=_Decisions([_finish()]),
        registry=_Registry(),
    )

    result = loop.run(
        problem_context="解方程 2x+3=7",
        student_message="物体为什么会做匀速直线运动？",
        active_subject="数学",
        active_stage="初中",
    )

    assert result.classification.subject.value == "物理"
    assert result.classification.is_new_problem is True


def test_off_topic_finishes_without_knowledge_tools() -> None:
    classification = RequestClassification(
        scope="off_topic",
        subject=None,
        stage="未知",
        is_new_problem=True,
        answer_mode="brief_redirect",
        reason="company product question",
    )
    registry = _Registry()
    provider = _Decisions([_finish()])
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(classification),
        action_provider=provider,
        registry=registry,
    )

    result = loop.run(
        problem_context="",
        student_message="网易企业邮箱是什么",
        active_subject="",
        active_stage="",
    )

    assert registry.calls == []
    assert provider.calls == 0
    assert result.termination_reason == "off_topic_redirect"
    assert result.routing_guidance["answer_mode"] == "brief_redirect"
    assert result.teaching_contract.response_mode is TeachingResponseMode.BRIEF_REDIRECT
    assert result.teaching_contract.may_apply_to_active_problem is False
    assert result.teaching_contract.may_reveal_final_answer is False


def test_formula_question_finishes_with_concept_only_contract() -> None:
    proposed = TeachingContract(
        response_mode="concept_explanation",
        requested_target="解释求根公式",
        allowed_content=["通用公式", "符号含义"],
        forbidden_content=["代入当前题", "计算当前题的根"],
        may_apply_to_active_problem=False,
        may_reveal_final_answer=False,
        reason="concept request",
    )
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_Decisions([_finish(proposed)]),
        registry=_Registry(),
    )

    result = loop.run(
        "3x^2+2x-11=0",
        "求根公式是什么？",
        "数学",
        "初中",
    )

    assert result.teaching_contract == proposed
    assert result.trace[-1].event_type == "teaching_contract"
    assert result.trace[-1].details == proposed.model_dump(mode="json")


def test_finish_normalizes_untrusted_full_solution_contract() -> None:
    proposed = TeachingContract(
        response_mode="full_solution",
        requested_target="解答当前题",
        allowed_content=["完整步骤", "最终答案"],
        may_apply_to_active_problem=True,
        may_reveal_final_answer=True,
        reason="model selected a full solution",
    )
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_Decisions([_finish(proposed)]),
        registry=_Registry(),
    )

    result = loop.run("3x+2=11", "这道题怎么解？", "数学", "初中")

    assert result.teaching_contract.response_mode is TeachingResponseMode.ONE_STEP
    assert result.teaching_contract.may_reveal_final_answer is False
    assert result.trace[-1].details == result.teaching_contract.model_dump(mode="json")


def test_provider_error_returns_safe_contract() -> None:
    class _BrokenProvider:
        def decide(self, payload):
            raise ValueError("malformed decision")

    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_BrokenProvider(),
        registry=_Registry(),
    )

    result = loop.run("3x+2=11", "这道题怎么解？", "数学", "初中")

    assert result.termination_reason == "provider_error"
    assert result.teaching_contract.response_mode is TeachingResponseMode.ONE_STEP
    assert result.teaching_contract.may_reveal_final_answer is False


def test_provider_error_concept_question_returns_non_applying_contract() -> None:
    class _BrokenProvider:
        def decide(self, payload):
            raise ValueError("malformed decision")

    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_BrokenProvider(),
        registry=_Registry(),
    )

    result = loop.run("3x^2+2x-11=0", "求根公式是什么？", "数学", "初中")

    assert result.teaching_contract.response_mode is TeachingResponseMode.CONCEPT_EXPLANATION
    assert result.teaching_contract.may_apply_to_active_problem is False
    assert result.teaching_contract.may_reveal_final_answer is False


def test_provider_error_non_solve_fallback_does_not_apply_to_active_problem() -> None:
    class _BrokenProvider:
        def decide(self, payload):
            raise ValueError("malformed decision")

    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_BrokenProvider(),
        registry=_Registry(),
    )

    result = loop.run("3x+2=11", "给我一点提示", "数学", "初中")

    assert result.teaching_contract.response_mode is TeachingResponseMode.GUIDED_SOLUTION
    assert result.teaching_contract.may_apply_to_active_problem is False
    assert result.teaching_contract.may_reveal_final_answer is False


def test_provider_error_keeps_explicit_full_request_with_why_cue() -> None:
    class _BrokenProvider:
        def decide(self, payload):
            raise ValueError("malformed decision")

    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_BrokenProvider(),
        registry=_Registry(),
    )

    result = loop.run(
        "3x+2=11",
        "请完整解答这道题，并解释为什么这样做",
        "数学",
        "初中",
    )

    assert result.teaching_contract.response_mode is TeachingResponseMode.FULL_SOLUTION
    assert result.teaching_contract.may_apply_to_active_problem is True
    assert result.teaching_contract.may_reveal_final_answer is True


def test_loop_stops_after_six_decisions() -> None:
    repeated = AgentDecision(
        kind="call_tool",
        tool_name="search_local_knowledge",
        arguments={"query": "力", "subject": "物理", "stage": "初中"},
        reason="try local evidence",
    )
    provider = _Decisions([repeated.model_copy(deep=True) for _ in range(8)])
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("物理", new=True)),
        action_provider=provider,
        registry=_Registry(),
        max_iterations=6,
    )

    result = loop.run(
        problem_context="",
        student_message="力是什么",
        active_subject="",
        active_stage="",
    )

    decision_events = [event for event in result.trace if event.event_type == "decision"]
    assert len(decision_events) == 6
    assert provider.calls == 6
    assert result.termination_reason == "iteration_budget_exhausted"
    assert result.teaching_contract.response_mode is TeachingResponseMode.CONCEPT_EXPLANATION
    assert result.teaching_contract.may_apply_to_active_problem is False
    assert result.teaching_contract.may_reveal_final_answer is False


def test_classifier_repairs_once_then_falls_back_off_topic() -> None:
    class BrokenClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat_json(self, **kwargs):
            self.calls += 1
            raise ValueError("malformed JSON")

    client = BrokenClient()
    result = RequestClassifier(client).classify("hello", "", "", "")

    assert client.calls == 2
    assert result.scope.value == "off_topic"
    assert result.answer_mode.value == "brief_redirect"
