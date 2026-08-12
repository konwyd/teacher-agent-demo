from __future__ import annotations

from prototypes.llm_led_teacher.orchestration_models import (
    AgentDecision,
    RequestClassification,
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


def _finish() -> AgentDecision:
    return AgentDecision(kind="finish", reason="evidence is enough", plan="guided")


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
