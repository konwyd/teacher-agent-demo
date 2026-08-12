from __future__ import annotations

from prototypes.llm_led_teacher.orchestration_models import (
    RequestClassification,
    ToolResult,
)
from prototypes.llm_led_teacher.orchestrator import (
    MultisubjectAgentLoop,
    RequestClassifier,
    RetrievalDecisionProvider,
)
from prototypes.llm_led_teacher.retrieval_policy import RetrievalDecision


class _Classifier:
    def __init__(self, classification: RequestClassification) -> None:
        self.classification = classification

    def classify(self, *args, **kwargs) -> RequestClassification:
        return self.classification


class _Routes:
    def __init__(self, decision: RetrievalDecision) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, payload) -> RetrievalDecision:
        self.calls += 1
        return self.decision


class _Registry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name, arguments, runtime) -> ToolResult:
        self.calls.append((name, arguments))
        if name == "search_local_knowledge":
            runtime.local_search_succeeded = True
        return ToolResult(
            name=name,
            ok=True,
            data={"text": f"{name} evidence", "hits": []},
            elapsed_seconds=0.25,
        )


def _k9(subject: str, *, new: bool) -> RequestClassification:
    return RequestClassification(
        scope="k9",
        subject=subject,
        stage="初中",
        is_new_problem=new,
        answer_mode="guided",
        reason="test route",
    )


def _no_retrieval() -> RetrievalDecision:
    return RetrievalDecision(mode="none", reason="无需外部证据", confidence=1.0)


def test_followup_inherits_math_subject_when_classifier_marks_same_problem() -> None:
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        retrieval_provider=_Routes(_no_retrieval()),
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
        retrieval_provider=_Routes(_no_retrieval()),
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
    provider = _Routes(_no_retrieval())
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(classification),
        retrieval_provider=provider,
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


def test_creative_guidance_skips_all_retrieval_tools() -> None:
    registry = _Registry()
    routes = _Routes(
        RetrievalDecision(
            mode="local_then_web",
            reason="模型提出搜索",
            confidence=0.8,
        )
    )
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("语文", new=True)),
        retrieval_provider=routes,
        registry=registry,
    )

    result = loop.run("", "把这句话润色成作文开头", "", "")

    assert result.retrieval_decision.mode.value == "none"
    assert routes.calls == 1
    assert registry.calls == []


def test_local_then_web_executes_each_tool_at_most_once() -> None:
    registry = _Registry()
    routes = _Routes(
        RetrievalDecision(
            mode="local_then_web",
            reason="需要核对最新政策",
            confidence=0.9,
            needs_current_information=True,
        )
    )
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("道法", new=True)),
        retrieval_provider=routes,
        registry=registry,
    )

    result = loop.run("", "查询今天最新教材政策", "", "")

    assert [name for name, _ in registry.calls] == [
        "search_local_knowledge",
        "search_web",
    ]
    assert [
        event.tool_name for event in result.trace if event.event_type == "tool_result"
    ] == ["search_local_knowledge", "search_web"]


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


def test_retrieval_provider_requests_typed_decision_schema() -> None:
    class Client:
        def __init__(self) -> None:
            self.schema = None

        def chat_json(self, **kwargs):
            self.schema = kwargs["schema"]
            return {
                "mode": "local",
                "reason": "需要教材证据",
                "confidence": 0.9,
            }

    client = Client()

    result = RetrievalDecisionProvider(client).decide({"student_message": "什么是力"})

    assert client.schema is RetrievalDecision
    assert result == RetrievalDecision(
        mode="local",
        reason="需要教材证据",
        confidence=0.9,
    )
