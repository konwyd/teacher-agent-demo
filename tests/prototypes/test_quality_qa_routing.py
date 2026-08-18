from __future__ import annotations

import pytest

from prototypes.llm_led_teacher import qa_cache
from prototypes.llm_led_teacher.fast_routing import choose_fast_route
from prototypes.llm_led_teacher.orchestration_models import (
    AgentActionKind,
    AgentDecision,
    AnswerMode,
    RequestClassification,
    RequestScope,
    SchoolStage,
    Subject,
    TeachingContract,
    TeachingResponseMode,
)
from prototypes.llm_led_teacher.orchestration_tools import (
    OrchestrationRuntime,
    OrchestrationToolRegistry,
)
from prototypes.llm_led_teacher.orchestrator import MultisubjectAgentLoop
from prototypes.llm_led_teacher.rag_hook import RagContext


def _classification(*, use_quality_qa: bool) -> RequestClassification:
    return RequestClassification(
        scope=RequestScope.K9,
        subject=Subject.MATH,
        stage=SchoolStage.MIDDLE,
        is_new_problem=True,
        answer_mode=AnswerMode.GUIDED,
        use_quality_qa=use_quality_qa,
        quality_qa_reason="test route",
        reason="test",
    )


def test_non_k9_classification_cannot_enable_quality_qa() -> None:
    with pytest.raises(ValueError):
        RequestClassification(
            scope=RequestScope.SOCIAL,
            subject=None,
            stage=SchoolStage.UNKNOWN,
            is_new_problem=False,
            answer_mode=AnswerMode.BRIEF_REDIRECT,
            use_quality_qa=True,
            reason="invalid",
        )


def test_fast_route_keeps_quality_qa_independent_from_textbook_route() -> None:
    experiential = choose_fast_route(
        "初中数学里移项总是写错怎么办？",
        "",
        "",
        "",
    )
    routine = choose_fast_route(
        "初中数学解方程 2x=4",
        "",
        "",
        "",
    )

    assert experiential is not None
    assert experiential.classification.use_quality_qa is True
    assert routine is not None
    assert routine.classification.use_quality_qa is False


def test_registry_rejects_quality_search_when_route_is_off() -> None:
    calls = 0

    def handler(**_kwargs):
        nonlocal calls
        calls += 1
        return RagContext(source="quality_qa")

    registry = OrchestrationToolRegistry(quality_qa_handler=handler)
    runtime = OrchestrationRuntime(
        classification=_classification(use_quality_qa=False),
        problem_context="",
        student_message="问题",
    )

    result = registry.execute(
        "search_quality_qa",
        {
            "query": "问题",
            "subject": "数学",
            "stage": "初中",
            "top_k": 3,
        },
        runtime,
    )

    assert result.ok is False
    assert "route is disabled" in result.error
    assert calls == 0


class _Classifier:
    def classify(self, *_args, **_kwargs):
        return _classification(use_quality_qa=True)


class _Finisher:
    def decide(self, _payload):
        return AgentDecision(
            kind=AgentActionKind.FINISH,
            reason="enough",
            plan="guided",
            contract=TeachingContract(
                response_mode=TeachingResponseMode.GUIDED_SOLUTION,
                requested_target="解释常见错误",
                allowed_content=["解释"],
                forbidden_content=[],
                may_apply_to_active_problem=True,
                may_reveal_final_answer=False,
                reason="test",
            ),
        )


def test_orchestrator_fetches_quality_qa_only_after_positive_route(
    monkeypatch,
) -> None:
    monkeypatch.setattr(qa_cache, "LOOKUP_ENABLED", True)
    calls = 0

    def handler(**_kwargs):
        nonlocal calls
        calls += 1
        return RagContext(
            text="历史问题：移项为什么容易错？\n历史回答：先说明等式两边同做运算。",
            refs=["[quality_qa:test]"],
            source="quality_qa",
            hits=[{"id": "test", "similarity": 0.9}],
            diagnostics={"quality_qa_used": True, "quality_qa_version": 3},
        )

    registry = OrchestrationToolRegistry(quality_qa_handler=handler)
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(),
        action_provider=_Finisher(),
        registry=registry,
        enable_fast_routing=False,
    )

    result = loop.run("", "移项为什么总写错？", "", "")

    assert calls == 1
    assert any(item.name == "search_quality_qa" for item in result.observations)
    assert "历史问题" in result.rag_context.text
    assert result.rag_context.diagnostics["quality_qa_version"] == 3
