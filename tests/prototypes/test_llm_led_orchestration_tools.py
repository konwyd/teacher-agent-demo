from __future__ import annotations

import pytest

from prototypes.llm_led_teacher import rag_hook
from prototypes.llm_led_teacher.orchestration_models import SchoolStage, Subject
from prototypes.llm_led_teacher.orchestration_models import RequestClassification
from prototypes.llm_led_teacher.orchestration_tools import (
    OrchestrationRuntime,
    OrchestrationToolRegistry,
)


def test_local_adapter_uses_requested_subject_and_never_online(monkeypatch) -> None:
    calls = []

    def fake_local(problem, subject, stage, topk, sources):
        calls.append((problem, subject, stage, topk, sources))
        return {
            "hits": [],
            "query": problem,
            "filters": {"subject": subject, "stage": stage},
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(rag_hook, "_retrieve_local_hits", fake_local)
    monkeypatch.setattr(
        rag_hook,
        "_tavily_search",
        lambda *args, **kwargs: pytest.fail("local adapter called online search"),
    )

    result = rag_hook.retrieve_local_context(
        "力为什么能改变运动状态",
        subject=Subject.PHYSICS,
        stage=SchoolStage.MIDDLE,
    )

    assert calls == [
        ("力为什么能改变运动状态", "物理", "初中", 5, ("textbook", "exercise"))
    ]
    assert result.source == "local_empty"
    assert result.online_triggered is False
    assert result.diagnostics["filters"] == {"subject": "物理", "stage": "初中"}


@pytest.mark.parametrize(
    ("subject", "exercise_subject"),
    [
        (Subject.MATH, "Math"),
        (Subject.CHINESE, "Chinese"),
        (Subject.ENGLISH, "English"),
        (Subject.PHYSICS, "Physics"),
        (Subject.CHEMISTRY, "Chemistry"),
        (Subject.BIOLOGY, "Biology"),
        (Subject.HISTORY, "History"),
        (Subject.GEOGRAPHY, "Geography"),
        (Subject.POLITICS, "Politics"),
    ],
)
def test_nine_subject_metadata_mapping(subject: Subject, exercise_subject: str) -> None:
    filters = rag_hook.subject_metadata_filters(subject, SchoolStage.MIDDLE)

    assert filters["textbook"] == {"subject": subject.value, "stage": "初中"}
    assert filters["exercise"] == {"subject": exercise_subject}


def test_local_adapter_preserves_transparent_scores(monkeypatch) -> None:
    raw_hit = {
        "text": "力可以改变物体的运动状态。",
        "metadata": {"subject": "物理", "stage": "初中"},
        "source": "textbook",
        "vec_sim": 0.71,
        "bm25_score": 3.2,
        "rrf_normalized": 0.83,
        "sim": 0.746,
    }
    monkeypatch.setattr(
        rag_hook,
        "_retrieve_local_hits",
        lambda *args, **kwargs: {
            "hits": [raw_hit],
            "query": "物理 力",
            "filters": {"subject": "物理", "stage": "初中"},
            "elapsed_seconds": 0.02,
        },
    )

    result = rag_hook.retrieve_local_context(
        "力的作用效果",
        subject=Subject.PHYSICS,
        stage=SchoolStage.MIDDLE,
    )

    assert result.source == "local"
    assert result.hits[0] == {
        **raw_hit,
        "document": raw_hit["text"],
        "similarity": raw_hit["sim"],
        "reference": result.refs[0],
    }
    assert result.diagnostics["hit_count"] == 1


def _classification(scope: str = "k9") -> RequestClassification:
    if scope == "off_topic":
        return RequestClassification(
            scope="off_topic",
            subject=None,
            stage="未知",
            is_new_problem=True,
            answer_mode="brief_redirect",
            reason="not a school question",
        )
    return RequestClassification(
        scope="k9",
        subject="物理",
        stage="初中",
        is_new_problem=True,
        answer_mode="guided",
        reason="physics question",
    )


def _runtime(scope: str = "k9") -> OrchestrationRuntime:
    return OrchestrationRuntime(
        classification=_classification(scope),
        problem_context="",
        student_message="力为什么能改变运动状态",
        skill_catalog={"concept_explanation": "解释概念并联系例子"},
    )


def test_web_requires_k9_and_successful_local_search() -> None:
    registry = OrchestrationToolRegistry(
        local_handler=lambda **kwargs: {"hits": []},
        web_handler=lambda **kwargs: {"hits": []},
    )
    runtime = _runtime()

    result = registry.execute(
        "search_web",
        {
            "query": "最新教材变化",
            "subject": "物理",
            "stage": "初中",
            "evidence_gap": "本地无新课标",
        },
        runtime,
    )

    assert not result.ok
    assert result.error == "web search requires successful local search"


def test_off_topic_cannot_use_local_or_web() -> None:
    calls = []
    registry = OrchestrationToolRegistry(
        local_handler=lambda **kwargs: calls.append("local"),
        web_handler=lambda **kwargs: calls.append("web"),
    )
    runtime = _runtime("off_topic")

    assert not registry.execute("search_local_knowledge", {}, runtime).ok
    assert not registry.execute("search_web", {}, runtime).ok
    assert calls == []


def test_one_web_call_maximum_after_successful_local() -> None:
    registry = OrchestrationToolRegistry(
        local_handler=lambda **kwargs: {"hits": [{"text": "local"}]},
        web_handler=lambda **kwargs: {"hits": [{"text": "web"}]},
    )
    runtime = _runtime()
    local_args = {
        "query": "力",
        "subject": "物理",
        "stage": "初中",
        "top_k": 5,
        "sources": ["textbook"],
    }
    web_args = {
        "query": "力",
        "subject": "物理",
        "stage": "初中",
        "evidence_gap": "本地内容不足",
    }

    assert registry.execute("search_local_knowledge", local_args, runtime).ok
    assert registry.execute("search_web", web_args, runtime).ok
    repeated = registry.execute("search_web", web_args, runtime)

    assert not repeated.ok
    assert repeated.error == "web search budget exhausted"


def test_duplicate_local_query_and_filters_are_rejected() -> None:
    registry = OrchestrationToolRegistry(local_handler=lambda **kwargs: {"hits": []})
    runtime = _runtime()
    args = {
        "query": "力",
        "subject": "物理",
        "stage": "初中",
        "top_k": 5,
        "sources": ["textbook", "exercise"],
    }

    assert registry.execute("search_local_knowledge", args, runtime).ok
    repeated = registry.execute("search_local_knowledge", args, runtime)

    assert not repeated.ok
    assert repeated.error == "duplicate local search"


def test_distinct_second_local_query_exhausts_turn_budget() -> None:
    registry = OrchestrationToolRegistry(local_handler=lambda **kwargs: {"hits": []})
    runtime = _runtime()

    first = registry.execute(
        "search_local_knowledge",
        {"query": "力", "subject": "物理", "stage": "初中"},
        runtime,
    )
    second = registry.execute(
        "search_local_knowledge",
        {"query": "运动状态", "subject": "物理", "stage": "初中"},
        runtime,
    )

    assert first.ok
    assert not second.ok
    assert second.error == "local search budget exhausted"


def test_tool_timeout_is_a_recoverable_result() -> None:
    def timeout_runner(handler, kwargs, timeout_seconds):
        raise TimeoutError

    registry = OrchestrationToolRegistry(
        local_handler=lambda **kwargs: {"hits": []},
        timeout_runner=timeout_runner,
    )

    result = registry.execute(
        "search_local_knowledge",
        {"query": "力", "subject": "物理", "stage": "初中"},
        _runtime(),
    )

    assert not result.ok
    assert result.error == "tool timed out"


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_timeout"),
    [
        (
            "search_local_knowledge",
            {"query": "力", "subject": "物理", "stage": "初中"},
            3,
        ),
        (
            "search_web",
            {
                "query": "最新教材变化",
                "subject": "物理",
                "stage": "初中",
                "evidence_gap": "需要核对最新变化",
            },
            12,
        ),
    ],
)
def test_tool_uses_source_specific_timeout(
    tool_name, arguments, expected_timeout
) -> None:
    observed_timeouts = []

    def timeout_runner(handler, kwargs, timeout_seconds):
        observed_timeouts.append(timeout_seconds)
        return {"hits": []}

    registry = OrchestrationToolRegistry(
        local_handler=lambda **kwargs: {"hits": []},
        web_handler=lambda **kwargs: {"hits": []},
        timeout_runner=timeout_runner,
        local_timeout_seconds=3,
        web_timeout_seconds=12,
    )
    runtime = _runtime()
    runtime.local_search_succeeded = True

    result = registry.execute(tool_name, arguments, runtime)

    assert result.ok
    assert observed_timeouts == [expected_timeout]


@pytest.mark.parametrize("sources", [[], ["教材", "题库"], ["unknown"]])
def test_invalid_or_empty_local_sources_do_not_count_as_success(sources) -> None:
    calls = []
    registry = OrchestrationToolRegistry(
        local_handler=lambda **kwargs: calls.append(kwargs) or {"hits": []}
    )
    runtime = _runtime()

    result = registry.execute(
        "search_local_knowledge",
        {
            "query": "力",
            "subject": "物理",
            "stage": "初中",
            "sources": sources,
        },
        runtime,
    )

    assert not result.ok
    assert result.error.startswith("invalid arguments:")
    assert not runtime.local_search_succeeded
    assert calls == []


def test_local_search_subject_must_match_classification() -> None:
    registry = OrchestrationToolRegistry(local_handler=lambda **kwargs: {"hits": []})
    runtime = _runtime()

    result = registry.execute(
        "search_local_knowledge",
        {"query": "力", "subject": "数学", "stage": "初中"},
        runtime,
    )

    assert not result.ok
    assert result.error == "local search subject does not match classification"
    assert not runtime.local_search_succeeded
