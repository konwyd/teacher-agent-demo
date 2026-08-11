"""Tests for optional GraphRAG context providers and no-graph evaluation."""

from app.evaluation.evaluator import SingleTurnCase
from app.evaluation.mode_runner import EvalMode, run_mode
from app.retrieval.context_provider import (
    EmptyGraphContextProvider,
    GraphContextRequest,
)


def test_empty_graph_context_provider_returns_no_data_without_a_reader() -> None:
    provider = EmptyGraphContextProvider()

    context = provider.get_context(
        GraphContextRequest(problem_context="2x+3=7", concept_id="c_linear")
    )
    evidence = provider.retrieve_pedagogical_evidence("c_linear", mastery_score=0.5)

    assert context.available is False
    assert context.provider == "disabled"
    assert provider.get_concept("c_linear") is None
    assert provider.get_prerequisites("c_linear") == []
    assert provider.get_related_exercises("c_linear") == []
    assert provider.get_misconceptions("c_linear") == []
    assert evidence.retrieval_backend == "disabled"
    assert evidence.evidence == []


def test_agent_no_graph_never_instantiates_local_graph_readers(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("agent_no_graph must not read local graph data")

    monkeypatch.setattr(
        "app.retrieval.context_provider.get_retriever", fail_if_called
    )
    monkeypatch.setattr(
        "app.retrieval.context_provider.get_local_pedagogical_graph_retriever",
        fail_if_called,
    )

    result = run_mode(
        EvalMode.AGENT_NO_GRAPH,
        golden_cases=[],
        single_turn_cases=[
            SingleTurnCase(
                case_id="no_graph_reader",
                message="老师 3x+4=7 怎么解？",
            )
        ],
    )[0]

    assert result.passed is True
    assert result.retrieval_backend == "disabled"
    assert result.diagnosis


def test_agent_no_graph_preserves_existing_offline_eval_without_graph_readers(
    monkeypatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("agent_no_graph must not read local graph data")

    monkeypatch.setattr(
        "app.retrieval.context_provider.get_retriever", fail_if_called
    )
    monkeypatch.setattr(
        "app.retrieval.context_provider.get_local_pedagogical_graph_retriever",
        fail_if_called,
    )

    results = run_mode(EvalMode.AGENT_NO_GRAPH)

    assert all(item.passed for item in results)
    assert all(item.retrieval_backend == "disabled" for item in results)
