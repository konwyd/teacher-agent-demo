"""Local Pedagogical Graph traversal and evidence-scoring tests."""

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent import math_checker as mc
from app.agent.workflow import run_turn
from app.api.main import app
from app.retrieval.graph_retriever import LocalPedagogicalGraphRetriever


def _evidence_of_type(result, node_type: str):
    return [item for item in result.evidence if item.node_type == node_type]


def test_linear_equation_basic_returns_ranked_hints_and_practice() -> None:
    result = LocalPedagogicalGraphRetriever().retrieve_evidence(
        "linear_equation_basic",
        mastery_score=0.5,
    )

    assert result.concept_id == "linear_equation"
    assert _evidence_of_type(result, "Hint")
    assert _evidence_of_type(result, "PracticeProblem")
    assert [item.score for item in result.evidence] == sorted(
        (item.score for item in result.evidence),
        reverse=True,
    )
    assert all(item.relation_path and item.reason for item in result.evidence)


def test_misconception_returns_remediating_hint() -> None:
    retriever = LocalPedagogicalGraphRetriever()

    hints = retriever.get_remediating_hints("move_terms_forget_sign")

    assert hints
    assert all(item.node_type == "Hint" for item in hints)
    assert all("REMEDIATES" in item.relation_path for item in hints)
    assert hints[0].score >= 0.9


def test_low_mastery_returns_easy_hint_and_practice() -> None:
    result = LocalPedagogicalGraphRetriever().retrieve_evidence(
        "linear_equation",
        mastery_score=0.2,
    )
    hints = _evidence_of_type(result, "Hint")
    practices = _evidence_of_type(result, "PracticeProblem")

    assert hints[0].content["level"] == 1
    assert practices[0].content["difficulty"] == 1
    assert "mastery threshold match" in hints[0].reason


def test_high_mastery_returns_harder_variant_practice() -> None:
    result = LocalPedagogicalGraphRetriever().retrieve_evidence(
        "linear_equation",
        mastery_score=0.8,
    )
    practices = _evidence_of_type(result, "PracticeProblem")

    assert practices[0].node_id == "ex_linear_two_side_01"
    assert practices[0].content["difficulty"] == 2
    assert "harder_variant" in practices[0].reason


def test_request_type_overrides_mastery_difficulty() -> None:
    retriever = LocalPedagogicalGraphRetriever()
    easier = retriever.retrieve_evidence(
        "linear_equation",
        mastery_score=0.8,
        request_type="generate_easier_exercise",
    )
    harder = retriever.retrieve_evidence(
        "linear_equation",
        mastery_score=0.2,
        request_type="generate_harder_exercise",
    )

    assert _evidence_of_type(easier, "PracticeProblem")[0].content["difficulty"] == 1
    assert _evidence_of_type(harder, "PracticeProblem")[0].content["difficulty"] == 2


def test_prerequisite_chain_uses_requires_edges() -> None:
    chain = LocalPedagogicalGraphRetriever().get_prerequisite_chain(
        "fractional_coefficient_equation",
        depth=2,
    )

    ids = [item.node_id for item in chain]
    assert "linear_equation" in ids
    assert "equation_properties" in ids
    assert all("REQUIRES" in item.relation_path for item in chain)


def test_turn_exposes_retrieval_evidence_without_polluting_original_problem() -> None:
    state = run_turn(
        "graph_evidence_student",
        "老师 3x+6=10 怎么解？",
        conversation_id="graph_evidence_conversation",
    )

    assert state.retrieval_backend == "local_json_graph"
    assert state.retrieval_evidence
    assert state.retrieved_concepts
    assert state.retrieved_hints
    assert state.top_evidence_reasons
    assert not mc.uses_new_example(state.response, state.target_equation)


def test_turn_api_contains_retrieval_evidence_fields() -> None:
    response = TestClient(app).post(
        "/turn",
        json={
            "student_id": "graph_evidence_api",
            "conversation_id": "graph_evidence_api_conversation",
            "message": "我不会解 2x+3=7",
        },
    )

    assert response.status_code == 200
    data = response.json()
    for field in (
        "retrieval_backend",
        "retrieval_evidence",
        "retrieved_concepts",
        "retrieved_misconceptions",
        "retrieved_hints",
        "retrieved_practice_problem_ids",
        "top_evidence_reasons",
    ):
        assert field in data


def test_turn_log_contains_retrieval_evidence() -> None:
    run_turn(
        "graph_evidence_log",
        "老师 5x-4=11 怎么解？",
        conversation_id="graph_evidence_log_conversation",
    )
    log_path = Path(os.environ["TEACHER_LOG_DIR"]) / "turn_logs.jsonl"
    entry = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])

    assert entry["retrieval_backend"] == "local_json_graph"
    assert entry["retrieval_evidence"]
    assert entry["retrieved_concepts"]
    assert entry["retrieved_misconceptions"]
    assert entry["retrieved_hints"]
    assert entry["retrieved_practice_problem_ids"]
    assert entry["top_evidence_reasons"]
