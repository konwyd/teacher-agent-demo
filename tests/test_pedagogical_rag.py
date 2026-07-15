"""Pedagogical Graph RAG retriever and workflow integration tests."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent import math_checker as mc
from app.agent.legacy_action_compat import LegacyTeachingAction as TeachingAction
from app.agent.workflow import run_turn
from app.api.main import app
from app.memory.student_state import get_store
from app.retrieval.pedagogical_retriever import JsonPedagogicalRetriever, get_pedagogical_retriever

GRAPH_PATH = Path(__file__).resolve().parents[1] / "data" / "sample" / "pedagogical_graph.json"
CORE_CONCEPT_IDS = {
    "linear_equation",
    "equation_properties",
    "move_terms",
    "divide_by_coefficient",
    "remove_parentheses",
    "fractional_coefficient_equation",
}
NODE_COLLECTIONS = {
    "concepts",
    "skills",
    "misconceptions",
    "diagnostic_questions",
    "hints",
    "worked_examples",
    "practice_problems",
    "teaching_strategies",
    "mastery_thresholds",
}
EDGE_TYPES = {
    "REQUIRES",
    "HAS_SKILL",
    "HAS_MISCONCEPTION",
    "DIAGNOSE_WITH",
    "HINT_FOR",
    "EXAMPLE_FOR",
    "PRACTICE_FOR",
    "REMEDIATES",
    "EASIER_THAN",
    "HARDER_THAN",
    "NEXT_CONCEPT",
    "USE_STRATEGY_WHEN",
}


def _set_mastery(student_id: str, concept_id: str, target: str) -> None:
    store = get_store()
    if target == "low":
        for _ in range(4):
            store.update_mastery_after_turn(student_id, concept_id, "incorrect_step")
    elif target == "high":
        for _ in range(2):
            store.update_mastery_after_turn(student_id, concept_id, "correct_final_answer")


def test_pedagogical_graph_loads() -> None:
    retriever = JsonPedagogicalRetriever()
    context = retriever.get_concept_context("linear_equation")
    assert context is not None
    assert context.concept_id == "linear_equation"
    assert context.name == "一元一次方程"
    assert context.diagnostic_questions
    assert context.hint_levels


def test_pedagogical_graph_has_normalized_nodes_and_edges() -> None:
    raw = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))

    assert raw["schema_version"] == "1.0.0"
    assert NODE_COLLECTIONS <= set(raw["nodes"])
    assert EDGE_TYPES <= {edge["type"] for edge in raw["edges"]}


def test_each_core_concept_has_pedagogical_decision_references() -> None:
    raw = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    concepts = {item["concept_id"]: item for item in raw["nodes"]["concepts"]}

    assert CORE_CONCEPT_IDS <= set(concepts)
    for concept_id in CORE_CONCEPT_IDS:
        concept = concepts[concept_id]
        for field in (
            "prerequisites",
            "skills",
            "misconceptions",
            "diagnostic_questions",
            "leveled_hints",
            "practice_problems",
            "teaching_strategies",
            "mastery_thresholds",
        ):
            assert field in concept, f"{concept_id} missing {field}"
        assert concept["skills"], f"{concept_id} has no skills"
        assert concept["misconceptions"], f"{concept_id} has no misconceptions"
        assert concept["leveled_hints"], f"{concept_id} has no hints"
        assert concept["practice_problems"], f"{concept_id} has no practice problems"


def test_old_retriever_interface_reads_normalized_graph(tmp_path: Path) -> None:
    raw = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    raw.pop("concepts", None)
    raw.pop("teaching_strategies", None)
    graph_path = tmp_path / "pedagogical_graph.json"
    graph_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    retriever = JsonPedagogicalRetriever(data_dir=tmp_path)

    for concept_id in CORE_CONCEPT_IDS:
        context = retriever.get_concept_context(concept_id)
        assert context is not None
        assert retriever.get_misconceptions(concept_id)
        assert retriever.get_diagnostic_questions(concept_id)
        assert retriever.get_hint_levels(concept_id)
        assert retriever.get_recommended_exercises(concept_id)
        assert retriever.get_mastery_based_strategy(concept_id, 0.2)


def test_get_concept_context() -> None:
    retriever = get_pedagogical_retriever()
    by_old_id = retriever.get_concept_context("c_linear")
    by_name = retriever.get_concept_context("一元一次方程")
    assert by_old_id is not None
    assert by_name is not None
    assert by_old_id.concept_id == by_name.concept_id == "linear_equation"
    assert retriever.get_prerequisites("c_linear")
    assert retriever.get_misconceptions("c_linear")
    assert retriever.get_diagnostic_questions("c_linear")
    assert retriever.get_hint_levels("c_linear")
    assert retriever.get_recommended_exercises("c_linear")
    assert "诊断" in retriever.get_mastery_based_strategy("c_linear", 0.2)


def test_low_mastery_uses_diagnostic_question() -> None:
    sid = "ped_low"
    _set_mastery(sid, "c_linear", "low")
    state = run_turn(sid, "老师 3x+6=10 怎么解？")
    assert state.action is TeachingAction.ASK_DIAGNOSTIC_QUESTION
    assert state.pedagogical_context_used is True
    assert state.diagnostic_question_used
    assert state.selected_hint_level == 1
    assert "两边" in state.response


def test_medium_mastery_uses_hint() -> None:
    state = run_turn("ped_medium", "老师 3x+6=10 怎么解？")
    assert state.action is TeachingAction.GIVE_HINT_LEVEL_1
    assert state.pedagogical_context_used is True
    assert state.selected_hint_level == 1
    assert state.diagnostic_question_used == ""


def test_high_mastery_recommends_exercise() -> None:
    sid = "ped_high"
    _set_mastery(sid, "c_linear", "high")
    state = run_turn(sid, "再来一题一元一次方程")
    assert state.action is TeachingAction.GENERATE_PRACTICE_QUESTION
    assert state.pedagogical_context_used is True
    assert state.recommended_exercise
    assert state.generated_practice_problem


def test_math_checker_priority_over_pedagogical_rag() -> None:
    sid = "ped_checker_priority"
    _set_mastery(sid, "c_linear", "low")
    run_turn(sid, "老师 3x+4=7 怎么解？")
    state = run_turn(sid, "先都减四，然后同时除以3")
    assert state.action is TeachingAction.GRADE_AND_FEEDBACK
    assert state.student_step_validity in ("correct_step", "correct_final_answer")
    assert "正确" in state.response or "思路" in state.response


def test_no_retrieved_example_pollution() -> None:
    state = run_turn("ped_no_pollution", "老师 3x+6=10 怎么解？")
    assert state.pedagogical_context_used is True
    assert not mc.uses_new_example(state.response, state.target_equation), state.response
    assert "2x + 5 = 11" not in state.response


def test_turn_response_contains_pedagogical_fields() -> None:
    client = TestClient(app)
    resp = client.post(
        "/turn",
        json={"student_id": "ped_api", "message": "我不会解 2x+3=7"},
    )
    assert resp.status_code == 200
    data = resp.json()
    for field in (
        "pedagogical_context_used",
        "selected_hint_level",
        "matched_misconception",
        "diagnostic_question_used",
        "recommended_exercise",
    ):
        assert field in data
    assert data["pedagogical_context_used"] is True
