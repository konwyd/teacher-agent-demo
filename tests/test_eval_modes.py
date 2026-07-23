"""Evaluation mode comparison tests."""

import json

from app.evaluation.evaluator import GoldenCase, GoldenTurn, SingleTurnCase
from app.evaluation.mode_runner import (
    EvalMode,
    get_mode_configuration,
    run_mode,
    write_mode_results,
)


def _single_case(case_id: str = "mode_case") -> SingleTurnCase:
    return SingleTurnCase(
        case_id=case_id,
        message="老师 3x+4=7 怎么解？",
        forbidden_patterns=["2x+3=7"],
    )


def test_mode_configurations_disable_expected_components() -> None:
    deepseek = get_mode_configuration(EvalMode.DEEPSEEK_ONLY)
    no_graph = get_mode_configuration(EvalMode.AGENT_NO_GRAPH)
    with_graph = get_mode_configuration(EvalMode.AGENT_WITH_GRAPH)

    assert deepseek.uses_workflow is False
    assert deepseek.uses_math_checker is False
    assert deepseek.uses_memory is False
    assert deepseek.uses_graph_retrieval is False
    assert no_graph.uses_workflow is True
    assert no_graph.uses_math_checker is True
    assert no_graph.uses_memory is True
    assert no_graph.uses_graph_retrieval is False
    assert with_graph.uses_graph_retrieval is True


def test_deepseek_only_does_not_call_workflow(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("workflow must not run in deepseek_only mode")

    monkeypatch.setattr("app.evaluation.mode_runner.run_turn", fail_if_called)
    result = run_mode(
        EvalMode.DEEPSEEK_ONLY,
        golden_cases=[],
        single_turn_cases=[_single_case("deepseek_only_case")],
        use_real_llm=False,
    )[0]

    assert result.mode == EvalMode.DEEPSEEK_ONLY
    assert result.final_response
    assert result.diagnosis == {}
    assert result.teaching_action == ""
    assert result.retrieval_backend == "disabled"
    assert result.teacher_quality_metrics.total_checks > 0


def test_deepseek_only_is_scored_on_observable_teaching_behavior() -> None:
    case = GoldenCase(
        case_id="deepseek_step_quality",
        original_problem="3x+4=7",
        turns=[
            GoldenTurn(role="student", content="老师 3x+4=7 怎么解？"),
            GoldenTurn(role="student", content="3x=3"),
        ],
        expected_final_answer="x=1",
        required_behaviors=["must_keep_original_problem"],
    )

    result = run_mode(
        EvalMode.DEEPSEEK_ONLY,
        golden_cases=[case],
        single_turn_cases=[],
        use_real_llm=False,
    )[0]

    assert result.passed is False
    assert (
        result.teacher_quality_metrics.check_results[
            "turn2.recognizes_correct_step"
        ]
        is False
    )


def test_agent_no_graph_disables_retrieval(monkeypatch) -> None:
    result = run_mode(
        EvalMode.AGENT_NO_GRAPH,
        golden_cases=[],
        single_turn_cases=[_single_case("agent_no_graph_case")],
        use_real_llm=False,
    )[0]

    assert result.mode == EvalMode.AGENT_NO_GRAPH
    assert result.diagnosis
    assert result.teaching_action
    assert result.retrieval_backend == "disabled"
    assert result.grounding_fields["original_problem"]


def test_agent_with_graph_records_graph_backend() -> None:
    result = run_mode(
        EvalMode.AGENT_WITH_GRAPH,
        golden_cases=[],
        single_turn_cases=[_single_case("agent_with_graph_case")],
        use_real_llm=False,
    )[0]

    assert result.mode == EvalMode.AGENT_WITH_GRAPH
    assert result.retrieval_backend == "local_json_graph"
    assert result.grounding_fields["original_problem"]


def test_mode_results_use_unified_schema_and_jsonl(tmp_path) -> None:
    results = []
    for mode in EvalMode:
        results.extend(
            run_mode(
                mode,
                golden_cases=[],
                single_turn_cases=[_single_case(f"schema_{mode.value}")],
                use_real_llm=False,
            )
        )

    output = tmp_path / "eval_mode_results.jsonl"
    write_mode_results(results, output)
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 3
    required = {
        "mode",
        "case_id",
        "passed",
        "teacher_quality_metrics",
        "final_response",
        "diagnosis",
        "teaching_action",
        "retrieval_backend",
        "grounding_fields",
    }
    assert all(required <= set(row) for row in rows)
