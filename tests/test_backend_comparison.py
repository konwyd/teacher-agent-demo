"""Offline tests for the local/API comparison runner."""

from __future__ import annotations

import json

import pytest


def test_load_cases_rejects_duplicate_ids(tmp_path) -> None:
    from scripts.run_backend_comparison import load_cases

    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {"case_id": "same", "prompt": "问题一"},
                {"case_id": "same", "prompt": "问题二"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_cases(path)


def test_run_matrix_uses_every_backend_repeat_and_persists_immediately() -> None:
    from scripts.run_backend_comparison import BenchmarkCase, run_matrix

    cases = [BenchmarkCase(case_id="q1", prompt="固定问题")]
    calls: list[tuple[str, str, int]] = []
    persisted: list[dict[str, object]] = []

    def execute(case: BenchmarkCase, backend: str, repeat: int) -> dict[str, object]:
        calls.append((case.case_id, backend, repeat))
        return {
            "status": "ok",
            "backend_total_seconds": float(repeat),
            "response": f"{backend}-{repeat}",
        }

    records = run_matrix(
        cases,
        backends=("local", "api"),
        repeats=3,
        execute=execute,
        persist=persisted.append,
    )

    assert calls == [
        ("q1", "local", 1),
        ("q1", "local", 2),
        ("q1", "local", 3),
        ("q1", "api", 1),
        ("q1", "api", 2),
        ("q1", "api", 3),
    ]
    assert persisted == records
    assert [record["response"] for record in records] == [
        "local-1",
        "local-2",
        "local-3",
        "api-1",
        "api-2",
        "api-3",
    ]


def test_run_matrix_records_one_failure_and_continues() -> None:
    from scripts.run_backend_comparison import BenchmarkCase, run_matrix

    def execute(case: BenchmarkCase, backend: str, repeat: int) -> dict[str, object]:
        if backend == "local":
            raise RuntimeError("simulated failure")
        return {
            "status": "ok",
            "backend_total_seconds": 2.0,
            "response": "API回答",
        }

    records = run_matrix(
        [BenchmarkCase(case_id="q1", prompt="固定问题")],
        backends=("local", "api"),
        repeats=1,
        execute=execute,
    )

    assert records[0]["status"] == "error"
    assert records[0]["error_type"] == "RuntimeError"
    assert "simulated failure" in records[0]["error"]
    assert records[1]["status"] == "ok"


def test_summarize_results_calculates_mean_and_median() -> None:
    from scripts.run_backend_comparison import summarize_results

    records = [
        {
            "case_id": "q1",
            "backend": "local",
            "status": "ok",
            "backend_total_seconds": value,
        }
        for value in (9.0, 3.0, 6.0)
    ] + [
        {
            "case_id": "q1",
            "backend": "api",
            "status": "error",
            "backend_total_seconds": 0.0,
        }
    ]

    rows = summarize_results(records)

    local = next(
        row for row in rows if row["case_id"] == "q1" and row["backend"] == "local"
    )
    api = next(
        row for row in rows if row["case_id"] == "q1" and row["backend"] == "api"
    )
    assert local["success_count"] == 3
    assert local["error_count"] == 0
    assert local["mean_seconds"] == 6.0
    assert local["median_seconds"] == 6.0
    assert local["min_seconds"] == 3.0
    assert local["max_seconds"] == 9.0
    assert api["success_count"] == 0
    assert api["error_count"] == 1
    assert api["mean_seconds"] == ""


def test_markdown_report_contains_timings_and_all_answers(tmp_path) -> None:
    from scripts.run_backend_comparison import (
        BenchmarkCase,
        render_markdown_report,
        summarize_results,
    )

    records = [
        {
            "case_id": "q1",
            "prompt": "固定问题",
            "backend": "local",
            "repeat": 1,
            "status": "ok",
            "backend_total_seconds": 4.2,
            "response": "本地回答",
            "stage_timings_seconds": {"generating": 3.0},
            "repair_performed": False,
            "llm_call_count": 3,
        },
        {
            "case_id": "q1",
            "prompt": "固定问题",
            "backend": "api",
            "repeat": 1,
            "status": "ok",
            "backend_total_seconds": 2.1,
            "response": "API回答",
            "stage_timings_seconds": {"generating": 1.0},
            "repair_performed": True,
            "llm_call_count": 4,
        },
    ]

    report = render_markdown_report(
        cases=[BenchmarkCase(case_id="q1", prompt="固定问题")],
        records=records,
        summary=summarize_results(records),
        run_id="test-run",
    )

    assert "test-run" in report
    assert "本地回答" in report
    assert "API回答" in report
    assert "4.200" in report
    assert "2.100" in report
