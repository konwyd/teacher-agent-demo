"""Static visual report generation tests."""

from __future__ import annotations

import json

from scripts.render_hybrid_report import build_report, load_run_summary


def _write_run(
    directory,
    run_id: str,
    mode: str,
    success: float,
    *,
    use_real_teacher_llm: bool = True,
) -> None:
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "complete": True,
                "trace_count": 18,
                "score_count": 18,
                "use_real_teacher_llm": use_real_teacher_llm,
                "use_real_student_wording_llm": False,
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "conversation_count": 18,
        "task_success_rate": success,
        "teacher_quality_pass_rate": 0.8,
        "solution_progress": 0.9,
        "teacher_turns": 4.0,
        "termination_distribution": {
            "solved_and_verified": {"count": 16, "rate": success},
            "student_stuck": {"count": 1, "rate": 0.05},
            "student_gave_up": {"count": 1, "rate": 0.05},
            "max_turns": {"count": 0, "rate": 0.0},
        },
    }
    (directory / "hybrid_summary.json").write_text(
        json.dumps(
            {
                "total_conversations": 18,
                "groups": {
                    "by_mode": {mode: metrics},
                    "by_mode_mastery": {
                        f"{mode}|low": metrics,
                        f"{mode}|medium": metrics,
                        f"{mode}|high": metrics,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_report_loads_runs_and_renders_comparison_charts(tmp_path) -> None:
    no_graph = tmp_path / "no_graph"
    with_graph = tmp_path / "with_graph"
    mock = tmp_path / "mock"
    _write_run(no_graph, "real_no_graph", "agent_no_graph", 0.8)
    _write_run(with_graph, "real_with_graph", "agent_with_graph", 0.9)
    _write_run(
        mock,
        "mock_direct",
        "deepseek_only",
        0.2,
        use_real_teacher_llm=False,
    )

    runs = [
        load_run_summary(no_graph),
        load_run_summary(with_graph),
        load_run_summary(mock),
    ]
    html = build_report(runs, title="真实 LLM 对比")

    assert "真实 LLM 对比" in html
    assert "agent_no_graph" in html
    assert "agent_with_graph" in html
    assert "deepseek_only" in html
    assert "真实 DeepSeek" in html
    assert "离线 mock" in html
    assert "完成并验算" in html
    assert "终止分布" in html
    assert "bar-fill" in html
