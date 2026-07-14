"""Offline coverage and live-safety tests for generic math evaluation."""

import importlib
import os

from app.evaluation.generic_math_evaluator import run_generic_math_eval
from scripts import run_eval
from scripts import run_generic_math_smoke as live_smoke


def test_offline_generic_eval_covers_five_domains_without_answer_leakage() -> None:
    results = run_generic_math_eval()

    assert {item.domain for item in results} >= {
        "ratio",
        "algebra",
        "geometry",
        "statistics",
        "probability",
    }
    assert len(results) == 12
    assert all(item.passed for item in results)


def test_live_smoke_refuses_without_explicit_flag(capsys) -> None:
    assert live_smoke.main([]) == 2

    assert "--live" in capsys.readouterr().out


def test_live_smoke_rejects_more_than_the_budgeted_case_limit(capsys) -> None:
    assert live_smoke.main(["--live", "--max-cases", "13"]) == 2

    assert "1 到 12" in capsys.readouterr().out


def test_default_eval_reports_generic_math_smoke_cases(capsys) -> None:
    assert run_eval.main() == 0

    assert "(generic_math_smoke) ratio_hint" in capsys.readouterr().out


def test_eval_module_and_run_restore_runtime_environment(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-for-test")
    monkeypatch.delenv("TEACHER_FORCE_MOCK", raising=False)

    module = importlib.reload(run_eval)

    assert os.getenv("DEEPSEEK_API_KEY") == "fake-key-for-test"
    assert os.getenv("TEACHER_FORCE_MOCK") is None
    assert module.main() == 0
    capsys.readouterr()
    assert os.getenv("DEEPSEEK_API_KEY") == "fake-key-for-test"
    assert os.getenv("TEACHER_FORCE_MOCK") is None
