"""Run dynamic Hybrid Student Simulator Evaluation.

Default execution is fully offline and produces 18 scenarios x 3 Teacher
modes = 54 persisted traces followed by 54 deterministic scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from app.evaluation.hybrid_runner import (
    TeacherSessionFactory,
    run_hybrid_matrix,
    write_hybrid_traces,
)
from app.evaluation.hybrid_scorer import score_trace_file
from app.evaluation.mode_runner import EvalMode
from app.evaluation.student_simulator import (
    MasteryLevel,
    SUPPORTED_CONCEPTS,
    StudentScenario,
    load_student_scenarios,
)

load_dotenv()


def select_scenarios(
    scenarios: list[StudentScenario],
    *,
    concept: str = "all",
    mastery: str = "all",
) -> list[StudentScenario]:
    selected = [
        scenario
        for scenario in scenarios
        if (concept == "all" or scenario.concept_id == concept)
        and (mastery == "all" or scenario.mastery_level.value == mastery)
    ]
    if not selected:
        raise ValueError("scenario filters selected no cases")
    return selected


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_hybrid_pipeline(
    *,
    scenarios: list[StudentScenario],
    modes: list[EvalMode],
    output_dir: Path,
    run_id: str,
    max_turns: int = 8,
    use_real_teacher_llm: bool = False,
    use_real_student_wording_llm: bool = False,
    teacher_session_factory: TeacherSessionFactory | None = None,
) -> dict[str, Any]:
    if not scenarios:
        raise ValueError("no scenarios selected")
    if not modes:
        raise ValueError("no Teacher modes selected")
    if not 1 <= max_turns <= 8:
        raise ValueError("max_turns must be between 1 and 8")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    trace_path = output_dir / "hybrid_traces.jsonl"
    score_path = output_dir / "hybrid_scores.jsonl"
    summary_path = output_dir / "hybrid_summary.json"
    expected_count = len(scenarios) * len(modes)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "modes": [mode.value for mode in modes],
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "max_turns": max_turns,
        "use_real_teacher_llm": use_real_teacher_llm,
        "use_real_student_wording_llm": use_real_student_wording_llm,
        "expected_count": expected_count,
        "trace_count": 0,
        "score_count": 0,
    }
    _write_manifest(manifest_path, manifest)

    try:
        traces = run_hybrid_matrix(
            scenarios,
            modes,
            run_id=run_id,
            max_turns=max_turns,
            use_real_teacher_llm=use_real_teacher_llm,
            use_real_student_wording_llm=use_real_student_wording_llm,
            teacher_session_factory=teacher_session_factory,
        )
        unique_keys = {(trace.scenario_id, trace.mode) for trace in traces}
        if len(traces) != expected_count or len(unique_keys) != expected_count:
            raise RuntimeError(
                f"trace count mismatch: expected {expected_count}, "
                f"got {len(traces)} ({len(unique_keys)} unique)"
            )

        # This context closes before score_trace_file re-opens the trace path.
        write_hybrid_traces(traces, trace_path)
        manifest["trace_count"] = len(traces)
        manifest["traces_persisted_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)

        scores, summary = score_trace_file(
            trace_path,
            score_path,
            summary_path,
        )
        if len(scores) != expected_count:
            raise RuntimeError(
                f"score count mismatch: expected {expected_count}, got {len(scores)}"
            )
        manifest["score_count"] = len(scores)
        manifest["complete"] = True
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        return {
            **manifest,
            "summary": summary,
            "output_dir": str(output_dir),
        }
    except Exception as exc:
        manifest["complete"] = False
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(manifest_path, manifest)
        raise


def _bounded_turns(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 8:
        raise argparse.ArgumentTypeError("max turns must be between 1 and 8")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic Hybrid Student Simulator Evaluation."
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=("all", *(mode.value for mode in EvalMode)),
    )
    parser.add_argument(
        "--concept",
        default="all",
        choices=("all", *sorted(SUPPORTED_CONCEPTS)),
    )
    parser.add_argument(
        "--mastery",
        default="all",
        choices=("all", *(level.value for level in MasteryLevel)),
    )
    parser.add_argument("--max-turns", type=_bounded_turns, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _configure_environment(
    output_dir: Path,
    *,
    use_real_teacher_llm: bool,
    use_real_student_wording_llm: bool,
) -> None:
    runtime_dir = output_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEACHER_LOG_DIR"] = str(runtime_dir)
    os.environ["TEACHER_PROFILE_PATH"] = str(runtime_dir / "student_profiles.json")
    os.environ["TEACHER_EPISODE_PATH"] = str(runtime_dir / "episode_states.json")
    os.environ["TEACHER_LEARNING_EVENTS_PATH"] = str(
        runtime_dir / "learning_events.jsonl"
    )
    os.environ["TEACHER_THREE_LAYER_PROFILE_PATH"] = str(
        runtime_dir / "student_profiles_v2.json"
    )

    if use_real_teacher_llm:
        os.environ.pop("TEACHER_FORCE_MOCK", None)
    else:
        os.environ["TEACHER_FORCE_MOCK"] = "1"
    if not use_real_teacher_llm and not use_real_student_wording_llm:
        os.environ.pop("DEEPSEEK_API_KEY", None)


def main() -> int:
    args = _parse_args()
    run_id = args.run_id or _new_run_id()
    log_root = Path(os.getenv("TEACHER_LOG_DIR", "logs"))
    output_dir = args.output_dir or (log_root / "hybrid_eval" / run_id)
    use_real_teacher_llm = (
        os.getenv("USE_REAL_LLM_FOR_EVAL", "").lower() == "true"
    )
    use_real_student_wording_llm = (
        os.getenv("USE_REAL_STUDENT_SIMULATOR_LLM", "").lower() == "true"
    )
    _configure_environment(
        output_dir,
        use_real_teacher_llm=use_real_teacher_llm,
        use_real_student_wording_llm=use_real_student_wording_llm,
    )

    scenarios = select_scenarios(
        load_student_scenarios(),
        concept=args.concept,
        mastery=args.mastery,
    )
    modes = list(EvalMode) if args.mode == "all" else [EvalMode(args.mode)]
    result = run_hybrid_pipeline(
        scenarios=scenarios,
        modes=modes,
        output_dir=output_dir,
        run_id=run_id,
        max_turns=args.max_turns,
        use_real_teacher_llm=use_real_teacher_llm,
        use_real_student_wording_llm=use_real_student_wording_llm,
    )

    print(f"Hybrid run：{run_id}")
    print(
        f"Teacher LLM：{'真实 DeepSeek' if use_real_teacher_llm else '离线 mock'}"
    )
    print(
        "Student wording："
        f"{'真实 DeepSeek' if use_real_student_wording_llm else '确定性模板'}"
    )
    print(
        f"场景×模式：{result['expected_count']}  "
        f"trace：{result['trace_count']}  score：{result['score_count']}"
    )
    print(f"结果目录：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

