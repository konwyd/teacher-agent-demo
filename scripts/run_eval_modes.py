"""Run comparable Teacher Agent evaluation modes.

Examples:
    python scripts/run_eval_modes.py --mode deepseek_only
    python scripts/run_eval_modes.py --mode agent_no_graph
    python scripts/run_eval_modes.py --mode agent_with_graph

Set USE_REAL_LLM_FOR_EVAL=true to use the configured DeepSeek API. The default
is fully offline and never calls the real API.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DeepSeek-only and Teacher Agent ablation modes."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("deepseek_only", "agent_no_graph", "agent_with_graph"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path (default: <TEACHER_LOG_DIR>/eval_mode_results.jsonl)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    use_real = os.getenv("USE_REAL_LLM_FOR_EVAL", "").lower() == "true"
    root_log_dir = Path(os.getenv("TEACHER_LOG_DIR", "logs"))
    runtime_dir = root_log_dir / "eval_modes" / args.mode
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if not use_real:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ["TEACHER_FORCE_MOCK"] = "1"

    # Keep ablation memory and turn logs isolated from real student data.
    os.environ["TEACHER_LOG_DIR"] = str(runtime_dir)
    os.environ["TEACHER_PROFILE_PATH"] = str(runtime_dir / "student_profiles.json")
    os.environ["TEACHER_EPISODE_PATH"] = str(runtime_dir / "episode_states.json")
    os.environ["TEACHER_LEARNING_EVENTS_PATH"] = str(
        runtime_dir / "learning_events.jsonl"
    )
    os.environ["TEACHER_THREE_LAYER_PROFILE_PATH"] = str(
        runtime_dir / "student_profiles_v2.json"
    )

    from app.evaluation.mode_runner import (  # noqa: PLC0415
        EvalMode,
        get_mode_configuration,
        run_mode,
        write_mode_results,
    )

    mode = EvalMode(args.mode)
    config = get_mode_configuration(mode)
    results = run_mode(mode, use_real_llm=use_real)
    output_path = args.output or (root_log_dir / "eval_mode_results.jsonl")
    write_mode_results(results, output_path)

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    print(f"模式：{mode.value}")
    print(f"LLM：{'真实 DeepSeek' if use_real else '离线 mock'}")
    print(
        "组件："
        f"workflow={config.uses_workflow}, "
        f"math_checker={config.uses_math_checker}, "
        f"memory={config.uses_memory}, "
        f"graph={config.uses_graph_retrieval}"
    )
    print(f"总 case 数：{len(results)}  通过：{passed}  失败：{failed}")
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        print(
            f"[{mark}] ({result.case_type}) {result.case_id} "
            f"quality={result.teacher_quality_metrics.pass_rate:.2f}"
        )
        for failure in result.failures:
            print(f"       - {failure}")
    print(f"统一结果已保存：{output_path}")
    print(f"该模式运行时数据：{runtime_dir}")

    # This is a comparison runner, so case failures are data rather than a
    # process failure. scripts/run_eval.py remains the strict regression gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
