"""Command-line demo for the isolated tool-loop prototype."""

from __future__ import annotations

import argparse
import json

from app.llm.deepseek_client import DeepSeekClient
from prototypes.tool_loop_teacher.loop import AgentLoop, DeepSeekActionProvider


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, help="稳定的原题上下文")
    parser.add_argument("--message", required=True, help="学生当前消息")
    parser.add_argument("--student-id", default="prototype_student")
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument(
        "--decision-max-tokens",
        type=int,
        default=1200,
        help="每次结构化工具决策的最大输出 token（含本地模型思考）",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="打印工具选择和安全收口轨迹",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    client = DeepSeekClient()
    loop = AgentLoop(
        DeepSeekActionProvider(client, max_tokens=args.decision_max_tokens),
        response_client=client,
        max_iterations=args.max_iterations,
    )
    result = loop.run_turn(
        student_id=args.student_id,
        problem_context=args.problem,
        student_message=args.message,
    )
    print(result.response)
    if args.show_trace:
        print(
            json.dumps(
                {
                    "action_plan": result.action_plan.model_dump(mode="json"),
                    "verification": result.verification.model_dump(mode="json"),
                    "loaded_skills": result.loaded_skills,
                    "termination_reason": result.termination_reason,
                    "fallback_used": result.fallback_used,
                    "trace": [event.model_dump(mode="json") for event in result.trace],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
