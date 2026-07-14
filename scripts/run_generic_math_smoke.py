"""Run a bounded, manually reviewed DeepSeek smoke test for generic math."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.generic_math_reasoning import GenericMathReasoner
from app.agent.teaching_harness import run_generic_teaching_turn
from app.evaluation.generic_math_evaluator import load_generic_math_smoke_cases
from app.llm.deepseek_client import DeepSeekClient, DeepSeekError, is_configured


MAX_CASES = 12
BUDGET_CNY = 2


class _CountingReasoner:
    def __init__(self) -> None:
        self._reasoner = GenericMathReasoner(client=DeepSeekClient(max_retries=0))
        self.model_call_count = 0
        self.error = ""

    def analyze(self, message, problem_context, analysis):
        if problem_context.strip() and not analysis.needs_clarification:
            self.model_call_count += 1
        result = self._reasoner.analyze(message, problem_context, analysis)
        if result.uncertainty_note:
            self.error = result.uncertainty_note
        return result


class _CountingResponseClient:
    def __init__(self) -> None:
        self._client = DeepSeekClient(max_retries=0)
        self.model_call_count = 0
        self.error = ""

    def chat(self, system, user, temperature=0.2, max_tokens=None) -> str:
        self.model_call_count += 1
        try:
            return self._client.chat(
                system,
                user,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except DeepSeekError as exc:
            self.error = str(exc)
            raise


def _safety_checks(case, state) -> dict[str, bool]:
    candidate_visible = case.candidate_conclusion in state.response
    return {
        "one_final_question": (
            state.response.count("？") + state.response.count("?") == 1
            and state.response.endswith(("？", "?"))
        ),
        "no_unapproved_candidate": (
            case.student_requests_answer
            or not candidate_visible
        ),
        "no_verified_claim_without_verifier": (
            state.verification.status.value not in {"verified_correct", "verified_incorrect"}
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="明确允许实际调用 DeepSeek")
    parser.add_argument("--max-cases", type=int, help="运行 1 到 12 条固定 smoke 样本")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.live or args.max_cases is None:
        print("拒绝运行：必须同时提供 --live 和 --max-cases。")
        return 2
    if not 1 <= args.max_cases <= MAX_CASES:
        print("拒绝运行：--max-cases 必须在 1 到 12 之间。")
        return 2
    if not is_configured():
        print("拒绝运行：未配置 DEEPSEEK_API_KEY。")
        return 2

    cases = load_generic_math_smoke_cases()[: args.max_cases]
    output_path = Path(os.getenv("TEACHER_LOG_DIR", "logs")) / "generic_math_live_smoke.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_calls = 0
    print(f"真实 smoke：最多 {2 * len(cases)} 次模型调用，预算上限 {BUDGET_CNY} 元，无重试。")
    with output_path.open("a", encoding="utf-8") as output:
        for case in cases:
            reasoner = _CountingReasoner()
            response_client = _CountingResponseClient()
            state = run_generic_teaching_turn(
                case.student_id,
                case.message,
                reasoner=reasoner,
                response_client=response_client,
                analysis_allow_llm=False,
            )
            model_call_count = reasoner.model_call_count + response_client.model_call_count
            total_calls += model_call_count
            record = {
                "case_id": case.id,
                "domain": case.domain,
                "model_call_count": model_call_count,
                "error": reasoner.error or response_client.error,
                "confidence": state.math_reasoning.confidence,
                "verification_status": state.verification.status.value,
                "action": state.action_plan.action.value if state.action_plan else "",
                "response": state.response,
                "safety_checks": _safety_checks(case, state),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{case.id}] calls={model_call_count} action={record['action']}")
    print(f"完成：实际模型调用 {total_calls} 次；结果写入 {output_path}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
