"""Run the bounded LLM-led Teacher against the local 8080/8002 stack.

This is a manually reviewed smoke test, not part of the offline evaluation gate.
It deliberately refuses non-loopback model endpoints so ``--live`` cannot spend
money through a cloud API by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.deepseek_client import DeepSeekClient, DeepSeekSettings
from prototypes.llm_led_teacher.logic import PrototypeEpisode, run_turn
from prototypes.llm_led_teacher.rag_hook import warmup_rag


MAX_SCENARIOS = 12
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "deepseek-v4-flash"


_PLACEHOLDER_FALLBACKS = {
    "我会只说明这个概念本身和相关符号的含义，不展开当前题的计算。",
    "这一步先只完成一个必要变换，并检查变换前后是否保持等价。",
    "我们先确定当前要推进的局部目标，再只处理这一部分。",
    "我会围绕你提出的学习目标，给出简洁且不超出约定范围的说明。",
    "这次先保留在约定的教学范围内。",
    "请继续提出一个相关学习问题。",
    "暂不展开。",
    "。",
}


@dataclass(frozen=True)
class SmokeScenario:
    id: str
    category: str
    messages: tuple[str, ...]
    expected_subject: str | None
    expected_scope: str = "k9"


SCENARIOS: tuple[SmokeScenario, ...] = (
    SmokeScenario(
        "simple_math_guidance",
        "简单数学引导",
        ("初中数学：解方程 2x+3=7，我不会，请只提示第一步。",),
        "数学",
    ),
    SmokeScenario(
        "math_concept",
        "数学概念解释",
        ("初中数学里，等式的基本性质是什么？请解释概念，不要套具体题。",),
        "数学",
    ),
    SmokeScenario(
        "incorrect_math_step",
        "学生错误步骤",
        (
            "初中数学：解方程 2x+3=7，请先引导我做第一步。",
            "我把两边都减去3，得到2x=5，对吗？",
        ),
        "数学",
    ),
    SmokeScenario(
        "physics",
        "物理",
        ("初中物理：为什么汽车突然刹车时人会向前倾？",),
        "物理",
    ),
    SmokeScenario(
        "chemistry",
        "化学",
        ("初中化学：质量守恒定律应该怎样理解？",),
        "化学",
    ),
    SmokeScenario(
        "chinese",
        "语文",
        ("初中语文：比喻和拟人有什么区别？各举一个简短例子。",),
        "语文",
    ),
    SmokeScenario(
        "english",
        "英语",
        ("初中英语：一般过去时和现在完成时有什么区别？",),
        "英语",
    ),
    SmokeScenario(
        "geography",
        "历史地理生物代表样本",
        ("初中地理：季风气候为什么会有明显的雨季和旱季？",),
        "地理",
    ),
    SmokeScenario(
        "follow_up",
        "追问续答",
        (
            "初中物理：请解释浮力产生的原因。",
            "那为什么物体上下表面的压力会不一样？",
        ),
        "物理",
    ),
    SmokeScenario(
        "full_solution",
        "明确索要完整答案",
        ("初中数学：请完整解答方程 3x-5=10，写出步骤和最终答案。",),
        "数学",
    ),
    SmokeScenario(
        "off_topic",
        "课外问题",
        ("请推荐一款适合打游戏的手机。",),
        None,
        expected_scope="off_topic",
    ),
    SmokeScenario(
        "invalid_input",
        "无效输入",
        ("asdfghjkl ？？？",),
        None,
        expected_scope="off_topic",
    ),
)


def _is_loopback_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _tool_calls(turn) -> list[dict[str, object]]:
    return [
        {
            "step": event.step,
            "event_type": event.event_type,
            "tool_name": event.tool_name,
            "ok": event.ok,
            "elapsed_seconds": event.elapsed_seconds,
            "summary": event.summary,
        }
        for event in turn.agent_trace
        if event.tool_name or event.event_type in {"fast_route", "fallback"}
    ]


def _turn_record(turn, *, elapsed_seconds: float) -> dict[str, object]:
    classification = turn.classification
    metadata = turn.model_generation_metadata
    return {
        "turn_id": turn.turn_id,
        "student_message": turn.student_message,
        "teacher_response": turn.teacher_output.response,
        "classification": (
            classification.model_dump(mode="json") if classification else None
        ),
        "loaded_skills": list(turn.loaded_skills),
        "tool_calls": _tool_calls(turn),
        "agent_termination_reason": turn.agent_termination_reason,
        "teaching_contract": (
            turn.teaching_contract.model_dump(mode="json")
            if turn.teaching_contract
            else None
        ),
        "verification_status": turn.student_verification.status.value,
        "boundary_audit_passed": (
            turn.boundary_audit.passed if turn.boundary_audit else None
        ),
        "boundary_violations": (
            [violation.value for violation in turn.boundary_audit.violations]
            if turn.boundary_audit
            else []
        ),
        "post_repair_boundary_audit_passed": (
            turn.post_repair_boundary_audit.passed
            if turn.post_repair_boundary_audit
            else None
        ),
        "boundary_fallback_reason": turn.boundary_fallback_reason,
        "repair_performed": turn.repair_performed,
        "remaining_warnings": list(turn.remaining_warnings),
        "rag_source": turn.rag_source,
        "rag_hit_count": len(turn.rag_hits),
        "stage_timings_seconds": dict(turn.stage_timings_seconds),
        "backend_total_seconds": turn.backend_total_seconds,
        # Safe display starts only after run_turn returns and auditing is complete.
        "safe_output_ready_seconds": round(elapsed_seconds, 3),
        "llm_call_count": int(metadata.get("llm_call_count", 0) or 0),
        "total_tokens": int(metadata.get("total_tokens", 0) or 0),
        "prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(metadata.get("completion_tokens", 0) or 0),
        "cache_hit_rate": float(metadata.get("cache_hit_rate", 0.0) or 0.0),
    }


def _scenario_checks(
    scenario: SmokeScenario,
    records: list[dict[str, object]],
) -> dict[str, bool]:
    classifications = [record.get("classification") or {} for record in records]
    last_classification = classifications[-1]
    contracts = [record.get("teaching_contract") or {} for record in records]
    no_duplicate_public_tool_events = True
    for record in records:
        tool_calls = record.get("tool_calls", [])
        signatures = {
            (call.get("event_type"), call.get("tool_name"), call.get("summary"))
            for call in tool_calls
        }
        if len(signatures) != len(tool_calls):
            no_duplicate_public_tool_events = False
            break
    no_final_answer_without_permission = all(
        bool(contract.get("may_reveal_final_answer"))
        == (contract.get("response_mode") == "full_solution")
        for contract in contracts
    )
    return {
        "expected_scope": last_classification.get("scope") == scenario.expected_scope,
        "expected_subject": last_classification.get("subject") == scenario.expected_subject,
        "follow_up_continuity": (
            len(records) == 1
            or classifications[-1].get("is_new_problem") is False
        ),
        "no_duplicate_public_tool_events": no_duplicate_public_tool_events,
        "no_final_answer_without_permission": no_final_answer_without_permission,
        "no_placeholder_fallback": all(
            (record.get("classification") or {}).get("scope") == "off_topic"
            or record.get("teacher_response") not in _PLACEHOLDER_FALLBACKS
            for record in records
        ),
        "no_remaining_warnings": all(
            not record.get("remaining_warnings") for record in records
        ),
        "terminated": all(bool(record.get("agent_termination_reason")) for record in records),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="明确允许调用本地真实模型")
    parser.add_argument(
        "--max-scenarios",
        type=int,
        help="运行前 1 到 12 组固定场景",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/llm_led_agent_live_smoke.jsonl"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.live or args.max_scenarios is None:
        print("拒绝运行：必须同时提供 --live 和 --max-scenarios。")
        return 2
    if not 1 <= args.max_scenarios <= MAX_SCENARIOS:
        print("拒绝运行：--max-scenarios 必须在 1 到 12 之间。")
        return 2
    if not _is_loopback_url(args.base_url):
        print("拒绝运行：真实 Agent smoke 只允许 127.0.0.1/localhost 模型地址。")
        return 2

    settings = DeepSeekSettings(
        deepseek_api_key="local-llama-server",
        deepseek_base_url=args.base_url,
        deepseek_model=args.model,
        deepseek_timeout_seconds=args.timeout,
    )
    client = DeepSeekClient(settings=settings, max_retries=0)
    warmup = warmup_rag()
    print(
        "RAG 预热："
        f"ok={warmup.get('ok')} device={warmup.get('device')} "
        f"elapsed={warmup.get('elapsed_seconds')}s"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = SCENARIOS[: args.max_scenarios]
    failed = 0
    with args.output.open("a", encoding="utf-8") as output:
        for scenario in selected:
            episode = PrototypeEpisode()
            records: list[dict[str, object]] = []
            scenario_started = perf_counter()
            error = ""
            for message in scenario.messages:
                turn_started = perf_counter()
                try:
                    turn = run_turn(
                        episode,
                        message,
                        client=client,
                        enable_agent_loop=True,
                    )
                except Exception as exc:  # noqa: BLE001 - smoke must preserve evidence
                    error = f"{type(exc).__name__}: {exc}"
                    break
                records.append(
                    _turn_record(
                        turn,
                        elapsed_seconds=perf_counter() - turn_started,
                    )
                )
            checks = _scenario_checks(scenario, records) if records else {}
            passed = not error and bool(checks) and all(checks.values())
            failed += int(not passed)
            record = {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "scenario_id": scenario.id,
                "category": scenario.category,
                "passed": passed,
                "error": error,
                "warmup": warmup,
                "elapsed_seconds": round(perf_counter() - scenario_started, 3),
                "checks": checks,
                "turns": records,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            total_calls = sum(int(item["llm_call_count"]) for item in records)
            total_tokens = sum(int(item["total_tokens"]) for item in records)
            print(
                f"[{scenario.id}] passed={passed} turns={len(records)} "
                f"calls={total_calls} tokens={total_tokens} "
                f"elapsed={record['elapsed_seconds']}s"
            )
            if error:
                print(f"  error={error}")
    print(f"完成：{len(selected) - failed}/{len(selected)} 组通过；结果写入 {args.output}。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
