"""Run the current LLM-led Teacher against local llama.cpp and a hosted API.

The benchmark keeps the application workflow unchanged. Every measured run uses
a fresh ``PrototypeEpisode`` and switches only the ``DeepSeekSettings`` backend
profile. Results are persisted after every turn.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Callable, Iterator, Sequence
from urllib.parse import urlparse

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CASES_PATH = ROOT / "data" / "eval" / "backend_comparison_cases.json"
DEFAULT_OUTPUT_ROOT = ROOT / "logs" / "backend_comparison"
DEFAULT_MODEL_SCRIPT = ROOT.parent / "run_deepseek_v4_8x4080.sh"
DEFAULT_DB_PATH = ROOT / "data" / "china_textbook_k9_vectordb"
BACKENDS = ("local", "api")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    prompt: str


def load_cases(path: Path) -> list[BenchmarkCase]:
    """Load and validate the fixed question set."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("benchmark case file must contain a non-empty JSON list")
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"case {index} must be a JSON object")
        case_id = str(item.get("case_id", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        if not case_id or not prompt:
            raise ValueError(f"case {index} requires non-empty case_id and prompt")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        cases.append(BenchmarkCase(case_id=case_id, prompt=prompt))
    return cases


def run_matrix(
    cases: Sequence[BenchmarkCase],
    *,
    backends: Sequence[str],
    repeats: int,
    execute: Callable[[BenchmarkCase, str, int], dict[str, object]],
    persist: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    """Run backend/case/repeat combinations and continue after isolated errors."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    unknown = [backend for backend in backends if backend not in BACKENDS]
    if unknown:
        raise ValueError(f"unsupported backends: {unknown}")

    records: list[dict[str, object]] = []
    for backend in backends:
        for case in cases:
            for repeat in range(1, repeats + 1):
                base: dict[str, object] = {
                    "case_id": case.case_id,
                    "prompt": case.prompt,
                    "backend": backend,
                    "repeat": repeat,
                }
                try:
                    details = execute(case, backend, repeat)
                    record = {**base, **details}
                    record.setdefault("status", "ok")
                except Exception as exc:  # noqa: BLE001 - persist failure and continue
                    record = {
                        **base,
                        "status": "error",
                        "backend_total_seconds": 0.0,
                        "response": "",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                records.append(record)
                if persist is not None:
                    persist(record)
    return records


def _aggregate(records: Sequence[dict[str, object]]) -> dict[str, object]:
    successes = [record for record in records if record.get("status") == "ok"]
    timings = [float(record["backend_total_seconds"]) for record in successes]
    return {
        "success_count": len(successes),
        "error_count": len(records) - len(successes),
        "mean_seconds": round(fmean(timings), 3) if timings else "",
        "median_seconds": round(median(timings), 3) if timings else "",
        "min_seconds": round(min(timings), 3) if timings else "",
        "max_seconds": round(max(timings), 3) if timings else "",
    }


def summarize_results(
    records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize successful backend latency per case without unstable p95s."""

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record["case_id"]), str(record["backend"]))
        grouped.setdefault(key, []).append(record)
    return [
        {"case_id": case_id, "backend": backend, **_aggregate(items)}
        for (case_id, backend), items in grouped.items()
    ]


def _format_number(value: object) -> str:
    return "" if value == "" else f"{float(value):.3f}"


def render_markdown_report(
    *,
    cases: Sequence[BenchmarkCase],
    records: Sequence[dict[str, object]],
    summary: Sequence[dict[str, object]],
    run_id: str,
) -> str:
    """Render latency tables and every raw answer for direct human comparison."""

    lines = [
        "# 本地模型与 API 自动对比报告",
        "",
        f"- Run ID：`{run_id}`",
        f"- 正式回合：{len(records)}",
        "- 口径：每次均为全新空对话；沿用当前 Agent、RAG、prompt 和温度。",
        "",
        "## 总体耗时",
        "",
        "| 后端 | 成功 | 失败 | 平均秒 | 中位秒 | 最快秒 | 最慢秒 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in BACKENDS:
        items = [record for record in records if record.get("backend") == backend]
        if not items:
            continue
        row = _aggregate(items)
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                backend,
                row["success_count"],
                row["error_count"],
                _format_number(row["mean_seconds"]),
                _format_number(row["median_seconds"]),
                _format_number(row["min_seconds"]),
                _format_number(row["max_seconds"]),
            )
        )

    summary_by_key = {
        (str(row["case_id"]), str(row["backend"])): row for row in summary
    }
    for case in cases:
        lines.extend(
            [
                "",
                f"## {case.case_id}",
                "",
                f"> {case.prompt}",
                "",
                "| 后端 | 成功 | 失败 | 平均秒 | 中位秒 | 最快秒 | 最慢秒 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for backend in BACKENDS:
            row = summary_by_key.get((case.case_id, backend))
            if row is None:
                continue
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    backend,
                    row["success_count"],
                    row["error_count"],
                    _format_number(row["mean_seconds"]),
                    _format_number(row["median_seconds"]),
                    _format_number(row["min_seconds"]),
                    _format_number(row["max_seconds"]),
                )
            )

        case_records = [
            record for record in records if record.get("case_id") == case.case_id
        ]
        for record in case_records:
            backend = record["backend"]
            repeat = record["repeat"]
            seconds = float(record.get("backend_total_seconds", 0.0) or 0.0)
            status = record.get("status", "error")
            lines.extend(
                [
                    "",
                    f"### {backend} 第 {repeat} 次 — {seconds:.3f}s — {status}",
                    "",
                    "- LLM 调用次数：{}".format(record.get("llm_call_count", "")),
                    "- 是否修正：{}".format(record.get("repair_performed", "")),
                    "- RAG 来源：{}".format(record.get("rag_source", "")),
                    "- 阶段耗时：`{}`".format(
                        json.dumps(
                            record.get("stage_timings_seconds", {}),
                            ensure_ascii=False,
                        )
                    ),
                    "",
                ]
            )
            if status == "ok":
                lines.extend(
                    [
                        "<pre>",
                        html.escape(str(record.get("response", ""))),
                        "</pre>",
                    ]
                )
            else:
                lines.append(
                    "错误：`{}: {}`".format(
                        html.escape(str(record.get("error_type", "Error"))),
                        html.escape(str(record.get("error", ""))),
                    )
                )
    return "\n".join(lines) + "\n"


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "case_id",
        "backend",
        "success_count",
        "error_count",
        "mean_seconds",
        "median_seconds",
        "min_seconds",
        "max_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _start_or_reuse_chroma(*, port: int, db_path: Path) -> subprocess.Popen | None:
    from scripts.run_llm_led_teacher import port_open

    if port_open("127.0.0.1", port):
        print(f"复用已有 Chroma：127.0.0.1:{port}")
        return None
    chroma = shutil.which("chroma")
    if chroma is None:
        raise RuntimeError("找不到 chroma 命令；请安装 chromadb")
    if not db_path.exists():
        raise RuntimeError(f"向量库路径不存在：{db_path}")
    print(f"启动 Chroma：{db_path} -> 127.0.0.1:{port}")
    process = subprocess.Popen(
        [
            chroma,
            "run",
            "--path",
            str(db_path),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
        cwd=ROOT,
    )
    for _ in range(60):
        if port_open("127.0.0.1", port):
            return process
        if process.poll() is not None:
            raise RuntimeError(f"Chroma 在就绪前退出，exit_code={process.returncode}")
        time.sleep(0.5)
    _terminate_process(process)
    raise RuntimeError("Chroma 未能在30秒内启动")


@contextmanager
def managed_services(args: argparse.Namespace, backends: Sequence[str]) -> Iterator[None]:
    """Start/reuse llama.cpp and Chroma, cleaning up only owned processes."""

    from app.llm.deepseek_client import DeepSeekSettings
    from scripts.run_llm_led_teacher import (
        build_model_env,
        ensure_model_server,
        load_service_base_env,
    )

    model_process: subprocess.Popen | None = None
    chroma_process: subprocess.Popen | None = None
    try:
        if "local" in backends:
            settings = DeepSeekSettings(teacher_llm_backend="local")
            parsed = urlparse(settings.deepseek_base_url)
            model_host = parsed.hostname or "127.0.0.1"
            model_port = parsed.port or 8080
            if model_host not in {"127.0.0.1", "localhost", "::1"}:
                raise RuntimeError(
                    "local profile must point to a localhost llama.cpp endpoint"
                )
            model_env = build_model_env(
                load_service_base_env(os.environ.copy()),
                model_host=model_host,
                model_port=model_port,
                ctx_size=settings.deepseek_context_window_tokens,
                parallel=1,
                batch_size=128,
                ubatch_size=128,
                cache_type_k="q8_0",
                cache_type_v="q4_0",
            )
            print(
                f"启动或复用本地模型：{settings.deepseek_base_url} "
                f"(model={settings.deepseek_model})"
            )
            model_process = ensure_model_server(
                host=model_host,
                port=model_port,
                model_name=settings.deepseek_model,
                model_script=args.model_script,
                startup_timeout=args.model_startup_timeout,
                model_env=model_env,
            )
        chroma_process = _start_or_reuse_chroma(
            port=args.rag_port,
            db_path=args.db_path,
        )
        yield
    finally:
        _terminate_process(chroma_process)
        _terminate_process(model_process)


def _validate_backend_settings(backends: Sequence[str]) -> dict[str, dict[str, str]]:
    from app.llm.deepseek_client import DeepSeekSettings

    safe: dict[str, dict[str, str]] = {}
    for backend in backends:
        settings = DeepSeekSettings(teacher_llm_backend=backend)
        if not settings.deepseek_api_key:
            raise RuntimeError(f"{backend} 后端没有配置 API key")
        safe[backend] = {
            "base_url": settings.deepseek_base_url,
            "model": settings.deepseek_model,
        }
    return safe


def _warm_backend(backend: str) -> None:
    from app.llm.deepseek_client import DeepSeekClient, DeepSeekSettings

    client = DeepSeekClient(
        settings=DeepSeekSettings(teacher_llm_backend=backend)
    )
    client.chat(
        "你正在进行连接预热。",
        "只回复：好",
        temperature=0.0,
        max_tokens=4,
    )


def _execute_real_turn(
    case: BenchmarkCase,
    backend: str,
    repeat: int,
) -> dict[str, object]:
    """Execute the exact current application turn with a fresh empty episode."""

    from app.llm.deepseek_client import DeepSeekClient, DeepSeekSettings
    from prototypes.llm_led_teacher.logic import PrototypeEpisode, run_turn

    client = DeepSeekClient(
        settings=DeepSeekSettings(teacher_llm_backend=backend)
    )
    episode = PrototypeEpisode()
    turn = run_turn(
        episode,
        case.prompt,
        client=client,
        enable_agent_loop=True,
    )
    usage = turn.model_generation_metadata
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend_total_seconds": turn.backend_total_seconds,
        "stage_timings_seconds": turn.stage_timings_seconds,
        "response": turn.teacher_output.response,
        "final_character_count": turn.final_character_count,
        "repair_performed": turn.repair_performed,
        "repair_reasons": turn.repair_reasons,
        "remaining_warnings": turn.remaining_warnings,
        "llm_call_count": int(usage.get("llm_call_count", 0) or 0),
        "model_generation_metadata": usage,
        "rag_source": turn.rag_source,
        "rag_refs": turn.rag_refs,
        "rag_hit_count": len(turn.rag_hits),
        "rag_online_triggered": turn.rag_online_triggered,
        "rag_online_status": turn.rag_online_status,
        "classification": (
            turn.classification.model_dump(mode="json")
            if turn.classification is not None
            else None
        ),
        "retrieval_decision": turn.retrieval_decision.model_dump(mode="json"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键运行本地 llama.cpp 与云端 API 的完整 Teacher 对比。"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--backend",
        choices=("both", *BACKENDS),
        default="both",
        help="默认依次运行 local 和 api；也可只跑一个后端。",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--no-manage-services",
        action="store_true",
        help="不自动启动 llama.cpp/Chroma，只复用已运行的服务。",
    )
    parser.add_argument("--rag-port", type=int, default=8002)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--model-script", type=Path, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument("--model-startup-timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def main() -> int:
    args = _parse_args()
    load_dotenv(ROOT / ".env")
    os.environ["TEACHER_KV_PERSIST"] = "0"
    os.environ.setdefault("TEACHER_RAG_DEVICE", "cpu")
    os.environ.setdefault("TEACHER_RAG_CPU_THREADS", "4")
    os.environ["CHROMA_HOST"] = "127.0.0.1"
    os.environ["CHROMA_PORT"] = str(args.rag_port)
    os.environ["CHROMA_SERVER"] = f"127.0.0.1:{args.rag_port}"

    backends = BACKENDS if args.backend == "both" else (args.backend,)
    cases = load_cases(args.cases)
    safe_settings = _validate_backend_settings(backends)
    run_id = _new_run_id()
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw_results.jsonl"
    summary_path = output_dir / "latency_summary.csv"
    report_path = output_dir / "comparison_report.md"
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "case_ids": [case.case_id for case in cases],
        "backends": list(backends),
        "repeats": args.repeats,
        "expected_turns": len(cases) * len(backends) * args.repeats,
        "backend_settings": safe_settings,
        "temperature_policy": "unchanged application defaults",
        "fresh_episode_per_turn": True,
        "kv_persistence": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    @contextmanager
    def service_scope() -> Iterator[None]:
        if args.no_manage_services:
            yield
        else:
            with managed_services(args, backends):
                yield

    records: list[dict[str, object]] = []
    try:
        with service_scope():
            from prototypes.llm_led_teacher.rag_hook import warmup_rag

            rag_status = warmup_rag()
            manifest["rag_warmup"] = rag_status
            print(
                "RAG预热：ok={} device={} elapsed={}s".format(
                    rag_status.get("ok"),
                    rag_status.get("device"),
                    rag_status.get("elapsed_seconds"),
                )
            )
            if not args.skip_warmup:
                for backend in backends:
                    print(f"预热 {backend} 后端（不计入正式结果）...")
                    _warm_backend(backend)

            def persist(record: dict[str, object]) -> None:
                _append_jsonl(raw_path, record)
                status = record.get("status")
                seconds = float(record.get("backend_total_seconds", 0.0) or 0.0)
                print(
                    "[{}] {} {} 第{}次 {:.3f}s".format(
                        status,
                        record["backend"],
                        record["case_id"],
                        record["repeat"],
                        seconds,
                    )
                )
                if status != "ok":
                    print(
                        "       {}: {}".format(
                            record.get("error_type"), record.get("error")
                        ),
                        file=sys.stderr,
                    )

            records = run_matrix(
                cases,
                backends=backends,
                repeats=args.repeats,
                execute=_execute_real_turn,
                persist=persist,
            )

        summary = summarize_results(records)
        _write_summary_csv(summary_path, summary)
        report_path.write_text(
            render_markdown_report(
                cases=cases,
                records=records,
                summary=summary,
                run_id=run_id,
            ),
            encoding="utf-8",
        )
        error_count = sum(1 for record in records if record.get("status") != "ok")
        manifest.update(
            {
                "complete": error_count == 0,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(records),
                "error_count": error_count,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果目录：{output_dir}")
        print(f"回答对比：{report_path}")
        print(f"耗时汇总：{summary_path}")
        return 0 if error_count == 0 else 1
    except Exception as exc:
        manifest.update(
            {
                "complete": False,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "record_count": len(records),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"自动对比失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"已保留结果目录：{output_dir}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
