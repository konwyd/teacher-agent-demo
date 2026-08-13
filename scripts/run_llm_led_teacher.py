"""One-command launcher for local DeepSeek-V4, Chroma, and the LLM-led Teacher."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from typing import Callable
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "china_textbook_k9_vectordb"
DEFAULT_MODEL_SCRIPT = ROOT.parent / "run_deepseek_v4_8x4080.sh"
DEFAULT_KV_CACHE_DIR = ROOT / "data" / "runtime" / "kv-states"
DEFAULT_MODEL_HOST = "127.0.0.1"
DEFAULT_MODEL_PORT = 8080
DEFAULT_MODEL_NAME = "deepseek-v4-flash"
DEFAULT_CTX_SIZE = 524288
DEFAULT_PARALLEL = 1
DEFAULT_BATCH_SIZE = 128
DEFAULT_UBATCH_SIZE = 128


def direct_urlopen(request: Request, timeout: float):
    """Open a local service request without environment proxy routing."""

    return build_opener(ProxyHandler({})).open(request, timeout=timeout)


class ModelStartupError(RuntimeError):
    """The required local model could not be safely started or reused."""


class AppStartupError(RuntimeError):
    """The Streamlit application cannot be safely started."""


def load_service_base_env(
    base: dict[str, str],
    *,
    env_file: Path = ROOT / ".env",
) -> dict[str, str]:
    """Load local service secrets before RAG warmup; explicit shell values win."""
    env = dict(base)
    for name, value in dotenv_values(env_file).items():
        if value is not None:
            env.setdefault(name, value)
    return env


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def ensure_app_port_available(
    app_port: int,
    *,
    port_check: Callable[[str, int], bool] = port_open,
) -> None:
    if port_check("127.0.0.1", app_port):
        raise AppStartupError(
            f"页面端口 {app_port} 已被占用；请复用现有页面或先停止旧启动器。"
        )


def build_model_env(
    base: dict[str, str],
    *,
    model_host: str = DEFAULT_MODEL_HOST,
    model_port: int,
    ctx_size: int,
    parallel: int,
    batch_size: int,
    ubatch_size: int,
    cache_type_k: str,
    cache_type_v: str,
    kv_cache_dir: Path = DEFAULT_KV_CACHE_DIR,
) -> dict[str, str]:
    env = dict(base)
    env["PORT"] = str(model_port)
    env["CTX_SIZE"] = str(ctx_size)
    env["PARALLEL"] = str(parallel)
    env["BATCH_SIZE"] = str(batch_size)
    env["UBATCH_SIZE"] = str(ubatch_size)
    env["LLAMA_ARG_CACHE_TYPE_K"] = cache_type_k
    env["LLAMA_ARG_CACHE_TYPE_V"] = cache_type_v
    env["KV_SLOT_SAVE_PATH"] = f"{kv_cache_dir.resolve()}/"
    no_proxy = {
        item.strip()
        for item in (env.get("NO_PROXY") or env.get("no_proxy") or "").split(",")
        if item.strip()
    }
    no_proxy.update({"127.0.0.1", "localhost", model_host})
    env["NO_PROXY"] = env["no_proxy"] = ",".join(sorted(no_proxy))
    return env


def build_runtime_env(
    base: dict[str, str],
    *,
    rag_host: str,
    rag_port: int,
    model_host: str = DEFAULT_MODEL_HOST,
    model_port: int = DEFAULT_MODEL_PORT,
    model_name: str = DEFAULT_MODEL_NAME,
    request_timeout: float = 300.0,
    context_window_tokens: int = DEFAULT_CTX_SIZE,
    kv_cache_dir: Path = DEFAULT_KV_CACHE_DIR,
    parallel: int = DEFAULT_PARALLEL,
) -> dict[str, str]:
    env = dict(base)
    env["CHROMA_HOST"] = rag_host
    env["CHROMA_PORT"] = str(rag_port)
    env["CHROMA_SERVER"] = f"{rag_host}:{rag_port}"
    env["DEEPSEEK_API_KEY"] = "local"
    env["DEEPSEEK_BASE_URL"] = f"http://{model_host}:{model_port}/v1"
    env["DEEPSEEK_MODEL"] = model_name
    env["DEEPSEEK_TIMEOUT_SECONDS"] = str(request_timeout)
    env["DEEPSEEK_CONTEXT_WINDOW_TOKENS"] = str(context_window_tokens)
    env["TEACHER_KV_PERSIST"] = "1"
    env["TEACHER_KV_CACHE_DIR"] = str(kv_cache_dir.resolve())
    env["TEACHER_KV_SERVER_URL"] = f"http://{model_host}:{model_port}"
    env["TEACHER_LLAMA_PARALLEL"] = str(parallel)
    env.setdefault("TEACHER_RAG_DEVICE", "cpu")
    env.setdefault("TEACHER_RAG_CPU_THREADS", "4")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, env["TEACHER_RAG_CPU_THREADS"])
    no_proxy = {
        item.strip()
        for item in (env.get("NO_PROXY") or env.get("no_proxy") or "").split(",")
        if item.strip()
    }
    no_proxy.update({"127.0.0.1", "localhost", rag_host, model_host})
    env["NO_PROXY"] = env["no_proxy"] = ",".join(sorted(no_proxy))
    return env


# Backward-compatible name for existing callers outside the one-command launcher.
build_service_env = build_runtime_env


def model_server_ready(
    host: str,
    port: int,
    model_name: str,
    *,
    urlopen: Callable = direct_urlopen,
    timeout: float = 2.0,
    inference_timeout: float = 30.0,
) -> bool:
    base_url = f"http://{host}:{port}"
    try:
        with urlopen(Request(f"{base_url}/health"), timeout=timeout) as response:
            if not 200 <= int(getattr(response, "status", 200)) < 300:
                return False
            response.read()
        with urlopen(Request(f"{base_url}/v1/models"), timeout=timeout) as response:
            if not 200 <= int(getattr(response, "status", 200)) < 300:
                return False
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("data", []) if isinstance(payload, dict) else []
        if not any(
            isinstance(model, dict) and model.get("id") == model_name
            for model in models
        ):
            return False

        probe_body = json.dumps(
            {
                "model": model_name,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0,
            }
        ).encode("utf-8")
        probe = Request(
            f"{base_url}/v1/chat/completions",
            data=probe_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(probe, timeout=inference_timeout) as response:
            if not 200 <= int(getattr(response, "status", 200)) < 300:
                return False
            probe_payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False

    return bool(
        isinstance(probe_payload, dict)
        and isinstance(probe_payload.get("choices"), list)
        and probe_payload["choices"]
    )


def _terminate_started_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    wait = getattr(process, "wait", None)
    if wait is None:
        return
    try:
        wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def ensure_model_server(
    *,
    host: str,
    port: int,
    model_name: str,
    model_script: Path,
    startup_timeout: float,
    model_env: dict[str, str],
    port_check: Callable[[str, int], bool] = port_open,
    ready_check: Callable[[str, int, str], bool] = model_server_ready,
    popen_factory: Callable = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> subprocess.Popen | None:
    if port_check(host, port):
        if ready_check(host, port, model_name):
            return None
        raise ModelStartupError(
            f"模型端口 {host}:{port} 已被占用，但服务不健康或模型别名不匹配。"
        )

    if not model_script.is_file():
        raise ModelStartupError(f"本地模型启动脚本不存在：{model_script}")

    process = popen_factory(
        [str(model_script)],
        cwd=model_script.parent,
        env=model_env,
    )
    deadline = monotonic() + startup_timeout
    while True:
        if ready_check(host, port, model_name):
            return process
        exit_code = process.poll()
        if exit_code is not None:
            raise ModelStartupError(
                f"本地模型进程在健康检查前退出，exit_code={exit_code}。"
            )
        if monotonic() >= deadline:
            _terminate_started_process(process)
            raise ModelStartupError(
                f"等待本地模型健康检查超时（{startup_timeout:g} 秒）。"
            )
        sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-host", default=DEFAULT_MODEL_HOST)
    parser.add_argument("--model-port", type=int, default=DEFAULT_MODEL_PORT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-script", type=Path, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument("--model-startup-timeout", type=float, default=900.0)
    parser.add_argument("--llm-request-timeout", type=float, default=300.0)
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE)
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help="llama-server slots; keep 1 on 8x16 GB GPUs to avoid deferred CUDA OOM",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ubatch-size", type=int, default=DEFAULT_UBATCH_SIZE)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--rag-port", type=int, default=8002)
    parser.add_argument("--app-port", type=int, default=8502)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    try:
        ensure_app_port_available(args.app_port)
    except AppStartupError as exc:
        print(f"页面启动失败：{exc}", file=sys.stderr)
        return 1

    owned: list[subprocess.Popen] = []
    owned_names: list[str] = []

    def stop_owned() -> None:
        for proc in reversed(owned):
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 5
        for proc in reversed(owned):
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if proc.poll() is None:
                proc.kill()

    def handle_signal(*_args: object) -> None:
        stop_owned()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    DEFAULT_KV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    model_env = build_model_env(
        os.environ.copy(),
        model_host=args.model_host,
        model_port=args.model_port,
        ctx_size=args.ctx_size,
        parallel=args.parallel,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v,
        kv_cache_dir=DEFAULT_KV_CACHE_DIR,
    )
    print(
        "启动或复用本地 DeepSeek-V4："
        f"http://{args.model_host}:{args.model_port} "
        f"(model={args.model_name}, ctx={args.ctx_size}, parallel={args.parallel}, "
        f"batch={args.batch_size}/{args.ubatch_size}, timeout={args.model_startup_timeout:g}s)"
    )
    try:
        model_process = ensure_model_server(
            host=args.model_host,
            port=args.model_port,
            model_name=args.model_name,
            model_script=args.model_script,
            startup_timeout=args.model_startup_timeout,
            model_env=model_env,
        )
    except ModelStartupError as exc:
        print(f"本地模型启动失败：{exc}", file=sys.stderr)
        stop_owned()
        return 1
    if model_process is None:
        print(f"复用已有本地模型：{args.model_host}:{args.model_port}")
    else:
        owned.append(model_process)
        owned_names.append("本地模型")
        print(f"本地模型健康检查通过：{args.model_host}:{args.model_port}")

    if not port_open("127.0.0.1", args.rag_port):
        chroma = shutil.which("chroma")
        if chroma is None:
            print("找不到 chroma 命令；请先安装 chromadb。", file=sys.stderr)
            stop_owned()
            return 1
        if not args.db_path.exists():
            print(f"向量库路径不存在：{args.db_path}", file=sys.stderr)
            stop_owned()
            return 1
        print(f"启动 Chroma：{args.db_path} -> 0.0.0.0:{args.rag_port}")
        owned.append(
            subprocess.Popen(
                [chroma, "run", "--path", str(args.db_path), "--host", "0.0.0.0", "--port", str(args.rag_port)],
                cwd=ROOT,
            )
        )
        owned_names.append("Chroma")
        for _ in range(30):
            if port_open("127.0.0.1", args.rag_port):
                break
            if owned[-1].poll() is not None:
                print("Chroma 进程在就绪前退出。", file=sys.stderr)
                stop_owned()
                return 1
            time.sleep(0.5)
        else:
            print("Chroma 未能在限定时间内监听端口。", file=sys.stderr)
            stop_owned()
            return 1
    else:
        print(f"复用已有 Chroma：127.0.0.1:{args.rag_port}")

    env = build_runtime_env(
        load_service_base_env(os.environ.copy()),
        rag_host="127.0.0.1",
        rag_port=args.rag_port,
        model_host=args.model_host,
        model_port=args.model_port,
        model_name=args.model_name,
        request_timeout=args.llm_request_timeout,
        context_window_tokens=args.ctx_size,
        kv_cache_dir=DEFAULT_KV_CACHE_DIR,
        parallel=args.parallel,
    )
    print(
        "启动 LLM-led Teacher Prototype："
        f"http://localhost:{args.app_port} "
        f"(local model={env['DEEPSEEK_BASE_URL']}, "
        f"request timeout={env['DEEPSEEK_TIMEOUT_SECONDS']}s, "
        f"RAG device={env['TEACHER_RAG_DEVICE']})"
    )
    owned.append(
        subprocess.Popen(
            [
                sys.executable,
                "scripts/run_warmed_streamlit.py",
                "--app-port",
                str(args.app_port),
            ],
            cwd=ROOT,
            env=env,
        )
    )
    owned_names.append("Streamlit")
    try:
        while True:
            for name, proc in zip(owned_names, owned):
                exit_code = proc.poll()
                if exit_code is not None:
                    print(
                        f"{name} 进程意外退出，exit_code={exit_code}；"
                        "正在停止本启动器拥有的其余服务。",
                        file=sys.stderr,
                    )
                    return exit_code or 1
            time.sleep(1)
    finally:
        stop_owned()


if __name__ == "__main__":
    raise SystemExit(main())
