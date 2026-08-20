import json
from pathlib import Path

import pytest


class FakeProcess:
    def __init__(self, poll_values=None):
        self._poll_values = iter(poll_values or [None])
        self.terminated = False
        self.killed = False

    def poll(self):
        return next(self._poll_values, None)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_direct_urlopen_disables_environment_proxies(monkeypatch):
    from scripts import run_llm_led_teacher as launcher

    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(("open", request.full_url, timeout))
            return "response"

    def proxy_handler(config):
        calls.append(("proxy", config))
        return "handler"

    def build_opener(handler):
        calls.append(("opener", handler))
        return Opener()

    monkeypatch.setattr(launcher, "ProxyHandler", proxy_handler, raising=False)
    monkeypatch.setattr(launcher, "build_opener", build_opener, raising=False)

    request = launcher.Request("http://127.0.0.1:8080/health")
    assert launcher.direct_urlopen(request, timeout=7) == "response"
    assert calls == [
        ("proxy", {}),
        ("opener", "handler"),
        ("open", "http://127.0.0.1:8080/health", 7),
    ]


def test_app_port_preflight_rejects_duplicate_launcher():
    from scripts.run_llm_led_teacher import AppStartupError, ensure_app_port_available

    with pytest.raises(AppStartupError, match="8502.*占用"):
        ensure_app_port_available(
            8502,
            port_check=lambda host, port: host == "127.0.0.1" and port == 8502,
        )


def test_service_environment_forces_local_model_and_preserves_explicit_rag_device():
    from scripts.run_llm_led_teacher import build_runtime_env

    env = build_runtime_env(
        {
            "DEEPSEEK_API_KEY": "paid-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_MODEL": "deepseek-chat",
            "TEACHER_RAG_DEVICE": "cpu",
        },
        rag_host="127.0.0.1",
        rag_port=8002,
        model_host="127.0.0.1",
        model_port=8080,
        model_name="deepseek-v4-flash",
        request_timeout=300.0,
        context_window_tokens=524288,
    )

    assert env["DEEPSEEK_API_KEY"] == "local"
    assert env["DEEPSEEK_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert env["DEEPSEEK_MODEL"] == "deepseek-v4-flash"
    assert env["DEEPSEEK_TIMEOUT_SECONDS"] == "300.0"
    assert env["DEEPSEEK_CONTEXT_WINDOW_TOKENS"] == "524288"
    assert env["TEACHER_RAG_DEVICE"] == "cpu"
    assert env["CHROMA_SERVER"] == "127.0.0.1:8002"


def test_launcher_passes_cli_context_size_to_runtime_environment():
    source = Path("scripts/run_llm_led_teacher.py").read_text()
    call_start = source.index("    env = build_runtime_env(")
    call_end = source.index("    )", call_start)
    runtime_call = source[call_start:call_end]

    assert "context_window_tokens=args.ctx_size," in runtime_call


def test_service_base_env_loads_dotenv_before_rag_warmup(tmp_path):
    from scripts.run_llm_led_teacher import load_service_base_env

    env_file = tmp_path / ".env"
    env_file.write_text("TAVILY_API_KEY=from-file\nKEEP=file\nONLY_FILE=loaded\n")

    env = load_service_base_env(
        {"TAVILY_API_KEY": "from-shell", "KEEP": "shell"},
        env_file=env_file,
    )

    assert env["TAVILY_API_KEY"] == "from-shell"
    assert env["KEEP"] == "shell"

    assert env["ONLY_FILE"] == "loaded"

def test_local_launcher_defaults_to_unified_512k_context():
    from scripts.run_llm_led_teacher import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_CACHE_RAM_MIB,
        DEFAULT_CTX_SIZE,
        DEFAULT_PARALLEL,
        DEFAULT_UBATCH_SIZE,
        build_model_env,
    )

    assert DEFAULT_CTX_SIZE == 524288
    assert DEFAULT_PARALLEL == 4
    assert DEFAULT_CACHE_RAM_MIB == 65536
    assert DEFAULT_BATCH_SIZE == 128
    assert DEFAULT_UBATCH_SIZE == 128

    env = build_model_env(
        {"PATH": "/usr/bin"},
        model_port=8080,
        ctx_size=DEFAULT_CTX_SIZE,
        parallel=DEFAULT_PARALLEL,
        batch_size=DEFAULT_BATCH_SIZE,
        ubatch_size=DEFAULT_UBATCH_SIZE,
        cache_type_k="q8_0",
        cache_type_v="q4_0",
        unified_kv=True,
        continuous_batching=True,
        cache_ram_mib=DEFAULT_CACHE_RAM_MIB,
        cache_idle_slots=True,
    )

    assert env["PORT"] == "8080"
    assert env["CTX_SIZE"] == "524288"
    assert env["PARALLEL"] == "4"
    assert env["KV_UNIFIED"] == "1"
    assert env["CONT_BATCHING"] == "1"
    assert env["CACHE_RAM_MIB"] == "65536"
    assert env["CACHE_IDLE_SLOTS"] == "1"
    assert env["BATCH_SIZE"] == "128"
    assert env["UBATCH_SIZE"] == "128"
    assert env["LLAMA_ARG_CACHE_TYPE_K"] == "q8_0"
    assert env["LLAMA_ARG_CACHE_TYPE_V"] == "q4_0"
    assert set(env["NO_PROXY"].split(",")) >= {"127.0.0.1", "localhost"}
    assert env["no_proxy"] == env["NO_PROXY"]


def test_kv_environment_reuses_existing_model_address_and_parallel(tmp_path):
    from scripts.run_llm_led_teacher import build_model_env, build_runtime_env

    service_env = build_runtime_env(
        {},
        rag_host="127.0.0.1",
        rag_port=8002,
        model_host="10.0.0.8",
        model_port=9099,
        kv_cache_dir=tmp_path,
        parallel=4,
    )
    model_env = build_model_env(
        {},
        model_host="10.0.0.8",
        model_port=9099,
        ctx_size=131072,
        parallel=4,
        batch_size=128,
        ubatch_size=128,
        cache_type_k="q8_0",
        cache_type_v="q4_0",
        kv_cache_dir=tmp_path,
    )

    assert service_env["DEEPSEEK_BASE_URL"] == "http://10.0.0.8:9099/v1"
    assert service_env["TEACHER_KV_SERVER_URL"] == "http://10.0.0.8:9099"
    assert service_env["TEACHER_LLAMA_PARALLEL"] == "4"
    assert service_env["TEACHER_LLAMA_UNIFIED_KV"] == "1"
    assert service_env["TEACHER_MAX_CONCURRENT_GENERATIONS"] == "4"
    assert service_env["TEACHER_KV_PERSIST"] == "1"
    assert service_env["TEACHER_KV_KEEP_VERSIONS"] == "1"
    assert service_env["TEACHER_KV_CACHE_DIR"] == str(tmp_path.resolve())
    assert model_env["KV_SLOT_SAVE_PATH"] == f"{tmp_path.resolve()}/"


def test_model_script_consumes_unified_kv_and_ram_cache_flags():
    from scripts.run_llm_led_teacher import DEFAULT_MODEL_SCRIPT

    source = DEFAULT_MODEL_SCRIPT.read_text()

    assert "--kv-unified" in source
    assert "--cont-batching" in source
    assert "--cache-ram" in source
    assert "--cache-idle-slots" in source


def test_model_script_resolution_supports_linked_worktree_and_explicit_env(tmp_path):
    from scripts.run_llm_led_teacher import resolve_model_script_path

    workspace = tmp_path / "workspace"
    main_checkout = workspace / "teacher-agent-demo"
    linked_checkout = main_checkout / ".worktrees" / "spike"
    common_git_dir = main_checkout / ".git"
    configured = tmp_path / "configured-model.sh"

    assert resolve_model_script_path(
        repo_root=linked_checkout,
        environ={"TEACHER_MODEL_SCRIPT": str(configured)},
        git_common_dir=common_git_dir,
    ) == configured.resolve()
    assert resolve_model_script_path(
        repo_root=linked_checkout,
        environ={},
        git_common_dir=common_git_dir,
    ) == (workspace / "run_deepseek_v4_8x4080.sh").resolve()


def test_model_health_requires_health_expected_alias_and_real_inference():
    from scripts.run_llm_led_teacher import model_server_ready

    calls = []

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def urlopen(request, timeout):
        calls.append((request.full_url, request.get_method(), request.data, timeout))
        if request.full_url.endswith("/health"):
            return Response({"status": "ok"})
        if request.full_url.endswith("/v1/models"):
            return Response({"data": [{"id": "deepseek-v4-flash"}]})
        return Response({"choices": [{"message": {"content": "ok"}}]})

    assert model_server_ready(
        "127.0.0.1",
        8080,
        "deepseek-v4-flash",
        urlopen=urlopen,
    )
    assert [url for url, _, _, _ in calls] == [
        "http://127.0.0.1:8080/health",
        "http://127.0.0.1:8080/v1/models",
        "http://127.0.0.1:8080/v1/chat/completions",
    ]
    inference_call = calls[-1]
    assert inference_call[1] == "POST"
    assert json.loads(inference_call[2]) == {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }


def test_model_health_rejects_server_that_crashes_on_first_inference():
    from scripts.run_llm_led_teacher import model_server_ready

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def urlopen(request, timeout):
        if request.full_url.endswith("/health"):
            return Response({"status": "ok"})
        if request.full_url.endswith("/v1/models"):
            return Response({"data": [{"id": "deepseek-v4-flash"}]})
        raise ConnectionResetError("llama-server aborted during CUDA allocation")

    assert not model_server_ready(
        "127.0.0.1",
        8080,
        "deepseek-v4-flash",
        urlopen=urlopen,
    )


def test_model_health_bypasses_proxy_for_loopback_server(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from scripts.run_llm_led_teacher import model_server_ready

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = (
                {"status": "ok"}
                if self.path == "/health"
                else {"data": [{"id": "deepseek-v4-flash"}]}
            )
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            encoded = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    try:
        assert model_server_ready(
            "127.0.0.1",
            server.server_port,
            "deepseek-v4-flash",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_existing_healthy_model_is_reused_without_starting_process(tmp_path):
    from scripts.run_llm_led_teacher import ensure_model_server

    started = []
    result = ensure_model_server(
        host="127.0.0.1",
        port=8080,
        model_name="deepseek-v4-flash",
        model_script=tmp_path / "model.sh",
        startup_timeout=900,
        model_env={},
        port_check=lambda host, port: True,
        ready_check=lambda host, port, name: True,
        popen_factory=lambda *args, **kwargs: started.append(args),
    )

    assert result is None
    assert started == []


def test_occupied_unhealthy_model_port_fails_closed(tmp_path):
    from scripts.run_llm_led_teacher import ModelStartupError, ensure_model_server

    with pytest.raises(ModelStartupError, match="占用"):
        ensure_model_server(
            host="127.0.0.1",
            port=8080,
            model_name="deepseek-v4-flash",
            model_script=tmp_path / "model.sh",
            startup_timeout=900,
            model_env={},
            port_check=lambda host, port: True,
            ready_check=lambda host, port, name: False,
        )


def test_new_model_process_is_returned_only_after_health_check(tmp_path):
    from scripts.run_llm_led_teacher import ensure_model_server

    script = tmp_path / "model.sh"
    script.write_text("#!/bin/sh\n")
    process = FakeProcess()
    readiness = iter([False, True])
    popen_calls = []

    result = ensure_model_server(
        host="127.0.0.1",
        port=8080,
        model_name="deepseek-v4-flash",
        model_script=script,
        startup_timeout=900,
        model_env={"PORT": "8080"},
        port_check=lambda host, port: False,
        ready_check=lambda host, port, name: next(readiness),
        popen_factory=lambda *args, **kwargs: popen_calls.append((args, kwargs)) or process,
        monotonic=iter([0.0, 1.0, 2.0]).__next__,
        sleep=lambda seconds: None,
    )

    assert result is process
    assert popen_calls[0][0] == ([str(script)],)
    assert popen_calls[0][1]["env"]["PORT"] == "8080"


def test_model_startup_timeout_terminates_owned_process(tmp_path):
    from scripts.run_llm_led_teacher import ModelStartupError, ensure_model_server

    script = tmp_path / "model.sh"
    script.write_text("#!/bin/sh\n")
    process = FakeProcess()

    with pytest.raises(ModelStartupError, match="超时"):
        ensure_model_server(
            host="127.0.0.1",
            port=8080,
            model_name="deepseek-v4-flash",
            model_script=script,
            startup_timeout=1,
            model_env={},
            port_check=lambda host, port: False,
            ready_check=lambda host, port, name: False,
            popen_factory=lambda *args, **kwargs: process,
            monotonic=iter([0.0, 2.0]).__next__,
            sleep=lambda seconds: None,
        )

    assert process.terminated is True


def test_explicit_runtime_env_can_disable_heavy_kv_persistence(tmp_path):
    from scripts.run_llm_led_teacher import build_runtime_env

    env = build_runtime_env(
        {"TEACHER_KV_PERSIST": "0"},
        rag_host="127.0.0.1",
        rag_port=8002,
        kv_cache_dir=tmp_path,
    )

    assert env["TEACHER_KV_PERSIST"] == "0"
