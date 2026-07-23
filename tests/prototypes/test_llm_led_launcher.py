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


def test_service_environment_forces_local_model_and_preserves_explicit_rag_device():
    from scripts.run_llm_led_teacher import build_service_env

    env = build_service_env(
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
    )

    assert env["DEEPSEEK_API_KEY"] == "local"
    assert env["DEEPSEEK_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert env["DEEPSEEK_MODEL"] == "deepseek-v4-flash"
    assert env["DEEPSEEK_TIMEOUT_SECONDS"] == "300.0"
    assert env["TEACHER_RAG_DEVICE"] == "cpu"
    assert env["CHROMA_SERVER"] == "127.0.0.1:8002"


def test_model_environment_contains_verified_large_context_defaults():
    from scripts.run_llm_led_teacher import build_model_env

    env = build_model_env(
        {"PATH": "/usr/bin"},
        model_port=8080,
        ctx_size=524288,
        parallel=1,
        cache_type_k="q8_0",
        cache_type_v="q4_0",
    )

    assert env["PORT"] == "8080"
    assert env["CTX_SIZE"] == "524288"
    assert env["PARALLEL"] == "1"
    assert env["LLAMA_ARG_CACHE_TYPE_K"] == "q8_0"
    assert env["LLAMA_ARG_CACHE_TYPE_V"] == "q4_0"


def test_model_health_requires_health_and_expected_alias():
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
        calls.append((request.full_url, timeout))
        if request.full_url.endswith("/health"):
            return Response({"status": "ok"})
        return Response({"data": [{"id": "deepseek-v4-flash"}]})

    assert model_server_ready(
        "127.0.0.1",
        8080,
        "deepseek-v4-flash",
        urlopen=urlopen,
    )
    assert [url for url, _ in calls] == [
        "http://127.0.0.1:8080/health",
        "http://127.0.0.1:8080/v1/models",
    ]


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
