import importlib
import sys
from pathlib import Path

import pytest
import torch


LLM_TEST_DIR = Path(__file__).resolve().parents[2] / "llm-test"
if str(LLM_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_TEST_DIR))


def _module():
    try:
        return importlib.import_module("rag_device")
    except ModuleNotFoundError:
        pytest.fail("rag_device module is missing")


def test_explicit_cpu_device_is_respected(monkeypatch):
    monkeypatch.setenv("TEACHER_RAG_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _module().resolve_rag_device() == "cpu"


def test_cuda_zero_is_default_when_cuda_is_available(monkeypatch):
    monkeypatch.delenv("TEACHER_RAG_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    assert _module().resolve_rag_device() == "cuda:0"


def test_requested_cuda_falls_back_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setenv("TEACHER_RAG_DEVICE", "cuda:3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _module().resolve_rag_device() == "cpu"


def test_out_of_range_cuda_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv("TEACHER_RAG_DEVICE", "cuda:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    assert _module().resolve_rag_device() == "cpu"


def test_configure_cpu_threads_uses_environment(monkeypatch):
    calls = []
    monkeypatch.setenv("TEACHER_RAG_CPU_THREADS", "4")
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    assert _module().configure_cpu_threads() == 4
    assert calls == [4]


class _FakeCollection:
    def count(self):
        return 1


class _FakeChromaClient:
    def get_collection(self, name, embedding_function):
        return _FakeCollection()


def _force_cuda_zero(monkeypatch):
    monkeypatch.setenv("TEACHER_RAG_DEVICE", "cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)


def test_local_retrievers_pass_shared_cuda_device_to_embeddings(monkeypatch):
    import rag_retriever
    from chromadb.utils import embedding_functions

    _force_cuda_zero(monkeypatch)
    devices = []

    class FakeEmbeddingFunction:
        def __init__(self, model_name, device="cpu", normalize_embeddings=False):
            devices.append(device)

    monkeypatch.setattr(
        embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    monkeypatch.setattr(rag_retriever, "ensure_chroma_compat", lambda: None)
    monkeypatch.setattr(rag_retriever, "get_chroma_client", lambda: _FakeChromaClient())

    rag_retriever.TextbookRAG()
    rag_retriever.ExerciseRAG()

    assert devices == ["cuda:0", "cuda:0"]


def test_local_retriever_falls_back_to_cpu_if_cuda_model_init_fails(monkeypatch):
    import rag_retriever
    from chromadb.utils import embedding_functions

    _force_cuda_zero(monkeypatch)
    devices = []

    class FakeEmbeddingFunction:
        def __init__(self, model_name, device="cpu", normalize_embeddings=False):
            devices.append(device)
            if device == "cuda:0":
                raise RuntimeError("CUDA init failed")

    monkeypatch.setattr(
        embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    monkeypatch.setattr(rag_retriever, "ensure_chroma_compat", lambda: None)
    monkeypatch.setattr(rag_retriever, "get_chroma_client", lambda: _FakeChromaClient())

    rag_retriever.TextbookRAG()

    assert devices == ["cuda:0", "cpu"]


def test_online_reranker_uses_shared_cuda_device(monkeypatch):
    import online_rag
    import sentence_transformers

    _force_cuda_zero(monkeypatch)
    devices = []

    class FakeModel:
        def __init__(self, model_name, device="cpu"):
            devices.append(device)

        def encode(self, texts, normalize_embeddings=True):
            return [[0.0]] * len(texts)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    online_rag._embed_model = None

    online_rag._get_embed_model()

    assert devices == ["cuda:0"]


def test_online_reranker_falls_back_to_cpu(monkeypatch):
    import online_rag
    import sentence_transformers

    _force_cuda_zero(monkeypatch)
    devices = []

    class FakeModel:
        def __init__(self, model_name, device="cpu"):
            devices.append(device)
            if device == "cuda:0":
                raise RuntimeError("CUDA init failed")

        def encode(self, texts, normalize_embeddings=True):
            return [[0.0]] * len(texts)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    online_rag._embed_model = None

    online_rag._get_embed_model()

    assert devices == ["cuda:0", "cpu"]


def test_same_process_warmup_is_cached_and_never_uses_online(monkeypatch):
    from prototypes.llm_led_teacher import rag_hook

    calls = []

    def fake_retrieve(problem, subject, topk, online_mode, verbose):
        calls.append((problem, subject, topk, online_mode, verbose))
        return ("context", ["ref"], False)

    monkeypatch.setattr(rag_hook, "_original_retrieve_for_exam", fake_retrieve)
    monkeypatch.setattr(rag_hook, "resolve_rag_device", lambda: "cuda:0", raising=False)
    monkeypatch.setattr(rag_hook, "configure_cpu_threads", lambda: 4, raising=False)
    rag_hook.warmup_rag.cache_clear()

    first = rag_hook.warmup_rag()
    second = rag_hook.warmup_rag()

    assert calls == [("一元一次方程", "初中数学", 1, "off", False)]
    assert first == second
    assert first["ok"] is True
    assert first["device"] == "cuda:0"
    assert first["cpu_threads"] == 4


def test_warmup_failure_is_reported_without_raising(monkeypatch):
    from prototypes.llm_led_teacher import rag_hook

    def fail(*args, **kwargs):
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(rag_hook, "_original_retrieve_for_exam", fail)
    monkeypatch.setattr(rag_hook, "resolve_rag_device", lambda: "cuda:0", raising=False)
    monkeypatch.setattr(rag_hook, "configure_cpu_threads", lambda: 4, raising=False)
    rag_hook.warmup_rag.cache_clear()

    result = rag_hook.warmup_rag()

    assert result["ok"] is False
    assert result["device"] == "cuda:0"
    assert result["error"] == "RuntimeError"


def test_launcher_environment_defaults_to_cpu_rag_and_bounded_threads():
    from scripts.run_llm_led_teacher import build_service_env

    env = build_service_env({"PATH": "/usr/bin"}, rag_host="127.0.0.1", rag_port=8002)

    assert env["TEACHER_RAG_DEVICE"] == "cpu"
    assert env["TEACHER_RAG_CPU_THREADS"] == "4"
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["MKL_NUM_THREADS"] == "4"
    assert "127.0.0.1" in env["NO_PROXY"].split(",")
    assert "localhost" in env["NO_PROXY"].split(",")


def test_warmed_streamlit_runs_warmup_before_cli(monkeypatch):
    from scripts import run_warmed_streamlit

    events = []

    def fake_warmup():
        events.append("warmup")
        return {"ok": True, "device": "cuda:0", "elapsed_seconds": 1.25}

    def fake_cli():
        events.append("streamlit")
        return 0

    result = run_warmed_streamlit.main(
        app_port=8502,
        warmup_fn=fake_warmup,
        cli_main=fake_cli,
    )

    assert result == 0
    assert events == ["warmup", "streamlit"]
    assert sys.argv == [
        "streamlit",
        "run",
        "prototypes/llm_led_teacher/streamlit_app.py",
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8502",
        "--server.headless",
        "true",
    ]
