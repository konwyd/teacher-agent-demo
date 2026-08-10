"""Optional compute-device selection for semantic retrieval embeddings."""

from __future__ import annotations

import os
import re
from typing import Any


def _load_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def resolve_rag_device(
    requested: str | None = None,
    *,
    torch_module: Any | None = None,
) -> str:
    runtime = torch_module if torch_module is not None else _load_torch()
    value = (requested or os.environ.get("TEACHER_RAG_DEVICE", "")).strip().lower()
    if value == "cpu":
        return "cpu"
    if runtime is None:
        return "cpu"
    if not value:
        return "cuda:0" if runtime.cuda.is_available() else "cpu"
    match = re.fullmatch(r"cuda(?::(\d+))?", value)
    if match is None:
        return "cpu"
    index = int(match.group(1) or 0)
    if not runtime.cuda.is_available() or index >= runtime.cuda.device_count():
        return "cpu"
    return f"cuda:{index}"


def configure_cpu_threads(
    raw: str | None = None,
    *,
    torch_module: Any | None = None,
) -> int:
    value = raw if raw is not None else os.environ.get("TEACHER_RAG_CPU_THREADS", "4")
    try:
        threads = max(1, int(value))
    except (TypeError, ValueError):
        threads = 4
    runtime = torch_module if torch_module is not None else _load_torch()
    if runtime is not None:
        runtime.set_num_threads(threads)
    return threads
