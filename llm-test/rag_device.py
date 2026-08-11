"""Shared compute-device selection for local and online RAG embeddings."""

from __future__ import annotations

import logging
import os
import re

import torch

logger = logging.getLogger(__name__)


def resolve_rag_device(requested: str | None = None) -> str:
    value = (requested or os.environ.get("TEACHER_RAG_DEVICE", "")).strip().lower()
    if not value:
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    match = re.fullmatch(r"cuda(?::(\d+))?", value)
    if match is None:
        logger.warning("Unsupported RAG device %r; falling back to CPU", value)
        return "cpu"
    index = int(match.group(1) or 0)
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        logger.warning("RAG CUDA device %s is unavailable; falling back to CPU", value)
        return "cpu"
    return f"cuda:{index}"


def configure_cpu_threads() -> int:
    raw = os.environ.get("TEACHER_RAG_CPU_THREADS", "4")
    try:
        threads = max(1, int(raw))
    except ValueError:
        threads = 4
    torch.set_num_threads(threads)
    return threads
