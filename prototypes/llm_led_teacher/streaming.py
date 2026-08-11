"""Presentation helpers for safely displaying audited teacher responses."""

from __future__ import annotations

from collections.abc import Iterator
from time import sleep as _sleep


def iter_display_chunks(
    text: str,
    chunk_size: int = 1,
    delay_seconds: float = 0.02,
) -> Iterator[str]:
    """Yield an audited response with a visible typewriter cadence."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]
        has_more_text = start + chunk_size < len(text)
        if has_more_text and delay_seconds > 0:
            _sleep(delay_seconds)
