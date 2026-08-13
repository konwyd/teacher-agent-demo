"""Presentation helpers for safely displaying audited teacher responses."""

from __future__ import annotations

from collections.abc import Iterator
import re
from time import sleep as _sleep


_DISPLAY_MATH_BLOCK = re.compile(r"\$\$.*?\$\$", re.DOTALL)


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


def iter_markdown_display_chunks(
    text: str,
    text_chunk_size: int = 1,
    delay_seconds: float = 0.02,
) -> Iterator[str]:
    """Yield text incrementally while keeping each complete display formula whole."""

    if text_chunk_size <= 0:
        raise ValueError("text_chunk_size must be positive")

    cursor = 0
    for match in _DISPLAY_MATH_BLOCK.finditer(text):
        yield from iter_display_chunks(
            text[cursor : match.start()],
            chunk_size=text_chunk_size,
            delay_seconds=delay_seconds,
        )
        yield match.group()
        cursor = match.end()

    yield from iter_display_chunks(
        text[cursor:],
        chunk_size=text_chunk_size,
        delay_seconds=delay_seconds,
    )
