"""Normalize high-confidence textbook math Markdown without touching code."""

from __future__ import annotations

import re


_FENCED_CODE_BLOCK = re.compile(r"(```.*?```)", re.DOTALL)
_BRACKET_LATEX_BLOCK = re.compile(
    r"(?ms)(?P<prefix>^|\n{2,})[ \t]*\["
    r"(?P<body>(?=[^\]]*(?:\\frac|\\sqrt|\\pm|\\times|\\neq|\\le|\\ge|\^|_))[^\]]*?)"
    r"\][ \t]*(?=\n{2,}|\Z)"
)


def normalize_math_markdown(text: str) -> tuple[str, bool]:
    """Convert unambiguous model LaTex delimiters to Markdown math syntax."""

    normalized_segments = []
    for segment in _FENCED_CODE_BLOCK.split(text):
        if segment.startswith("```"):
            normalized_segments.append(segment)
            continue

        segment = re.sub(
            r"\\{1,2}\[(.*?)\\{1,2}\]",
            lambda match: f"$${match.group(1).strip()}$$",
            segment,
            flags=re.DOTALL,
        )
        segment = re.sub(
            r"\\{1,2}\((.*?)\\{1,2}\)",
            lambda match: f"${match.group(1).strip()}$",
            segment,
            flags=re.DOTALL,
        )
        segment = _BRACKET_LATEX_BLOCK.sub(
            lambda match: f"{match.group('prefix')}$${match.group('body').strip()}$$",
            segment,
        )
        normalized_segments.append(segment)

    normalized = "".join(normalized_segments)
    return normalized, normalized != text
