"""Normalize high-confidence textbook math Markdown without touching code."""

from __future__ import annotations

import re


_BRACKET_LATEX_BLOCK = re.compile(
    r"(?ms)(?P<prefix>^|\n{2,})[ \t]*\["
    r"(?P<body>(?=[^\]]*(?:\\frac|\\sqrt|\\pm|\\times|\\neq|\\le|\\ge|\^|_))[^\]]*?)"
    r"\][ \t]*(?=\n{2,}|\Z)"
)


def _iter_fenced_code_segments(text: str):
    """Yield segments paired with whether they are fenced code."""

    text_lines = []
    code_lines = []
    opening_marker = None
    for line in text.splitlines(keepends=True):
        if opening_marker is None:
            opening = re.match(
                r"[ \t]{0,3}(?P<marker>`{3,}|~{3,})[^\r\n]*(?:\r?\n|$)",
                line,
            )
            if opening is None:
                text_lines.append(line)
                continue

            if text_lines:
                yield "".join(text_lines), False
                text_lines = []
            opening_marker = opening.group("marker")
            code_lines = [line]
            continue

        code_lines.append(line)
        closing = re.match(
            rf"[ \t]{{0,3}}{re.escape(opening_marker[0])}{{{len(opening_marker)},}}[ \t]*(?:\r?\n|$)",
            line,
        )
        if closing is not None:
            yield "".join(code_lines), True
            code_lines = []
            opening_marker = None

    if code_lines:
        yield "".join(code_lines), True
    elif text_lines:
        yield "".join(text_lines), False


def normalize_math_markdown(text: str) -> tuple[str, bool]:
    """Convert unambiguous model LaTex delimiters to Markdown math syntax."""

    normalized_segments = []
    for segment, is_fenced_code in _iter_fenced_code_segments(text):
        if is_fenced_code:
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
