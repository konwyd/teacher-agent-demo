"""Warm RAG in-process, then start the Streamlit CLI with shared model caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(
    *,
    app_port: int,
    warmup_fn: Callable[[], dict[str, object]] | None = None,
    cli_main: Callable[[], int] | None = None,
) -> int:
    if warmup_fn is None:
        from prototypes.llm_led_teacher.rag_hook import warmup_rag
        warmup_fn = warmup_rag
    if cli_main is None:
        from streamlit.web.cli import main as streamlit_main
        cli_main = streamlit_main

    status = warmup_fn()
    print(
        "RAG 预热："
        f"device={status.get('device')} "
        f"elapsed={status.get('elapsed_seconds')}s "
        f"ok={status.get('ok')}"
    )
    sys.argv = [
        "streamlit",
        "run",
        "prototypes/llm_led_teacher/streamlit_app.py",
        "--server.address",
        "0.0.0.0",
        "--server.port",
        str(app_port),
        "--server.headless",
        "true",
    ]
    return cli_main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-port", type=int, default=8502)
    args = parser.parse_args()
    raise SystemExit(main(app_port=args.app_port))
