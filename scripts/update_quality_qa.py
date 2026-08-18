"""Run the curated Q&A incremental pipeline once or during a nightly window."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_pipeline.quality_qa_pipeline import (  # noqa: E402
    DeepSeekQualityQaExtractor,
    QualityQaPipeline,
    QualityQaPipelineState,
)
from app.llm.deepseek_client import DeepSeekClient  # noqa: E402
from app.memory.conversation_archive import LocalConversationStore  # noqa: E402
from prototypes.llm_led_teacher import qa_cache  # noqa: E402


def _run_once() -> int:
    archive = LocalConversationStore(
        os.getenv(
            "TEACHER_CONVERSATION_DB",
            "data/runtime/conversations.sqlite3",
        )
    )
    state = QualityQaPipelineState(
        os.getenv(
            "TEACHER_QUALITY_QA_STATE_DB",
            qa_cache.STATE_DB_PATH,
        )
    )
    pipeline = QualityQaPipeline(
        archive,
        DeepSeekQualityQaExtractor(DeepSeekClient()),
        state=state,
    )
    result = pipeline.run()
    print(json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True))
    return 2 if result.failed else 0


def _with_process_lock(handler) -> int:
    lock_path = Path(
        os.getenv(
            "TEACHER_QUALITY_QA_LOCK_FILE",
            "data/runtime/quality_qa_pipeline.lock",
        )
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('{"status":"skipped","reason":"pipeline_already_running"}')
            return 0
        return int(handler())


def _inside_window(now: datetime, start_hour: int, end_hour: int) -> bool:
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= now.hour < end_hour
    return now.hour >= start_hour or now.hour < end_hour


def _run_daemon(
    *,
    start_hour: int,
    end_hour: int,
    poll_seconds: int,
) -> int:
    last_successful_date = ""
    while True:
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        if (
            today != last_successful_date
            and _inside_window(now, start_hour, end_hour)
        ):
            result = _with_process_lock(_run_once)
            if result == 0:
                last_successful_date = today
        time.sleep(max(30, poll_seconds))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="增量提炼历史对话并发布优质问答库新版本"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="常驻并在指定 UTC 时间窗口每天运行一次",
    )
    parser.add_argument("--window-start-hour", type=int, default=0)
    parser.add_argument("--window-end-hour", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if not 0 <= args.window_start_hour <= 23:
        parser.error("--window-start-hour must be between 0 and 23")
    if not 0 <= args.window_end_hour <= 23:
        parser.error("--window-end-hour must be between 0 and 23")
    if args.daemon:
        return _run_daemon(
            start_hour=args.window_start_hour,
            end_hour=args.window_end_hour,
            poll_seconds=args.poll_seconds,
        )
    return _with_process_lock(_run_once)


if __name__ == "__main__":
    raise SystemExit(main())
