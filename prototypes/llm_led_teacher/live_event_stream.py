"""Bridge synchronous teacher work to incremental UI event rendering."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Callable, TypeVar


EventT = TypeVar("EventT")
ResultT = TypeVar("ResultT")


def run_with_event_stream(
    task: Callable[[Callable[[EventT], None]], ResultT],
    on_event: Callable[[EventT], None],
    *,
    poll_interval_seconds: float = 0.05,
) -> ResultT:
    """Run blocking work off the UI thread and render its events on the UI thread."""

    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")

    event_queue: Queue[EventT] = Queue()
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="teacher-turn",
    ) as executor:
        future = executor.submit(task, event_queue.put)
        while True:
            try:
                event = event_queue.get(
                    timeout=0 if future.done() else poll_interval_seconds
                )
            except Empty:
                if future.done():
                    break
                continue
            on_event(event)
        return future.result()
