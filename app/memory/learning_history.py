"""Layer 3 learning history stored as local JSONL events."""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.memory.schemas import LearningEvent

DEFAULT_LEARNING_EVENTS_PATH = "data/runtime/learning_events.jsonl"


def _path() -> Path:
    return Path(os.getenv("TEACHER_LEARNING_EVENTS_PATH", DEFAULT_LEARNING_EVENTS_PATH))


def append_learning_event(student_id: str, event: LearningEvent) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if event.student_id != student_id:
        event = event.model_copy(update={"student_id": student_id})
    with path.open("a", encoding="utf-8") as f:
        f.write(event.model_dump_json() + "\n")


def _read_all() -> list[LearningEvent]:
    path = _path()
    if not path.exists():
        return []
    events: list[LearningEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(LearningEvent.model_validate(json.loads(line)))
    return events


def get_learning_events(student_id: str, limit: int | None = None) -> list[LearningEvent]:
    events = [e for e in _read_all() if e.student_id == student_id]
    return events[-limit:] if limit is not None else events


def get_recent_learning_events(
    student_id: str, concept_id: str | None = None, limit: int = 20
) -> list[LearningEvent]:
    events = get_learning_events(student_id)
    if concept_id is not None:
        events = [e for e in events if e.concept_id == concept_id]
    return events[-limit:]


def summarize_learning_history(student_id: str) -> dict:
    events = get_learning_events(student_id)
    type_counts: dict[str, int] = {}
    concept_counts: dict[str, int] = {}
    for event in events:
        type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
        if event.concept_id:
            concept_counts[event.concept_id] = concept_counts.get(event.concept_id, 0) + 1
    return {
        "student_id": student_id,
        "total_events": len(events),
        "event_type_counts": type_counts,
        "concept_counts": concept_counts,
        "recent_events": [e.model_dump(mode="json") for e in events[-5:]],
    }


def reset_learning_history(student_id: str) -> None:
    path = _path()
    if not path.exists():
        return
    kept = [e for e in _read_all() if e.student_id != student_id]
    path.write_text(
        "\n".join(e.model_dump_json() for e in kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )
