"""Layer 2 episode state store keyed by (student_id, conversation_id)."""

from __future__ import annotations

import json
import os
import uuid
from functools import lru_cache
from pathlib import Path

from app.memory.schemas import EpisodeState

DEFAULT_EPISODE_PATH = "data/runtime/episode_states.json"


def new_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex[:12]}"


class JsonEpisodeStateStore:
    def __init__(self, path: Path | str = DEFAULT_EPISODE_PATH) -> None:
        self._path = Path(path)
        self._episodes: dict[str, EpisodeState] = {}
        self._load()

    def _key(self, student_id: str, conversation_id: str) -> str:
        return f"{student_id}::{conversation_id}"

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._episodes = {
                key: EpisodeState.model_validate(value)
                for key, value in raw.get("episodes", {}).items()
            }
        except (json.JSONDecodeError, ValueError):
            self._episodes = {}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"episodes": {k: v.model_dump(mode="json") for k, v in self._episodes.items()}}
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, student_id: str, conversation_id: str | None = None) -> EpisodeState:
        cid = conversation_id or student_id
        key = self._key(student_id, cid)
        if key not in self._episodes:
            self._episodes[key] = EpisodeState(conversation_id=cid, student_id=student_id)
            self._flush()
        return self._episodes[key]

    def find(self, student_id: str, conversation_id: str | None = None) -> EpisodeState | None:
        """Return an existing episode without creating or persisting a new one."""

        cid = conversation_id or student_id
        return self._episodes.get(self._key(student_id, cid))

    def save(self, episode: EpisodeState) -> None:
        self._episodes[self._key(episode.student_id, episode.conversation_id)] = episode
        self._flush()

    def reset(self, student_id: str, conversation_id: str | None = None) -> EpisodeState:
        cid = conversation_id or new_conversation_id()
        episode = EpisodeState(conversation_id=cid, student_id=student_id)
        self.save(episode)
        return episode

    def list_conversations(self, student_id: str, limit: int = 20) -> list[EpisodeState]:
        episodes = [e for e in self._episodes.values() if e.student_id == student_id]
        episodes.sort(key=lambda e: e.recent_turns[-1].get("timestamp", "") if e.recent_turns else "")
        return list(reversed(episodes[-limit:]))

    def clear_student(self, student_id: str) -> None:
        self._episodes = {k: v for k, v in self._episodes.items() if v.student_id != student_id}
        self._flush()


_stores: dict[str, JsonEpisodeStateStore] = {}


def get_episode_store() -> JsonEpisodeStateStore:
    path = os.getenv("TEACHER_EPISODE_PATH", DEFAULT_EPISODE_PATH)
    if path not in _stores:
        _stores[path] = JsonEpisodeStateStore(path)
    return _stores[path]


def create_conversation(student_id: str) -> EpisodeState:
    return get_episode_store().reset(student_id, new_conversation_id())
