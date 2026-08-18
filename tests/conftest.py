"""pytest 全局配置：默认离线（mock LLM）+ 日志写入临时目录。"""

from pathlib import Path

import pytest
from app.memory.learner_memory import reset_student_memory_services



@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """pytest 默认不带 API key（走 mock LLM），日志写到临时目录。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    # An explicit empty value overrides a real .env backend selection. Deleting
    # it is insufficient because DeepSeekSettings also reads .env directly.
    monkeypatch.setenv("TEACHER_LLM_BACKEND", "")
    monkeypatch.setenv("TEACHER_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TEACHER_PROFILE_PATH", str(tmp_path / "student_profiles.json"))
    monkeypatch.setenv("TEACHER_EPISODE_PATH", str(tmp_path / "episode_states.json"))
    monkeypatch.setenv("TEACHER_LEARNING_EVENTS_PATH", str(tmp_path / "learning_events.jsonl"))
    monkeypatch.setenv("TEACHER_THREE_LAYER_PROFILE_PATH", str(tmp_path / "student_profiles_v2.json"))
    monkeypatch.setenv("TEACHER_STUDENT_MEMORY_DB", str(tmp_path / "student_memory.sqlite3"))
    monkeypatch.setenv("TEACHER_CONVERSATION_DB", str(tmp_path / "conversations.sqlite3"))
    yield
    reset_student_memory_services()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify tests by migration directory without editing frozen modules."""

    for item in items:
        path = Path(str(item.path)).as_posix()
        if "/tests/contracts/" in f"/{path}":
            item.add_marker(pytest.mark.safety_contract)
        elif "/tests/behavior/" in f"/{path}":
            item.add_marker(pytest.mark.teaching_behavior)
        else:
            item.add_marker(pytest.mark.legacy_lock)
