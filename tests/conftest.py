"""pytest 全局配置：默认离线（mock LLM）+ 日志写入临时目录。"""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """pytest 默认不带 API key（走 mock LLM），日志写到临时目录。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("TEACHER_LOG_DIR", str(tmp_path / "logs"))
