from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from streamlit.testing.v1 import AppTest

from app.auth.local_accounts import LocalAccountStore, get_local_account_store
from app.memory.conversation_archive import (
    LocalConversationStore,
    get_local_conversation_store,
)
from prototypes.llm_led_teacher.logic import PrototypeEpisode
from prototypes.llm_led_teacher import logic, rag_hook
from app.llm.deepseek_client import DeepSeekError


ROOT = Path(__file__).resolve().parents[2]


class LegacyPrototypeEpisode(BaseModel):
    """Session object created before pending messages became part of the model."""

    problem_context: str = ""
    previous_verified_step: str = ""
    turns: list = Field(default_factory=list)


def test_logged_out_page_shows_login_and_registration_without_teacher_ui(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TEACHER_ACCOUNT_DB", str(tmp_path / "users.sqlite3"))
    monkeypatch.setenv(
        "TEACHER_CONVERSATION_DB",
        str(tmp_path / "conversations.sqlite3"),
    )
    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )

    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["登录", "注册"]
    assert any(title.value == "📚 Teacher Agent" for title in app.title)
    assert not any(title.value == "🧪 LLM 主导 Teacher Prototype" for title in app.title)


def test_login_restores_the_users_most_recent_conversation(
    tmp_path,
    monkeypatch,
) -> None:
    account_db = tmp_path / "users.sqlite3"
    conversation_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setenv("TEACHER_ACCOUNT_DB", str(account_db))
    monkeypatch.setenv("TEACHER_CONVERSATION_DB", str(conversation_db))
    monkeypatch.setattr(
        rag_hook,
        "warmup_rag",
        lambda: {"ok": True, "device": "test", "elapsed_seconds": 0.0},
    )
    get_local_account_store.cache_clear()
    get_local_conversation_store.cache_clear()

    account = LocalAccountStore(account_db).register(
        "history_student",
        "history-password",
    )
    episode = PrototypeEpisode(
        problem_context="历史题目：物体受力",
        active_subject="物理",
        active_stage="初中",
    )
    conversation = LocalConversationStore(conversation_db).create(
        account.user_id,
        episode.model_dump(mode="json"),
        title="历史题目",
    )
    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )
    app.run()

    app.text_input[0].set_value("history_student")
    app.text_input[1].set_value("history-password")
    app.button[0].click().run()

    assert not app.exception
    assert app.session_state.prototype_conversation_id == conversation.conversation_id
    assert app.session_state.prototype_episode.problem_context == "历史题目：物体受力"
    assert app.session_state.prototype_episode.active_subject == "物理"
    assert app.session_state.prototype_episode.active_stage == "初中"
    assert app.session_state.prototype_user_id == account.user_id
    assert any(title.value == "🧪 LLM 主导 Teacher Prototype" for title in app.title)


def test_interrupted_generation_keeps_submitted_message_and_offers_retry(
    tmp_path,
    monkeypatch,
) -> None:
    account_db = tmp_path / "users.sqlite3"
    conversation_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setenv("TEACHER_ACCOUNT_DB", str(account_db))
    monkeypatch.setenv("TEACHER_CONVERSATION_DB", str(conversation_db))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        rag_hook,
        "warmup_rag",
        lambda: {"ok": True, "device": "test", "elapsed_seconds": 0.0},
    )
    get_local_account_store.cache_clear()
    get_local_conversation_store.cache_clear()
    account = LocalAccountStore(account_db).register(
        "interrupted_student",
        "history-password",
    )
    observed_pending: list[str] = []

    def interrupted_turn(episode, message, **kwargs):
        summary = LocalConversationStore(conversation_db).list_for_user(account.user_id)[0]
        saved = LocalConversationStore(conversation_db).load(
            account.user_id,
            summary.conversation_id,
        )
        observed_pending.append(saved.episode.get("pending_student_message", ""))
        raise DeepSeekError("模拟连接中断")

    monkeypatch.setattr(logic, "run_turn", interrupted_turn)
    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )
    app.run()
    app.text_input[0].set_value("interrupted_student")
    app.text_input[1].set_value("history-password")
    app.button[0].click().run()

    app.chat_input[0].set_value("这条消息不能丢").run()

    assert not app.exception
    assert observed_pending == ["这条消息不能丢"]
    conversation_id = app.session_state.prototype_conversation_id
    saved = LocalConversationStore(conversation_db).load(
        account.user_id,
        conversation_id,
    )
    assert saved.episode["pending_student_message"] == "这条消息不能丢"
    app.run()
    assert any(button.label == "重新生成上一条消息" for button in app.button)


def test_hot_reload_migrates_legacy_episode_without_losing_pending_message(
    tmp_path,
    monkeypatch,
) -> None:
    account_db = tmp_path / "users.sqlite3"
    conversation_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setenv("TEACHER_ACCOUNT_DB", str(account_db))
    monkeypatch.setenv("TEACHER_CONVERSATION_DB", str(conversation_db))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        rag_hook,
        "warmup_rag",
        lambda: {"ok": True, "device": "test", "elapsed_seconds": 0.0},
    )
    get_local_account_store.cache_clear()
    get_local_conversation_store.cache_clear()
    account = LocalAccountStore(account_db).register(
        "legacy_session_student",
        "history-password",
    )

    def interrupted_turn(episode, message, **kwargs):
        raise DeepSeekError("模拟连接中断")

    monkeypatch.setattr(logic, "run_turn", interrupted_turn)
    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )
    app.run()
    app.text_input[0].set_value("legacy_session_student")
    app.text_input[1].set_value("history-password")
    app.button[0].click().run()

    app.session_state.prototype_episode = LegacyPrototypeEpisode(
        problem_context="旧会话里的题目"
    )
    app.session_state.prototype_session_schema_version = 10
    app.run()

    assert not app.exception
    app.chat_input[0].set_value("旧会话热更新后发送的消息").run()
    assert not app.exception
    saved = LocalConversationStore(conversation_db).load(
        account.user_id,
        app.session_state.prototype_conversation_id,
    )
    assert saved.episode["pending_student_message"] == "旧会话热更新后发送的消息"
