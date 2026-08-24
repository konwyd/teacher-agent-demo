from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field
from streamlit.testing.v1 import AppTest

from app.auth.local_accounts import LocalAccountStore, get_local_account_store
from app.agent.state import VerificationResult, VerificationStatus
from app.memory.conversation_archive import (
    LocalConversationStore,
    get_local_conversation_store,
)
from prototypes.llm_led_teacher.logic import (
    PrototypeEpisode,
    PrototypeTurnRecord,
    TeacherResponse,
)
from prototypes.llm_led_teacher import logic, rag_hook
from prototypes.llm_led_teacher.public_thinking import (
    PublicThinkingEvent,
    PublicThinkingUpdate,
    PublicThinkingKind,
    PublicThinkingSection,
    PublicThinkingStatus,
    ReasoningTraceEvidence,
    ReasoningTraceSource,
    ReasoningTraceStage,
    ReasoningTraceStageId,
)
from prototypes.llm_led_teacher.desk_theme import (
    DESK_THEME_CSS,
    render_reasoning_trace,
)
from prototypes.llm_led_teacher.thinking_state import thinking_key
from app.llm.deepseek_client import DeepSeekError


ROOT = Path(__file__).resolve().parents[2]


def _reasoning_stage(
    *,
    sequence: int = 1,
    value: str = "数学",
    summary: str = "已识别为初中数学概念解释任务。",
) -> ReasoningTraceStage:
    return ReasoningTraceStage(
        sequence=sequence,
        stage_id=ReasoningTraceStageId.REQUEST_ANALYSIS,
        stage="问题分析",
        icon="🔍",
        summary=summary,
        content=[
            ReasoningTraceEvidence(
                label="学科",
                value=value,
                source=ReasoningTraceSource.REQUEST_CLASSIFICATION,
                source_field="RequestClassification.subject",
            )
        ],
        status=PublicThinkingStatus.COMPLETED,
    )


def test_reasoning_trace_renderer_is_continuous_public_summary_and_escaped() -> None:
    rendered = []

    class FakeContainer:
        def markdown(self, value, **kwargs):
            rendered.append((value, kwargs))

    html = render_reasoning_trace(
        [
            _reasoning_stage(
                value="<script>alert(1)</script>",
                summary="<b>不能作为 HTML</b>",
            )
        ],
        live=True,
        container=FakeContainer(),
    )

    assert "<details" not in html
    assert "<summary" not in html
    assert '<div class="desk-reasoning-flow"' in html
    assert '<div class="desk-reasoning-line"' in html
    assert "🧠 AI教学决策过程" in html
    assert "RequestClassification.subject" not in html
    assert "RuntimeStage.name" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" not in html
    assert "&lt;b&gt;不能作为 HTML&lt;/b&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert rendered == [(html, {"unsafe_allow_html": True})]


def test_reasoning_trace_css_has_continuous_flow_states_and_reduced_motion() -> None:
    assert "@keyframes desk-reasoning-enter" in DESK_THEME_CSS
    assert ".desk-reasoning-line[data-status=\"in_progress\"]" in DESK_THEME_CSS
    assert ".desk-reasoning-line[data-status=\"completed\"]" in DESK_THEME_CSS
    assert ".desk-reasoning-line > summary" not in DESK_THEME_CSS
    assert "@media (prefers-reduced-motion: reduce)" in DESK_THEME_CSS


def test_legacy_section_fallback_does_not_render_free_form_reason() -> None:
    from prototypes.llm_led_teacher.desk_theme import render_thinking_sections

    rendered = []

    class FakeContainer:
        def markdown(self, value, **kwargs):
            rendered.append(value)

    class LegacyTurn:
        sections = [
            PublicThinkingSection(
                kind=PublicThinkingKind.RETRIEVAL,
                title="资料",
                summary="已确定本轮资料使用方式。",
                details=["未使用 RAG", "SECRET FREE FORM REASON"],
                status=PublicThinkingStatus.COMPLETED,
            )
        ]

    render_thinking_sections(LegacyTurn(), container=FakeContainer())

    assert "🧠 AI教学决策过程" in rendered[0]
    assert "知识判断" not in rendered[0]
    assert "<details" not in rendered[0]
    assert "已确定本轮资料使用方式。" in rendered[0]
    assert "未使用 RAG" in rendered[0]
    assert "SECRET FREE FORM REASON" not in rendered[0]


class LegacyPrototypeTurnRecord(BaseModel):
    """Turn object created before structured memory and audit fields existed."""

    turn_id: int
    student_message: str
    teacher_output: dict[str, str]
    student_verification: dict[str, str]
    evidence_level: str


class LegacyPrototypeEpisode(BaseModel):
    """Session object created before pending messages became part of the model."""

    problem_context: str = ""
    previous_verified_step: str = ""
    turns: list = Field(default_factory=list)


def _turn(message: str, response: str = "测试回答") -> PrototypeTurnRecord:
    return PrototypeTurnRecord(
        turn_id=1,
        student_message=message,
        teacher_output=TeacherResponse(response=response),
        student_verification=VerificationResult(
            status=VerificationStatus.UNABLE_TO_VERIFY,
            summary="测试中不运行确定性验证。",
            verifier_name="none",
        ),
        evidence_level="unverified",
    )


def _successful_turn(episode, message, **kwargs):
    if not episode.problem_context:
        episode.problem_context = message
    turn = _turn(message, response=f"讲解：{message}")
    turn.turn_id = len(episode.turns) + 1
    episode.turns.append(turn)
    return turn


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
    assert any(
        "desk-auth-page-marker" in markdown.value
        and "数字教师书桌" in markdown.value
        for markdown in app.markdown
    )
    assert any(
        "desk-auth-panel-header" in markdown.value and "欢迎回来" in markdown.value
        for markdown in app.markdown
    )
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
        turns=[_turn("历史题目：物体受力")],
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
    assert app.chat_input
    assert any(button.label == "开启新对话" for button in app.button)


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
    conversation = LocalConversationStore(conversation_db).create(
        account.user_id,
        PrototypeEpisode(
            problem_context="旧会话里的题目",
            turns=[_turn("旧会话里的题目")],
        ).model_dump(mode="json"),
        title="旧会话里的题目",
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
        conversation.conversation_id,
    )
    assert saved.episode["pending_student_message"] == "旧会话热更新后发送的消息"


def test_new_public_thinking_remains_visible_without_status_or_expander(
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
    LocalAccountStore(account_db).register("thinking_student", "history-password")

    public_event = PublicThinkingEvent(
        sequence=1,
        kind=PublicThinkingKind.COMPLETE,
        status=PublicThinkingStatus.COMPLETED,
        summary="思考过程已完成。",
        elapsed_seconds=6.8,
    )
    reasoning_stage = _reasoning_stage()

    def successful_turn(episode, message, **kwargs):
        kwargs["thinking_update_callback"](
            PublicThinkingUpdate(
                event=public_event,
                reasoning_trace=[reasoning_stage],
            )
        )
        turn = PrototypeTurnRecord(
            turn_id=len(episode.turns) + 1,
            student_message=message,
            teacher_output=TeacherResponse(response="这是审计后的回答。"),
            student_verification=VerificationResult(
                status=VerificationStatus.UNABLE_TO_VERIFY,
                summary="当前题型暂无确定性 verifier。",
                verifier_name="none",
            ),
            evidence_level="unverified",
            public_thinking_events=[public_event],
            reasoning_trace=[reasoning_stage],
            backend_total_seconds=6.8,
        )
        episode.turns.append(turn)
        return turn

    monkeypatch.setattr(logic, "run_turn", successful_turn)
    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )
    app.run()
    app.text_input[0].set_value("thinking_student")
    app.text_input[1].set_value("history-password")
    app.button[0].click().run()

    app.chat_input[0].set_value("请解释这道题").run()

    assert not [
        status
        for status in app.status
        if status.label == "🧠 AI教学决策完成（用时 7 秒）"
    ]
    assert not [
        expander
        for expander in app.expander
        if expander.label == "🧠 AI教学决策完成（用时 7 秒）"
    ]
    assert any(
        "AI教学决策过程" in markdown.value
        and "已识别为初中数学概念解释任务。" in markdown.value
        and "RequestClassification.subject" not in markdown.value
        for markdown in app.markdown
    )
    assert any(
        markdown.value == "这是审计后的回答。"
        for markdown in app.markdown
    )

    app.run()

    assert any(
        "AI教学决策过程" in markdown.value
        and "已识别为初中数学概念解释任务。" in markdown.value
        and "RequestClassification.subject" not in markdown.value
        for markdown in app.markdown
    )
    assert any(
        markdown.value == "这是审计后的回答。"
        for markdown in app.markdown
    )

def test_new_chat_remains_an_unpersisted_draft_across_reruns(
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
    account = LocalAccountStore(account_db).register("draft_student", "history-password")
    old = LocalConversationStore(conversation_db).create(
        account.user_id,
        PrototypeEpisode(problem_context="你好", turns=[_turn("你好")]).model_dump(
            mode="json"
        ),
        title="你好",
    )
    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )
    app.run()
    app.text_input[0].set_value("draft_student")
    app.text_input[1].set_value("history-password")
    app.button[0].click().run()

    assert app.session_state.prototype_conversation_id == old.conversation_id
    next(button for button in app.button if button.label == "开启新对话").click().run()

    assert app.session_state.prototype_conversation_id is None
    assert app.session_state.prototype_episode.turns == []
    assert len(LocalConversationStore(conversation_db).list_for_user(account.user_id)) == 1

    app.run()
    app.run()

    assert app.session_state.prototype_conversation_id is None
    assert app.session_state.prototype_episode.turns == []
    assert len(LocalConversationStore(conversation_db).list_for_user(account.user_id)) == 1


def test_first_message_and_follow_up_stay_isolated_in_the_new_conversation(
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
    monkeypatch.setattr(logic, "run_turn", _successful_turn)
    get_local_account_store.cache_clear()
    get_local_conversation_store.cache_clear()
    account = LocalAccountStore(account_db).register("follow_student", "history-password")
    old = LocalConversationStore(conversation_db).create(
        account.user_id,
        PrototypeEpisode(problem_context="你好", turns=[_turn("你好")]).model_dump(
            mode="json"
        ),
        title="你好",
    )
    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )
    app.run()
    app.text_input[0].set_value("follow_student")
    app.text_input[1].set_value("history-password")
    app.button[0].click().run()

    next(button for button in app.button if button.label == "开启新对话").click().run()
    app.chat_input[0].set_value("数学有什么非常毒的定理吗").run()
    new_id = app.session_state.prototype_conversation_id
    app.chat_input[0].set_value("我想让你给我讲一下").run()

    assert new_id is not None
    assert new_id != old.conversation_id
    assert app.session_state.prototype_conversation_id == new_id
    saved = LocalConversationStore(conversation_db).load(account.user_id, new_id)
    assert saved.title == "数学有什么非常毒的定理吗"
    assert [turn["student_message"] for turn in saved.episode["turns"]] == [
        "数学有什么非常毒的定理吗",
        "我想让你给我讲一下",
    ]
    original = LocalConversationStore(conversation_db).load(
        account.user_id,
        old.conversation_id,
    )
    assert [turn["student_message"] for turn in original.episode["turns"]] == ["你好"]


def test_new_conversation_defers_topic_selection_and_passes_memory_identity(
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
    observed: list[tuple[str, str, str, str, str]] = []

    def memory_aware_turn(episode, message, **kwargs):
        memory = episode.cross_session_memory
        identity = kwargs["memory_identity"]
        observed.append(
            (
                episode.problem_context,
                episode.active_subject,
                memory.source_conversation_id if memory else "",
                identity.user_id,
                identity.conversation_id,
            )
        )
        return _successful_turn(episode, message, **kwargs)

    monkeypatch.setattr(logic, "run_turn", memory_aware_turn)
    get_local_account_store.cache_clear()
    get_local_conversation_store.cache_clear()
    account = LocalAccountStore(account_db).register(
        "memory_student",
        "history-password",
    )
    old = LocalConversationStore(conversation_db).create(
        account.user_id,
        PrototypeEpisode(
            problem_context="怎么写好《难忘的一天》这篇作文？",
            active_subject="语文",
            active_stage="小学",
            turns=[_turn("怎么写好《难忘的一天》这篇作文？")],
        ).model_dump(mode="json"),
        title="难忘的一天作文",
    )
    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )
    app.run()
    app.text_input[0].set_value("memory_student")
    app.text_input[1].set_value("history-password")
    app.button[0].click().run()

    next(button for button in app.button if button.label == "开启新对话").click().run()
    app.chat_input[0].set_value("这个作文的开头怎么写？").run()

    new_conversation_id = app.session_state.prototype_conversation_id
    assert new_conversation_id != old.conversation_id
    assert observed == [
        ("", "", "", account.user_id, new_conversation_id)
    ]


def test_history_switch_and_schema_reload_preserve_one_authoritative_id(
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
    account = LocalAccountStore(account_db).register("switch_student", "history-password")
    older = LocalConversationStore(conversation_db).create(
        account.user_id,
        PrototypeEpisode(
            problem_context="旧会话问题",
            turns=[_turn("旧会话消息")],
        ).model_dump(mode="json"),
        title="旧会话问题",
    )
    newer = LocalConversationStore(conversation_db).create(
        account.user_id,
        PrototypeEpisode(
            problem_context="杨辉三角是什么",
            turns=[_turn("杨辉三角是什么")],
        ).model_dump(mode="json"),
        title="杨辉三角是什么",
    )
    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )
    app.run()
    app.text_input[0].set_value("switch_student")
    app.text_input[1].set_value("history-password")
    app.button[0].click().run()

    assert app.session_state.prototype_conversation_id == newer.conversation_id
    next(button for button in app.button if button.label == "旧会话问题").click().run()

    assert app.session_state.prototype_conversation_id == older.conversation_id
    assert any(markdown.value == "旧会话消息" for markdown in app.markdown)
    assert not any(markdown.value == "杨辉三角是什么" for markdown in app.markdown)

    app.session_state.prototype_session_schema_version = 0
    app.run()

    assert app.session_state.prototype_conversation_id == older.conversation_id
    assert any(markdown.value == "旧会话消息" for markdown in app.markdown)


def test_sidebar_renders_grouped_direct_history_and_collapsed_debug_details(
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
    account = LocalAccountStore(account_db).register("sidebar_student", "history-password")
    store = LocalConversationStore(conversation_db)
    records = [
        store.create(
            account.user_id,
            PrototypeEpisode(turns=[_turn("杨辉三角是什么")]).model_dump(mode="json"),
            title="杨辉三角是什么",
        ),
        store.create(
            account.user_id,
            PrototypeEpisode(turns=[_turn("三天前的问题")]).model_dump(mode="json"),
            title="三天前的问题",
        ),
        store.create(
            account.user_id,
            PrototypeEpisode(turns=[_turn("十二天前的问题")]).model_dump(mode="json"),
            title="十二天前的问题",
        ),
        store.create(
            account.user_id,
            PrototypeEpisode(turns=[_turn("四十天前的问题")]).model_dump(mode="json"),
            title="四十天前的问题",
        ),
    ]
    now = datetime.now(timezone.utc)
    timestamps = [now, now - timedelta(days=3), now - timedelta(days=12), now - timedelta(days=40)]
    with sqlite3.connect(conversation_db) as connection:
        for record, timestamp in zip(records, timestamps):
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (timestamp.isoformat(), record.conversation_id),
            )

    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )
    app.run()
    app.text_input[0].set_value("sidebar_student")
    app.text_input[1].set_value("history-password")
    app.button[0].click().run()

    assert not app.exception
    assert any(button.label == "开启新对话" for button in app.button)
    assert not app.selectbox
    assert {heading.value for heading in app.subheader} >= {
        "今天",
        "7 天内",
        "30 天内",
        "更早",
    }
    debug = next(expander for expander in app.expander if expander.label == "调试详情")
    assert debug.proto.expanded is False
    next(button for button in app.button if button.label == "杨辉三角是什么").click().run()

    assert app.session_state.prototype_conversation_id == records[0].conversation_id
    assert any(markdown.value == "杨辉三角是什么" for markdown in app.markdown)


def test_hot_reload_upgrades_legacy_turn_before_rendering_details(
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
    LocalAccountStore(account_db).register("legacy_turn_student", "history-password")

    app = AppTest.from_file(
        str(ROOT / "prototypes" / "llm_led_teacher" / "streamlit_app.py"),
        default_timeout=10,
    )
    app.run()
    app.text_input[0].set_value("legacy_turn_student")
    app.text_input[1].set_value("history-password")
    app.button[0].click().run()

    legacy_turn = LegacyPrototypeTurnRecord(
        turn_id=1,
        student_message="旧版本回合",
        teacher_output={"response": "旧版本回答"},
        student_verification={
            "status": "unable_to_verify",
            "summary": "旧回合没有新版验证字段。",
            "verifier_name": "none",
        },
        evidence_level="unverified",
    )
    app.session_state.prototype_episode = LegacyPrototypeEpisode(
        problem_context="旧版本回合",
        turns=[legacy_turn],
    )
    app.session_state.prototype_session_schema_version = 14
    app.session_state.prototype_conversation_id = None
    app.run()

    assert not app.exception
    upgraded_turn = app.session_state.prototype_episode.turns[0]
    assert upgraded_turn.memory_resolution is None
    assert upgraded_turn.memory_writeback_status == "not_scheduled"
    assert any(
        expander.label == "🔎 决策、工具与验证" for expander in app.expander
    )
