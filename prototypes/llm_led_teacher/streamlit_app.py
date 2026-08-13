"""Standalone Streamlit UI for the throwaway LLM-led teacher prototype."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototypes.llm_led_teacher.desk_theme import (
    DESK_THEME_CSS,
    render_brand_header,
    render_thinking_sections as desk_render_thinking_sections,
)

from app.auth.local_accounts import AccountError, AuthenticationError, get_local_account_store
from app.llm.deepseek_client import DeepSeekError, is_configured
from app.llm.kv_cache import KvCacheIdentity
from app.memory.conversation_archive import (
    ConversationArchiveError,
    ConversationRecord,
    get_local_conversation_store,
)
from prototypes.llm_led_teacher.context_usage import (
    build_context_usage_display,
    build_kv_persistence_display,
)
from prototypes.llm_led_teacher.conversation_ui import (
    build_conversation_history,
    episode_has_content,
)
from prototypes.llm_led_teacher.logic import PrototypeEpisode, PrototypeTurnRecord, run_turn
from prototypes.llm_led_teacher.live_progress import LiveThinkingPanel
from prototypes.llm_led_teacher.public_thinking import format_public_thinking_event
from prototypes.llm_led_teacher.rag_hook import warmup_rag
from prototypes.llm_led_teacher.streaming import iter_display_chunks
from prototypes.llm_led_teacher.thinking_state import (
    ThinkingPanelState,
    begin_thinking,
    complete_thinking,
    fail_thinking,
    set_user_expanded,
    thinking_key,
)


_SESSION_SCHEMA_VERSION = 12


st.set_page_config(page_title="Teacher Agent", page_icon="🧠", layout="wide")

_DEEPSEEK_STYLE = """
<style>
:root {
    --teacher-blue: #4d6bfe;
    --teacher-blue-soft: #e9efff;
    --teacher-sidebar: #f7f8fa;
    --teacher-text-muted: #7a8194;
}
[data-testid="stToolbar"] {
    visibility: hidden;
    height: 0;
}
[data-testid="stSidebar"] {
    background: var(--teacher-sidebar);
    border-right: 1px solid rgba(0, 0, 0, 0.045);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.1rem;
}
[data-testid="stSidebar"] .stButton > button {
    border: 0;
    border-radius: 0.65rem;
    box-shadow: none;
    justify-content: flex-start;
    min-height: 2.35rem;
    padding: 0.42rem 0.7rem;
    text-align: left;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(77, 107, 254, 0.08);
    color: #2746d8;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--teacher-blue-soft);
    color: #2746d8;
}
[data-testid="stSidebar"] h3 {
    color: var(--teacher-text-muted);
    font-size: 0.77rem;
    font-weight: 500;
    margin: 1rem 0 0.2rem;
}
[data-testid="stMainBlockContainer"] {
    max-width: 980px;
    padding-bottom: 9rem;
    padding-top: 1.4rem;
}
.teacher-page-title {
    color: #242735;
    font-size: 1.05rem;
    font-weight: 650;
    line-height: 1.4;
    margin: 0.1rem 0 0;
}
.teacher-page-subtitle {
    color: var(--teacher-text-muted);
    font-size: 0.78rem;
    margin: 0.1rem 0 1.6rem;
}
[data-testid="stChatMessage"] {
    background: transparent;
    border: 0;
    padding: 0.6rem 0;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #f1f4fb;
    border-radius: 1rem;
    margin-left: auto;
    max-width: 76%;
    padding: 0.65rem 0.9rem;
    width: fit-content;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageAvatarUser"] {
    display: none;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
    padding: 0;
}
[data-testid="stBottomBlockContainer"] {
    background: rgba(255, 255, 255, 0.96);
    margin: 0 auto;
    max-width: 980px;
    padding-bottom: 0.8rem;
}
[data-testid="stChatInput"] {
    border-radius: 1rem;
    box-shadow: 0 8px 28px rgba(31, 35, 48, 0.08);
}
@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        max-width: 90%;
    }
}
</style>
"""
st.markdown(DESK_THEME_CSS, unsafe_allow_html=True)

_PENDING_STUDENT_MESSAGE_KEY = "prototype_pending_student_message"


def _get_pending_student_message(current_episode: object) -> str:
    """Read pending text from both current and pre-hot-reload episode objects."""

    model_value = getattr(current_episode, "pending_student_message", None)
    if model_value is not None:
        return str(model_value)
    return str(st.session_state.get(_PENDING_STUDENT_MESSAGE_KEY, ""))


def _set_pending_student_message(current_episode: object, message: str) -> None:
    """Keep pending text writable when Streamlit retains an older Pydantic model."""

    st.session_state[_PENDING_STUDENT_MESSAGE_KEY] = message
    model_fields = getattr(type(current_episode), "model_fields", {})
    if "pending_student_message" in model_fields:
        setattr(current_episode, "pending_student_message", message)


def _episode_payload(current_episode: PrototypeEpisode) -> dict[str, object]:
    payload = current_episode.model_dump(mode="json")
    payload["pending_student_message"] = _get_pending_student_message(current_episode)
    return payload


def _activate_conversation(record: ConversationRecord) -> None:
    st.session_state.prototype_episode = PrototypeEpisode.model_validate(record.episode)
    st.session_state[_PENDING_STUDENT_MESSAGE_KEY] = str(
        record.episode.get("pending_student_message", "") or ""
    )
    st.session_state.prototype_conversation_id = record.conversation_id
    st.session_state.prototype_user_id = record.user_id
    st.session_state.prototype_session_schema_version = _SESSION_SCHEMA_VERSION


def _activate_draft(user_id: str) -> None:
    st.session_state.prototype_episode = PrototypeEpisode()
    st.session_state[_PENDING_STUDENT_MESSAGE_KEY] = ""
    st.session_state.prototype_conversation_id = None
    st.session_state.prototype_user_id = user_id
    st.session_state.prototype_session_schema_version = _SESSION_SCHEMA_VERSION


def _visible_conversation_records(user_id: str) -> list[ConversationRecord]:
    conversation_store = get_local_conversation_store()
    records = [
        conversation_store.load(user_id, summary.conversation_id)
        for summary in conversation_store.list_for_user(user_id)
    ]
    return [record for record in records if episode_has_content(record.episode)]


def _open_latest_or_draft(user_id: str) -> None:
    records = _visible_conversation_records(user_id)
    if records:
        _activate_conversation(records[0])
        return
    _activate_draft(user_id)


def _restore_current_or_draft(user_id: str) -> None:
    conversation_id = st.session_state.get("prototype_conversation_id")
    if conversation_id:
        try:
            record = get_local_conversation_store().load(user_id, conversation_id)
        except ConversationArchiveError:
            st.session_state.prototype_conversation_restore_error = (
                "原对话无法恢复，已进入新的空白对话。"
            )
        else:
            _activate_conversation(record)
            return
    _activate_draft(user_id)


def _start_authenticated_session(user_id: str, username: str) -> None:
    st.session_state.authenticated_user = {
        "user_id": user_id,
        "username": username,
    }
    _open_latest_or_draft(user_id)


class ConversationStateError(RuntimeError):
    """Raised when a turn tries to save into a no-longer-active conversation."""


def _conversation_title(current_episode: PrototypeEpisode, fallback: str = "") -> str:
    if current_episode.turns:
        title = current_episode.turns[0].student_message
    else:
        title = _get_pending_student_message(current_episode) or fallback
    normalized = " ".join(str(title or "").split()).strip()
    return (normalized or "新对话")[:80]


def _assert_active_target(target_conversation_id: str) -> None:
    if st.session_state.get("prototype_conversation_id") != target_conversation_id:
        raise ConversationStateError("会话状态已变化，请重试。")


def _ensure_target_conversation(
    current_episode: PrototypeEpisode,
    first_message: str,
) -> str:
    conversation_id = st.session_state.get("prototype_conversation_id")
    if conversation_id:
        return str(conversation_id)
    user_id = st.session_state.authenticated_user["user_id"]
    record = get_local_conversation_store().create(
        user_id,
        _episode_payload(current_episode),
        title=first_message,
    )
    st.session_state.prototype_conversation_id = record.conversation_id
    st.session_state.prototype_user_id = user_id
    st.session_state.prototype_session_schema_version = _SESSION_SCHEMA_VERSION
    return record.conversation_id


def _save_conversation(
    target_conversation_id: str,
    current_episode: PrototypeEpisode,
    *,
    fallback_title: str = "",
) -> None:
    _assert_active_target(target_conversation_id)
    get_local_conversation_store().save(
        st.session_state.authenticated_user["user_id"],
        target_conversation_id,
        _episode_payload(current_episode),
        title=_conversation_title(current_episode, fallback_title),
    )


def _activate_conversation_by_id(conversation_id: str) -> None:
    user_id = st.session_state.authenticated_user["user_id"]
    try:
        record = get_local_conversation_store().load(user_id, conversation_id)
    except ConversationArchiveError as exc:
        st.session_state.prototype_conversation_switch_error = str(exc)
    else:
        _activate_conversation(record)


def _render_auth_gate() -> None:
    st.title("📚 Teacher Agent")
    st.caption("登录后进入你的专属学习空间")
    login_tab, register_tab = st.tabs(["登录", "注册"])
    account_store = get_local_account_store()

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            submitted = st.form_submit_button("登录", use_container_width=True)
        if submitted:
            try:
                account = account_store.authenticate(username, password)
            except AuthenticationError as exc:
                st.error(str(exc))
            else:
                _start_authenticated_session(account.user_id, account.username)
                st.rerun()

    with register_tab:
        with st.form("register_form"):
            username = st.text_input(
                "设置用户名",
                help="2–32 个字符，可使用中文、字母、数字、下划线、短横线或点。",
                key="register_username",
            )
            password = st.text_input(
                "设置密码",
                type="password",
                help="至少 8 个字符。密码只保存加盐哈希，不保存明文。",
                key="register_password",
            )
            password_confirm = st.text_input(
                "再次输入密码",
                type="password",
                key="register_password_confirm",
            )
            submitted = st.form_submit_button("注册并登录", use_container_width=True)
        if submitted:
            if password != password_confirm:
                st.error("两次输入的密码不一致。")
            else:
                try:
                    account = account_store.register(username, password)
                except AccountError as exc:
                    st.error(str(exc))
                else:
                    _start_authenticated_session(account.user_id, account.username)
                    st.rerun()


if "authenticated_user" not in st.session_state:
    _render_auth_gate()
    st.stop()

authenticated_user = st.session_state.authenticated_user
warmup_status = warmup_rag()

if (
    st.session_state.get("prototype_session_schema_version") != _SESSION_SCHEMA_VERSION
    or st.session_state.get("prototype_user_id") != authenticated_user["user_id"]
):
    if st.session_state.get("prototype_user_id") != authenticated_user["user_id"]:
        _open_latest_or_draft(authenticated_user["user_id"])
    else:
        _restore_current_or_draft(authenticated_user["user_id"])
elif "prototype_episode" not in st.session_state:
    _restore_current_or_draft(authenticated_user["user_id"])

episode: PrototypeEpisode = st.session_state.prototype_episode

with st.sidebar:
    st.markdown(
        """
        <div class="desk-directory">
          <div class="desk-directory-kicker">Teacher Agent</div>
          <div class="desk-directory-title">数字教师书桌 · 新建一课</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    current_conversation_id = st.session_state.get("prototype_conversation_id")
    st.button(
        "开启新对话",
        help="开启一次全新的课程对话",
        use_container_width=True,
        type="secondary",
        on_click=_activate_draft,
        args=(authenticated_user["user_id"],),
    )
    conversation_records = _visible_conversation_records(
        authenticated_user["user_id"]
    )
    history_items = build_conversation_history(
        conversation_records,
        now=datetime.now(timezone.utc),
        current_conversation_id=current_conversation_id,
    )
    last_group = ""
    for item in history_items:
        if item.group != last_group:
            st.subheader(item.group)
            last_group = item.group
        st.button(
            item.title,
            key=f"history_{item.conversation_id}",
            type=(
                "primary"
                if item.conversation_id == current_conversation_id
                else "tertiary"
            ),
            use_container_width=True,
            on_click=_activate_conversation_by_id,
            args=(item.conversation_id,),
        )
        st.caption(item.timestamp_label)
    st.divider()
    with st.expander("调试详情", expanded=False):
        st.caption(f"学生 ID：{authenticated_user['user_id']}")
        st.caption(f"当前对话：{current_conversation_id or '尚未保存'}")
        if warmup_status["ok"]:
            st.write(
                "**RAG 状态**",
                f"{warmup_status['device']} · 预热 {warmup_status['elapsed_seconds']}s",
            )
        else:
            st.warning(f"RAG 预热失败，已降级：{warmup_status['error']}")
        st.write("**active problem**", episode.problem_context or "尚未开始")
        st.write("**回合数**", len(episode.turns))
    st.caption(f"你好，{authenticated_user['username']}")
    if st.button("退出登录", use_container_width=True, type="tertiary"):
        for key in (
            "authenticated_user",
            "prototype_episode",
            "prototype_conversation_id",
            "prototype_user_id",
            "prototype_session_schema_version",
            _PENDING_STUDENT_MESSAGE_KEY,
        ):
            st.session_state.pop(key, None)
        st.rerun()

if st.session_state.pop("prototype_conversation_switch_error", ""):
    st.error("历史对话加载失败，已保留当前对话。")
restore_error = st.session_state.pop("prototype_conversation_restore_error", "")
if restore_error:
    st.warning(restore_error)

render_brand_header()
st.caption("备课批注 · 题意 / 写法 / 资料 / 成稿")
if not is_configured():
    st.warning(
        "当前进程没有配置 DEEPSEEK_API_KEY。请在启动 Streamlit 的同一 PowerShell "
        "窗口中设置环境变量后重新启动。"
    )


_TIMING_LABELS = {
    "observing": "Observer 判断",
    "retrieving": "RAG 检索",
    "generating": "教师初稿",
    "auditing": "确定性审计",
    "repairing": "安全修正",
}


def _format_token_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


_CONTEXT_USAGE_STYLE = """
<style>
.st-key-teacher_composer_actions > div[data-testid="stVerticalBlock"] {
    gap: 8px;
    min-width: 0;
}
.st-key-teacher_composer_controls [data-testid="stHorizontalBlock"] {
    align-items: center;
    gap: 10px;
}
.st-key-context_usage_control {
    position: static;
    width: max-content !important;
}
.st-key-context_usage_control [data-testid="stPopover"] > button {
    background: transparent;
    border: 0;
    box-shadow: none;
    color: rgba(49, 51, 63, 0.72);
    padding: 0.3rem 0.45rem;
}
.st-key-context_usage_control [data-testid="stPopover"] > button:hover {
    background: rgba(151, 166, 195, 0.12);
    color: rgb(49, 51, 63);
}
@media (max-width: 640px) {
    .st-key-teacher_composer_actions > div[data-testid="stVerticalBlock"] {
        gap: 6px;
    }
    .st-key-context_usage_control [data-testid="stPopover"] > button p {
        display: none;
    }
    .st-key-context_usage_control [data-testid="stPopover"] > button::after {
        content: "上下文";
    }
}
</style>
"""


def render_context_usage(current_episode: PrototypeEpisode) -> None:
    """Render the conversation context meter inside Streamlit's bottom root."""

    metadata = (
        current_episode.turns[-1].model_generation_metadata
        if current_episode.turns
        else {}
    )
    configured_window = int(os.getenv("DEEPSEEK_CONTEXT_WINDOW_TOKENS", "131072"))
    display = build_context_usage_display(
        metadata,
        current_window_tokens=configured_window,
    )
    label = (
        f"{display.usage_ratio:.1%} · {_format_token_count(display.used_tokens)} / "
        f"{_format_token_count(display.current_window_tokens)}"
    )
    with st.popover(
        label,
        icon=":material/data_usage:",
        help="本轮最后一次模型请求占用 / 当前模型上限",
        type="tertiary",
        key="context_usage_popover",
    ):
        st.markdown("#### 上下文占用")
        st.progress(min(max(display.usage_ratio, 0.0), 1.0))
        st.write("**本轮请求占用**", _format_token_count(display.used_tokens))
        st.write(
            "**当前模型上限**",
            _format_token_count(display.current_window_tokens),
        )
        if display.historical_window_tokens is None:
            historical_label = "暂无记录"
        else:
            historical_label = _format_token_count(display.historical_window_tokens)
            if display.historical_limit_changed:
                historical_label += "（旧配置记录）"
        st.write("**该对话历史上限**", historical_label)
        st.caption(
            "占用值来自本轮最后一次模型请求，不是整段对话累计 Token，"
            "也不是 GPU 显存占用。"
        )


def render_turn_token_usage(turn: PrototypeTurnRecord) -> None:
    """Show token and KV-cache counters belonging only to this reply turn."""

    metadata = turn.model_generation_metadata
    if not metadata:
        return

    total_tokens = int(metadata.get("total_tokens", 0) or 0)
    prompt_tokens = int(metadata.get("prompt_tokens", 0) or 0)
    completion_tokens = int(metadata.get("completion_tokens", 0) or 0)
    reasoning_tokens = int(metadata.get("reasoning_tokens", 0) or 0)
    response_tokens = int(
        metadata.get(
            "response_tokens",
            max(completion_tokens - reasoning_tokens, 0),
        )
        or 0
    )
    cache_hits = int(metadata.get("prompt_cache_hit_tokens", 0) or 0)
    cache_misses = int(metadata.get("prompt_cache_miss_tokens", 0) or 0)
    cache_denominator = cache_hits + cache_misses
    cache_rate = cache_hits / cache_denominator if cache_denominator else 0.0
    cache_label = (
        f"缓存 {cache_rate:.1%}"
        if metadata.get("cache_usage_available")
        else "缓存暂无数据"
    )

    with st.popover(f"🪙 Tokens: {total_tokens:,}　⚡ {cache_label}"):
        st.markdown("#### 本轮 Token 消耗")
        st.caption(
            f"仅统计这条教师回复对应的一轮，共 "
            f"{int(metadata.get('llm_call_count', 1) or 1)} 次模型调用。"
        )
        st.write("**输入**", f"{prompt_tokens:,}")
        st.write("　🟩 缓存命中", f"{cache_hits:,}")
        st.write("　🟥 缓存未命中", f"{cache_misses:,}")
        if metadata.get("cache_write_tokens") is not None:
            st.write("　🟨 缓存写入", f"{int(metadata['cache_write_tokens'] or 0):,}")
        st.divider()
        st.write("**输出**", f"{completion_tokens:,}")
        st.write("　思考过程", f"{reasoning_tokens:,}")
        st.write("　回复内容", f"{response_tokens:,}")
        st.divider()
        st.markdown("#### ⚡ 本轮 KV 缓存命中")
        if metadata.get("cache_usage_available"):
            if cache_hits > 0:
                st.success(f"命中 {cache_hits:,} token（{cache_rate:.1%}）")
            else:
                st.info("本轮未命中。新对话的首轮请求通常不会命中。")
            st.progress(min(max(cache_rate, 0.0), 1.0))
        else:
            st.caption("模型服务没有返回这一轮的缓存统计。")
        st.caption(
            "命中指显存中的 KV/prompt cache 前缀复用，不代表模型权重是否已装入显存。"
        )
        kv_display = build_kv_persistence_display(metadata)
        if kv_display.enabled:
            st.divider()
            st.markdown("#### 💾 KV 持久化")
            st.write("**llama-server slot**", kv_display.slot_id)
            st.write("**恢复状态**", kv_display.restore_status)
            st.write("**保存状态**", kv_display.save_status)
            if kv_display.filename:
                st.write("**本地文件**", kv_display.filename)
            if kv_display.directory:
                st.caption(f"保存目录：{kv_display.directory}")
            if kv_display.error:
                st.warning(f"KV 已降级，不影响本轮回答：{kv_display.error}")


def render_turn_details(turn: PrototypeTurnRecord) -> None:
    with st.expander("验证详情"):
        if turn.stage_timings_seconds:
            st.markdown("### ⏱ 处理耗时")
            timing_rows = [
                {
                    "阶段": _TIMING_LABELS.get(stage, stage),
                    "耗时": f"{seconds:.3f} 秒",
                }
                for stage, seconds in turn.stage_timings_seconds.items()
            ]
            if turn.display_seconds > 0:
                timing_rows.append(
                    {"阶段": "审计后逐字展示", "耗时": f"{turn.display_seconds:.3f} 秒"}
                )
            st.table(timing_rows)
            total_seconds = turn.backend_total_seconds + turn.display_seconds
            st.metric(
                "本轮总耗时",
                f"{total_seconds:.3f} 秒",
                help="包括后端处理和审计后的逐字符展示。",
            )

        st.markdown("### 🧠 决策与审计摘要")
        st.caption(
            "这里展示可复核的决策依据和审计结果；不展示模型私有的逐字思维链。"
        )
        st.write("**观察器原始输出**", turn.observer_raw_guidance.model_dump(mode="json"))
        st.write("**规范化后的有效 guidance**", turn.turn_guidance.model_dump(mode="json"))
        if turn.observer_error:
            st.warning(f"观察器已降级，但主回答未被阻断：{turn.observer_error}")
        st.write("**证据等级**", turn.evidence_level)
        if turn.evidence_level == "unverified":
            st.info("当前没有可用的确定性验证证据；教师回答仍正常展示。")
        st.write("**学生步骤验证**", turn.student_verification.status.value)
        st.write("**verifier**", turn.student_verification.verifier_name or "none")
        st.write("**验证摘要**", turn.student_verification.summary)
        st.write("**自动提取的方程**", turn.extracted_equations or [])
        st.write(
            "**生成摘要**",
            {
                "初稿字符数": turn.draft_character_count,
                "最终字符数": turn.final_character_count,
                "是否触发修正": turn.repair_performed,
            },
        )
        st.markdown("#### 模型处理摘要")
        st.caption(
            "原始逐字思维链不会展示；下面展示可复核的处理步骤、运行状态与 token 统计。"
        )
        model_generation_metadata = turn.model_generation_metadata
        processing_steps = [
            f"识别请求并确定节奏：{turn.turn_guidance.pacing.value}",
            f"检索参考：{turn.rag_source or 'none'}，命中 {len(turn.rag_hits)} 条",
            f"审计回答：提取 {len(turn.extracted_equations)} 个方程，"
            f"{'执行过修正' if turn.repair_performed else '无需修正'}",
        ]
        for step in processing_steps:
            st.markdown(f"- {step}")
        if model_generation_metadata:
            st.write(
                "**生成运行信息**",
                {
                    "生成硬上限": "未设置",
                    "结束原因": model_generation_metadata.get("finish_reason") or "未提供",
                    "输出 token": model_generation_metadata.get("completion_tokens", 0),
                    "推理 token": model_generation_metadata.get("reasoning_tokens", 0),
                    "模型提供了推理通道": bool(
                        model_generation_metadata.get("reasoning_available", False)
                    ),
                },
            )
        else:
            st.caption("当前模型接口没有返回 token 使用量或结束原因。")


        # RAG 信息：用指标 + 命中卡片呈现，保留相似度和可读的 Markdown/LaTeX。
        st.divider()
        st.markdown("### 📚 RAG 检索")
        if turn.rag_online_triggered:
            online_status = turn.rag_online_status
            if online_status == "used":
                st.success("🌐 本地命中不足，已触发并使用联网搜索")
            elif online_status == "unavailable_missing_key":
                st.warning(
                    "🌐 已触发联网搜索，但当前进程未配置 TAVILY_API_KEY，因此只保留本地结果。"
                )
            elif online_status == "no_results":
                st.warning("🌐 已触发联网搜索，但没有返回可用结果。")
            else:
                st.warning(f"🌐 已触发联网搜索，当前状态：{online_status}")
            if turn.rag_online_reason:
                st.caption(f"触发原因：{turn.rag_online_reason}")
        else:
            st.caption("本轮未触发联网搜索。")
        if turn.rag_local_threshold > 0:
            st.caption(
                f"本地最高相似度 {turn.rag_local_top_score:.0%}；"
                f"联网判定阈值 {turn.rag_local_threshold:.0%}。"
            )
        if turn.rag_hits:
            best_similarity = max(float(hit.get("similarity", 0.0)) for hit in turn.rag_hits)
            c1, c2, c3 = st.columns(3)
            c1.metric("检索来源", turn.rag_source or "local")
            c2.metric("命中片段", str(len(turn.rag_hits)))
            c3.metric("最高相似度", f"{best_similarity:.0%}")
            if turn.rag_exact_match:
                st.success("🎯 找到同题或高度相似题目")
            for index, hit in enumerate(turn.rag_hits, start=1):
                similarity = float(hit.get("similarity", 0.0))
                title = hit.get("reference", "检索片段")
                with st.expander(f"片段 {index} · 相似度 {similarity:.0%} · {title}"):
                    document = hit.get("document", "")
                    if document:
                        st.markdown(document)
                    else:
                        st.caption("该引用的正文已合并到下方实际 RAG 上下文。")
            with st.expander("📋 传给模型的实际 RAG 上下文"):
                st.markdown(turn.rag_text)
            if turn.rag_refs:
                st.caption("来源：" + "；".join(turn.rag_refs))
        else:
            st.info(f"暂未检索到可用参考（状态：{turn.rag_source or 'none'}）")
        if turn.claim_audits:
            st.write(
                "**方程检查**",
                [
                    {
                        "claim": audit.claim,
                        "status": audit.verification.status.value,
                        "summary": audit.verification.summary,
                    }
                    for audit in turn.claim_audits
                ],
            )
        st.write("**触发的修正**", turn.repair_reasons or [])
        st.write("**修正后仍有警告**", turn.remaining_warnings or [])


def render_public_thinking(
    turn: PrototypeTurnRecord,
    *,
    current_conversation_id: str,
) -> None:
    """Render saved public events without exposing internal trace payloads."""

    if not turn.public_thinking_events and not turn.sections:
        return
    state_key = thinking_key(current_conversation_id, turn.turn_id)
    panel_state = st.session_state.get(state_key)
    if not isinstance(panel_state, ThinkingPanelState):
        panel_state = complete_thinking(None)
        st.session_state[state_key] = panel_state
    expanded_key = f"{state_key}:expanded"
    if expanded_key in st.session_state:
        panel_state = set_user_expanded(
            panel_state,
            bool(st.session_state[expanded_key]),
        )
        st.session_state[state_key] = panel_state
    seconds = max(round(turn.backend_total_seconds), 0)
    with st.expander(
        f"🧠 已思考（用时 {seconds} 秒）",
        expanded=panel_state.expanded,
        key=expanded_key,
        on_change=_remember_thinking_expansion,
        args=(state_key, expanded_key),
    ):
        render_thinking_sections(turn, expanded=panel_state.expanded)
        for event in turn.public_thinking_events:
            st.markdown(format_public_thinking_event(event))


def _remember_thinking_expansion(state_key: str, expanded_key: str) -> None:
    """Persist a historical panel choice without changing its phase."""

    panel_state = st.session_state.get(state_key)
    if not isinstance(panel_state, ThinkingPanelState):
        panel_state = complete_thinking(None)
    st.session_state[state_key] = set_user_expanded(
        panel_state,
        bool(st.session_state[expanded_key]),
    )


def render_thinking_sections(
    turn: object,
    expanded: bool = True,
    *,
    container: object | None = None,
) -> None:
    """Keep the page-level rendering contract explicit for the desk rail."""

    _ = turn.sections
    # desk_theme renders escape(section.summary) and escape(detail) for every public slot.
    return desk_render_thinking_sections(
        turn,
        expanded=expanded,
        container=container,
    )


def render_desk_turn(turn: PrototypeTurnRecord, *, conversation_id: str) -> None:
    """Render one completed turn in the same desk layout for history and live output."""

    with st.container(key=f"desk_paper_sheet_{turn.turn_id}"):
        st.markdown(f'<div class="desk-user-note">{escape(turn.student_message)}</div>', unsafe_allow_html=True)
        rail, answer = st.columns([1, 2.25], gap="large")
        with rail:
            render_public_thinking(turn, current_conversation_id=conversation_id)
        with answer:
            with st.container(key=f"desk_answer_{turn.turn_id}"):
                st.markdown(turn.teacher_output.response)
                render_turn_token_usage(turn)
                render_turn_details(turn)

for turn in episode.turns:
    with st.container(key=f"desk_compat_question_{turn.turn_id}"):
        st.markdown(turn.student_message)
    render_desk_turn(turn, conversation_id=current_conversation_id or "draft")

retry_message: str | None = None
pending_student_message = _get_pending_student_message(episode)
if pending_student_message:
    with st.container(key="pending_desk_turn"):
        st.markdown(f'<div class="desk-user-note">{escape(pending_student_message)}</div>', unsafe_allow_html=True)
        st.warning("上一条消息的回复生成被中断，消息已经保留。")
        if st.button("重新生成上一条消息", type="secondary"):
            retry_message = pending_student_message

st.markdown(_CONTEXT_USAGE_STYLE, unsafe_allow_html=True)
with st.bottom:
    with st.container(key="teacher_composer_actions"):
        message = st.chat_input(
            "输入一道小学或初中学习问题，或继续追问……",
            disabled=not is_configured(),
        )
        with st.container(key="teacher_composer_controls"):
            context_column, mode_column = st.columns(
                [1.35, 3.65],
                vertical_alignment="center",
            )
            with context_column:
                with st.container(key="context_usage_control", width="content"):
                    render_context_usage(episode)
            with mode_column:
                st.markdown(
                    """
                    <div class="desk-composer-status">
                      <span class="desk-mode-chip">深度思考 · 已启用</span>
                      <span class="desk-mode-chip is-auto">联网检索 · 按需触发</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
message = retry_message or message
if message:
    is_retry = retry_message is not None
    _set_pending_student_message(episode, message)
    try:
        target_conversation_id = _ensure_target_conversation(episode, message)
        _save_conversation(
            target_conversation_id,
            episode,
            fallback_title=message,
        )
    except (ConversationArchiveError, ConversationStateError) as exc:
        st.error(f"消息保存失败，已停止生成以避免内容丢失：{exc}")
        st.stop()

    with st.container(key=f"desk_paper_sheet_live_{len(episode.turns) + 1}"):
        st.markdown(f'<div class="desk-user-note">{escape(message)}</div>', unsafe_allow_html=True)
        live_rail, live_answer = st.columns([1, 2.25], gap="large")
        thinking_state_key = thinking_key(
            target_conversation_id,
            len(episode.turns) + 1,
        )
        previous_thinking_state = st.session_state.get(thinking_state_key)
        if not isinstance(previous_thinking_state, ThinkingPanelState):
            previous_thinking_state = None
        st.session_state[thinking_state_key] = begin_thinking(
            previous_thinking_state
        )
        with live_rail:
            progress_status = st.status("🧠 思考中…（已用时 0 秒）", expanded=True)
            live_thinking = LiveThinkingPanel(progress_status)

        try:
            turn = run_turn(
                episode,
                message,
                enable_agent_loop=True,
                thinking_callback=live_thinking.append,
                kv_identity=KvCacheIdentity(
                    authenticated_user["user_id"],
                    target_conversation_id,
                ),
            )
        except (DeepSeekError, ValueError) as exc:
            st.session_state[thinking_state_key] = fail_thinking(
                st.session_state[thinking_state_key]
            )
            live_thinking.fail()
            st.error(str(exc))
        else:
            _set_pending_student_message(episode, "")
            history_save_error = ""
            try:
                _save_conversation(
                    target_conversation_id,
                    episode,
                    fallback_title=message,
                )
            except (ConversationArchiveError, ConversationStateError) as exc:
                history_save_error = str(exc)
            if history_save_error:
                st.warning(f"历史记录保存失败：{history_save_error}")
            if turn.sections:
                with progress_status:
                    render_thinking_sections(
                        turn,
                        expanded=True,
                        container=progress_status,
                    )
            st.session_state[thinking_state_key] = complete_thinking(
                st.session_state[thinking_state_key]
            )
            live_thinking.complete(turn.backend_total_seconds)
            display_started = perf_counter()
            with live_answer:
                with st.container(key=f"desk_answer_live_{turn.turn_id}"):
                    st.write_stream(iter_display_chunks(turn.teacher_output.response))
            turn.display_seconds = round(perf_counter() - display_started, 3)
            with live_answer:
                render_turn_token_usage(turn)
                render_turn_details(turn)
