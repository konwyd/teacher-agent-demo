"""Standalone Streamlit UI for the throwaway LLM-led teacher prototype."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.state import VerificationStatus
from app.auth.local_accounts import AccountError, AuthenticationError, get_local_account_store
from app.llm.deepseek_client import DeepSeekError, is_configured
from app.memory.conversation_archive import (
    ConversationArchiveError,
    ConversationRecord,
    get_local_conversation_store,
)
from prototypes.llm_led_teacher.context_usage import build_context_usage_display
from prototypes.llm_led_teacher.logic import PrototypeEpisode, PrototypeTurnRecord, run_turn
from prototypes.llm_led_teacher.live_progress import LiveProgressTrace, agent_trace_rows
from prototypes.llm_led_teacher.orchestration_models import AgentTraceEvent
from prototypes.llm_led_teacher.rag_hook import warmup_rag
from prototypes.llm_led_teacher.streaming import iter_markdown_display_chunks


_SESSION_SCHEMA_VERSION = 11
_PROGRESS_LABELS = {
    "classification": "Agent · 正在识别请求……",
    "planning": "Agent · 正在规划与调用工具……",
    "observing": "Agent · 正在识别请求……",
    "retrieving": "Agent · 正在规划与调用工具……",
    "generating": "Answer Guardrails · 正在生成回答……",
    "auditing": "Answer Guardrails · 正在检查教学边界……",
    "boundary_check": "Answer Guardrails · 正在检查教学边界……",
    "repairing": "Answer Guardrails · 正在修正回答……",
    "formatting": "Answer Guardrails · 正在格式化 Markdown……",
    "complete": "Agent Loop + Answer Guardrails · 完成",
}


st.set_page_config(page_title="LLM-led Teacher Prototype", page_icon="🧪", layout="wide")

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


def _episode_payload(current_episode: object) -> dict[str, object]:
    payload = current_episode.model_dump(mode="json")
    payload["pending_student_message"] = _get_pending_student_message(current_episode)
    return payload


def _activate_conversation(record: ConversationRecord) -> None:
    st.session_state.prototype_episode = PrototypeEpisode.model_validate(record.episode)
    st.session_state[_PENDING_STUDENT_MESSAGE_KEY] = str(
        record.episode.get("pending_student_message", "") or ""
    )
    st.session_state.prototype_conversation_id = record.conversation_id
    st.session_state.conversation_selector = record.conversation_id
    st.session_state.prototype_user_id = record.user_id
    st.session_state.prototype_session_schema_version = _SESSION_SCHEMA_VERSION


def _open_latest_or_create_conversation(user_id: str) -> None:
    conversation_store = get_local_conversation_store()
    conversations = conversation_store.list_for_user(user_id)
    if conversations:
        record = conversation_store.load(user_id, conversations[0].conversation_id)
    else:
        episode = PrototypeEpisode()
        record = conversation_store.create(
            user_id,
            episode.model_dump(mode="json"),
        )
    _activate_conversation(record)


def _start_authenticated_session(user_id: str, username: str) -> None:
    st.session_state.authenticated_user = {
        "user_id": user_id,
        "username": username,
    }
    _open_latest_or_create_conversation(user_id)


def _save_active_conversation(current_episode: PrototypeEpisode) -> None:
    get_local_conversation_store().save(
        st.session_state.authenticated_user["user_id"],
        st.session_state.prototype_conversation_id,
        _episode_payload(current_episode),
        title=current_episode.problem_context or "新对话",
    )


def _create_new_conversation() -> None:
    user_id = st.session_state.authenticated_user["user_id"]
    episode = PrototypeEpisode()
    record = get_local_conversation_store().create(
        user_id,
        episode.model_dump(mode="json"),
    )
    _activate_conversation(record)


def _switch_conversation() -> None:
    user_id = st.session_state.authenticated_user["user_id"]
    conversation_id = st.session_state.conversation_selector
    record = get_local_conversation_store().load(user_id, conversation_id)
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

if st.session_state.get("prototype_user_id") != authenticated_user["user_id"]:
    _open_latest_or_create_conversation(authenticated_user["user_id"])
elif st.session_state.get("prototype_session_schema_version") != _SESSION_SCHEMA_VERSION:
    if "prototype_episode" in st.session_state:
        st.session_state.prototype_episode = PrototypeEpisode.model_validate(
            _episode_payload(st.session_state.prototype_episode)
        )
        st.session_state.prototype_session_schema_version = _SESSION_SCHEMA_VERSION
    else:
        _open_latest_or_create_conversation(authenticated_user["user_id"])
elif "prototype_episode" not in st.session_state:
    _open_latest_or_create_conversation(authenticated_user["user_id"])

episode: PrototypeEpisode = st.session_state.prototype_episode

st.title("🧪 LLM 主导 Teacher Prototype")
st.caption("九学科 Agent Loop + 本地知识优先 · 确定性验证与审计不绕过")

with st.sidebar:
    st.subheader(f"你好，{authenticated_user['username']}")
    st.caption(f"学生 ID：{authenticated_user['user_id']}")
    st.caption(f"当前对话：{st.session_state.prototype_conversation_id}")
    if st.button("退出登录", use_container_width=True):
        for key in (
            "authenticated_user",
            "prototype_episode",
            "prototype_conversation_id",
            "prototype_user_id",
            "prototype_session_schema_version",
            "conversation_selector",
            _PENDING_STUDENT_MESSAGE_KEY,
        ):
            st.session_state.pop(key, None)
        st.rerun()
    st.divider()
    st.subheader("历史对话")
    st.button(
        "新建对话",
        use_container_width=True,
        on_click=_create_new_conversation,
    )
    conversation_summaries = get_local_conversation_store().list_for_user(
        authenticated_user["user_id"]
    )
    conversation_labels = {
        summary.conversation_id: summary.title
        for summary in conversation_summaries
    }
    st.selectbox(
        "选择要继续的对话",
        options=[summary.conversation_id for summary in conversation_summaries],
        format_func=lambda conversation_id: conversation_labels[conversation_id],
        key="conversation_selector",
        on_change=_switch_conversation,
    )
    st.divider()
    st.subheader("RAG 知识检索")
    st.write("📚 本地教材库 + 题库")
    st.write("🌐 联网搜索（新题触发）")
    if warmup_status["ok"]:
        st.caption(
            f"⚡ RAG {warmup_status['device']} · 预热 {warmup_status['elapsed_seconds']}s"
        )
    else:
        st.caption(f"⚠️ RAG 预热失败，已降级：{warmup_status['error']}")
    st.write("**当前 active problem**")
    st.write(episode.problem_context or "尚未开始")
    st.write("**回合数**", len(episode.turns))

if not is_configured():
    st.warning(
        "当前进程没有配置 DEEPSEEK_API_KEY。请在启动 Streamlit 的同一 PowerShell "
        "窗口中设置环境变量后重新启动。"
    )


_TIMING_LABELS = {
    "classification": "Agent · 请求分类",
    "planning": "Agent · 规划与工具",
    "observing": "Legacy Observer 判断",
    "retrieving": "Legacy RAG 检索",
    "generating": "Answer Guardrails · 生成回答",
    "auditing": "Answer Guardrails · 边界检查",
    "repairing": "Answer Guardrails · 修正",
}


def _format_token_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


_CONTEXT_USAGE_STYLE = """
<style>
.st-key-context_usage_control {
    position: absolute;
    right: 4.35rem;
    top: 50%;
    transform: translateY(-50%);
    width: max-content !important;
    z-index: 10;
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

        st.markdown("### 🧭 Agent Loop 与教学边界")
        st.caption(
            "这里持久展示公开的 Agent/Tool 事件、教学契约与边界审计；"
            "不展示 prompt、私有 details 或模型逐字思维链。"
        )

        classification = getattr(turn, "classification", None)
        if classification is not None:
            st.write(
                "**请求分类**",
                {
                    "范围": classification.scope.value,
                    "学科": (
                        classification.subject.value
                        if classification.subject is not None
                        else "不适用"
                    ),
                    "学段": classification.stage.value,
                    "是否新题": classification.is_new_problem,
                    "回答模式": classification.answer_mode.value,
                },
            )

        st.markdown("#### 教学契约")
        teaching_contract = getattr(turn, "teaching_contract", None)
        if teaching_contract is None:
            st.caption("该历史记录生成于教学契约持久化之前。")
        else:
            st.write("**请求目标**", teaching_contract.requested_target)
            st.write("**回答模式**", teaching_contract.response_mode.value)
            st.write(
                "**权限**",
                {
                    "可应用到当前题": teaching_contract.may_apply_to_active_problem,
                    "可披露最终答案": teaching_contract.may_reveal_final_answer,
                },
            )
            st.write("**允许内容**", teaching_contract.allowed_content or ["未列出"])
            st.write("**禁止内容**", teaching_contract.forbidden_content or ["未列出"])
            st.write("**契约依据**", teaching_contract.reason)

        st.markdown("#### Agent / Tool Trace")
        trace_rows = agent_trace_rows(turn)
        if trace_rows:
            st.table(trace_rows)
        else:
            st.caption("该历史记录没有持久化 Agent trace。")

        st.markdown("#### 教学边界审计")
        boundary_audit = getattr(turn, "boundary_audit", None)
        if boundary_audit is None:
            st.caption("该历史记录没有语义边界审计。")
        else:
            if boundary_audit.passed:
                st.success("通过：回答符合本轮教学契约。")
            else:
                st.warning("未通过：回答已进入边界修正流程。")
            st.write(
                "**违规项**",
                [violation.value for violation in boundary_audit.violations]
                or ["无"],
            )
            st.write("**证据摘要**", boundary_audit.evidence or ["无"])
            st.write("**修正指令**", boundary_audit.repair_instructions or ["无"])
            if boundary_audit.audit_error:
                st.warning(f"边界审计已降级：{boundary_audit.audit_error}")

        st.markdown("#### 可选确定性验证")
        verification = turn.student_verification
        if verification.status is VerificationStatus.NOT_APPLICABLE:
            st.info("本轮不适用确定性 verifier；这不是教学边界审计失败。")
        elif verification.status is VerificationStatus.VERIFIED_CORRECT:
            st.success("确定性 verifier 已确认相关数学步骤。")
        elif verification.status is VerificationStatus.VERIFIED_INCORRECT:
            st.warning("确定性 verifier 发现学生步骤不成立。")
        else:
            st.info("确定性 verifier 未能提供结论；教师回答仍受教学边界约束。")
        st.write("**状态**", verification.status.value)
        st.write("**Verifier**", verification.verifier_name or "none")
        st.write("**验证摘要**", verification.summary or "未提供")
        st.write("**验证证据**", verification.evidence or ["无"])
        st.write("**自动提取的方程**", turn.extracted_equations or [])
        st.write("**整体证据等级**", turn.evidence_level)
        st.write(
            "**生成摘要**",
            {
                "初稿字符数": turn.draft_character_count,
                "最终字符数": turn.final_character_count,
                "是否触发修正": turn.repair_performed,
            },
        )
        st.markdown("#### 模型生成信息")
        st.caption(
            "原始逐字思维链不会展示；这里只展示模型接口返回的运行元数据。"
        )
        model_generation_metadata = turn.model_generation_metadata
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


for turn in episode.turns:
    with st.chat_message("user"):
        st.markdown(turn.student_message)
    with st.chat_message("assistant"):
        st.markdown(turn.teacher_output.response)
        render_turn_token_usage(turn)
        render_turn_details(turn)

retry_message: str | None = None
pending_student_message = _get_pending_student_message(episode)
if pending_student_message:
    with st.chat_message("user"):
        st.markdown(pending_student_message)
    st.warning("上一条消息的回复生成被中断，消息已经保留。")
    if st.button("重新生成上一条消息", type="secondary"):
        retry_message = pending_student_message

st.markdown(_CONTEXT_USAGE_STYLE, unsafe_allow_html=True)
with st.bottom:
    with st.container(key="context_usage_control", width="content"):
        render_context_usage(episode)
    message = st.chat_input(
        "输入一道小学或初中学习问题，或继续追问……",
        disabled=not is_configured(),
    )
message = retry_message or message
if message:
    is_retry = retry_message is not None
    if not is_retry:
        with st.chat_message("user"):
            st.markdown(message)

    _set_pending_student_message(episode, message)
    try:
        _save_active_conversation(episode)
    except ConversationArchiveError as exc:
        st.error(f"消息保存失败，已停止生成以避免内容丢失：{exc}")
        st.stop()

    with st.chat_message("assistant"):
        progress_status = st.status(_PROGRESS_LABELS["classification"], expanded=True)
        live_trace = LiveProgressTrace(progress_status)

        def update_progress(stage: str | AgentTraceEvent) -> None:
            if isinstance(stage, AgentTraceEvent):
                live_trace.update_agent_event(stage)
                event_label = {
                    "classification": "Agent · 请求分类",
                    "decision": "Agent · 规划",
                    "tool_result": "Agent · Tool",
                    "teaching_contract": "Agent · 教学契约",
                    "fallback": "Agent · 安全收口",
                }.get(stage.event_type, f"Agent · {stage.event_type}")
                progress_status.update(label=event_label)
                return
            if stage == "complete":
                live_trace.update("formatting")
                progress_status.update(label=_PROGRESS_LABELS["formatting"])
                return
            live_trace.update(stage)
            progress_status.update(label=_PROGRESS_LABELS.get(stage, stage))

        try:
            turn = run_turn(
                episode,
                message,
                enable_agent_loop=True,
                progress_callback=update_progress,
            )
        except (DeepSeekError, ValueError) as exc:
            progress_status.update(
                label="本轮处理失败",
                state="error",
                expanded=True,
            )
            st.error(str(exc))
        else:
            _set_pending_student_message(episode, "")
            history_save_error = ""
            try:
                _save_active_conversation(episode)
            except ConversationArchiveError as exc:
                history_save_error = str(exc)
            if turn.boundary_audit is not None:
                audit_label = "通过" if turn.boundary_audit.passed else "已触发修正"
                progress_status.write(f"🛡️ **教学边界审计**：{audit_label}。")
            if history_save_error:
                progress_status.write(
                    f"⚠️ **历史记录保存失败**：{history_save_error}"
                )
            else:
                progress_status.write("💾 **历史记录**：本轮对话已自动保存。")
            display_started = perf_counter()
            st.write_stream(iter_markdown_display_chunks(turn.teacher_output.response))
            turn.display_seconds = round(perf_counter() - display_started, 3)
            live_trace.update("complete")
            progress_status.update(
                label=_PROGRESS_LABELS["complete"],
                state="complete",
                expanded=True,
            )
            render_turn_token_usage(turn)
            render_turn_details(turn)
            st.rerun()
