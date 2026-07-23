"""Standalone Streamlit UI for the throwaway LLM-led teacher prototype."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.llm.deepseek_client import DeepSeekError, is_configured
from prototypes.llm_led_teacher.logic import PrototypeEpisode, PrototypeTurnRecord, run_turn
from prototypes.llm_led_teacher.live_progress import LiveProgressTrace
from prototypes.llm_led_teacher.rag_hook import warmup_rag
from prototypes.llm_led_teacher.streaming import iter_display_chunks


_SESSION_SCHEMA_VERSION = 6
_PROGRESS_LABELS = {
    "observing": "正在判断题目和本轮节奏……",
    "retrieving": "正在检索相关教材与题目……",
    "generating": "教师正在组织讲解……",
    "auditing": "正在核验数学步骤和教学节奏……",
    "repairing": "检测到需要修正，正在生成安全版本……",
    "complete": "回答已核验",
}


st.set_page_config(page_title="LLM-led Teacher Prototype", page_icon="🧪", layout="wide")
warmup_status = warmup_rag()

if st.session_state.get("prototype_session_schema_version") != _SESSION_SCHEMA_VERSION:
    st.session_state.prototype_episode = PrototypeEpisode()
    st.session_state.prototype_session_schema_version = _SESSION_SCHEMA_VERSION
elif "prototype_episode" not in st.session_state:
    st.session_state.prototype_episode = PrototypeEpisode()

episode: PrototypeEpisode = st.session_state.prototype_episode

st.title("🧪 LLM 主导 Teacher Prototype")
st.caption("LLM 回合观察 + DeepSeek 纯文本教学 · 验证缺失不阻断")

with st.sidebar:
    st.subheader("RAG 知识检索")
    st.write("📚 本地教材库 + 题库")
    st.write("🌐 联网搜索（新题触发）")
    if warmup_status["ok"]:
        st.caption(
            f"⚡ RAG {warmup_status['device']} · 预热 {warmup_status['elapsed_seconds']}s"
        )
    else:
        st.caption(f"⚠️ RAG 预热失败，已降级：{warmup_status['error']}")
    if st.button("新建对话", use_container_width=True):
        st.session_state.prototype_episode = PrototypeEpisode()
        st.session_state.prototype_session_schema_version = _SESSION_SCHEMA_VERSION
        st.rerun()
    st.divider()
    st.write("**当前 active problem**")
    st.write(episode.problem_context or "尚未开始")
    st.write("**回合数**", len(episode.turns))

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

        # RAG 信息：用指标 + 命中卡片呈现，保留相似度和可读的 Markdown/LaTeX。
        st.divider()
        st.markdown("### 📚 RAG 检索")
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
        render_turn_details(turn)

message = st.chat_input("输入一道初中数学题或继续回答……", disabled=not is_configured())
if message:
    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        progress_status = st.status(_PROGRESS_LABELS["observing"], expanded=True)
        live_trace = LiveProgressTrace(progress_status)

        def update_progress(stage: str) -> None:
            live_trace.update(stage)
            progress_status.update(label=_PROGRESS_LABELS.get(stage, stage))

        try:
            turn = run_turn(
                episode,
                message,
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
            progress_status.write(
                "🧭 **Observer 结论**："
                f"starts_new_problem={turn.turn_guidance.starts_new_problem}，"
                f"pacing={turn.turn_guidance.pacing.value}；"
                f"{turn.turn_guidance.reason or '未提供额外理由'}"
            )
            if turn.rag_source.startswith("retrieval_error"):
                progress_status.write(
                    f"⚠️ **RAG 已降级**：{turn.rag_source}；教学主回答未被阻断。"
                )
            else:
                progress_status.write(
                    f"📚 **RAG 结果**：来源={turn.rag_source or 'none'}，"
                    f"命中={len(turn.rag_hits)} 条。"
                )
            progress_status.write(
                f"🧮 **审计结果**：提取 {len(turn.extracted_equations)} 个方程，"
                f"修正={'是' if turn.repair_performed else '否'}。"
            )
            progress_status.update(
                label=_PROGRESS_LABELS["complete"],
                state="complete",
                expanded=True,
            )
            display_started = perf_counter()
            st.write_stream(iter_display_chunks(turn.teacher_output.response))
            turn.display_seconds = round(perf_counter() - display_started, 3)
            render_turn_details(turn)
