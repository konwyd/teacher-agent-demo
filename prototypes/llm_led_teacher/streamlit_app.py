"""Streamlit UI for the autonomous LLM-led Teacher Agent."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.llm.deepseek_client import DeepSeekError, is_configured
from prototypes.llm_led_teacher.live_progress import LiveProgressTrace
from prototypes.llm_led_teacher.logic import PrototypeEpisode, run_turn


_SESSION_SCHEMA_VERSION = 5


st.set_page_config(page_title="LLM-led Teacher Agent", page_icon="🧪", layout="wide")

if st.session_state.get("prototype_session_schema_version") != _SESSION_SCHEMA_VERSION:
    st.session_state.prototype_episode = PrototypeEpisode()
    st.session_state.prototype_session_schema_version = _SESSION_SCHEMA_VERSION
elif "prototype_episode" not in st.session_state:
    st.session_state.prototype_episode = PrototypeEpisode()

episode: PrototypeEpisode = st.session_state.prototype_episode

st.title("🧪 LLM 主导 Teacher Agent")
st.caption("LLM 自主教学 · 可选软约束 RAG · 确定性数学审计")

with st.sidebar:
    st.subheader("自主教学边界")
    st.write("- TeachingPlanner")
    st.write("- 策略族 / StrategyRegistry")
    st.write("- 掌握度硬流程控制")
    st.write("- RAG 只提供软参考，不决定教学动作或数学正误")
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

for turn in episode.turns:
    with st.chat_message("user"):
        st.markdown(turn.student_message)
    with st.chat_message("assistant"):
        st.markdown(turn.teacher_output.response)
        with st.expander("验证详情"):
            st.write("**观察器：是否新题**", turn.turn_guidance.starts_new_problem)
            st.write("**观察器：讲解节奏**", turn.turn_guidance.pacing.value)
            st.write("**观察器理由**", turn.turn_guidance.reason or "无")
            if turn.observer_error:
                st.warning(f"观察器已降级，但主回答未被阻断：{turn.observer_error}")
            st.write("**证据等级**", turn.evidence_level)
            if turn.evidence_level == "unverified":
                st.info("当前没有可用的确定性验证证据；教师回答仍正常展示。")
            st.write(
                "**学生步骤验证**",
                turn.student_verification.status.value,
            )
            st.write(
                "**verifier**",
                turn.student_verification.verifier_name or "none",
            )
            st.write("**验证摘要**", turn.student_verification.summary)
            st.write("**自动提取的方程**", turn.extracted_equations or [])
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
            st.divider()
            st.write("**RAG 状态**", turn.retrieval_status.value)
            st.write("**RAG provider**", turn.retrieval_provider or "none")
            st.write("**RAG query**", turn.retrieval_query or "无")
            st.write("**本轮允许例题**", turn.retrieval_examples_allowed)
            st.write("**RAG 耗时（秒）**", turn.retrieval_elapsed_seconds)
            if turn.retrieval_hits:
                st.write(
                    "**RAG 来源**",
                    [
                        {
                            "reference_id": hit.reference_id,
                            "source": hit.source_label,
                            "collection": hit.collection,
                            "kind": hit.content_kind.value,
                            "similarity": hit.similarity,
                        }
                        for hit in turn.retrieval_hits
                    ],
                )
            if turn.retrieval_error_code:
                st.warning(f"RAG 已安全降级：{turn.retrieval_error_code}")
            st.write("**阶段耗时（秒）**", turn.stage_timings_seconds)
            st.write("**后端总耗时（秒）**", turn.backend_total_seconds)

message = st.chat_input("输入一道初中数学题或继续回答……", disabled=not is_configured())
if message:
    try:
        with st.status("正在完成教学回合……", expanded=True) as status_box:
            progress = LiveProgressTrace(status_box)
            run_turn(episode, message, progress_callback=progress.update)
            status_box.update(label="教学回合已完成", state="complete", expanded=False)
    except (DeepSeekError, ValueError) as exc:
        st.error(str(exc))
    else:
        st.rerun()
