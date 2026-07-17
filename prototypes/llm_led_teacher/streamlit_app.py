"""Standalone Streamlit UI for the throwaway LLM-led teacher prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.llm.deepseek_client import DeepSeekError, is_configured
from prototypes.llm_led_teacher.logic import PrototypeEpisode, run_turn


_SESSION_SCHEMA_VERSION = 3


st.set_page_config(page_title="LLM-led Teacher Prototype", page_icon="🧪", layout="wide")

if st.session_state.get("prototype_session_schema_version") != _SESSION_SCHEMA_VERSION:
    st.session_state.prototype_episode = PrototypeEpisode()
    st.session_state.prototype_session_schema_version = _SESSION_SCHEMA_VERSION
elif "prototype_episode" not in st.session_state:
    st.session_state.prototype_episode = PrototypeEpisode()

episode: PrototypeEpisode = st.session_state.prototype_episode

st.title("🧪 LLM 主导 Teacher Prototype")
st.caption("DeepSeek 纯文本教学 · Agent 能验证则验证，不能验证也不阻断")

with st.sidebar:
    st.subheader("本原型刻意绕过")
    st.write("- TeachingPlanner")
    st.write("- 策略族 / StrategyRegistry")
    st.write("- Pedagogical RAG")
    st.write("- 掌握度流程控制")
    if st.button("新建对话", use_container_width=True):
        st.session_state.prototype_episode = PrototypeEpisode()
        st.session_state.prototype_session_schema_version = _SESSION_SCHEMA_VERSION
        st.rerun()
    st.divider()
    st.write("**稳定原题**")
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

message = st.chat_input("输入一道初中数学题或继续回答……", disabled=not is_configured())
if message:
    try:
        with st.spinner("DeepSeek 正在自主组织教学，Agent 正在核验……"):
            run_turn(episode, message)
    except (DeepSeekError, ValueError) as exc:
        st.error(str(exc))
    else:
        st.rerun()
