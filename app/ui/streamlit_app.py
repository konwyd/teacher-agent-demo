"""Minimal Streamlit chat UI for the Teacher Agent demo.

Run: streamlit run app/ui/streamlit_app.py
(FastAPI backend must be running first.)
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL: str = os.getenv("TEACHER_API_URL", "http://127.0.0.1:8000")


def call_turn(student_id: str, message: str) -> dict[str, Any]:
    resp = requests.post(
        f"{API_URL}/turn",
        json={"student_id": student_id, "message": message},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def render_meta(meta: dict[str, Any]) -> None:
    with st.expander("诊断与检索详情"):
        st.write("**detected_concept**:", meta.get("detected_concept"))
        st.write("**intent**:", meta.get("intent"))
        st.write("**student_gap**:", meta.get("student_gap"))
        st.write("**teaching_action**:", meta.get("teaching_action"))
        st.write("**retrieved_prerequisites**:", meta.get("retrieved_prerequisites"))
        st.write("**retrieved_exercises**:", meta.get("retrieved_exercises"))
        st.write("**updated_student_state**:", meta.get("updated_student_state"))
        st.write("**original_problem**:", meta.get("original_problem"))
        st.write("**student_step_validity**:", meta.get("student_step_validity"))
        st.write("**log_id**:", meta.get("log_id"))


def main() -> None:
    st.set_page_config(page_title="Teacher Agent Demo", page_icon="📘")
    st.title("Teacher Agent 研究 Demo")
    st.caption("中文初中数学 · 诊断卡点 → 检索前置知识 → 策略选择 → 分层引导")

    student_id: str = st.sidebar.text_input("student_id", value="demo_student")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 历史对话
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                render_meta(msg["meta"])

    # 输入
    user_input: str | None = st.chat_input("输入你的数学问题，例如：我不会解 2x+3=7")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            try:
                data = call_turn(student_id, user_input)
            except requests.RequestException as exc:
                st.error(f"请求后端失败：{exc}\n请确认 FastAPI 已启动（uvicorn app.api.main:app --reload）")
                return
            st.write(data.get("response", ""))
            render_meta(data)

        st.session_state.messages.append(
            {"role": "assistant", "content": data.get("response", ""), "meta": data}
        )


if __name__ == "__main__":
    main()
