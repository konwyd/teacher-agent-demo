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


def call_turn(student_id: str, message: str, use_rag: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"student_id": student_id, "message": message}
    if use_rag is not None:
        payload["use_rag"] = use_rag
    resp = requests.post(
        f"{API_URL}/turn",
        json=payload,
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
        st.write("**教材参考(textbook_refs)**:", meta.get("textbook_refs") or "无")
        st.write("**题库参考(exercise_refs)**:", meta.get("exercise_refs") or "无")
        st.write("**课标参考(curriculum_refs)**:", meta.get("curriculum_refs") or "无")
        st.write("**教资参考(cert_refs)**:", meta.get("cert_refs") or "无")
        image_urls = meta.get("textbook_image_urls") or []
        if image_urls:
            st.write("**📷 教材关联图片**:")
            cols = st.columns(min(len(image_urls), 4))
            for i, url in enumerate(image_urls[:4]):
                with cols[i % 4]:
                    try:
                        st.image(url, caption=f"教材图{i+1}", use_container_width=True)
                    except Exception:
                        st.caption(f"[图片{i+1}]({url})")
        st.write("**updated_student_state**:", meta.get("updated_student_state"))
        st.write("**original_problem**:", meta.get("original_problem"))
        before = meta.get("student_mastery_before")
        after = meta.get("student_mastery_after")
        delta = meta.get("mastery_delta", 0.0)
        if after is not None:
            sign = "+" if (delta or 0) >= 0 else ""
            st.write("**当前知识点掌握度**:", f"{before} → {after}（本轮 {sign}{delta}）")
        st.write("**常见错因**:", meta.get("known_misconceptions") or "无")
        st.write("**推荐下一步**:", meta.get("recommended_next_action") or "-")
        st.write("**student_step_validity**:", meta.get("student_step_validity"))
        st.write("**log_id**:", meta.get("log_id"))


def main() -> None:
    st.set_page_config(page_title="Teacher Agent Demo", page_icon="📘")
    st.title("Teacher Agent 研究 Demo")
    st.caption("中文初中数学 · 诊断卡点 → 检索前置知识 → 策略选择 → 分层引导")

    student_id: str = st.sidebar.text_input("student_id", value="demo_student")

    # 新建对话：清空当前对话历史
    if st.sidebar.button("🗑 新建对话", help="清空当前对话历史，开始新一轮教学"):
        st.session_state.messages = []
        st.rerun()

    # RAG 开关：None=跟随环境变量默认；勾选=强制开启；取消勾选=强制关闭
    rag_default = True
    rag_col1, rag_col2 = st.sidebar.columns([3, 1])
    with rag_col1:
        use_rag_checkbox = st.checkbox("教材 RAG 检索", value=rag_default, help="开启后教师讲解会引用 K9 教材原文")
    with rag_col2:
        if st.button("跟随默认", help="不传 use_rag 参数，由环境变量 TEACHER_RAG_ENABLED 决定"):
            st.session_state["_rag_follow_default"] = True
        else:
            st.session_state["_rag_follow_default"] = False

    # use_rag: None=跟随环境变量；True/False=强制
    if st.session_state.get("_rag_follow_default"):
        use_rag: bool | None = None
    else:
        use_rag = use_rag_checkbox

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
                data = call_turn(student_id, user_input, use_rag=use_rag)
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
