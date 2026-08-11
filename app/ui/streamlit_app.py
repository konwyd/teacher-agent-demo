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


def call_api(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    resp = requests.request(method, f"{API_URL}{path}", timeout=60, **kwargs)
    resp.raise_for_status()
    return resp.json()


def new_conversation(student_id: str) -> str:
    data = call_api("POST", "/conversations/new", json={"student_id": student_id})
    return str(data["conversation_id"])


def call_turn(student_id: str, conversation_id: str, message: str) -> dict[str, Any]:
    return call_api(
        "POST",
        "/turn",
        json={
            "student_id": student_id,
            "conversation_id": conversation_id,
            "message": message,
        },
    )


def reset_chat_window(conversation_id: str) -> None:
    st.session_state.messages = []
    st.session_state.conversation_id = conversation_id


def render_meta(meta: dict[str, Any]) -> None:
    episode = meta.get("episode_state_summary") or {}
    context = meta.get("context_resolution") or {}
    action_plan = meta.get("teaching_action_plan") or {}
    constraints = meta.get("safety_constraints") or {}
    orchestration = meta.get("orchestration_trace") or {}
    shadow_plan = orchestration.get("shadow_plan") or {}
    with st.expander("诊断与检索详情"):
        st.write("**conversation_id**:", meta.get("conversation_id"))
        st.write("**current_phase**:", episode.get("current_phase"))
        st.write("**last_teacher_question**:", episode.get("last_teacher_question") or "无")
        st.write("**expected_student_response**:", episode.get("expected_student_response") or "无")
        st.write("**pending_question_type**:", episode.get("pending_question_type") or "无")
        st.write("**dialogue_act**:", context.get("dialogue_act"))
        st.write("**student_request_type**:", meta.get("student_request_type") or context.get("student_request_type") or "无")
        st.write("**is_reply_to_previous_teacher**:", context.get("is_reply_to_previous_teacher"))
        st.write("**context_reason**:", context.get("reason"))
        st.write("**generated_practice_problem**:", meta.get("generated_practice_problem") or "无")
        st.write("**practice_difficulty**:", meta.get("practice_difficulty") or "无")
        st.write("**source_concept**:", meta.get("source_concept") or "无")
        st.write("**whether_new_episode_started**:", meta.get("whether_new_episode_started"))
        st.write("**practice_generation_reason**:", meta.get("practice_generation_reason") or "无")

        st.divider()
        st.write("**detected_concept**:", meta.get("detected_concept"))
        st.write("**intent**:", meta.get("intent"))
        st.write("**student_gap**:", meta.get("student_gap"))
        st.write("**teaching_action**:", meta.get("teaching_action"))
        st.write("**unified_action**:", action_plan.get("action") or "无")
        st.write("**unified_scaffold_level**:", action_plan.get("scaffold_level", 0))
        st.write(
            "**disclosure_ceiling**:",
            constraints.get("disclosure_ceiling") or "无",
        )
        st.write("**shadow_action**:", shadow_plan.get("action") or "无")
        st.write("**llm_call_count**:", orchestration.get("llm_call_count", 0))
        for decision in orchestration.get("decisions") or []:
            st.write(
                f"**module:{decision.get('module', 'unknown')}**:",
                {
                    "called": decision.get("called"),
                    "reason": decision.get("reason"),
                    "llm_calls": decision.get("llm_calls", 0),
                },
            )
        st.write("**socratic_stage**:", meta.get("socratic_stage"))
        st.write("**socratic_move**:", meta.get("socratic_move"))
        st.write("**socratic_reason**:", meta.get("socratic_reason"))

        st.divider()
        st.write("**retrieved_prerequisites**:", meta.get("retrieved_prerequisites"))
        st.write("**retrieved_exercises**:", meta.get("retrieved_exercises"))
        st.write("**retrieval_backend**:", meta.get("retrieval_backend") or "无")
        st.write("**retrieved_concepts**:", meta.get("retrieved_concepts") or [])
        st.write(
            "**retrieved_misconceptions**:",
            meta.get("retrieved_misconceptions") or [],
        )
        st.write("**retrieved_hints**:", meta.get("retrieved_hints") or [])
        st.write(
            "**retrieved_practice_problem_ids**:",
            meta.get("retrieved_practice_problem_ids") or [],
        )
        st.write("**top_evidence_reasons**:", meta.get("top_evidence_reasons") or [])
        st.write("**retrieval_evidence**:", meta.get("retrieval_evidence") or [])
        st.write("**Pedagogical RAG used**:", meta.get("pedagogical_context_used"))
        st.write("**Pedagogical 前置知识**:", meta.get("pedagogical_prerequisites") or "无")
        st.write("**Pedagogical 常见误区**:", meta.get("pedagogical_misconceptions") or "无")
        st.write("**Pedagogical 分层提示**:", meta.get("pedagogical_hint_levels") or "无")
        st.write("**本轮实际提示层级**:", meta.get("selected_hint_level") or "无")
        st.write("**本轮匹配误区**:", meta.get("matched_misconception") or "无")
        st.write("**本轮诊断问题**:", meta.get("diagnostic_question_used") or "无")
        st.write("**推荐下一题**:", meta.get("recommended_exercise") or "无")

        st.divider()
        st.write("**learning_events_this_turn**:", meta.get("learning_events_this_turn") or [])
        st.write("**updated_profile_summary**:", meta.get("updated_profile_summary") or {})
        st.write("**updated_student_state**:", meta.get("updated_student_state"))
        st.write("**original_problem**:", meta.get("original_problem"))
        before = meta.get("student_mastery_before")
        after = meta.get("student_mastery_after")
        delta = meta.get("mastery_delta", 0.0)
        if after is not None:
            sign = "+" if (delta or 0) >= 0 else ""
            st.write("**当前知识点掌握度**:", f"{before} -> {after} (本轮 {sign}{delta})")
        st.write("**常见错因**:", meta.get("known_misconceptions") or "无")
        st.write("**推荐下一步**:", meta.get("recommended_next_action") or "-")
        st.write("**student_step_validity**:", meta.get("student_step_validity"))
        st.write("**log_id**:", meta.get("log_id"))


def render_sidebar(student_id: str) -> None:
    st.sidebar.caption("会话与记忆")
    st.sidebar.write("当前 conversation_id")
    st.sidebar.code(st.session_state.conversation_id)

    if st.sidebar.button("新建会话", use_container_width=True):
        try:
            reset_chat_window(new_conversation(student_id))
            st.rerun()
        except requests.RequestException as exc:
            st.sidebar.error(f"新建会话失败：{exc}")

    if st.sidebar.button("清空当前聊天窗口", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.sidebar.expander("危险操作"):
        confirm = st.checkbox("确认清空当前学生数据")
        if st.button("清空长期记忆", disabled=not confirm, use_container_width=True):
            try:
                call_api("POST", f"/students/{student_id}/reset-profile")
                st.success("长期记忆已清空")
            except requests.RequestException as exc:
                st.error(f"清空失败：{exc}")
        if st.button("清空学习历史", disabled=not confirm, use_container_width=True):
            try:
                call_api("POST", f"/students/{student_id}/reset-learning-history")
                st.success("学习历史已清空")
            except requests.RequestException as exc:
                st.error(f"清空失败：{exc}")

    try:
        profile = call_api("GET", f"/students/{student_id}/profile")
        st.sidebar.write("长期画像摘要")
        st.sidebar.json(profile, expanded=False)
    except requests.RequestException:
        st.sidebar.caption("后端启动后会显示长期画像。")


def main() -> None:
    st.set_page_config(page_title="LLM主导 Teaching Prototype", page_icon="📚")
    st.title("LLM主导 Teaching Prototype")
    st.caption("中文初中数学 · Context Resolver · 三层记忆 · Pedagogical RAG")

    student_id: str = st.sidebar.text_input("student_id", value="demo_student")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        try:
            st.session_state.conversation_id = new_conversation(student_id)
        except requests.RequestException:
            st.session_state.conversation_id = f"local_{student_id}"

    render_sidebar(student_id)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                render_meta(msg["meta"])

    user_input: str | None = st.chat_input("输入你的数学问题，例如：我不会解 2x+3=7")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            try:
                data = call_turn(student_id, st.session_state.conversation_id, user_input)
            except requests.RequestException as exc:
                st.error(f"请求后端失败：{exc}\n请确认 FastAPI 已启动：uvicorn app.api.main:app --reload")
                return
            st.session_state.conversation_id = data.get(
                "conversation_id", st.session_state.conversation_id
            )
            st.write(data.get("response", ""))
            render_meta(data)

        st.session_state.messages.append(
            {"role": "assistant", "content": data.get("response", ""), "meta": data}
        )


if __name__ == "__main__":
    main()
