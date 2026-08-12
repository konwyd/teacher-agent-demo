"""Interactive UI for the isolated local-model tool-loop prototype."""

from __future__ import annotations

import os
import time
import uuid

import streamlit as st

from app.llm.deepseek_client import DeepSeekClient, DeepSeekSettings
from app.memory.schemas import EpisodeState
from prototypes.tool_loop_teacher.loop import AgentLoop, DeepSeekActionProvider


MODEL_BASE_URL = os.getenv(
    "TOOL_LOOP_MODEL_BASE_URL",
    "http://127.0.0.1:8080/v1",
)
MODEL_NAME = os.getenv("TOOL_LOOP_MODEL_NAME", "deepseek-v4-flash")


def _bypass_proxy_for_local_model() -> None:
    for variable in ("NO_PROXY", "no_proxy"):
        values = {
            item.strip()
            for item in os.getenv(variable, "").split(",")
            if item.strip()
        }
        values.update({"127.0.0.1", "localhost"})
        os.environ[variable] = ",".join(sorted(values))


def _new_conversation() -> None:
    st.session_state.messages = []
    st.session_state.turn_details = []
    st.session_state.conversation_id = f"prototype_{uuid.uuid4().hex[:10]}"
    st.session_state.episode = None


def _ensure_state() -> None:
    if "messages" not in st.session_state:
        _new_conversation()
    if "problem_context" not in st.session_state:
        st.session_state.problem_context = "解方程 2x+3=7"


def _build_client() -> DeepSeekClient:
    _bypass_proxy_for_local_model()
    settings = DeepSeekSettings(
        deepseek_api_key="local",
        deepseek_base_url=MODEL_BASE_URL,
        deepseek_model=MODEL_NAME,
        deepseek_timeout_seconds=600.0,
    )
    return DeepSeekClient(settings=settings)


def _episode(problem_context: str) -> EpisodeState:
    current = st.session_state.episode
    if current is None or current.problem_context != problem_context:
        current = EpisodeState(
            conversation_id=st.session_state.conversation_id,
            student_id="human_tester",
            original_problem=problem_context,
            problem_context=problem_context,
        )
        st.session_state.episode = current
    return current


st.set_page_config(
    page_title="Teacher Tool Loop Prototype",
    page_icon="🧪",
    layout="wide",
)
_ensure_state()

st.title("🧪 Teacher Agent Tool Loop")
st.caption(
    "本地 DeepSeek-V4 动态选择 Skills / Tools；math_checker 与安全门不可绕过。"
)

with st.sidebar:
    st.subheader("体验设置")
    problem_context = st.text_area(
        "当前原题",
        key="problem_context",
        height=120,
        help="原题在本次会话内保持稳定；修改后会创建新的原型 episode。",
    ).strip()
    max_iterations = st.slider("最大 Agent Loop 轮数", 2, 8, 6)
    decision_tokens = st.slider("单次决策 token 预算", 600, 2400, 1200, 100)
    show_trace = st.checkbox("每轮显示工具轨迹", value=True)
    if st.button("清空对话", use_container_width=True):
        _new_conversation()
        st.rerun()
    st.divider()
    st.success(f"模型：{MODEL_NAME}")
    st.code(MODEL_BASE_URL, language=None)
    st.caption("该页面强制使用本地地址，不读取付费 API endpoint。")

if not problem_context:
    st.warning("请先在左侧填写当前原题。")

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            detail_index = message.get("detail_index")
            if show_trace and detail_index is not None:
                details = st.session_state.turn_details[detail_index]
                with st.expander("查看本轮 Agent 轨迹"):
                    columns = st.columns(4)
                    columns[0].metric("动作", details["action"])
                    columns[1].metric("工具数", details["tool_count"])
                    columns[2].metric("耗时", f"{details['elapsed_seconds']:.1f}s")
                    columns[3].metric(
                        "降级",
                        "是" if details["fallback_used"] else "否",
                    )
                    st.json(details)

prompt = st.chat_input(
    "输入学生回答或追问……",
    disabled=not bool(problem_context),
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("本地模型正在选择 Skill / Tool……"):
            started = time.perf_counter()
            try:
                client = _build_client()
                result = AgentLoop(
                    DeepSeekActionProvider(
                        client,
                        max_tokens=decision_tokens,
                    ),
                    response_client=client,
                    max_iterations=max_iterations,
                ).run_turn(
                    student_id="human_tester",
                    conversation_id=st.session_state.conversation_id,
                    problem_context=problem_context,
                    student_message=prompt,
                    episode=_episode(problem_context),
                )
                response = result.response
                elapsed = time.perf_counter() - started
                details = {
                    "action": result.action_plan.action.value,
                    "action_plan": result.action_plan.model_dump(mode="json"),
                    "verification": result.verification.model_dump(mode="json"),
                    "loaded_skills": result.loaded_skills,
                    "tools": [
                        observation.tool_name for observation in result.observations
                    ],
                    "tool_count": len(result.observations),
                    "termination_reason": result.termination_reason,
                    "fallback_used": result.fallback_used,
                    "elapsed_seconds": elapsed,
                    "trace": [
                        event.model_dump(mode="json") for event in result.trace
                    ],
                }
            except Exception as exc:  # noqa: BLE001 - surface prototype failure
                elapsed = time.perf_counter() - started
                response = "原型本轮调用失败，请在轨迹中查看错误。"
                details = {
                    "action": "error",
                    "tool_count": 0,
                    "elapsed_seconds": elapsed,
                    "fallback_used": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        detail_index = len(st.session_state.turn_details)
        st.session_state.turn_details.append(details)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "detail_index": detail_index,
            }
        )
        st.markdown(response)
        if show_trace:
            with st.expander("查看本轮 Agent 轨迹"):
                st.json(details)
