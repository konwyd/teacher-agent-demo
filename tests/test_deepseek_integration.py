"""DeepSeek 接入测试：全部离线，不真实调用外部 API。"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import app.agent.workflow as wf
from app.agent.state import (
    DiagnosisResult,
    RetrievalResult,
    StudentGap,
    StudentIntent,
    TeacherState,
    TeachingAction,
)
from app.llm.deepseek_client import DeepSeekError, is_configured


class _ReplySchema(BaseModel):
    text: str


class _RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"text":"ok"}'))]
        )


def _make_state() -> TeacherState:
    return TeacherState(
        student_id="s1",
        message="我不会解 2x+3=7",
        diagnosis=DiagnosisResult(
            intent=StudentIntent.PROBLEM_SOLVING,
            detected_concept="一元一次方程",
            student_gap=StudentGap.UNKNOWN,
            confidence=0.8,
        ),
        detected_concept_id="c_linear",
        detected_concept_name="一元一次方程",
        retrieval=RetrievalResult(
            concept_id="c_linear",
            concept_name="一元一次方程",
            prerequisites=["等式性质", "移项"],
            exercises=["解方程：2x + 3 = 7"],
            misconceptions=["移项时忘记变号。"],
        ),
        action=TeachingAction.GIVE_HINT_LEVEL_1,
    )


def test_is_configured_false_without_key() -> None:
    assert is_configured() is False  # conftest 已删除 key


def test_is_configured_true_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-for-test")
    assert is_configured() is True


def test_no_key_uses_mock() -> None:
    result = wf.teach_node(_make_state())
    assert "一元一次方程" in result["response"]
    assert "[错误]" not in result["response"]


def test_with_key_calls_deepseek_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """有 key 时必须走 _deepseek_response；用打桩替代真实网络调用。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-for-test")
    calls: list[str] = []

    def fake_deepseek_response(state: TeacherState) -> str:
        calls.append(state.message)
        return "canned-llm-reply，你觉得呢？"  # 自带一个问句，postprocess 不会改动

    monkeypatch.setattr(wf, "_deepseek_response", fake_deepseek_response)
    result = wf.teach_node(_make_state())
    assert calls == ["我不会解 2x+3=7"]
    assert result["response"] == "canned-llm-reply，你觉得呢？"


def test_api_failure_returns_clear_error_not_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """API 失败时返回清晰错误，而不是静默降级为 mock 文案。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-for-test")

    def failing_response(state: TeacherState) -> str:
        raise DeepSeekError("模拟的网络超时")

    monkeypatch.setattr(wf, "_deepseek_response", failing_response)
    result = wf.teach_node(_make_state())
    assert "[错误]" in result["response"]
    assert "模拟的网络超时" in result["response"]
    assert "你觉得第一步应该做什么" not in result["response"]


def test_diagnose_with_key_uses_llm_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """有 key 时 diagnose_node 走 LLM 路径（打桩验证，不碰网络）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-for-test")
    stub = DiagnosisResult(
        intent=StudentIntent.PROBLEM_SOLVING,
        detected_concept="一元一次方程",
        student_gap=StudentGap.CONCEPT_MISSING,
        confidence=0.95,
        reason="llm-stub",
    )

    def fake_llm_diagnosis(student_id: str, message: str):
        concept = wf.get_retriever().get_concept("c_linear")
        return stub, concept

    monkeypatch.setattr(wf, "_llm_diagnosis", fake_llm_diagnosis)
    out = wf.diagnose_node(TeacherState(student_id="s1", message="我不会解 2x+3=7"))
    assert out["diagnosis"].reason == "llm-stub"
    assert out["detected_concept_id"] == "c_linear"


def test_client_raises_without_key() -> None:
    from app.llm.deepseek_client import DeepSeekClient, DeepSeekSettings

    with pytest.raises(DeepSeekError):
        DeepSeekClient(settings=DeepSeekSettings(deepseek_api_key=""))


def test_chat_json_forwards_optional_max_tokens() -> None:
    from app.llm.deepseek_client import DeepSeekClient, DeepSeekSettings

    completions = _RecordingCompletions()
    client = DeepSeekClient(settings=DeepSeekSettings(deepseek_api_key="test-key"))
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    reply = client.chat_json("system", "user", _ReplySchema, max_tokens=450)

    assert reply.text == "ok"
    assert completions.calls[-1]["max_tokens"] == 450


def test_client_can_disable_transport_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.llm import deepseek_client

    calls: list[dict] = []

    class _RecordingOpenAI:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(deepseek_client, "OpenAI", _RecordingOpenAI)

    deepseek_client.DeepSeekClient(
        settings=deepseek_client.DeepSeekSettings(deepseek_api_key="test-key"),
        max_retries=0,
    )

    assert calls[-1]["max_retries"] == 0
