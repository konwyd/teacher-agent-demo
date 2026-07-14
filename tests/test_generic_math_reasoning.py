"""Tests for the structured, topic-agnostic math reasoning boundary."""

import app.agent.math_analysis as math_analysis
from app.agent.generic_math_reasoning import GenericMathReasoner
from app.agent.state import MathAnalysis
from app.llm.deepseek_client import DeepSeekError


class _ReasoningClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return schema(
            problem_interpretation="总份数为5。",
            reasoning_steps=["把3:2看成5个相等份。"],
            next_step_hint="先计算每份是多少。",
            candidate_conclusion="男生18人，女生12人。",
            confidence=0.9,
        )


def test_reasoner_returns_validated_reasoning_with_a_token_cap() -> None:
    client = _ReasoningClient()

    result = GenericMathReasoner(client=client).analyze(
        "我不会做。",
        "男生和女生人数的比是3:2，总数30人。",
        MathAnalysis(),
    )

    assert result.confidence == 0.9
    assert result.reasoning_steps == ["把3:2看成5个相等份。"]
    assert client.calls[-1]["max_tokens"] == 450
    assert "不要选择教学动作" in client.calls[-1]["system"]


def test_reasoner_declines_ambiguous_context_without_calling_client() -> None:
    client = _ReasoningClient()

    result = GenericMathReasoner(client=client).analyze(
        "怎么算？",
        "",
        MathAnalysis(needs_clarification=True),
    )

    assert result.needs_clarification is True
    assert result.reasoning_steps == []
    assert client.calls == []


def test_analysis_can_disable_its_llm_for_budgeted_smoke(
    monkeypatch,
) -> None:
    monkeypatch.setattr(math_analysis, "is_configured", lambda: True)

    class _UnexpectedClient:
        def __init__(self) -> None:
            raise AssertionError("analysis client must not be created")

    monkeypatch.setattr(math_analysis, "DeepSeekClient", _UnexpectedClient)

    result = math_analysis.analyze_turn(
        "为什么要这样算？",
        "比是3:2，总数30人。",
        allow_llm=False,
    )

    assert result.intent.value == "asks_why"


def test_unconfigured_reasoner_leaves_a_safe_generic_hint_available() -> None:
    result = GenericMathReasoner().analyze(
        "我不知道先想什么。",
        "男生和女生人数的比是3:2，总数30人。",
        MathAnalysis(),
    )

    assert result.needs_clarification is False
    assert result.reasoning_steps == []
    assert result.uncertainty_note


def test_reasoner_preserves_a_client_error_summary_for_audit() -> None:
    class _FailingClient:
        def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
            raise DeepSeekError("simulated transport failure")

    reasoner = GenericMathReasoner(client=_FailingClient())

    result = reasoner.analyze(
        "我不会做。",
        "男生和女生人数的比是3:2，总数30人。",
        MathAnalysis(),
    )

    assert result.needs_clarification is True
    assert "DeepSeekError: simulated transport failure" in reasoner.last_error
