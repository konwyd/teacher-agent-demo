"""Structured, topic-agnostic mathematical reasoning for the teaching harness."""

from __future__ import annotations

from typing import Protocol

from app.agent.state import GenericMathReasoning, MathAnalysis
from app.llm.deepseek_client import DeepSeekClient, DeepSeekError, is_configured


class _ReasoningClient(Protocol):
    def chat_json(
        self,
        system: str,
        user: str,
        schema,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> GenericMathReasoning:
        """Return a Pydantic-validated structured response."""


class GenericMathReasoner:
    """Produce advisory reasoning without choosing actions or verifying answers."""

    def __init__(self, client: _ReasoningClient | None = None) -> None:
        self._client = client
        self.last_error = ""

    def analyze(
        self,
        message: str,
        problem_context: str,
        analysis: MathAnalysis,
    ) -> GenericMathReasoning:
        """Return structured reasoning or a conservative clarification signal."""

        self.last_error = ""
        if not problem_context.strip() or analysis.needs_clarification:
            return GenericMathReasoning(
                needs_clarification=True,
                uncertainty_note="缺少足以形成可靠推理的题目上下文。",
            )
        if self._client is None and not is_configured():
            return GenericMathReasoning(
                uncertainty_note="未配置通用数学推理模型。",
            )

        client = self._client or DeepSeekClient()
        try:
            return client.chat_json(
                system=(
                    "你是初中数学推理器。给出短小、可教学的推理，明确不确定性。"
                    "不要选择教学动作，不要给学生回复，不要声称 verifier 已验证，"
                    "题意不清或不能可靠推理时必须设置 needs_clarification=true。"
                    "在同一次 JSON 结果的 strategy_selection 中给出一个 primary "
                    "和最多两个 supporting；strategy_id 只能是 symbolic_transformation、"
                    "quantitative_modeling、geometric_reasoning、data_analysis 或 unknown。"
                ),
                user=f"题目：{problem_context}\n学生消息：{message}",
                schema=GenericMathReasoning,
                temperature=0.0,
                max_tokens=450,
            )
        except (DeepSeekError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return GenericMathReasoning(
                needs_clarification=True,
                uncertainty_note="通用数学推理不可用，已降级为澄清教学。",
            )
