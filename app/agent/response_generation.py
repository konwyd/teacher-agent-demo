"""Guarded response generation from a policy-approved teaching brief."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable
from typing import Protocol

from app.agent.teaching_actions import TeachingAction
from app.agent.teaching_brief import TeachingBrief
from app.llm.deepseek_client import DeepSeekClient, DeepSeekError, is_configured


_SAFE_RESPONSES = {
    TeachingAction.ELICIT: "请把题目完整发来，并写出你已经想到的一步。你现在最不确定哪一部分？",
    TeachingAction.FOCUS: "我先不假设题型，也不直接给答案。你觉得自己卡在理解题意、选择方法还是计算哪一步？",
    TeachingAction.SCAFFOLD: "先把题目中的已知条件、所求内容和它们的关系分别写出来。你准备先整理哪一条信息？",
    TeachingAction.EXPLAIN: "可以先把这一步依据的数学关系说清楚，再把它对应回原题。你认为这一步依据是什么？",
    TeachingAction.CORRECT: "这一步已有明确的检查证据，可以先定位需要修正的位置。你愿意先重新检查这一处变化吗？",
    TeachingAction.CHECK: "这个结论已经有验证依据了。你能用题目中的条件说明它为什么成立吗？",
    TeachingAction.MODEL: "我可以按相同思路示范，但先不替代你完成原题。你希望我示范的是读题、列关系还是检查步骤？",
    TeachingAction.PRACTICE: "可以安排练习，不过先确定要巩固的思路。你想练习理解题意、建立关系还是检查结果？",
    TeachingAction.CONSOLIDATE: "我们先把刚才确认过的方法收束成一句话。你能说出下一次遇到同类关系时先看什么吗？",
}


class _ResponseClient(Protocol):
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Return a teacher response."""


GroundingGuard = Callable[[str, TeachingBrief], bool]


def _generic_grounding_guard(response: str, brief: TeachingBrief) -> bool:
    _ = response, brief
    return True


def _normalize_for_disclosure(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split())


class ResponseGuard:
    """Validate response shape, disclosure, and optional grounding evidence."""

    @staticmethod
    def validate(
        response: str,
        brief: TeachingBrief,
        grounding_guard: GroundingGuard | None = None,
    ) -> bool:
        text = response.strip()
        if not text:
            return False
        if text.count("？") + text.count("?") != 1:
            return False
        if not text.endswith(("？", "?")):
            return False

        normalized_response = _normalize_for_disclosure(text)
        if any(
            normalized_forbidden
            and normalized_forbidden in normalized_response
            for normalized_forbidden in (
                _normalize_for_disclosure(item)
                for item in brief.forbidden_disclosures
            )
        ):
            return False

        active_grounding_guard = grounding_guard or _generic_grounding_guard
        if brief.action_plan.action not in {
            TeachingAction.MODEL,
            TeachingAction.PRACTICE,
        } and not active_grounding_guard(text, brief):
            return False
        return True


def _safe_response(brief: TeachingBrief) -> str:
    return _SAFE_RESPONSES[brief.action_plan.action]


def generate_response(
    brief: TeachingBrief,
    *,
    client: _ResponseClient | None = None,
) -> str:
    """Generate only from the approved brief or return its action fallback."""

    if not isinstance(brief, TeachingBrief):
        raise TypeError("generate_response requires a TeachingBrief")
    fallback = _safe_response(brief)
    if not brief.primary_next_step:
        return fallback
    if client is None and not is_configured():
        return fallback

    try:
        active_client = client or DeepSeekClient()
    except (DeepSeekError, ValueError):
        return fallback

    try:
        response = active_client.chat(
            system=(
                "你是初中数学老师。只能使用用户消息中 TeachingBrief 的获准内容，"
                "围绕稳定原题给一个可执行的下一步。不得披露禁止内容；"
                "只能保留一个结尾问句。"
            ),
            user=json.dumps(
                brief.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            temperature=0.2,
            max_tokens=350,
        ).strip()
    except (DeepSeekError, ValueError):
        return fallback

    if not ResponseGuard.validate(response, brief):
        return fallback
    return response
