"""Guarded response generation for the topic-agnostic teaching harness."""

from __future__ import annotations

from typing import Protocol

from app.agent.state import TeachingActionKind, TeachingState
from app.llm.deepseek_client import DeepSeekClient, DeepSeekError, is_configured


_SAFE_RESPONSES: dict[TeachingActionKind, str] = {
    TeachingActionKind.ASSESS: "请把题目完整发来，并写出你已经想到的一步。你现在最不确定哪一部分？",
    TeachingActionKind.DIAGNOSE: "我先不假设题型，也不直接给答案。你觉得自己卡在理解题意、选择方法还是计算哪一步？",
    TeachingActionKind.GIVE_HINT: "先把题目中的已知条件、所求内容和它们的关系分别写出来。你准备先整理哪一条信息？",
    TeachingActionKind.EXPLAIN_STEP: "可以先把这一步依据的数学关系说清楚，再把它对应回原题。你认为这一步依据是什么？",
    TeachingActionKind.CHECK_UNDERSTANDING: "这个结论已经有验证依据了。你能用题目中的条件说明它为什么成立吗？",
    TeachingActionKind.SHOW_WORKED_EXAMPLE: "我可以按相同思路示范，但先不替代你完成原题。你希望我示范的是读题、列关系还是检查步骤？",
    TeachingActionKind.OFFER_PRACTICE: "可以安排练习，不过先确定要巩固的思路。你想练习理解题意、建立关系还是检查结果？",
    TeachingActionKind.FINISH: "这一步可以先收束并保留你的思路记录。下次开始时你想先复习哪一个关键方法？",
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


def _safe_response(state: TeachingState) -> str:
    if state.action_plan is None:
        raise ValueError("TeachingState requires an action plan before response generation")
    return _SAFE_RESPONSES[state.action_plan.action]


def _candidate_is_authorized(state: TeachingState) -> bool:
    return bool(
        state.action_plan
        and state.action_plan.answer_disclosure_allowed
        and state.answer_disclosure_allowed
    )


def _approved_context(state: TeachingState) -> str:
    if state.action_plan is None:
        raise ValueError("TeachingState requires an action plan before response generation")

    reasoning = state.math_reasoning
    facts = [
        f"原题：{state.problem_context}",
        f"教学动作：{state.action_plan.action.value}",
        f"提示层级：{state.action_plan.hint_level}",
    ]
    if reasoning.reasoning_steps:
        facts.append("可使用的推理步骤：" + "\n".join(reasoning.reasoning_steps))
    if reasoning.next_step_hint:
        facts.append(f"可使用的下一步提示：{reasoning.next_step_hint}")
    if _candidate_is_authorized(state) and reasoning.candidate_conclusion:
        facts.append(f"获准披露的候选结论：{reasoning.candidate_conclusion}")
    return "\n".join(facts)


def _has_single_final_question(response: str) -> bool:
    question_count = response.count("？") + response.count("?")
    return question_count == 1 and response.rstrip().endswith(("？", "?"))


def _contains_unapproved_candidate(response: str, state: TeachingState) -> bool:
    candidate = state.math_reasoning.candidate_conclusion
    if not candidate or _candidate_is_authorized(state):
        return False
    return candidate.replace(" ", "") in response.replace(" ", "")


def generate_response(
    state: TeachingState,
    *,
    client: _ResponseClient | None = None,
) -> str:
    """Generate from policy-approved facts or return a one-question safe fallback."""

    fallback = _safe_response(state)
    if not state.math_reasoning.reasoning_steps and not state.math_reasoning.next_step_hint:
        return fallback
    if client is None and not is_configured():
        return fallback

    active_client = client or DeepSeekClient()
    try:
        response = active_client.chat(
            system=(
                "你是初中数学老师。只能使用输入中的获准事实，围绕原题给一个可执行的下一步。"
                "未获准时不得给出最终答案或等价结论；只能保留一个结尾问句；"
                "不得声称候选结论已经被 verifier 验证。"
            ),
            user=_approved_context(state),
            temperature=0.2,
            max_tokens=350,
        ).strip()
    except (DeepSeekError, ValueError):
        return fallback

    if not response or not _has_single_final_question(response):
        return fallback
    if _contains_unapproved_candidate(response, state):
        return fallback
    return response
