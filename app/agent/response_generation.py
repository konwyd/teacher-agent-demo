"""Guarded response generation for the topic-agnostic teaching harness."""

from __future__ import annotations

from app.agent.state import TeachingActionKind, TeachingState


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


def generate_response(state: TeachingState) -> str:
    """Return a single-question response without asserting an unverified answer."""

    if state.action_plan is None:
        raise ValueError("TeachingState requires an action plan before response generation")
    return _SAFE_RESPONSES[state.action_plan.action]
