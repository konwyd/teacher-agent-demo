"""Topic-agnostic analysis for the teaching harness."""

from __future__ import annotations

from app.agent.state import MathAnalysis, TeachingIntent
from app.llm.deepseek_client import DeepSeekClient, DeepSeekError, is_configured


_WHY_WORDS = ("为什么", "为啥", "怎么会")
_EXAMPLE_WORDS = ("举例", "例子", "示范")
_PRACTICE_WORDS = ("练习", "再来一题", "出一道题")
_ANSWER_WORDS = ("直接告诉我", "答案", "完整解答")
_CONFUSION_WORDS = ("不会", "不懂", "不明白", "不知道")


def _fallback_analysis(message: str, problem_context: str = "") -> MathAnalysis:
    """Return a conservative analysis when no configured LLM is available."""

    text = message.strip()
    if any(word in text for word in _WHY_WORDS):
        intent = TeachingIntent.ASKS_WHY
    elif any(word in text for word in _EXAMPLE_WORDS):
        intent = TeachingIntent.REQUESTS_EXAMPLE
    elif any(word in text for word in _PRACTICE_WORDS):
        intent = TeachingIntent.REQUESTS_PRACTICE
    elif any(word in text for word in _ANSWER_WORDS):
        intent = TeachingIntent.REQUESTS_ANSWER
    elif any(word in text for word in _CONFUSION_WORDS):
        intent = TeachingIntent.EXPRESSES_CONFUSION
    else:
        intent = TeachingIntent.UNKNOWN

    return MathAnalysis(
        problem_summary=problem_context or text,
        attempt_summary=text,
        intent=intent,
        likely_obstacle="学生需要确定下一步学习任务" if intent is TeachingIntent.EXPRESSES_CONFUSION else "",
        confidence=0.65 if intent is not TeachingIntent.UNKNOWN else 0.35,
        needs_clarification=not bool(problem_context or text),
    )


def analyze_turn(message: str, problem_context: str = "") -> MathAnalysis:
    """Analyze a turn without selecting a topic or assuming a solver.

    A configured DeepSeek client may supply the structured analysis. Any client
    error, unavailable credential, or invalid result falls back to the same
    conservative local analysis used by offline tests.
    """

    fallback = _fallback_analysis(message, problem_context)
    if not is_configured():
        return fallback

    try:
        result = DeepSeekClient().chat_json(
            system=(
                "你是初中数学教学分析器。只分析学生的当前意图、已给出的尝试、"
                "可能卡点和置信度；不要选择知识点，不要判断数学正误，不要给出答案。"
            ),
            user=(
                f"题目上下文：{problem_context or '未提供'}\n"
                f"学生消息：{message}"
            ),
            schema=MathAnalysis,
        )
    except (DeepSeekError, ValueError):
        return fallback

    return result.model_copy(
        update={
            "problem_summary": result.problem_summary or fallback.problem_summary,
            "attempt_summary": result.attempt_summary or fallback.attempt_summary,
        }
    )
