"""Resolve dialogue context before concept diagnosis and Pedagogical RAG."""

from __future__ import annotations

from app.agent import math_checker as mc
from app.memory.schemas import ContextResolution, EpisodeState, StudentProfile

_WHY_WORDS = ("为什么", "为啥", "怎么会")
_CONFUSION_WORDS = ("不会", "不懂", "不明白", "听不懂", "没懂")
_OFF_TOPIC_WORDS = ("天气", "吃饭", "电影", "游戏")
_VERIFICATION_SUCCESS_WORDS = ("一样", "相等", "对了", "等于", "两边一样", "两边相等")

_REQUEST_RULES: list[tuple[tuple[str, ...], str, str, str]] = [
    (("难一点", "更难", "进阶", "挑战"), "request_harder_exercise", "generate_harder_exercise", "学生请求更难的练习题。"),
    (("简单点", "简单一点", "容易点", "基础点", "简单的"), "request_easier_exercise", "generate_easier_exercise", "学生请求更简单的练习题。"),
    (("类似", "同类型", "同类", "差不多"), "request_similar_exercise", "generate_similar_exercise", "学生请求类似练习题。"),
    (("复习", "回顾"), "request_review", "review_concept", "学生请求复习。"),
    (("总结", "小结", "归纳"), "request_summary", "summarize_current_problem", "学生请求总结当前学习内容。"),
    (("换个知识点", "换一个知识点", "换知识点", "别的知识点"), "request_new_topic", "switch_topic", "学生请求切换知识点。"),
    (("结束这题", "结束这道题", "不做了", "先到这"), "request_end_episode", "end_current_episode", "学生请求结束当前 episode。"),
    (("出一道题", "出一题", "出个题", "再来一题", "再来一道", "再出一道", "练一个", "练一题", "练习一下", "给我一题", "给我一道", "来一道"), "request_practice", "generate_practice", "学生请求老师生成练习题。"),
]

_ADDITIONAL_PRACTICE_REQUEST_WORDS = (
    "\u51fa\u4e00\u9053",
    "\u7ee7\u7eed\u4e0b\u4e00\u9898",
    "\u4e0b\u4e00\u9898",
    "\u60f3\u7ec3\u4e00\u4e0b",
    "\u7ec3\u4e00\u4e0b",
)


class ContextResolver:
    """Deterministic resolver with an LLM-ready interface.

    Tests and eval run offline, so this class keeps a deterministic fallback.
    The priority is:
    pending teacher question -> student request -> math step/answer -> why
    question -> new problem -> off-topic/unknown.
    """

    def resolve(
        self,
        student_message: str,
        episode_state: EpisodeState,
        recent_turns: list[dict],
        student_profile: StudentProfile,
    ) -> ContextResolution:
        message = student_message.strip()

        if episode_state.pending_question_type:
            pending = episode_state.pending_question_type
            if pending == "verification_question" and any(
                word in message for word in _VERIFICATION_SUCCESS_WORDS
            ):
                return ContextResolution(
                    is_reply_to_previous_teacher=True,
                    dialogue_act="answer_pending_question",
                    reply_to_question_type="verification_question",
                    semantic_meaning="verification_success",
                    confidence=0.95,
                    should_start_new_problem=False,
                    reason="当前 episode 正在等待验算结果，学生短答表达两边相等。",
                )
            if pending == "next_step_question":
                kind = mc.classify_student_math_response(message)
                if kind in ("single_step", "multi_step_plan", "intermediate_equation", "final_answer"):
                    return ContextResolution(
                        is_reply_to_previous_teacher=True,
                        dialogue_act="answer_pending_question",
                        reply_to_question_type="next_step_question",
                        semantic_meaning=kind,
                        confidence=0.85,
                        should_start_new_problem=False,
                        reason="当前 episode 正在等待下一步，学生给出数学步骤或答案。",
                    )
            if pending == "diagnostic_question" and message:
                request = self._resolve_request(message)
                if request is not None:
                    return request
                return ContextResolution(
                    is_reply_to_previous_teacher=True,
                    dialogue_act="answer_pending_question",
                    reply_to_question_type="diagnostic_question",
                    semantic_meaning="diagnostic_answer",
                    confidence=0.7,
                    should_start_new_problem=False,
                    reason="当前 episode 正在等待诊断问题回答。",
                )

        request = self._resolve_request(message)
        if request is not None:
            return request

        kind = mc.classify_student_math_response(message)
        if kind == "why_question" or any(word in message for word in _WHY_WORDS):
            return ContextResolution(
                dialogue_act="ask_why",
                semantic_meaning="why_question",
                confidence=0.85,
                should_start_new_problem=False,
                reason="学生在问原因。",
            )

        if kind == "multi_step_plan":
            act = "provide_multi_step_plan"
        elif kind == "single_step":
            act = "provide_single_step"
        elif kind == "final_answer":
            act = "provide_final_answer"
        else:
            act = ""
        if act:
            return ContextResolution(
                dialogue_act=act,
                semantic_meaning=kind,
                confidence=0.75,
                should_start_new_problem=False,
                reason="学生给出数学相关回答。",
            )

        equation = mc.extract_equation(message)
        if equation and (
            not episode_state.original_problem
            or (episode_state.is_problem_solved and not mc.is_equivalent(equation, episode_state.original_problem))
        ):
            return ContextResolution(
                dialogue_act="ask_new_problem",
                semantic_meaning="new_problem",
                confidence=0.9,
                should_start_new_problem=True,
                reason="消息包含新方程，且当前没有未完成原题或上一题已完成。",
            )

        if any(word in message for word in _CONFUSION_WORDS):
            return ContextResolution(
                dialogue_act="express_confusion",
                semantic_meaning="confusion",
                confidence=0.75,
                should_start_new_problem=False,
                reason="学生表达困惑。",
            )
        if any(word in message for word in _OFF_TOPIC_WORDS):
            return ContextResolution(
                dialogue_act="off_topic",
                semantic_meaning="off_topic",
                confidence=0.7,
                should_start_new_problem=False,
                reason="消息与当前数学任务无关。",
            )
        return ContextResolution(
            dialogue_act="unknown",
            semantic_meaning="",
            confidence=0.4,
            should_start_new_problem=False,
            reason="没有 pending question，也没有明确数学动作或教学请求。",
        )

    def _resolve_request(self, message: str) -> ContextResolution | None:
        if any(word in message for word in _ADDITIONAL_PRACTICE_REQUEST_WORDS):
            return ContextResolution(
                is_reply_to_previous_teacher=False,
                dialogue_act="request_practice",
                student_request_type="generate_practice",
                semantic_meaning="generate_practice",
                confidence=0.9,
                should_start_new_problem=False,
                reason="\u5b66\u751f\u8bf7\u6c42\u8001\u5e08\u751f\u6210\u7ec3\u4e60\u9898\u3002",
            )
        for keywords, dialogue_act, request_type, reason in _REQUEST_RULES:
            if any(word in message for word in keywords):
                return ContextResolution(
                    is_reply_to_previous_teacher=False,
                    dialogue_act=dialogue_act,
                    student_request_type=request_type,
                    semantic_meaning=request_type,
                    confidence=0.9,
                    should_start_new_problem=False,
                    reason=reason,
                )
        return None
