"""跨轮对话状态：original_problem 等在一次辅导会话内持续存在。

原则：一旦设定 original_problem，后续轮次不得覆盖（除非上一题已解完且学生带来新题）。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field


class ConversationState(BaseModel):
    student_id: str
    original_problem: str = ""
    target_equation: str = ""
    lesson_goal: str = ""
    current_step: str = Field(default="", description="学生已确认到的方程状态")
    expected_next_step: str = ""
    solved_steps: list[str] = Field(default_factory=list)
    pending_teacher_question: str = ""
    last_student_answer: str = ""
    solved: bool = False
    consecutive_wrong_steps: int = 0


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._conversations: dict[str, ConversationState] = {}

    def get(self, student_id: str) -> ConversationState:
        if student_id not in self._conversations:
            self._conversations[student_id] = ConversationState(student_id=student_id)
        return self._conversations[student_id]

    def save(self, conv: ConversationState) -> None:
        self._conversations[conv.student_id] = conv

    def reset(self, student_id: str) -> ConversationState:
        conv = ConversationState(student_id=student_id)
        self._conversations[student_id] = conv
        return conv


@lru_cache
def get_conversation_store() -> InMemoryConversationStore:
    """进程级单例（demo 用）。"""
    return InMemoryConversationStore()
