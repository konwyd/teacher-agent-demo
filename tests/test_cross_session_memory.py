from __future__ import annotations

from app.memory.conversation_archive import LocalConversationStore
from app.memory.cross_session import recall_learning_context


def _episode(problem: str, *, subject: str = "语文") -> dict:
    return {
        "problem_context": problem,
        "active_subject": subject,
        "active_stage": "小学",
        "turns": [
            {
                "student_message": problem,
                "teacher_output": {"response": "先明确中心，再选择具体材料。"},
            }
        ],
    }


def test_recall_continues_an_essay_in_a_new_conversation(tmp_path) -> None:
    store = LocalConversationStore(tmp_path / "conversations.sqlite3")
    old = store.create(
        "user_a",
        _episode("怎么写好《难忘的一天》这篇作文？"),
        title="难忘的一天作文",
    )

    recalled = recall_learning_context(store, "user_a", "这个作文的开头怎么写？")

    assert recalled is not None
    assert recalled.source_conversation_id == old.conversation_id
    assert "难忘的一天" in recalled.problem_context
    assert recalled.active_subject == "语文"
    assert recalled.recent_teacher_response == "先明确中心，再选择具体材料。"


def test_recall_never_reads_another_users_conversation(tmp_path) -> None:
    store = LocalConversationStore(tmp_path / "conversations.sqlite3")
    store.create("user_b", _episode("怎么写好《我的老师》作文？"))

    assert recall_learning_context(store, "user_a", "继续说这个作文") is None


def test_ordinary_new_question_does_not_trigger_old_memory(tmp_path) -> None:
    store = LocalConversationStore(tmp_path / "conversations.sqlite3")
    store.create("user_a", _episode("怎么写好《我的老师》作文？"))

    assert recall_learning_context(store, "user_a", "怎么学好数学？") is None


def test_recall_uses_latest_matching_topic_not_latest_unrelated_topic(tmp_path) -> None:
    store = LocalConversationStore(tmp_path / "conversations.sqlite3")
    essay = store.create("user_a", _episode("怎么写好《春天》作文？"))
    store.create("user_a", _episode("浮力的方向是什么？", subject="物理"))

    recalled = recall_learning_context(store, "user_a", "接着聊这个作文")

    assert recalled is not None
    assert recalled.source_conversation_id == essay.conversation_id
