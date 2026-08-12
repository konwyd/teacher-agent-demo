from __future__ import annotations

import pytest

from app.memory.conversation_archive import (
    ConversationNotFoundError,
    LocalConversationStore,
)
from prototypes.llm_led_teacher.logic import PrototypeEpisode


def test_conversation_survives_store_recreation(tmp_path) -> None:
    db_path = tmp_path / "conversations.sqlite3"
    first_store = LocalConversationStore(db_path)
    created = first_store.create(
        "user_a",
        {"problem_context": "", "previous_verified_step": "", "turns": []},
    )
    payload = {
        "problem_context": "2x+3=7",
        "previous_verified_step": "",
        "turns": [{"turn_id": 1, "student_message": "这题怎么解？"}],
    }
    first_store.save("user_a", created.conversation_id, payload, title="2x+3=7")

    loaded = LocalConversationStore(db_path).load("user_a", created.conversation_id)

    assert loaded.episode == payload
    assert loaded.title == "2x+3=7"


def test_users_only_list_their_own_conversations(tmp_path) -> None:
    store = LocalConversationStore(tmp_path / "conversations.sqlite3")
    user_a = store.create("user_a", {"turns": []}, title="A 的对话")
    store.create("user_b", {"turns": []}, title="B 的对话")

    summaries = store.list_for_user("user_a")

    assert [summary.conversation_id for summary in summaries] == [
        user_a.conversation_id
    ]


def test_cross_user_load_and_save_are_rejected(tmp_path) -> None:
    store = LocalConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create("user_a", {"turns": []})

    with pytest.raises(ConversationNotFoundError):
        store.load("user_b", conversation.conversation_id)
    with pytest.raises(ConversationNotFoundError):
        store.save("user_b", conversation.conversation_id, {"turns": []})


def test_recently_updated_conversation_is_listed_first(tmp_path) -> None:
    store = LocalConversationStore(tmp_path / "conversations.sqlite3")
    older = store.create("user_a", {"turns": []}, title="较早")
    newer = store.create("user_a", {"turns": []}, title="较新")
    store.save("user_a", older.conversation_id, {"turns": [1]}, title="已继续")

    summaries = store.list_for_user("user_a")

    assert summaries[0].conversation_id == older.conversation_id
    assert {summary.conversation_id for summary in summaries} == {
        older.conversation_id,
        newer.conversation_id,
    }


def test_legacy_episode_and_turn_gain_multisubject_defaults() -> None:
    legacy = {
        "problem_context": "2x+3=7",
        "previous_verified_step": "",
        "turns": [
            {
                "turn_id": 1,
                "student_message": "怎么解？",
                "teacher_output": {"response": "先观察等式。"},
                "student_verification": {
                    "status": "not_applicable",
                    "summary": "旧记录",
                    "verifier_name": "none",
                },
                "evidence_level": "none",
            }
        ],
    }

    episode = PrototypeEpisode.model_validate(legacy)

    assert episode.active_subject == ""
    assert episode.active_stage == ""
    assert episode.turns[0].classification is None
    assert episode.turns[0].agent_trace == []
    assert episode.turns[0].loaded_skills == []
    assert episode.turns[0].agent_termination_reason == ""
    assert episode.turns[0].teaching_contract is None
    assert episode.turns[0].boundary_audit is None
    assert episode.turns[0].formula_normalized is False
