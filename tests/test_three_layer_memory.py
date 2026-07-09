"""Three-layer memory and context-resolution tests."""

from fastapi.testclient import TestClient

from app.agent.context_resolver import ContextResolver
from app.agent.workflow import run_turn
from app.api.main import app
from app.memory.episode_state import get_episode_store
from app.memory.learning_history import (
    append_learning_event,
    get_learning_events,
    get_recent_learning_events,
    summarize_learning_history,
)
from app.memory.profile_updater import get_profile_store, update_profile_from_event
from app.memory.schemas import EpisodeState, LearningEvent, StudentProfile


def test_episode_state_tracks_last_teacher_question() -> None:
    state = run_turn("ctx_episode", "老师 4x+5=9 怎么解？", conversation_id="conv_ep")
    episode = get_episode_store().get("ctx_episode", "conv_ep")
    assert episode.conversation_id == "conv_ep"
    assert episode.original_problem == "4x + 5 = 9"
    assert episode.last_teacher_question
    assert episode.pending_question_type == "next_step_question"
    assert episode.expected_student_response
    assert state.conversation_id == "conv_ep"


def test_context_resolver_uses_episode_state() -> None:
    episode = EpisodeState(
        conversation_id="conv_resolve",
        student_id="ctx_student",
        original_problem="4x + 5 = 9",
        target_equation="4x + 5 = 9",
        current_phase="verification",
        current_solution_state="x = 1",
        last_teacher_question="把 x=1 代回原方程检验一下，两边相等吗？",
        expected_student_response="verification_result",
        pending_question_type="verification_question",
    )
    resolution = ContextResolver().resolve(
        "两边一样了",
        episode_state=episode,
        recent_turns=episode.recent_turns,
        student_profile=StudentProfile(student_id="ctx_student"),
    )
    assert resolution.is_reply_to_previous_teacher is True
    assert resolution.dialogue_act == "answer_pending_question"
    assert resolution.reply_to_question_type == "verification_question"
    assert resolution.semantic_meaning == "verification_success"
    assert resolution.should_start_new_problem is False


def test_short_answer_resolved_as_pending_reply() -> None:
    sid = "ctx_short"
    conv_id = "conv_short"
    run_turn(sid, "老师 4x+5=9 怎么解？", conversation_id=conv_id)
    run_turn(sid, "x=1", conversation_id=conv_id)
    state = run_turn(sid, "两边一样了", conversation_id=conv_id)
    assert state.context_resolution.is_reply_to_previous_teacher is True
    assert state.context_resolution.semantic_meaning == "verification_success"
    assert state.context_resolution.should_start_new_problem is False
    assert state.diagnosis.detected_concept == "一元一次方程"
    assert "4x + 5 = 9" in state.response
    assert "已经完成" in state.response or "验算" in state.response


def test_same_short_answer_without_episode_state_not_forced() -> None:
    episode = EpisodeState(conversation_id="new", student_id="ctx_none")
    resolution = ContextResolver().resolve(
        "两边一样是什么意思？",
        episode_state=episode,
        recent_turns=[],
        student_profile=StudentProfile(student_id="ctx_none"),
    )
    assert resolution.is_reply_to_previous_teacher is False
    assert resolution.dialogue_act != "answer_pending_question"
    assert resolution.should_start_new_problem is False


def test_learning_event_created_after_correct_step() -> None:
    sid = "ctx_event_step"
    state = run_turn(sid, "老师 3x+6=10 怎么解？", conversation_id="conv_event_step")
    state = run_turn(sid, "3x=4", conversation_id="conv_event_step")
    events = get_recent_learning_events(sid, concept_id="c_linear", limit=5)
    assert any(event.event_type == "correct_step" for event in events)
    assert any("3x" in event.evidence_text for event in events)


def test_learning_event_created_after_verification_success() -> None:
    sid = "ctx_event_verify"
    conv_id = "conv_event_verify"
    run_turn(sid, "老师 4x+5=9 怎么解？", conversation_id=conv_id)
    run_turn(sid, "x=1", conversation_id=conv_id)
    run_turn(sid, "两边一样了", conversation_id=conv_id)
    events = get_learning_events(sid)
    assert any(event.event_type == "verification_success" for event in events)
    assert any(event.event_type == "completed_episode" for event in events)


def test_profile_updates_from_learning_events() -> None:
    event = LearningEvent(
        student_id="ctx_profile",
        conversation_id="conv_profile",
        concept_id="c_linear",
        skill_id="solve_linear_equation",
        event_type="solved_problem",
        evidence_text="x=1",
        problem="4x + 5 = 9",
        confidence=0.95,
    )
    profile = update_profile_from_event(event)
    assert profile.concept_mastery["c_linear"] > 0.5
    assert profile.skill_mastery["solve_linear_equation"] > 0.5


def test_repeated_misconception_updates_common_weaknesses() -> None:
    sid = "ctx_weakness"
    for _ in range(2):
        event = LearningEvent(
            student_id=sid,
            conversation_id="conv_weak",
            concept_id="c_linear",
            skill_id="divide_by_coefficient",
            event_type="misconception",
            evidence_text="只除了左边",
            problem="2x = 4",
            confidence=0.9,
            metadata={"misconception": "除以系数时没有两边同时除"},
        )
        profile = update_profile_from_event(event)
    assert profile.repeated_misconceptions["除以系数时没有两边同时除"] == 2
    assert "除以系数时没有两边同时除" in profile.common_weaknesses


def test_new_conversation_creates_new_episode_state() -> None:
    sid = "ctx_new_conv"
    run_turn(sid, "老师 2x+3=7 怎么解？", conversation_id="conv_a")
    run_turn(sid, "老师 4x+5=9 怎么解？", conversation_id="conv_b")
    store = get_episode_store()
    a = store.get(sid, "conv_a")
    b = store.get(sid, "conv_b")
    assert a.conversation_id != b.conversation_id
    assert a.original_problem == "2x + 3 = 7"
    assert b.original_problem == "4x + 5 = 9"


def test_new_conversation_preserves_student_profile() -> None:
    sid = "ctx_profile_preserved"
    run_turn(sid, "老师 4x+5=9 怎么解？", conversation_id="conv_a")
    run_turn(sid, "x=1", conversation_id="conv_a")
    before = get_profile_store().get_profile(sid)
    run_turn(sid, "老师 2x+3=7 怎么解？", conversation_id="conv_b")
    after = get_profile_store().get_profile(sid)
    assert after.concept_mastery == before.concept_mastery
    assert get_episode_store().get(sid, "conv_b").original_problem == "2x + 3 = 7"


def test_pedagogical_rag_runs_after_context_resolution() -> None:
    sid = "ctx_order"
    conv_id = "conv_order"
    run_turn(sid, "老师 4x+5=9 怎么解？", conversation_id=conv_id)
    run_turn(sid, "x=1", conversation_id=conv_id)
    state = run_turn(sid, "两边一样了", conversation_id=conv_id)
    assert state.context_resolution.is_reply_to_previous_teacher is True
    assert state.pedagogical_context_used is False
    assert state.trace.index("context_resolver_node") < state.trace.index("retrieve_node")


def test_turn_api_accepts_conversation_id_and_returns_memory_fields() -> None:
    client = TestClient(app)
    resp = client.post(
        "/turn",
        json={
            "student_id": "ctx_api",
            "conversation_id": "api_conv",
            "message": "老师 4x+5=9 怎么解？",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"] == "api_conv"
    assert "episode_state_summary" in data
    assert "context_resolution" in data
    assert "learning_events_this_turn" in data
    assert "updated_profile_summary" in data


def test_learning_history_summary() -> None:
    append_learning_event(
        "ctx_summary",
        LearningEvent(
            student_id="ctx_summary",
            conversation_id="conv_summary",
            concept_id="c_linear",
            skill_id=None,
            event_type="asked_why",
            evidence_text="为什么移项变号",
            problem="2x + 3 = 7",
            confidence=0.8,
        ),
    )
    summary = summarize_learning_history("ctx_summary")
    assert summary["total_events"] >= 1
    assert summary["event_type_counts"]["asked_why"] >= 1
