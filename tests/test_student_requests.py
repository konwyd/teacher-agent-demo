"""Student request handling tests.

These protect the distinction between a student's math question and a request
for the teacher to take an action, such as generating a practice exercise.
"""

import pytest

from app.agent.context_resolver import ContextResolver
from app.agent.workflow import run_turn
from app.memory.episode_state import get_episode_store
from app.memory.schemas import EpisodeState, StudentProfile
from app.memory.student_state import get_store


def test_context_resolver_detects_practice_request() -> None:
    resolution = ContextResolver().resolve(
        "你出一道题好吗？",
        episode_state=EpisodeState(
            conversation_id="req_resolve",
            student_id="req_student",
            current_phase="completed",
            is_problem_solved=True,
        ),
        recent_turns=[],
        student_profile=StudentProfile(student_id="req_student"),
    )
    assert resolution.dialogue_act == "request_practice"
    assert resolution.student_request_type == "generate_practice"
    assert resolution.is_reply_to_previous_teacher is False


def test_practice_request_after_completed_problem_generates_new_exercise() -> None:
    sid = "req_after_done"
    conv_id = "conv_req_after_done"
    run_turn(sid, "老师 2x+5=11 怎么解？", conversation_id=conv_id)
    run_turn(sid, "x=3", conversation_id=conv_id)
    run_turn(sid, "两边相等的", conversation_id=conv_id)

    state = run_turn(sid, "你出一道题好吗？", conversation_id=conv_id)
    episode = get_episode_store().get(sid, conv_id)

    assert state.context_resolution.dialogue_act == "request_practice"
    assert state.context_resolution.student_request_type == "generate_practice"
    assert state.action and state.action.value in {
        "generate_practice_question",
        "generate_similar_exercise",
    }
    assert state.generated_practice_problem
    assert state.generated_practice_problem != "2x + 5 = 11"
    assert "2x + 5 = 11" not in state.response
    assert state.generated_practice_problem in state.response
    assert episode.current_phase == "practice_solving"
    assert episode.original_problem == state.generated_practice_problem
    assert episode.target_equation == state.generated_practice_problem
    assert episode.pending_question_type == "next_step_question"


def test_request_similar_exercise() -> None:
    resolution = ContextResolver().resolve(
        "再来一道类似的",
        episode_state=EpisodeState(conversation_id="similar", student_id="req_similar"),
        recent_turns=[],
        student_profile=StudentProfile(student_id="req_similar"),
    )
    assert resolution.dialogue_act == "request_similar_exercise"
    assert resolution.student_request_type == "generate_similar_exercise"


def test_request_harder_exercise() -> None:
    resolution = ContextResolver().resolve(
        "给我来一道难一点的",
        episode_state=EpisodeState(conversation_id="harder", student_id="req_harder"),
        recent_turns=[],
        student_profile=StudentProfile(student_id="req_harder"),
    )
    assert resolution.dialogue_act == "request_harder_exercise"
    assert resolution.student_request_type == "generate_harder_exercise"


def test_request_summary() -> None:
    resolution = ContextResolver().resolve(
        "你总结一下",
        episode_state=EpisodeState(conversation_id="summary", student_id="req_summary"),
        recent_turns=[],
        student_profile=StudentProfile(student_id="req_summary"),
    )
    assert resolution.dialogue_act == "request_summary"
    assert resolution.student_request_type == "summarize_current_problem"


@pytest.mark.parametrize(
    "message",
    [
        "我想练一下",
        "继续下一题",
        "出一道去括号的",
    ],
)
def test_practice_request_variants_are_detected(message: str) -> None:
    resolution = ContextResolver().resolve(
        message,
        episode_state=EpisodeState(
            conversation_id="request_variants",
            student_id="request_variants_student",
            current_phase="completed",
            is_problem_solved=True,
        ),
        recent_turns=[],
        student_profile=StudentProfile(student_id="request_variants_student"),
    )
    assert resolution.dialogue_act == "request_practice"
    assert resolution.student_request_type == "generate_practice"


def test_requested_practice_concept_overrides_completed_problem_concept() -> None:
    sid = "requested_concept_student"
    conv_id = "requested_concept_conversation"
    episode = get_episode_store().get(sid, conv_id)
    episode.original_problem = "2x + 5 = 11"
    episode.target_equation = "2x + 5 = 11"
    episode.current_phase = "completed"
    episode.is_problem_solved = True
    get_episode_store().save(episode)

    state = run_turn(sid, "给我一题分数系数方程", conversation_id=conv_id)

    assert state.context_resolution.student_request_type == "generate_practice"
    assert state.detected_concept_id == "c_frac"
    assert state.source_concept == "c_frac"


def test_practice_generation_uses_mastery() -> None:
    low_sid = "req_low_mastery"
    high_sid = "req_high_mastery"
    low_conv = "conv_low_mastery"
    high_conv = "conv_high_mastery"

    get_store().update_mastery_after_turn(low_sid, "c_linear", "incorrect_step")
    get_store().update_mastery_after_turn(low_sid, "c_linear", "incorrect_step")
    get_store().update_mastery_after_turn(low_sid, "c_linear", "incorrect_step")

    get_store().update_mastery_after_turn(high_sid, "c_linear", "correct_final_answer")
    get_store().update_mastery_after_turn(high_sid, "c_linear", "correct_final_answer")
    get_store().update_mastery_after_turn(high_sid, "c_linear", "correct_final_answer")

    low_episode = get_episode_store().get(low_sid, low_conv)
    low_episode.original_problem = "2x + 5 = 11"
    low_episode.target_equation = "2x + 5 = 11"
    low_episode.current_phase = "completed"
    low_episode.is_problem_solved = True
    get_episode_store().save(low_episode)

    high_episode = get_episode_store().get(high_sid, high_conv)
    high_episode.original_problem = "2x + 5 = 11"
    high_episode.target_equation = "2x + 5 = 11"
    high_episode.current_phase = "completed"
    high_episode.is_problem_solved = True
    get_episode_store().save(high_episode)

    low_state = run_turn(low_sid, "你出一道题好吗？", conversation_id=low_conv)
    high_state = run_turn(high_sid, "你出一道题好吗？", conversation_id=high_conv)

    assert low_state.practice_difficulty == "easy"
    assert high_state.practice_difficulty == "hard"
    assert "(" not in low_state.generated_practice_problem
    assert ("(" in high_state.generated_practice_problem) or ("x" in high_state.generated_practice_problem.split("=")[1])
