"""FastAPI entrypoint.

Run: uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from app.agent.workflow import run_turn
from app.memory.episode_state import create_conversation, get_episode_store
from app.memory.learning_history import get_learning_events, reset_learning_history, summarize_learning_history
from app.memory.profile_updater import get_profile_store, summarize_profile
from app.memory.student_state import get_store

app = FastAPI(title="Teacher Agent (research demo)")


class TurnRequest(BaseModel):
    student_id: str
    message: str
    conversation_id: str | None = None


class TurnResponse(BaseModel):
    detected_concept: str | None = None
    intent: str | None = None
    student_gap: str | None = None
    teaching_action: str | None = None
    socratic_stage: str = ""
    socratic_move: str = ""
    socratic_reason: str = ""
    pedagogical_context_used: bool = False
    selected_hint_level: int = 0
    matched_misconception: str = ""
    diagnostic_question_used: str = ""
    recommended_exercise: str = ""
    retrieved_prerequisites: list[str] = Field(default_factory=list)
    retrieved_exercises: list[str] = Field(default_factory=list)
    pedagogical_prerequisites: list[str] = Field(default_factory=list)
    pedagogical_misconceptions: list[str] = Field(default_factory=list)
    pedagogical_hint_levels: list[str] = Field(default_factory=list)
    conversation_id: str = ""
    episode_state_summary: dict = Field(default_factory=dict)
    context_resolution: dict = Field(default_factory=dict)
    learning_events_this_turn: list[dict] = Field(default_factory=list)
    updated_profile_summary: dict = Field(default_factory=dict)
    student_request_type: str = ""
    generated_practice_problem: str = ""
    generated_exercise_id: str = ""
    practice_generation_reason: str = ""
    practice_difficulty: str = ""
    source_concept: str = ""
    whether_new_episode_started: bool = False
    response: str = ""
    updated_student_state: dict[str, float] = Field(default_factory=dict)
    log_id: str | None = None
    original_problem: str = ""
    student_step_validity: str = ""
    student_mastery_before: float = 0.0
    student_mastery_after: float = 0.0
    mastery_delta: float = 0.0
    known_misconceptions: list[str] = Field(default_factory=list)
    recommended_next_action: str = ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/turn", response_model=TurnResponse)
def turn(req: TurnRequest) -> TurnResponse:
    conversation_id = req.conversation_id or create_conversation(req.student_id).conversation_id
    state = run_turn(req.student_id, req.message, conversation_id=conversation_id)
    diagnosis = state.diagnosis
    retrieval = state.retrieval
    pedagogical = state.pedagogical_context
    return TurnResponse(
        detected_concept=state.detected_concept_name,
        intent=diagnosis.intent.value if diagnosis else None,
        student_gap=diagnosis.student_gap.value if diagnosis else None,
        teaching_action=state.action.value if state.action else None,
        socratic_stage=state.socratic_stage,
        socratic_move=state.socratic_move,
        socratic_reason=state.socratic_reason,
        pedagogical_context_used=state.pedagogical_context_used,
        selected_hint_level=state.selected_hint_level,
        matched_misconception=state.matched_misconception,
        diagnostic_question_used=state.diagnostic_question_used,
        recommended_exercise=state.recommended_exercise,
        retrieved_prerequisites=retrieval.prerequisites if retrieval else [],
        retrieved_exercises=retrieval.exercises if retrieval else [],
        pedagogical_prerequisites=pedagogical.prerequisites,
        pedagogical_misconceptions=pedagogical.common_misconceptions,
        pedagogical_hint_levels=pedagogical.hint_levels,
        conversation_id=state.conversation_id or conversation_id,
        episode_state_summary=(
            state.episode_state.model_dump(mode="json") if state.episode_state else {}
        ),
        context_resolution=state.context_resolution.model_dump(mode="json"),
        learning_events_this_turn=[
            event.model_dump(mode="json") for event in state.learning_events_this_turn
        ],
        updated_profile_summary=state.updated_profile_summary,
        student_request_type=state.context_resolution.student_request_type or state.student_request_type,
        generated_practice_problem=state.generated_practice_problem,
        generated_exercise_id=state.generated_exercise_id,
        practice_generation_reason=state.practice_generation_reason,
        practice_difficulty=state.practice_difficulty,
        source_concept=state.source_concept,
        whether_new_episode_started=state.whether_new_episode_started,
        response=state.response,
        updated_student_state=state.updated_student_state,
        log_id=state.log_id,
        original_problem=state.original_problem,
        student_step_validity=state.student_step_validity,
        student_mastery_before=state.mastery_update.mastery_before if state.mastery_update else 0.0,
        student_mastery_after=state.mastery_update.mastery_after if state.mastery_update else 0.0,
        mastery_delta=state.mastery_delta,
        known_misconceptions=state.known_misconceptions,
        recommended_next_action=state.recommended_next_action,
    )


class NewConversationRequest(BaseModel):
    student_id: str


@app.post("/conversations/new")
def new_conversation(req: NewConversationRequest) -> dict:
    episode = create_conversation(req.student_id)
    return {
        "student_id": req.student_id,
        "conversation_id": episode.conversation_id,
        "episode_state": episode.model_dump(mode="json"),
    }


@app.get("/students/{student_id}/conversations")
def list_conversations(student_id: str, limit: int = Query(default=20, ge=1, le=100)) -> dict:
    episodes = get_episode_store().list_conversations(student_id, limit=limit)
    return {
        "student_id": student_id,
        "conversations": [episode.model_dump(mode="json") for episode in episodes],
    }


@app.get("/students/{student_id}/profile")
def get_student_profile(student_id: str) -> dict:
    return summarize_profile(student_id)


@app.get("/students/{student_id}/learning-events")
def get_student_learning_events(
    student_id: str,
    limit: int | None = Query(default=None, ge=1, le=500),
) -> dict:
    events = get_learning_events(student_id, limit=limit)
    return {
        "student_id": student_id,
        "summary": summarize_learning_history(student_id),
        "events": [event.model_dump(mode="json") for event in events],
    }


@app.post("/students/{student_id}/reset-profile")
def reset_student_profile(student_id: str) -> dict:
    get_store().reset_student(student_id)
    profile = get_profile_store().reset_profile(student_id)
    return {
        "student_id": student_id,
        "profile": profile.model_dump(mode="json"),
        "status": "reset",
    }


@app.post("/students/{student_id}/reset-learning-history")
def reset_student_learning_history(student_id: str) -> dict:
    reset_learning_history(student_id)
    return {"student_id": student_id, "status": "reset"}
