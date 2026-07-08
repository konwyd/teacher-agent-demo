"""FastAPI entrypoint.

Run: uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agent.workflow import run_turn

app = FastAPI(title="Teacher Agent (research demo)")


class TurnRequest(BaseModel):
    student_id: str
    message: str


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
    state = run_turn(req.student_id, req.message)
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
