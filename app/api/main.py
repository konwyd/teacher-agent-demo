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
    retrieved_prerequisites: list[str] = Field(default_factory=list)
    retrieved_exercises: list[str] = Field(default_factory=list)
    response: str = ""
    updated_student_state: dict[str, float] = Field(default_factory=dict)
    log_id: str | None = None
    original_problem: str = ""
    student_step_validity: str = ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/turn", response_model=TurnResponse)
def turn(req: TurnRequest) -> TurnResponse:
    state = run_turn(req.student_id, req.message)
    diagnosis = state.diagnosis
    retrieval = state.retrieval
    return TurnResponse(
        detected_concept=state.detected_concept_name,
        intent=diagnosis.intent.value if diagnosis else None,
        student_gap=diagnosis.student_gap.value if diagnosis else None,
        teaching_action=state.action.value if state.action else None,
        retrieved_prerequisites=retrieval.prerequisites if retrieval else [],
        retrieved_exercises=retrieval.exercises if retrieval else [],
        response=state.response,
        updated_student_state=state.updated_student_state,
        log_id=state.log_id,
        original_problem=state.original_problem,
        student_step_validity=state.student_step_validity,
    )
