"""FastAPI entrypoint.

Run: uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent.workflow import run_turn
from app.retrieval.textbook_rag import is_rag_available

app = FastAPI(title="Teacher Agent (research demo)")

# 挂载教材图片目录：可通过 /images/parsed/{stem}/images/{filename} 访问
import os as _os
_parsed_root = "/data/education/data/china_textbook_k9_parsed"
if _os.path.isdir(_parsed_root):
    app.mount("/images/parsed", StaticFiles(directory=_parsed_root), name="textbook_images")


class TurnRequest(BaseModel):
    student_id: str
    message: str
    # RAG 开关：None=跟随环境变量默认；True=强制开启；False=强制关闭
    use_rag: bool | None = None


class TurnResponse(BaseModel):
    detected_concept: str | None = None
    intent: str | None = None
    student_gap: str | None = None
    teaching_action: str | None = None
    retrieved_prerequisites: list[str] = Field(default_factory=list)
    retrieved_exercises: list[str] = Field(default_factory=list)
    textbook_refs: list[str] = Field(default_factory=list)
    exercise_refs: list[str] = Field(default_factory=list)
    curriculum_refs: list[str] = Field(default_factory=list)
    cert_refs: list[str] = Field(default_factory=list)
    textbook_image_urls: list[str] = Field(default_factory=list)
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


@app.get("/rag/status")
def rag_status() -> dict:
    """查询 Textbook RAG 当前状态：是否启用、是否可用、配置参数。"""
    return {
        "enabled_by_env": is_rag_available(),
        "description": "RAG 开关可通过 3 个层级控制："
        "1) 环境变量 TEACHER_RAG_ENABLED（全局默认）；"
        "2) API 请求参数 use_rag（每轮覆盖，None=跟随默认）；"
        "3) UI 复选框（调用时传 use_rag）。",
    }


@app.post("/turn", response_model=TurnResponse)
def turn(req: TurnRequest) -> TurnResponse:
    state = run_turn(req.student_id, req.message, use_rag=req.use_rag)
    diagnosis = state.diagnosis
    retrieval = state.retrieval
    return TurnResponse(
        detected_concept=state.detected_concept_name,
        intent=diagnosis.intent.value if diagnosis else None,
        student_gap=diagnosis.student_gap.value if diagnosis else None,
        teaching_action=state.action.value if state.action else None,
        retrieved_prerequisites=retrieval.prerequisites if retrieval else [],
        retrieved_exercises=retrieval.exercises if retrieval else [],
        textbook_refs=retrieval.textbook_refs if retrieval else [],
        exercise_refs=retrieval.exercise_refs if retrieval else [],
        curriculum_refs=retrieval.curriculum_refs if retrieval else [],
        cert_refs=retrieval.cert_refs if retrieval else [],
        textbook_image_urls=retrieval.textbook_image_urls if retrieval else [],
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
