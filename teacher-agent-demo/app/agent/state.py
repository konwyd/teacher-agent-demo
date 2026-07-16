"""Teacher Agent schemas and workflow state (Pydantic)."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TeachingAction(str, Enum):
    """Allowed teaching actions (see CLAUDE.md)."""

    ASK_DIAGNOSTIC_QUESTION = "ask_diagnostic_question"
    EXPLAIN_CONCEPT = "explain_concept"
    GIVE_HINT_LEVEL_1 = "give_hint_level_1"
    GIVE_HINT_LEVEL_2 = "give_hint_level_2"
    SHOW_WORKED_EXAMPLE = "show_worked_example"
    GIVE_PRACTICE_QUESTION = "give_practice_question"
    GRADE_AND_FEEDBACK = "grade_and_feedback"
    SCHEDULE_REVIEW = "schedule_review"


class StudentIntent(str, Enum):
    CONCEPT_QUESTION = "concept_question"
    PROBLEM_SOLVING = "problem_solving"
    ANSWER_CHECKING = "answer_checking"
    REVIEW = "review"
    UNKNOWN = "unknown"
    # 围绕 original_problem 的细分意图（policy 用）
    ASKS_WHY = "student_asks_why"
    PROVIDES_STEP = "student_provides_step"
    PROVIDES_FINAL_ANSWER = "student_provides_final_answer"
    CONFUSED = "student_is_confused"
    REQUESTS_SOLUTION = "student_requests_solution"


class StudentGap(str, Enum):
    PREREQUISITE_MISSING = "prerequisite_missing"
    CONCEPT_MISSING = "concept_missing"
    CALCULATION_ERROR = "calculation_error"
    PROCEDURE_ERROR = "procedure_error"
    UNKNOWN = "unknown"


class DiagnosisResult(BaseModel):
    """诊断结果（DeepSeek JSON mode 输出或规则 fallback）。"""

    intent: StudentIntent = StudentIntent.UNKNOWN
    detected_concept: str = Field(default="", description="知识点名称")
    student_gap: StudentGap = StudentGap.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class RetrievalResult(BaseModel):
    """Graph RAG 检索结果 + 教材向量召回（Textbook RAG）。"""

    concept_id: str = ""
    concept_name: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    exercises: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    # 教材向量检索（K9 知识库 bge-m3 + ChromaDB）
    textbook_context: str = ""
    textbook_refs: list[str] = Field(default_factory=list)
    # 题库向量检索（k9_exercises 集合，B 方案多集合）
    exercise_context: str = ""
    exercise_refs: list[str] = Field(default_factory=list)
    # 课标向量检索（curriculum_std 集合）
    curriculum_context: str = ""
    curriculum_refs: list[str] = Field(default_factory=list)
    # 教资向量检索（teacher_cert 集合）
    cert_context: str = ""
    cert_refs: list[str] = Field(default_factory=list)
    # 多模态：教材检索关联的图片 URL
    textbook_image_urls: list[str] = Field(default_factory=list)


class TeacherResponse(BaseModel):
    """教学回复（动作 + 文本）。"""

    action: TeachingAction
    response: str


class StudentMasteryUpdate(BaseModel):
    """一轮交互后的掌握度变化。"""

    concept_id: str
    mastery_before: float = Field(ge=0.0, le=1.0)
    mastery_after: float = Field(ge=0.0, le=1.0)
    attempts: int = 0


class GradeResult(BaseModel):
    """LLM JSON-mode output: grading of a student answer (reserved for later)."""

    correct: bool
    score: float = Field(ge=0.0, le=1.0)
    feedback: str = ""


class TurnLog(BaseModel):
    """一轮交互的研究日志（写入 logs/turn_logs.jsonl）。"""

    log_id: str
    timestamp: str
    student_id: str
    user_message: str
    diagnosis: DiagnosisResult
    retrieval: RetrievalResult
    teaching_action: str
    teacher_response: str
    student_state_before: dict[str, float] = Field(default_factory=dict)
    student_state_after: dict[str, float] = Field(default_factory=dict)
    # grounding 相关
    original_problem: str = ""
    target_equation: str = ""
    student_step_validity: str = ""
    whether_response_used_new_example: bool = False
    whether_response_returned_to_original_problem: bool = False


class TeacherState(BaseModel):
    """State flowing through the LangGraph workflow (one turn)."""

    # input
    student_id: str
    message: str

    # diagnose
    diagnosis: Optional[DiagnosisResult] = None
    detected_concept_id: Optional[str] = None
    detected_concept_name: Optional[str] = None

    # retrieve
    retrieval: Optional[RetrievalResult] = None

    # RAG 开关：None=跟随环境变量默认；True=强制开启；False=强制关闭
    use_rag: Optional[bool] = None

    # grounding：围绕学生原题的跨轮状态（由 conversation store 载入/回写）
    original_problem: str = ""
    target_equation: str = ""
    lesson_goal: str = ""
    current_step: str = ""
    expected_next_step: str = ""
    solved_steps: list[str] = Field(default_factory=list)
    pending_teacher_question: str = ""
    last_student_answer: str = ""
    allowed_to_use_new_example: bool = False
    student_step_validity: str = ""
    student_equation: str = ""
    step_explanation: str = ""
    response_kind: str = ""
    plan_steps: list[str] = Field(default_factory=list)
    consecutive_wrong_steps: int = 0

    # policy + teach
    action: Optional[TeachingAction] = None
    response: str = ""

    # memory
    mastery_update: Optional[StudentMasteryUpdate] = None
    updated_student_state: dict[str, float] = Field(default_factory=dict)
    mastery_delta: float = 0.0
    known_misconceptions: list[str] = Field(default_factory=list)
    recommended_next_action: str = ""

    # log
    log_id: Optional[str] = None

    # research logs
    trace: list[str] = Field(default_factory=list)
