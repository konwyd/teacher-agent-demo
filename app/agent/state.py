"""Teacher Agent schemas and workflow state (Pydantic)."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.memory.schemas import ContextResolution, EpisodeState, LearningEvent
from app.retrieval.graph_retriever import RetrievalEvidence


class TeachingAction(str, Enum):
    """Allowed teaching actions (see CLAUDE.md)."""

    ASK_DIAGNOSTIC_QUESTION = "ask_diagnostic_question"
    EXPLAIN_CONCEPT = "explain_concept"
    GIVE_HINT_LEVEL_1 = "give_hint_level_1"
    GIVE_HINT_LEVEL_2 = "give_hint_level_2"
    SHOW_WORKED_EXAMPLE = "show_worked_example"
    GIVE_PRACTICE_QUESTION = "give_practice_question"
    GENERATE_PRACTICE_QUESTION = "generate_practice_question"
    GENERATE_SIMILAR_EXERCISE = "generate_similar_exercise"
    GENERATE_HARDER_EXERCISE = "generate_harder_exercise"
    GENERATE_EASIER_EXERCISE = "generate_easier_exercise"
    SUMMARIZE_THEN_PRACTICE = "summarize_then_practice"
    GRADE_AND_FEEDBACK = "grade_and_feedback"
    SCHEDULE_REVIEW = "schedule_review"


class TeachingActionKind(str, Enum):
    """Topic-agnostic actions used at the teaching-harness boundary.

    ``TeachingAction`` remains the legacy runtime enum in Stage 2.  This
    separate enum gives new callers a stable, domain-neutral vocabulary without
    changing the workflow's existing action dispatch.
    """

    ASSESS = "assess"
    DIAGNOSE = "diagnose"
    GIVE_HINT = "give_hint"
    EXPLAIN_STEP = "explain_step"
    CHECK_UNDERSTANDING = "check_understanding"
    SHOW_WORKED_EXAMPLE = "show_worked_example"
    OFFER_PRACTICE = "offer_practice"
    FINISH = "finish"


class TeachingActionPlan(BaseModel):
    """A neutral next-action decision, independent of a math topic."""

    action: TeachingActionKind
    hint_level: int = Field(default=0, ge=0, le=3)
    allow_worked_example: bool = False
    answer_disclosure_allowed: bool = False
    rationale: str = ""
    requires_student_response: bool = True


class TeachingIntent(str, Enum):
    """Topic-agnostic interaction intent inferred for one tutoring turn."""

    UNKNOWN = "unknown"
    ASKS_WHY = "asks_why"
    REQUESTS_EXAMPLE = "requests_example"
    REQUESTS_PRACTICE = "requests_practice"
    REQUESTS_ANSWER = "requests_answer"
    EXPRESSES_CONFUSION = "expresses_confusion"


class MathAnalysis(BaseModel):
    """Advisory mathematical analysis for a tutoring turn.

    Stage 2 only defines this data boundary.  No new LLM call populates it in
    the production workflow yet.
    """

    problem_summary: str = ""
    attempt_summary: str = ""
    likely_obstacle: str = ""
    intent: TeachingIntent = TeachingIntent.UNKNOWN
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False


class GenericMathReasoning(BaseModel):
    """Advisory reasoning for a math problem without topic-specific routing."""

    problem_interpretation: str = ""
    student_attempt_assessment: str = ""
    reasoning_steps: list[str] = Field(default_factory=list)
    next_step_hint: str = ""
    candidate_conclusion: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    uncertainty_note: str = ""


class VerificationStatus(str, Enum):
    """Whether an optional mathematical verifier produced usable evidence."""

    VERIFIED_CORRECT = "verified_correct"
    VERIFIED_INCORRECT = "verified_incorrect"
    NOT_APPLICABLE = "not_applicable"
    UNABLE_TO_VERIFY = "unable_to_verify"


class VerificationResult(BaseModel):
    """Verifier evidence represented without requiring a verifier call."""

    status: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)
    verifier_name: str = ""


class GraphContext(BaseModel):
    """Optional retrieval/graph context for a tutoring turn."""

    available: bool = False
    provider: str = ""
    evidence: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class TeachingState(BaseModel):
    """Topic-agnostic teaching state for future harness consumers.

    The model deliberately does not require an equation, concept identifier,
    verifier result, or retrieval result.  It is additive in Stage 2 and does
    not replace ``TeacherState`` inside the existing workflow.
    """

    student_id: str
    message: str
    conversation_id: str | None = None
    problem_context: str = ""
    learning_goal: str = ""
    progress_summary: str = ""
    confirmed_student_contributions: list[str] = Field(default_factory=list)
    pending_teacher_question: str = ""
    expected_student_response: str = ""
    current_phase: str = ""
    answer_disclosure_allowed: bool = False
    action_plan: TeachingActionPlan | None = None
    math_analysis: MathAnalysis = Field(default_factory=MathAnalysis)
    math_reasoning: GenericMathReasoning = Field(default_factory=GenericMathReasoning)
    verification: VerificationResult = Field(default_factory=VerificationResult)
    graph_context: GraphContext = Field(default_factory=GraphContext)
    response: str = ""
    trace: list[str] = Field(default_factory=list)


_LEGACY_TO_TEACHING_ACTION: dict[TeachingAction, TeachingActionKind] = {
    TeachingAction.ASK_DIAGNOSTIC_QUESTION: TeachingActionKind.DIAGNOSE,
    TeachingAction.EXPLAIN_CONCEPT: TeachingActionKind.EXPLAIN_STEP,
    TeachingAction.GIVE_HINT_LEVEL_1: TeachingActionKind.GIVE_HINT,
    TeachingAction.GIVE_HINT_LEVEL_2: TeachingActionKind.GIVE_HINT,
    TeachingAction.SHOW_WORKED_EXAMPLE: TeachingActionKind.SHOW_WORKED_EXAMPLE,
    TeachingAction.GIVE_PRACTICE_QUESTION: TeachingActionKind.OFFER_PRACTICE,
    TeachingAction.GENERATE_PRACTICE_QUESTION: TeachingActionKind.OFFER_PRACTICE,
    TeachingAction.GENERATE_SIMILAR_EXERCISE: TeachingActionKind.OFFER_PRACTICE,
    TeachingAction.GENERATE_HARDER_EXERCISE: TeachingActionKind.OFFER_PRACTICE,
    TeachingAction.GENERATE_EASIER_EXERCISE: TeachingActionKind.OFFER_PRACTICE,
    TeachingAction.SUMMARIZE_THEN_PRACTICE: TeachingActionKind.OFFER_PRACTICE,
    TeachingAction.GRADE_AND_FEEDBACK: TeachingActionKind.CHECK_UNDERSTANDING,
    TeachingAction.SCHEDULE_REVIEW: TeachingActionKind.FINISH,
}

_TEACHING_TO_LEGACY_ACTION: dict[TeachingActionKind, TeachingAction] = {
    TeachingActionKind.ASSESS: TeachingAction.ASK_DIAGNOSTIC_QUESTION,
    TeachingActionKind.DIAGNOSE: TeachingAction.ASK_DIAGNOSTIC_QUESTION,
    TeachingActionKind.GIVE_HINT: TeachingAction.GIVE_HINT_LEVEL_1,
    TeachingActionKind.EXPLAIN_STEP: TeachingAction.EXPLAIN_CONCEPT,
    TeachingActionKind.CHECK_UNDERSTANDING: TeachingAction.GRADE_AND_FEEDBACK,
    TeachingActionKind.SHOW_WORKED_EXAMPLE: TeachingAction.SHOW_WORKED_EXAMPLE,
    TeachingActionKind.OFFER_PRACTICE: TeachingAction.GIVE_PRACTICE_QUESTION,
    TeachingActionKind.FINISH: TeachingAction.SCHEDULE_REVIEW,
}


def project_teaching_action(action: TeachingAction | None) -> TeachingActionKind | None:
    """Map a legacy runtime action to the neutral harness action vocabulary."""

    if action is None:
        return None
    try:
        return _LEGACY_TO_TEACHING_ACTION[action]
    except KeyError as exc:
        raise ValueError(f"No neutral action mapping for legacy action: {action!r}") from exc


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
    """mock Graph RAG 检索结果。"""

    concept_id: str = ""
    concept_name: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    exercises: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)


class PedagogicalContext(BaseModel):
    """Pedagogical Graph RAG teaching-decision evidence."""

    concept_id: str = ""
    concept_name: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)
    diagnostic_questions: list[str] = Field(default_factory=list)
    hint_levels: list[str] = Field(default_factory=list)
    recommended_exercises: list[str] = Field(default_factory=list)
    mastery_based_strategy: str = ""


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


class SocraticDecision(BaseModel):
    """KELE-inspired consultant decision used as an advisory teaching signal."""

    stage: str = ""
    move: str = ""
    reason: str = ""
    must_not_override_checker: bool = True


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
    # consultant/teacher 分工相关：仅作为教学顾问信号，不裁决数学对错
    socratic_stage: str = ""
    socratic_move: str = ""
    socratic_reason: str = ""
    # Pedagogical RAG 使用痕迹：仅作为教学决策辅助，不裁决数学对错
    pedagogical_context_used: bool = False
    retrieval_backend: str = ""
    retrieval_evidence: list[RetrievalEvidence] = Field(default_factory=list)
    retrieved_concepts: list[str] = Field(default_factory=list)
    retrieved_misconceptions: list[str] = Field(default_factory=list)
    retrieved_hints: list[str] = Field(default_factory=list)
    retrieved_practice_problem_ids: list[str] = Field(default_factory=list)
    top_evidence_reasons: list[str] = Field(default_factory=list)
    selected_hint_level: int = 0
    matched_misconception: str = ""
    diagnostic_question_used: str = ""
    recommended_exercise: str = ""
    student_request_type: str = ""
    generated_practice_problem: str = ""
    generated_exercise_id: str = ""
    practice_generation_reason: str = ""
    practice_difficulty: str = ""
    source_concept: str = ""
    whether_new_episode_started: bool = False
    conversation_id: str = ""
    current_phase: str = ""
    pending_question_type: str = ""
    expected_student_response: str = ""
    context_resolution: ContextResolution = Field(default_factory=ContextResolution)
    learning_events_this_turn: list[LearningEvent] = Field(default_factory=list)
    updated_profile_summary: dict = Field(default_factory=dict)


class TeacherState(BaseModel):
    """State flowing through the LangGraph workflow (one turn)."""

    # input
    student_id: str
    message: str
    conversation_id: str | None = None

    # context resolution + episode memory
    episode_state: EpisodeState | None = None
    context_resolution: ContextResolution = Field(default_factory=ContextResolution)
    pending_reply_handled: bool = False

    # diagnose
    diagnosis: Optional[DiagnosisResult] = None
    detected_concept_id: Optional[str] = None
    detected_concept_name: Optional[str] = None

    # retrieve
    retrieval: Optional[RetrievalResult] = None
    pedagogical_context: PedagogicalContext = Field(default_factory=PedagogicalContext)
    pedagogical_context_used: bool = False
    retrieval_backend: str = ""
    retrieval_evidence: list[RetrievalEvidence] = Field(default_factory=list)
    retrieved_concepts: list[str] = Field(default_factory=list)
    retrieved_misconceptions: list[str] = Field(default_factory=list)
    retrieved_hints: list[str] = Field(default_factory=list)
    retrieved_practice_problem_ids: list[str] = Field(default_factory=list)
    top_evidence_reasons: list[str] = Field(default_factory=list)

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
    problem_solved: bool = False
    current_phase: str = ""
    pending_question_type: str = ""
    expected_student_response: str = ""

    # policy + teach
    action: Optional[TeachingAction] = None
    response: str = ""
    socratic_decision: Optional[SocraticDecision] = None
    socratic_stage: str = ""
    socratic_move: str = ""
    socratic_reason: str = ""
    selected_hint_level: int = 0
    matched_misconception: str = ""
    diagnostic_question_used: str = ""
    recommended_exercise: str = ""
    student_request_type: str = ""
    generated_practice_problem: str = ""
    generated_exercise_id: str = ""
    practice_generation_reason: str = ""
    practice_difficulty: str = ""
    source_concept: str = ""
    whether_new_episode_started: bool = False

    # memory
    mastery_update: Optional[StudentMasteryUpdate] = None
    updated_student_state: dict[str, float] = Field(default_factory=dict)
    mastery_delta: float = 0.0
    known_misconceptions: list[str] = Field(default_factory=list)
    recommended_next_action: str = ""
    learning_events_this_turn: list[LearningEvent] = Field(default_factory=list)
    updated_profile_summary: dict = Field(default_factory=dict)

    # log
    log_id: Optional[str] = None

    # research logs
    trace: list[str] = Field(default_factory=list)


def _project_verification_result(legacy_state: TeacherState) -> VerificationResult:
    """Translate existing checker output without invoking or replacing it."""

    validity = legacy_state.student_step_validity
    if validity in {"correct", "correct_step", "correct_final_answer", "verification_success"}:
        status = VerificationStatus.VERIFIED_CORRECT
    elif validity == "incorrect_step":
        status = VerificationStatus.VERIFIED_INCORRECT
    elif validity in {"not_a_step", "unparseable"}:
        status = VerificationStatus.UNABLE_TO_VERIFY
    else:
        status = VerificationStatus.NOT_APPLICABLE

    evidence = [legacy_state.step_explanation] if legacy_state.step_explanation else []
    return VerificationResult(
        status=status,
        summary=validity,
        evidence=evidence,
        verifier_name="legacy_math_checker" if validity else "",
    )


def _legacy_hint_level(legacy_state: TeacherState) -> int:
    """Preserve the hint level encoded by pre-existing legacy actions."""

    if legacy_state.action is TeachingAction.GIVE_HINT_LEVEL_2:
        return max(2, legacy_state.selected_hint_level)
    if legacy_state.action is TeachingAction.GIVE_HINT_LEVEL_1:
        return max(1, legacy_state.selected_hint_level)
    return legacy_state.selected_hint_level


def project_teacher_state(legacy_state: TeacherState) -> TeachingState:
    """Produce a neutral, read-only projection of the existing workflow state.

    This adapter is intentionally pure: it neither changes ``legacy_state`` nor
    calls an LLM, the checker, memory, or a retriever.  It gives API consumers a
    topic-neutral state contract while Stage 2 leaves runtime decisions on the
    legacy equation workflow.
    """

    diagnosis = legacy_state.diagnosis
    diagnosis_evidence = [diagnosis.reason] if diagnosis and diagnosis.reason else []
    math_analysis = MathAnalysis(
        problem_summary=legacy_state.original_problem,
        attempt_summary=legacy_state.last_student_answer or legacy_state.message,
        likely_obstacle=(diagnosis.student_gap.value if diagnosis else ""),
        evidence=diagnosis_evidence,
        confidence=diagnosis.confidence if diagnosis else 0.0,
        needs_clarification=not bool(diagnosis),
    )

    graph_evidence = list(legacy_state.top_evidence_reasons)
    if not graph_evidence:
        graph_evidence = [item.reason for item in legacy_state.retrieval_evidence]
    graph_context = GraphContext(
        available=bool(
            legacy_state.retrieval_backend
            or legacy_state.retrieval_evidence
            or legacy_state.retrieved_concepts
            or legacy_state.retrieved_misconceptions
            or legacy_state.retrieved_hints
        ),
        provider=legacy_state.retrieval_backend,
        evidence=graph_evidence,
        source_ids=[item.node_id for item in legacy_state.retrieval_evidence],
    )

    action = project_teaching_action(legacy_state.action)
    action_plan = (
        TeachingActionPlan(
            action=action,
            hint_level=_legacy_hint_level(legacy_state),
            allow_worked_example=legacy_state.allowed_to_use_new_example,
            rationale=legacy_state.socratic_reason,
            requires_student_response=not legacy_state.problem_solved,
        )
        if action is not None
        else None
    )

    return TeachingState(
        student_id=legacy_state.student_id,
        message=legacy_state.message,
        conversation_id=legacy_state.conversation_id,
        problem_context=legacy_state.original_problem,
        learning_goal=legacy_state.lesson_goal,
        progress_summary=legacy_state.current_step,
        confirmed_student_contributions=list(legacy_state.solved_steps),
        pending_teacher_question=legacy_state.pending_teacher_question,
        expected_student_response=legacy_state.expected_student_response,
        current_phase=legacy_state.current_phase,
        answer_disclosure_allowed=legacy_state.problem_solved,
        action_plan=action_plan,
        math_analysis=math_analysis,
        verification=_project_verification_result(legacy_state),
        graph_context=graph_context,
        response=legacy_state.response,
        trace=list(legacy_state.trace),
    )


def project_teaching_state_to_teacher_state(teaching_state: TeachingState) -> TeacherState:
    """Adapt the generic runtime result for callers of the legacy ``run_turn`` API.

    This projection is an output compatibility boundary only.  The generic
    harness never reads equation fields or legacy actions to make a decision.
    """

    action_plan = teaching_state.action_plan
    action = (
        _TEACHING_TO_LEGACY_ACTION[action_plan.action]
        if action_plan is not None
        else TeachingAction.ASK_DIAGNOSTIC_QUESTION
    )
    return TeacherState(
        student_id=teaching_state.student_id,
        message=teaching_state.message,
        conversation_id=teaching_state.conversation_id,
        diagnosis=DiagnosisResult(
            confidence=teaching_state.math_analysis.confidence,
            reason="generic_teaching_harness",
        ),
        original_problem=teaching_state.problem_context,
        lesson_goal=teaching_state.learning_goal,
        current_step=teaching_state.progress_summary,
        pending_teacher_question=teaching_state.pending_teacher_question,
        expected_student_response=teaching_state.expected_student_response,
        current_phase=teaching_state.current_phase,
        problem_solved=teaching_state.answer_disclosure_allowed,
        action=action,
        selected_hint_level=action_plan.hint_level if action_plan else 0,
        response=teaching_state.response,
        trace=list(teaching_state.trace),
    )
