"""Deterministic evaluation harness for the Teacher Agent research demo."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from app.agent import math_checker as mc
from app.agent.legacy_action_compat import LegacyTeachingAction as TeachingAction
from app.agent.math_checker import StepValidity
from app.agent.state import TeacherState
from app.agent.workflow import BASIC_LANGUAGE_MARKERS, is_step_message, run_turn

EVAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"

_EXAMPLE_ALLOWED_ACTIONS = (
    TeachingAction.SHOW_WORKED_EXAMPLE.value,
    TeachingAction.GIVE_PRACTICE_QUESTION.value,
    TeachingAction.GENERATE_PRACTICE_QUESTION.value,
    TeachingAction.GENERATE_SIMILAR_EXERCISE.value,
    TeachingAction.GENERATE_HARDER_EXERCISE.value,
    TeachingAction.GENERATE_EASIER_EXERCISE.value,
    TeachingAction.SUMMARIZE_THEN_PRACTICE.value,
)

_REQUEST_PRACTICE_ACTIONS = {
    TeachingAction.GENERATE_PRACTICE_QUESTION,
    TeachingAction.GENERATE_SIMILAR_EXERCISE,
    TeachingAction.GENERATE_HARDER_EXERCISE,
    TeachingAction.GENERATE_EASIER_EXERCISE,
    TeachingAction.SUMMARIZE_THEN_PRACTICE,
}


class GoldenTurn(BaseModel):
    role: str
    content: str


class GoldenCase(BaseModel):
    case_id: str
    original_problem: str
    turns: list[GoldenTurn]
    expected_concept: str = ""
    expected_final_answer: str = ""
    forbidden_patterns: list[str] = Field(default_factory=list)
    required_behaviors: list[str] = Field(default_factory=list)
    note: str = ""


class SingleTurnCase(BaseModel):
    case_id: str
    message: str
    expected_concept: str = ""
    expected_intent: str = ""
    expected_action: str = ""
    forbidden_patterns: list[str] = Field(default_factory=list)


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class CaseResult(BaseModel):
    case_id: str
    case_type: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    num_turns: int = 0


def load_golden_cases(path: Path | None = None) -> list[GoldenCase]:
    p = path or (EVAL_DATA_DIR / "golden_conversations.jsonl")
    cases = [
        GoldenCase.model_validate(json.loads(line))
        for line in p.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    if path is None:
        extra = EVAL_DATA_DIR / "three_layer_memory_conversations.jsonl"
        if extra.exists():
            cases.extend(
                GoldenCase.model_validate(json.loads(line))
                for line in extra.read_text(encoding="utf-8").strip().splitlines()
                if line.strip()
            )
    return cases


def load_single_turn_cases(path: Path | None = None) -> list[SingleTurnCase]:
    p = path or (EVAL_DATA_DIR / "single_turn_cases.jsonl")
    return [
        SingleTurnCase.model_validate(json.loads(line))
        for line in p.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]


def _norm(text: str) -> str:
    return text.replace(" ", "")


def _question_count(text: str) -> int:
    return text.count("？") + text.count("?")


def _equations_excluding_student(response: str, student_msg: str) -> list[str]:
    student_eqs = {_norm(e) for e in mc.find_equations(student_msg)}
    return [e for e in mc.find_equations(response) if _norm(e) not in student_eqs]


def _has_correct_ack(response: str) -> bool:
    return any(marker in response for marker in ("正确", "对", "很好", "思路"))


def _has_wrong_step_feedback(response: str) -> bool:
    return any(marker in response for marker in ("不是", "不对", "先停", "错"))


def evaluate_response(
    case: GoldenCase,
    student_msg: str,
    state: TeacherState,
    final_answer_already_given: bool = False,
    is_final_turn: bool = False,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    response = state.response
    action = state.action.value if state.action else ""
    behaviors = set(case.required_behaviors)

    practice_request_started = bool(state.generated_practice_problem)
    keeps = practice_request_started or (
        bool(state.original_problem) and mc.is_equivalent(
            state.original_problem, case.original_problem
        )
    )
    checks.append(CheckResult(
        name="keeps_original_problem",
        passed=keeps,
        detail=f"state.original_problem={state.original_problem!r}, expected={case.original_problem!r}",
    ))

    bad = [pat for pat in case.forbidden_patterns if _norm(pat) in _norm(response)]
    checks.append(CheckResult(
        name="no_forbidden_patterns",
        passed=not bad,
        detail=f"matched forbidden patterns: {bad}" if bad else "",
    ))

    if action not in _EXAMPLE_ALLOWED_ACTIONS:
        alien = [
            e for e in _equations_excluding_student(response, student_msg)
            if not mc.is_equivalent(e, case.original_problem)
        ]
        checks.append(CheckResult(
            name="no_unrelated_new_example",
            passed=not alien,
            detail=f"unrelated equations: {alien}" if alien else "",
        ))

    ref = mc.check_step(case.original_problem, student_msg) if is_step_message(student_msg) else None
    if ref is not None and ref.validity in (
        StepValidity.CORRECT_STEP,
        StepValidity.CORRECT_FINAL_ANSWER,
    ):
        recognized = state.student_step_validity == ref.validity.value and _has_correct_ack(response)
        checks.append(CheckResult(
            name="recognizes_correct_step",
            passed=recognized,
            detail=f"validity={state.student_step_validity!r}, response={response!r}" if not recognized else "",
        ))
    elif ref is not None and ref.validity is StepValidity.INCORRECT_STEP:
        rejected = state.student_step_validity == StepValidity.INCORRECT_STEP.value and _has_wrong_step_feedback(response)
        checks.append(CheckResult(
            name="rejects_wrong_step",
            passed=rejected,
            detail=f"validity={state.student_step_validity!r}, response={response!r}" if not rejected else "",
        ))

    if "为什么" in student_msg:
        answered = "两边同时" in response or "等式" in response or "保持相等" in response
        checks.append(CheckResult(
            name="answers_why_question",
            passed=answered,
            detail="why-question was not explained with equivalence/equality language" if not answered else "",
        ))

    n_q = _question_count(response)
    checks.append(CheckResult(
        name="asks_at_most_one_question",
        passed=n_q <= 1,
        detail=f"question_count={n_q}",
    ))
    if "must_ask_one_next_question" in behaviors:
        checks.append(CheckResult(
            name="asks_exactly_one_question",
            passed=n_q == 1,
            detail=f"question_count={n_q}",
        ))

    if (
        case.expected_final_answer
        and not final_answer_already_given
        and state.student_step_validity != StepValidity.CORRECT_FINAL_ANSWER.value
    ):
        leaked = _norm(case.expected_final_answer) in _norm(response)
        checks.append(CheckResult(
            name="no_premature_final_answer",
            passed=not leaked,
            detail=f"premature final answer {case.expected_final_answer}" if leaked else "",
        ))

    if is_final_turn and "must_adapt_low_mastery" in behaviors:
        ok = (
            state.action is TeachingAction.ASK_DIAGNOSTIC_QUESTION
            and any(marker in response for marker in BASIC_LANGUAGE_MARKERS)
        )
        checks.append(CheckResult(
            name="adapts_low_mastery",
            passed=ok,
            detail=f"expected diagnostic/basic language, action={action!r}, response={response!r}" if not ok else "",
        ))
    if is_final_turn and "must_not_overexplain_high_mastery" in behaviors:
        ok = state.action is TeachingAction.GIVE_PRACTICE_QUESTION or state.action in _REQUEST_PRACTICE_ACTIONS
        checks.append(CheckResult(
            name="no_overexplain_high_mastery",
            passed=ok,
            detail=f"expected practice question, action={action!r}" if not ok else "",
        ))
    if is_final_turn and "must_track_repeated_misconception" in behaviors:
        ok = any("变号" in m for m in state.known_misconceptions)
        checks.append(CheckResult(
            name="tracks_repeated_misconception",
            passed=ok,
            detail=f"known_misconceptions={state.known_misconceptions}" if not ok else "",
        ))
    if is_final_turn and "must_accept_multi_step_plan" in behaviors:
        ok = (
            state.student_step_validity in ("correct_step", "correct_final_answer")
            and ("思路" in response or "正确" in response)
        )
        checks.append(CheckResult(
            name="accepts_multi_step_plan",
            passed=ok,
            detail=f"validity={state.student_step_validity!r}, response={response!r}" if not ok else "",
        ))
    if is_final_turn and "must_not_force_single_step" in behaviors:
        ok = "除以" in response
        checks.append(CheckResult(
            name="not_forcing_single_step",
            passed=ok,
            detail=f"response={response!r}" if not ok else "",
        ))
    if is_final_turn and "must_not_repeat_already_answered_step" in behaviors:
        ok = all(marker not in response for marker in ("减去几", "减几", "第一步应该做什么"))
        checks.append(CheckResult(
            name="not_repeating_answered_step",
            passed=ok,
            detail=f"response={response!r}" if not ok else "",
        ))

    if is_final_turn and "must_resolve_pending_context" in behaviors:
        ok = (
            state.context_resolution.is_reply_to_previous_teacher
            and state.context_resolution.dialogue_act == "answer_pending_question"
        )
        checks.append(CheckResult(
            name="resolves_pending_context",
            passed=ok,
            detail=f"context_resolution={state.context_resolution.model_dump()}" if not ok else "",
        ))
    if is_final_turn and "must_not_start_new_problem" in behaviors:
        ok = not state.context_resolution.should_start_new_problem
        checks.append(CheckResult(
            name="does_not_start_new_problem",
            passed=ok,
            detail=f"context_resolution={state.context_resolution.model_dump()}" if not ok else "",
        ))
    if is_final_turn and "must_not_explain_unrelated_concept" in behaviors:
        ok = "等式性质" not in response and "等式的性质" not in response
        checks.append(CheckResult(
            name="does_not_explain_unrelated_concept",
            passed=ok,
            detail=f"response={response!r}" if not ok else "",
        ))
    if is_final_turn and "must_confirm_verification" in behaviors:
        ok = state.context_resolution.semantic_meaning == "verification_success" and (
            "验算" in response or "完成" in response or "两边" in response
        )
        checks.append(CheckResult(
            name="confirms_verification",
            passed=ok,
            detail=f"response={response!r}, semantic={state.context_resolution.semantic_meaning!r}" if not ok else "",
        ))
    if is_final_turn and "must_create_verification_success_event" in behaviors:
        ok = any(event.event_type == "verification_success" for event in state.learning_events_this_turn)
        checks.append(CheckResult(
            name="creates_verification_success_event",
            passed=ok,
            detail=f"events={[event.event_type for event in state.learning_events_this_turn]}" if not ok else "",
        ))
    if is_final_turn and "must_summarize_original_problem" in behaviors:
        ok = mc.mentions_equation(response, case.original_problem)
        checks.append(CheckResult(
            name="summarizes_original_problem",
            passed=ok,
            detail=f"response={response!r}, original={case.original_problem!r}" if not ok else "",
        ))
    if is_final_turn and "must_record_repeated_weakness" in behaviors:
        profile = state.updated_profile_summary or {}
        ok = bool(profile.get("common_weaknesses") or profile.get("repeated_misconceptions"))
        checks.append(CheckResult(
            name="records_repeated_weakness",
            passed=ok,
            detail=f"profile={profile}" if not ok else "",
        ))
    if is_final_turn and "must_update_student_profile" in behaviors:
        profile = state.updated_profile_summary or {}
        ok = bool(profile.get("concept_mastery") or profile.get("skill_mastery"))
        checks.append(CheckResult(
            name="updates_student_profile",
            passed=ok,
            detail=f"profile={profile}" if not ok else "",
        ))
    if is_final_turn and "must_detect_practice_request" in behaviors:
        ok = bool(state.context_resolution.student_request_type) and state.context_resolution.dialogue_act.startswith("request_")
        checks.append(CheckResult(
            name="detects_practice_request",
            passed=ok,
            detail=f"context_resolution={state.context_resolution.model_dump()}" if not ok else "",
        ))
    if is_final_turn and "must_generate_new_exercise" in behaviors:
        ok = bool(state.generated_practice_problem) and state.generated_practice_problem in response
        checks.append(CheckResult(
            name="generates_new_exercise",
            passed=ok,
            detail=f"generated={state.generated_practice_problem!r}, response={response!r}" if not ok else "",
        ))
    if is_final_turn and "must_not_continue_original_problem" in behaviors:
        ok = (
            bool(state.generated_practice_problem)
            and not mc.is_equivalent(state.generated_practice_problem, case.original_problem)
            and case.original_problem not in response
        )
        checks.append(CheckResult(
            name="does_not_continue_original_problem",
            passed=ok,
            detail=f"generated={state.generated_practice_problem!r}, response={response!r}" if not ok else "",
        ))
    if is_final_turn and "must_not_treat_request_as_math_question" in behaviors:
        ok = state.student_step_validity == "" and state.action in _REQUEST_PRACTICE_ACTIONS
        checks.append(CheckResult(
            name="does_not_treat_request_as_math_question",
            passed=ok,
            detail=f"validity={state.student_step_validity!r}, action={state.action}" if not ok else "",
        ))
    if is_final_turn and "must_use_student_profile_for_difficulty" in behaviors:
        ok = state.practice_difficulty in {"easy", "medium", "hard"} and bool(state.practice_generation_reason)
        checks.append(CheckResult(
            name="uses_student_profile_for_difficulty",
            passed=ok,
            detail=f"difficulty={state.practice_difficulty!r}, reason={state.practice_generation_reason!r}" if not ok else "",
        ))
    if is_final_turn and "must_update_episode_to_practice_solving" in behaviors:
        ok = state.current_phase == "practice_solving" and state.pending_question_type == "next_step_question"
        checks.append(CheckResult(
            name="updates_episode_to_practice_solving",
            passed=ok,
            detail=f"phase={state.current_phase!r}, pending={state.pending_question_type!r}" if not ok else "",
        ))
    if is_final_turn and "must_generate_harder_exercise" in behaviors:
        ok = (
            state.action is TeachingAction.GENERATE_HARDER_EXERCISE
            and state.practice_difficulty == "hard"
            and bool(state.generated_practice_problem)
        )
        checks.append(CheckResult(
            name="generates_harder_exercise",
            passed=ok,
            detail=f"action={state.action}, difficulty={state.practice_difficulty}, generated={state.generated_practice_problem!r}" if not ok else "",
        ))
    if is_final_turn and "should_use_related_concept_or_higher_difficulty" in behaviors:
        generated = state.generated_practice_problem or ""
        ok = state.practice_difficulty == "hard" and ("(" in generated or "x" in generated.split("=")[-1])
        checks.append(CheckResult(
            name="uses_related_or_higher_difficulty",
            passed=ok,
            detail=f"generated={generated!r}, difficulty={state.practice_difficulty!r}" if not ok else "",
        ))

    if action not in _EXAMPLE_ALLOWED_ACTIONS:
        halluc = [
            e for e in _equations_excluding_student(response, student_msg)
            if not mc.is_equivalent(e, case.original_problem)
        ]
        checks.append(CheckResult(
            name="no_hallucinated_equations",
            passed=not halluc,
            detail=f"hallucinated equations: {halluc}" if halluc else "",
        ))

    return checks


def run_golden_conversation(case: GoldenCase, student_id: str | None = None) -> CaseResult:
    sid = student_id or f"eval_{case.case_id}_{uuid.uuid4().hex[:8]}"
    failures: list[str] = []
    final_given = False
    first_state: TeacherState | None = None

    for i, turn in enumerate(case.turns):
        if turn.role != "student":
            continue
        state = run_turn(sid, turn.content)
        if first_state is None:
            first_state = state
        is_final_turn = i == len(case.turns) - 1
        for check in evaluate_response(
            case,
            turn.content,
            state,
            final_answer_already_given=final_given,
            is_final_turn=is_final_turn,
        ):
            if not check.passed:
                failures.append(f"turn{i + 1}[{turn.content}] {check.name}: {check.detail}")
        if state.student_step_validity == StepValidity.CORRECT_FINAL_ANSWER.value:
            final_given = True

    if case.expected_concept and first_state is not None:
        if first_state.detected_concept_name != case.expected_concept:
            failures.append(
                f"turn1 concept: expected {case.expected_concept}, got {first_state.detected_concept_name}"
            )

    return CaseResult(
        case_id=case.case_id,
        case_type="golden",
        passed=not failures,
        failures=failures,
        num_turns=len(case.turns),
    )


def run_single_case(case: SingleTurnCase) -> CaseResult:
    sid = f"eval_{case.case_id}_{uuid.uuid4().hex[:8]}"
    state = run_turn(sid, case.message)
    failures: list[str] = []

    if case.expected_concept and state.detected_concept_name != case.expected_concept:
        failures.append(f"concept: expected {case.expected_concept}, got {state.detected_concept_name}")
    if case.expected_intent:
        actual = state.diagnosis.intent.value if state.diagnosis else ""
        if actual not in case.expected_intent.split("|"):
            failures.append(f"intent: expected {case.expected_intent}, got {actual}")
    if case.expected_action:
        actual = state.action.value if state.action else ""
        if actual not in case.expected_action.split("|"):
            failures.append(f"action: expected {case.expected_action}, got {actual}")
    bad = [p for p in case.forbidden_patterns if _norm(p) in _norm(state.response)]
    if bad:
        failures.append(f"matched forbidden patterns: {bad}")
    if not state.response:
        failures.append("empty response")

    return CaseResult(
        case_id=case.case_id,
        case_type="single_turn",
        passed=not failures,
        failures=failures,
        num_turns=1,
    )


def run_all() -> list[CaseResult]:
    results = [run_golden_conversation(c) for c in load_golden_cases()]
    results += [run_single_case(c) for c in load_single_turn_cases()]
    return results
