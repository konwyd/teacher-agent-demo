"""Comparable evaluation runners for DeepSeek-only and Agent ablations."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from app.agent import math_checker as mc
from app.agent.math_checker import StepValidity
from app.agent.state import TeacherState
from app.agent.workflow import BASIC_LANGUAGE_MARKERS, is_step_message, run_turn
from app.evaluation.evaluator import (
    CheckResult,
    GoldenCase,
    SingleTurnCase,
    evaluate_response,
    load_golden_cases,
    load_single_turn_cases,
)
from app.llm.deepseek_client import DeepSeekClient, DeepSeekError, is_configured


class EvalMode(str, Enum):
    DEEPSEEK_ONLY = "deepseek_only"
    AGENT_NO_GRAPH = "agent_no_graph"
    AGENT_WITH_GRAPH = "agent_with_graph"


class ModeConfiguration(BaseModel):
    mode: EvalMode
    uses_workflow: bool
    uses_math_checker: bool
    uses_memory: bool
    uses_graph_retrieval: bool


class TeacherQualityMetrics(BaseModel):
    total_checks: int = 0
    passed_checks: int = 0
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    check_results: dict[str, bool] = Field(default_factory=dict)


class ModeCaseResult(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    mode: EvalMode
    case_id: str
    case_type: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    teacher_quality_metrics: TeacherQualityMetrics = Field(
        default_factory=TeacherQualityMetrics
    )
    final_response: str = ""
    diagnosis: dict[str, Any] = Field(default_factory=dict)
    teaching_action: str = ""
    retrieval_backend: str = "disabled"
    grounding_fields: dict[str, Any] = Field(default_factory=dict)
    turn_outputs: list[dict[str, Any]] = Field(default_factory=list)


def get_mode_configuration(mode: EvalMode | str) -> ModeConfiguration:
    selected = EvalMode(mode)
    if selected is EvalMode.DEEPSEEK_ONLY:
        return ModeConfiguration(
            mode=selected,
            uses_workflow=False,
            uses_math_checker=False,
            uses_memory=False,
            uses_graph_retrieval=False,
        )
    if selected is EvalMode.AGENT_NO_GRAPH:
        return ModeConfiguration(
            mode=selected,
            uses_workflow=True,
            uses_math_checker=True,
            uses_memory=True,
            uses_graph_retrieval=False,
        )
    return ModeConfiguration(
        mode=selected,
        uses_workflow=True,
        uses_math_checker=True,
        uses_memory=True,
        uses_graph_retrieval=True,
    )


@contextmanager
def _graph_setting(enabled: bool) -> Iterator[None]:
    previous = os.environ.get("USE_PEDAGOGICAL_RAG")
    os.environ["USE_PEDAGOGICAL_RAG"] = "true" if enabled else "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("USE_PEDAGOGICAL_RAG", None)
        else:
            os.environ["USE_PEDAGOGICAL_RAG"] = previous


def _metrics(checks: list[tuple[str, CheckResult]]) -> TeacherQualityMetrics:
    results = {label: check.passed for label, check in checks}
    total = len(checks)
    passed = sum(1 for _, check in checks if check.passed)
    return TeacherQualityMetrics(
        total_checks=total,
        passed_checks=passed,
        pass_rate=round(passed / total, 4) if total else 0.0,
        check_results=results,
    )


def _failure_messages(checks: list[tuple[str, CheckResult]]) -> list[str]:
    return [
        f"{label}: {check.detail or check.name}"
        for label, check in checks
        if not check.passed
    ]


def _agent_grounding(state: TeacherState) -> dict[str, Any]:
    return {
        "original_problem": state.original_problem,
        "target_equation": state.target_equation,
        "student_step_validity": state.student_step_validity,
        "whether_response_used_new_example": bool(
            state.generated_practice_problem or state.recommended_exercise
        ),
        "whether_response_returned_to_original_problem": bool(
            state.original_problem and state.original_problem in state.response
        ),
    }


def _agent_metadata(state: TeacherState, mode: EvalMode) -> dict[str, Any]:
    return {
        "diagnosis": (
            state.diagnosis.model_dump(mode="json") if state.diagnosis else {}
        ),
        "teaching_action": state.action.value if state.action else "",
        "retrieval_backend": (
            state.retrieval_backend
            if mode is EvalMode.AGENT_WITH_GRAPH and state.retrieval_backend
            else "disabled"
        ),
        "grounding_fields": _agent_grounding(state),
    }


def _run_agent_golden(case: GoldenCase, mode: EvalMode) -> ModeCaseResult:
    sid = f"mode_{mode.value}_{case.case_id}_{uuid.uuid4().hex[:8]}"
    conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
    checks: list[tuple[str, CheckResult]] = []
    outputs: list[dict[str, Any]] = []
    final_answer_already_given = False
    final_state: TeacherState | None = None

    for index, turn in enumerate(case.turns):
        if turn.role != "student":
            continue
        state = run_turn(sid, turn.content, conversation_id=conversation_id)
        final_state = state
        is_final_turn = index == len(case.turns) - 1
        turn_checks = evaluate_response(
            case,
            turn.content,
            state,
            final_answer_already_given=final_answer_already_given,
            is_final_turn=is_final_turn,
        )
        checks.extend(
            (f"turn{index + 1}.{check.name}", check) for check in turn_checks
        )
        outputs.append(
            {
                "student_message": turn.content,
                "teacher_response": state.response,
            }
        )
        if state.student_step_validity == "correct_final_answer":
            final_answer_already_given = True

    if final_state is None:
        checks.append(
            (
                "conversation.non_empty",
                CheckResult(
                    name="non_empty_conversation",
                    passed=False,
                    detail="case has no student turn",
                ),
            )
        )
        final_state = TeacherState(student_id=sid, message="")

    failures = _failure_messages(checks)
    metadata = _agent_metadata(final_state, mode)
    return ModeCaseResult(
        mode=mode,
        case_id=case.case_id,
        case_type="golden",
        passed=not failures,
        failures=failures,
        teacher_quality_metrics=_metrics(checks),
        final_response=final_state.response,
        turn_outputs=outputs,
        **metadata,
    )


def _single_agent_checks(
    case: SingleTurnCase,
    state: TeacherState,
) -> list[tuple[str, CheckResult]]:
    checks: list[tuple[str, CheckResult]] = []
    if case.expected_concept:
        checks.append(
            (
                "expected_concept",
                CheckResult(
                    name="expected_concept",
                    passed=state.detected_concept_name == case.expected_concept,
                    detail=(
                        f"expected {case.expected_concept}, "
                        f"got {state.detected_concept_name}"
                    ),
                ),
            )
        )
    if case.expected_intent:
        actual = state.diagnosis.intent.value if state.diagnosis else ""
        checks.append(
            (
                "expected_intent",
                CheckResult(
                    name="expected_intent",
                    passed=actual in case.expected_intent.split("|"),
                    detail=f"expected {case.expected_intent}, got {actual}",
                ),
            )
        )
    if case.expected_action:
        actual = state.action.value if state.action else ""
        checks.append(
            (
                "expected_action",
                CheckResult(
                    name="expected_action",
                    passed=actual in case.expected_action.split("|"),
                    detail=f"expected {case.expected_action}, got {actual}",
                ),
            )
        )
    bad = [item for item in case.forbidden_patterns if item.replace(" ", "") in state.response.replace(" ", "")]
    checks.extend(
        [
            (
                "no_forbidden_patterns",
                CheckResult(
                    name="no_forbidden_patterns",
                    passed=not bad,
                    detail=f"matched forbidden patterns: {bad}" if bad else "",
                ),
            ),
            (
                "non_empty_response",
                CheckResult(
                    name="non_empty_response",
                    passed=bool(state.response.strip()),
                    detail="empty response",
                ),
            ),
        ]
    )
    return checks


def _run_agent_single(case: SingleTurnCase, mode: EvalMode) -> ModeCaseResult:
    sid = f"mode_{mode.value}_{case.case_id}_{uuid.uuid4().hex[:8]}"
    state = run_turn(
        sid,
        case.message,
        conversation_id=f"conv_{uuid.uuid4().hex[:12]}",
    )
    checks = _single_agent_checks(case, state)
    failures = _failure_messages(checks)
    return ModeCaseResult(
        mode=mode,
        case_id=case.case_id,
        case_type="single_turn",
        passed=not failures,
        failures=failures,
        teacher_quality_metrics=_metrics(checks),
        final_response=state.response,
        turn_outputs=[
            {
                "student_message": case.message,
                "teacher_response": state.response,
            }
        ],
        **_agent_metadata(state, mode),
    )


_DEEPSEEK_ONLY_SYSTEM = """你是一名中文初中数学老师。
直接回应学生当前消息，不使用外部工具、知识图谱、学生记忆或结构化 Agent 流程。
尽量循序渐进，不要无故引入新题，不要一次提出多个问题。"""


def _offline_deepseek_only_response(student_message: str) -> str:
    """Deterministic direct-model baseline used when real DeepSeek is disabled."""
    if any(marker in student_message for marker in ("出一道", "来一题", "来一道", "练一下")):
        return "可以。你希望继续练同类型，还是换一个知识点？"
    if "为什么" in student_message:
        return "我会先解释你问的原因，再继续这道题。你最不理解的是哪一步？"
    return "我们先只看你当前最困惑的地方。你觉得第一步应该从哪里开始？"


def _direct_deepseek_response(
    student_message: str,
    transcript: list[dict[str, str]],
    use_real_llm: bool,
) -> str:
    if not use_real_llm or not is_configured():
        return _offline_deepseek_only_response(student_message)
    history = "\n".join(
        f"{item['role']}：{item['content']}" for item in transcript[-8:]
    )
    user_prompt = (
        f"此前对话：\n{history or '无'}\n\n"
        f"学生当前消息：{student_message}\n"
        "请直接给出教师回复。"
    )
    return DeepSeekClient().chat(
        system=_DEEPSEEK_ONLY_SYSTEM,
        user=user_prompt,
        temperature=0.2,
    )


def _response_only_checks(
    response: str,
    forbidden_patterns: list[str],
    expected_final_answer: str = "",
    final_answer_was_in_student_message: bool = False,
) -> list[tuple[str, CheckResult]]:
    normalized = response.replace(" ", "")
    bad = [
        item
        for item in forbidden_patterns
        if item.replace(" ", "") in normalized
    ]
    question_count = response.count("?") + response.count("？")
    leaked = bool(
        expected_final_answer
        and not final_answer_was_in_student_message
        and expected_final_answer.replace(" ", "") in normalized
    )
    return [
        (
            "non_empty_response",
            CheckResult(
                name="non_empty_response",
                passed=bool(response.strip()),
                detail="empty response",
            ),
        ),
        (
            "no_forbidden_patterns",
            CheckResult(
                name="no_forbidden_patterns",
                passed=not bad,
                detail=f"matched forbidden patterns: {bad}" if bad else "",
            ),
        ),
        (
            "asks_at_most_one_question",
            CheckResult(
                name="asks_at_most_one_question",
                passed=question_count <= 1,
                detail=f"question_count={question_count}",
            ),
        ),
        (
            "no_premature_final_answer",
            CheckResult(
                name="no_premature_final_answer",
                passed=not leaked,
                detail=(
                    f"premature final answer {expected_final_answer}"
                    if leaked
                    else ""
                ),
            ),
        ),
    ]


def _observable_golden_checks(
    case: GoldenCase,
    student_message: str,
    response: str,
    *,
    final_answer_already_given: bool,
    is_final_turn: bool,
) -> list[tuple[str, CheckResult]]:
    checks = _response_only_checks(
        response,
        case.forbidden_patterns,
        case.expected_final_answer,
        final_answer_was_in_student_message=(
            final_answer_already_given
            or (
                bool(case.expected_final_answer)
                and case.expected_final_answer.replace(" ", "")
                in student_message.replace(" ", "")
            )
        ),
    )
    student_equations = {
        equation.replace(" ", "") for equation in mc.find_equations(student_message)
    }
    response_equations = [
        equation
        for equation in mc.find_equations(response)
        if equation.replace(" ", "") not in student_equations
    ]
    unrelated = [
        equation
        for equation in response_equations
        if not mc.is_equivalent(equation, case.original_problem)
    ]
    checks.append(
        (
            "no_unrelated_new_example",
            CheckResult(
                name="no_unrelated_new_example",
                passed=not unrelated,
                detail=f"unrelated equations: {unrelated}" if unrelated else "",
            ),
        )
    )

    if is_step_message(student_message):
        judged = mc.check_step(case.original_problem, student_message)
        if judged.validity in (
            StepValidity.CORRECT_STEP,
            StepValidity.CORRECT_FINAL_ANSWER,
        ):
            recognized = any(
                marker in response for marker in ("正确", "对", "很好", "思路")
            )
            checks.append(
                (
                    "recognizes_correct_step",
                    CheckResult(
                        name="recognizes_correct_step",
                        passed=recognized,
                        detail="correct student step was not acknowledged",
                    ),
                )
            )
        elif judged.validity is StepValidity.INCORRECT_STEP:
            rejected = any(
                marker in response for marker in ("不是", "不对", "先停", "错")
            )
            checks.append(
                (
                    "rejects_wrong_step",
                    CheckResult(
                        name="rejects_wrong_step",
                        passed=rejected,
                        detail="incorrect student step was not corrected",
                    ),
                )
            )

    if "为什么" in student_message:
        explains_why = any(
            marker in response for marker in ("两边同时", "等式", "保持相等")
        )
        checks.append(
            (
                "answers_why_question",
                CheckResult(
                    name="answers_why_question",
                    passed=explains_why,
                    detail="why-question lacks equality-based explanation",
                ),
            )
        )

    behaviors = set(case.required_behaviors)
    question_count = response.count("?") + response.count("？")
    if "must_ask_one_next_question" in behaviors:
        checks.append(
            (
                "asks_exactly_one_question",
                CheckResult(
                    name="asks_exactly_one_question",
                    passed=question_count == 1,
                    detail=f"question_count={question_count}",
                ),
            )
        )
    if is_final_turn and "must_adapt_low_mastery" in behaviors:
        adapted = any(marker in response for marker in BASIC_LANGUAGE_MARKERS)
        checks.append(
            (
                "adapts_low_mastery",
                CheckResult(
                    name="adapts_low_mastery",
                    passed=adapted,
                    detail="response lacks basic-language adaptation",
                ),
            )
        )
    if is_final_turn and "must_not_overexplain_high_mastery" in behaviors:
        practice_like = any(
            marker in response for marker in ("练习", "来一题", "来一道", "变式")
        )
        checks.append(
            (
                "no_overexplain_high_mastery",
                CheckResult(
                    name="no_overexplain_high_mastery",
                    passed=practice_like,
                    detail="response does not shift high-mastery student to practice",
                ),
            )
        )
    if is_final_turn and "must_accept_multi_step_plan" in behaviors:
        accepted = any(
            marker in response for marker in ("正确", "对", "很好", "思路")
        )
        checks.append(
            (
                "accepts_multi_step_plan",
                CheckResult(
                    name="accepts_multi_step_plan",
                    passed=accepted,
                    detail="multi-step plan was not acknowledged",
                ),
            )
        )
    if is_final_turn and "must_not_force_single_step" in behaviors:
        checks.append(
            (
                "not_forcing_single_step",
                CheckResult(
                    name="not_forcing_single_step",
                    passed="除以" in response,
                    detail="response did not preserve the student's multi-step plan",
                ),
            )
        )
    if is_final_turn and "must_not_repeat_already_answered_step" in behaviors:
        repeated = any(
            marker in response for marker in ("减去几", "减几", "第一步应该做什么")
        )
        checks.append(
            (
                "not_repeating_answered_step",
                CheckResult(
                    name="not_repeating_answered_step",
                    passed=not repeated,
                    detail="response repeats an already answered step",
                ),
            )
        )
    if is_final_turn and "must_confirm_verification" in behaviors:
        confirmed = any(
            marker in response for marker in ("验算", "两边", "完成", "正确")
        )
        checks.append(
            (
                "confirms_verification",
                CheckResult(
                    name="confirms_verification",
                    passed=confirmed,
                    detail="verification result was not confirmed",
                ),
            )
        )
    if is_final_turn and "must_summarize_original_problem" in behaviors:
        summarized = mc.mentions_equation(response, case.original_problem)
        checks.append(
            (
                "summarizes_original_problem",
                CheckResult(
                    name="summarizes_original_problem",
                    passed=summarized,
                    detail="response does not summarize the original problem",
                ),
            )
        )

    memory_behaviors = {
        "must_track_repeated_misconception",
        "must_create_verification_success_event",
        "must_record_repeated_weakness",
        "must_update_student_profile",
        "must_update_episode_to_practice_solving",
        "must_use_student_profile_for_difficulty",
    }
    if is_final_turn:
        for behavior in sorted(behaviors & memory_behaviors):
            checks.append(
                (
                    behavior,
                    CheckResult(
                        name=behavior,
                        passed=False,
                        detail="deepseek_only has no memory or episode-state component",
                    ),
                )
            )

    if is_final_turn and "must_detect_practice_request" in behaviors:
        detected = any(
            marker in response for marker in ("题", "练习", "可以")
        )
        checks.append(
            (
                "detects_practice_request",
                CheckResult(
                    name="detects_practice_request",
                    passed=detected,
                    detail="practice request was not acknowledged",
                ),
            )
        )
    if is_final_turn and "must_generate_new_exercise" in behaviors:
        generated = any(
            not mc.is_equivalent(equation, case.original_problem)
            for equation in response_equations
        )
        checks.append(
            (
                "generates_new_exercise",
                CheckResult(
                    name="generates_new_exercise",
                    passed=generated,
                    detail="response does not contain a new exercise",
                ),
            )
        )
    if is_final_turn and "must_generate_harder_exercise" in behaviors:
        harder = any(
            "(" in equation
            or (
                "=" in equation
                and "x" in equation.split("=", maxsplit=1)[1]
            )
            for equation in response_equations
        )
        checks.append(
            (
                "generates_harder_exercise",
                CheckResult(
                    name="generates_harder_exercise",
                    passed=harder,
                    detail="response does not contain a harder variation",
                ),
            )
        )
    return checks


def _run_deepseek_golden(
    case: GoldenCase,
    use_real_llm: bool,
) -> ModeCaseResult:
    transcript: list[dict[str, str]] = []
    outputs: list[dict[str, Any]] = []
    checks: list[tuple[str, CheckResult]] = []
    final_response = ""
    final_answer_already_given = False
    try:
        for index, turn in enumerate(case.turns):
            if turn.role != "student":
                continue
            final_response = _direct_deepseek_response(
                turn.content,
                transcript,
                use_real_llm,
            )
            checks.extend(
                (
                    f"turn{index + 1}.{label}",
                    check,
                )
                for label, check in _observable_golden_checks(
                    case,
                    turn.content,
                    final_response,
                    final_answer_already_given=final_answer_already_given,
                    is_final_turn=index == len(case.turns) - 1,
                )
            )
            transcript.extend(
                [
                    {"role": "student", "content": turn.content},
                    {"role": "teacher", "content": final_response},
                ]
            )
            outputs.append(
                {
                    "student_message": turn.content,
                    "teacher_response": final_response,
                }
            )
            if (
                case.expected_final_answer
                and case.expected_final_answer.replace(" ", "")
                in turn.content.replace(" ", "")
            ):
                final_answer_already_given = True
    except DeepSeekError as exc:
        checks.append(
            (
                "deepseek_api",
                CheckResult(
                    name="deepseek_api",
                    passed=False,
                    detail=str(exc),
                ),
            )
        )
    failures = _failure_messages(checks)
    return ModeCaseResult(
        mode=EvalMode.DEEPSEEK_ONLY,
        case_id=case.case_id,
        case_type="golden",
        passed=not failures,
        failures=failures,
        teacher_quality_metrics=_metrics(checks),
        final_response=final_response,
        retrieval_backend="disabled",
        grounding_fields={
            "original_problem": case.original_problem,
            "target_equation": case.original_problem,
            "student_step_validity": "",
            "whether_response_used_new_example": False,
            "whether_response_returned_to_original_problem": False,
        },
        turn_outputs=outputs,
    )


def _run_deepseek_single(
    case: SingleTurnCase,
    use_real_llm: bool,
) -> ModeCaseResult:
    response = ""
    checks: list[tuple[str, CheckResult]]
    try:
        response = _direct_deepseek_response(case.message, [], use_real_llm)
        checks = _response_only_checks(response, case.forbidden_patterns)
    except DeepSeekError as exc:
        checks = [
            (
                "deepseek_api",
                CheckResult(
                    name="deepseek_api",
                    passed=False,
                    detail=str(exc),
                ),
            )
        ]
    failures = _failure_messages(checks)
    return ModeCaseResult(
        mode=EvalMode.DEEPSEEK_ONLY,
        case_id=case.case_id,
        case_type="single_turn",
        passed=not failures,
        failures=failures,
        teacher_quality_metrics=_metrics(checks),
        final_response=response,
        retrieval_backend="disabled",
        grounding_fields={
            "original_problem": "",
            "target_equation": "",
            "student_step_validity": "",
            "whether_response_used_new_example": False,
            "whether_response_returned_to_original_problem": False,
        },
        turn_outputs=[
            {
                "student_message": case.message,
                "teacher_response": response,
            }
        ],
    )


def run_mode(
    mode: EvalMode | str,
    *,
    golden_cases: list[GoldenCase] | None = None,
    single_turn_cases: list[SingleTurnCase] | None = None,
    use_real_llm: bool = False,
) -> list[ModeCaseResult]:
    selected = EvalMode(mode)
    golden = load_golden_cases() if golden_cases is None else golden_cases
    singles = (
        load_single_turn_cases()
        if single_turn_cases is None
        else single_turn_cases
    )
    if selected is EvalMode.DEEPSEEK_ONLY:
        return [
            *(_run_deepseek_golden(case, use_real_llm) for case in golden),
            *(_run_deepseek_single(case, use_real_llm) for case in singles),
        ]

    graph_enabled = selected is EvalMode.AGENT_WITH_GRAPH
    with _graph_setting(graph_enabled):
        return [
            *(_run_agent_golden(case, selected) for case in golden),
            *(_run_agent_single(case, selected) for case in singles),
        ]


def write_mode_results(
    results: list[ModeCaseResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    result.model_dump(mode="json"),
                    ensure_ascii=False,
                )
                + "\n"
            )
