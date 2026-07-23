"""Deterministic, disk-first scoring for hybrid student traces."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from app.agent import math_checker as mc
from app.agent.workflow import is_step_message
from app.evaluation.hybrid_runner import (
    HybridConversationTrace,
    load_hybrid_traces,
)
from app.evaluation.mode_runner import EvalMode
from app.evaluation.student_simulator import (
    TerminationReason,
    UnderstandingStatus,
)


class HybridConversationScore(BaseModel):
    schema_version: str = "1.0.0"
    run_id: str
    scenario_id: str
    concept_id: str
    mastery_level: str
    mode: EvalMode
    teacher_quality_pass_rate: float = Field(ge=0.0, le=1.0)
    teacher_quality_checks: dict[str, bool] = Field(default_factory=dict)
    task_success: bool
    solution_progress: float = Field(ge=0.0, le=1.0)
    misconception_resolved: bool | None
    understanding_gain: int
    confidence_change: float
    teacher_turns: int = Field(ge=1, le=8)
    hint_exposure: int = Field(ge=0)
    termination_reason: TerminationReason


_ACK_MARKERS = ("正确", "对", "很好", "思路")
_REJECT_MARKERS = ("不是", "不对", "先停", "错", "有问题")
_WHY_MARKERS = ("两边同时", "等式", "保持相等")
_UNDERSTANDING_RANK = {
    UnderstandingStatus.CONFUSED: 0,
    UnderstandingStatus.PARTIAL: 1,
    UnderstandingStatus.UNDERSTOOD: 2,
    UnderstandingStatus.VERIFIED: 3,
}


def _norm(text: str) -> str:
    return "".join(text.split()).lower()


def _observable_teacher_checks(
    trace: HybridConversationTrace,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    final_answer_already_given = False

    for turn in trace.turns:
        prefix = f"turn{turn.teacher_turn_index}"
        student_message = turn.student_message
        response = turn.teacher_response
        student_equations_found = mc.find_equations(student_message)
        presents_original_as_question = (
            bool(student_equations_found)
            and all(
                _norm(equation) == _norm(trace.original_problem)
                for equation in student_equations_found
            )
            and any(
                marker in student_message
                for marker in ("老师", "怎么", "不会", "为什么", "总是", "吗", "?", "？")
            )
        )
        judged = (
            mc.check_step(trace.original_problem, student_message)
            if is_step_message(student_message) and not presents_original_as_question
            else None
        )
        student_has_final = bool(
            judged
            and judged.validity is mc.StepValidity.CORRECT_FINAL_ANSWER
        )
        answer_allowed = final_answer_already_given or student_has_final

        checks[f"{prefix}.non_empty_response"] = bool(response.strip())
        checks[f"{prefix}.asks_at_most_one_question"] = (
            response.count("?") + response.count("？") <= 1
        )
        checks[f"{prefix}.no_premature_final_answer"] = not (
            not answer_allowed
            and _norm(trace.expected_final_answer) in _norm(response)
        )

        student_equations = {
            _norm(equation) for equation in student_equations_found
        }
        response_equations = [
            equation
            for equation in mc.find_equations(response)
            if _norm(equation) not in student_equations
        ]
        unrelated = [
            equation
            for equation in response_equations
            if not mc.is_equivalent(equation, trace.original_problem)
        ]
        checks[f"{prefix}.no_unrelated_new_equation"] = not unrelated

        if judged and judged.validity in {
            mc.StepValidity.CORRECT_STEP,
            mc.StepValidity.CORRECT_FINAL_ANSWER,
        }:
            checks[f"{prefix}.recognizes_correct_step"] = any(
                marker in response for marker in _ACK_MARKERS
            )
        elif judged and judged.validity is mc.StepValidity.INCORRECT_STEP:
            checks[f"{prefix}.rejects_wrong_step"] = any(
                marker in response for marker in _REJECT_MARKERS
            )

        if "为什么" in student_message:
            checks[f"{prefix}.answers_why_question"] = any(
                marker in response for marker in _WHY_MARKERS
            )

        final_answer_already_given = answer_allowed
    return checks


def score_hybrid_trace(
    trace: HybridConversationTrace,
) -> HybridConversationScore:
    checks = _observable_teacher_checks(trace)
    passed = sum(1 for value in checks.values() if value)
    quality = round(passed / len(checks), 4) if checks else 0.0
    initial = trace.initial_student_state
    final = trace.final_student_state
    misconception_resolved = (
        None
        if initial.active_misconception is None
        else final.active_misconception is None
    )
    return HybridConversationScore(
        run_id=trace.run_id,
        scenario_id=trace.scenario_id,
        concept_id=trace.concept_id,
        mastery_level=trace.mastery_level,
        mode=trace.mode,
        teacher_quality_pass_rate=quality,
        teacher_quality_checks=checks,
        task_success=(
            trace.termination_reason is TerminationReason.SOLVED_AND_VERIFIED
        ),
        solution_progress=round(
            min(
                final.current_solution_step / trace.solution_step_count,
                1.0,
            ),
            4,
        ),
        misconception_resolved=misconception_resolved,
        understanding_gain=(
            _UNDERSTANDING_RANK[final.understanding_status]
            - _UNDERSTANDING_RANK[initial.understanding_status]
        ),
        confidence_change=round(final.confidence - initial.confidence, 4),
        teacher_turns=trace.teacher_turn_count,
        hint_exposure=final.hint_exposure,
        termination_reason=trace.termination_reason,
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _group_metrics(
    scores: list[HybridConversationScore],
) -> dict[str, object]:
    applicable_misconceptions = [
        score.misconception_resolved
        for score in scores
        if score.misconception_resolved is not None
    ]
    termination_distribution: dict[str, dict[str, float | int]] = {}
    for reason in TerminationReason:
        count = sum(
            1 for score in scores if score.termination_reason is reason
        )
        termination_distribution[reason.value] = {
            "count": count,
            "rate": round(count / len(scores), 4),
        }
    return {
        "conversation_count": len(scores),
        "teacher_quality_pass_rate": _mean(
            [score.teacher_quality_pass_rate for score in scores]
        ),
        "task_success_rate": _mean(
            [1.0 if score.task_success else 0.0 for score in scores]
        ),
        "solution_progress": _mean(
            [score.solution_progress for score in scores]
        ),
        "misconception_resolution_rate": (
            _mean([1.0 if value else 0.0 for value in applicable_misconceptions])
            if applicable_misconceptions
            else None
        ),
        "understanding_gain": _mean(
            [float(score.understanding_gain) for score in scores]
        ),
        "confidence_change": _mean(
            [score.confidence_change for score in scores]
        ),
        "teacher_turns": _mean(
            [float(score.teacher_turns) for score in scores]
        ),
        "hint_exposure": _mean(
            [float(score.hint_exposure) for score in scores]
        ),
        "termination_distribution": termination_distribution,
    }


def _build_group(
    scores: list[HybridConversationScore],
    key_fn: Callable[[HybridConversationScore], str],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[HybridConversationScore]] = defaultdict(list)
    for score in scores:
        grouped[key_fn(score)].append(score)
    return {
        key: _group_metrics(items)
        for key, items in sorted(grouped.items())
    }


def aggregate_hybrid_scores(
    scores: list[HybridConversationScore],
) -> dict[str, object]:
    if not scores:
        raise ValueError("cannot aggregate an empty score list")
    return {
        "schema_version": "1.0.0",
        "run_ids": sorted({score.run_id for score in scores}),
        "total_conversations": len(scores),
        "overall": _group_metrics(scores),
        "groups": {
            "by_mode": _build_group(scores, lambda score: score.mode.value),
            "by_mode_concept": _build_group(
                scores,
                lambda score: f"{score.mode.value}|{score.concept_id}",
            ),
            "by_mode_mastery": _build_group(
                scores,
                lambda score: f"{score.mode.value}|{score.mastery_level}",
            ),
            "by_mode_concept_mastery": _build_group(
                scores,
                lambda score: (
                    f"{score.mode.value}|{score.concept_id}|"
                    f"{score.mastery_level}"
                ),
            ),
        },
    }


def write_hybrid_scores(
    scores: list[HybridConversationScore],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(
                json.dumps(score.model_dump(mode="json"), ensure_ascii=False)
                + "\n"
            )


def score_trace_file(
    trace_path: Path,
    score_path: Path,
    summary_path: Path,
) -> tuple[list[HybridConversationScore], dict[str, object]]:
    """Re-read closed traces from disk, then score and persist results."""
    traces = load_hybrid_traces(trace_path)
    if not traces:
        raise ValueError("trace file contains no conversations")
    scores = [score_hybrid_trace(trace) for trace in traces]
    summary = aggregate_hybrid_scores(scores)
    write_hybrid_scores(scores, score_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return scores, summary
