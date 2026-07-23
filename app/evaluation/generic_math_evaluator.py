"""Offline behavioural evaluation for the generic math teaching harness."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.agent.state import GenericMathReasoning, VerificationStatus
from app.agent.teaching_actions import DisclosureLevel
from app.agent.teaching_brief import TeachingBrief
from app.agent.teaching_harness import run_generic_teaching_turn


EVAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"
SMOKE_CASES_PATH = EVAL_DATA_DIR / "topic_agnostic_smoke.jsonl"


class GenericMathSmokeCase(BaseModel):
    id: str
    domain: str
    student_id: str
    message: str
    allowed_actions: list[str]
    expected_action: str
    candidate_conclusion: str
    student_requests_answer: bool
    forbidden_answer_tokens: list[str] = Field(default_factory=list)


class GenericMathCaseResult(BaseModel):
    case_id: str
    case_type: str = "generic_math_smoke"
    domain: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    response: str = ""


def load_generic_math_smoke_cases(
    path: Path | None = None,
) -> list[GenericMathSmokeCase]:
    source = path or SMOKE_CASES_PATH
    return [
        GenericMathSmokeCase.model_validate(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class _FixtureReasoner:
    def __init__(self, case: GenericMathSmokeCase) -> None:
        self._case = case

    def analyze(self, message, problem_context, analysis) -> GenericMathReasoning:
        del message, problem_context, analysis
        return GenericMathReasoning(
            problem_interpretation="题目给出了待整理的数量或关系。",
            reasoning_steps=["先把题目中的条件和所求内容对应起来。"],
            next_step_hint="先写出条件之间最直接的数量或图形关系。",
            candidate_conclusion=self._case.candidate_conclusion,
            confidence=0.9,
        )


class _FixtureResponseClient:
    def chat(self, system, user, temperature=0.2, max_tokens=None) -> str:
        del system, temperature, max_tokens
        brief = TeachingBrief.model_validate_json(user)
        if (
            brief.disclosure_ceiling is DisclosureLevel.FINAL
            and brief.allowed_content
        ):
            conclusion = brief.allowed_content[-1]
            return f"根据题目条件逐步整理，可以得到：{conclusion} 你能复述关键关系吗？"
        return "先把题目中的条件和所求内容对应起来。你准备先写出哪一条关系？"


def _evaluate_case(case: GenericMathSmokeCase) -> GenericMathCaseResult:
    state = run_generic_teaching_turn(
        case.student_id,
        case.message,
        reasoner=_FixtureReasoner(case),
        response_client=_FixtureResponseClient(),
        analysis_allow_llm=False,
    )
    failures: list[str] = []
    action = state.action_plan.action.value if state.action_plan else ""
    if action != case.expected_action:
        failures.append(f"expected action {case.expected_action}, got {action}")
    if state.response.count("？") + state.response.count("?") != 1:
        failures.append("response does not contain exactly one question")
    if not state.response.endswith(("？", "?")):
        failures.append("response question is not final")
    if state.verification.status in {
        VerificationStatus.VERIFIED_CORRECT,
        VerificationStatus.VERIFIED_INCORRECT,
    }:
        failures.append("generic turn claimed a verified mathematical verdict")
    candidate_visible = case.candidate_conclusion in state.response
    if case.student_requests_answer:
        if not state.answer_disclosure_allowed:
            failures.append("explicit answer request did not receive disclosure permission")
        if not candidate_visible:
            failures.append("authorized candidate conclusion is missing")
    else:
        if state.answer_disclosure_allowed:
            failures.append("candidate disclosure was authorized without an answer request")
        if candidate_visible:
            failures.append("candidate conclusion leaked without an answer request")
    return GenericMathCaseResult(
        case_id=case.id,
        domain=case.domain,
        passed=not failures,
        failures=failures,
        response=state.response,
    )


def run_generic_math_eval(
    path: Path | None = None,
) -> list[GenericMathCaseResult]:
    """Evaluate the generic harness with deterministic cross-domain fixtures."""

    return [_evaluate_case(case) for case in load_generic_math_smoke_cases(path)]
