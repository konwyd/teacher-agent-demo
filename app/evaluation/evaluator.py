"""Evaluation harness：评估 Teacher Agent 是否循序渐进、紧扣原题地教学。

- run_single_case():          单轮 case（诊断/动作/禁用模式）
- run_golden_conversation():  多轮 golden 对话回放
- evaluate_response():        单轮回复的行为检查

默认 mock LLM（不调真实 DeepSeek）；设 USE_REAL_LLM_FOR_EVAL=true 才走真实 API
（该开关由 scripts/run_eval.py 处理：非 true 时移除 DEEPSEEK_API_KEY）。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from app.agent import math_checker as mc
from app.agent.math_checker import StepValidity
from app.agent.state import TeacherState, TeachingAction
from app.agent.workflow import is_step_message, run_turn

EVAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"

_EXAMPLE_ALLOWED_ACTIONS = (
    TeachingAction.SHOW_WORKED_EXAMPLE.value,
    TeachingAction.GIVE_PRACTICE_QUESTION.value,
)


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
    case_type: str  # "golden" | "single_turn"
    passed: bool
    failures: list[str] = Field(default_factory=list)
    num_turns: int = 0


def load_golden_cases(path: Path | None = None) -> list[GoldenCase]:
    p = path or (EVAL_DATA_DIR / "golden_conversations.jsonl")
    return [GoldenCase.model_validate(json.loads(line))
            for line in p.read_text(encoding="utf-8").strip().splitlines()]


def load_single_turn_cases(path: Path | None = None) -> list[SingleTurnCase]:
    p = path or (EVAL_DATA_DIR / "single_turn_cases.jsonl")
    return [SingleTurnCase.model_validate(json.loads(line))
            for line in p.read_text(encoding="utf-8").strip().splitlines()]


def _norm(text: str) -> str:
    return text.replace(" ", "")


def _equations_excluding_student(response: str, student_msg: str) -> list[str]:
    """回复中的方程，排除学生本轮消息里已出现的（引用学生的式子不算引入新题）。"""
    student_eqs = {_norm(e) for e in mc.find_equations(student_msg)}
    return [e for e in mc.find_equations(response) if _norm(e) not in student_eqs]


def evaluate_response(
    case: GoldenCase,
    student_msg: str,
    state: TeacherState,
    final_answer_already_given: bool = False,
    is_final_turn: bool = False,
) -> list[CheckResult]:
    """对一轮教师回复做行为检查（全部确定性，不依赖 LLM）。"""
    checks: list[CheckResult] = []
    response = state.response
    action = state.action.value if state.action else ""
    behaviors = set(case.required_behaviors)

    # 1) 是否保持 original_problem
    keeps = bool(state.original_problem) and mc.is_equivalent(state.original_problem, case.original_problem)
    checks.append(CheckResult(
        name="keeps_original_problem", passed=keeps,
        detail=f"state.original_problem={state.original_problem!r}，期望等价于 {case.original_problem!r}"))

    # 2) 禁用模式（无关新题/固定坏话术）
    bad = [pat for pat in case.forbidden_patterns if _norm(pat) in _norm(response)]
    checks.append(CheckResult(
        name="no_forbidden_patterns", passed=not bad, detail=f"命中禁用模式：{bad}" if bad else ""))

    # 3) 是否错误引入无关新题（引用学生自己的式子不算；worked_example/practice 允许）
    if action not in _EXAMPLE_ALLOWED_ACTIONS:
        alien = [e for e in _equations_excluding_student(response, student_msg)
                 if not mc.is_equivalent(e, case.original_problem)]
        checks.append(CheckResult(
            name="no_unrelated_new_example", passed=not alien,
            detail=f"出现与原题不同解的新方程：{alien}" if alien else ""))

    # 4/5) 步骤判定：以 math_checker 对学生消息的独立判定为参照
    # 仅对“步骤回答”类消息检查（陈述题目/求助不算步骤，与 diagnose 同一门控）
    ref = mc.check_step(case.original_problem, student_msg) if is_step_message(student_msg) \
        else None
    if ref is not None and ref.validity in (StepValidity.CORRECT_STEP, StepValidity.CORRECT_FINAL_ANSWER):
        recognized = state.student_step_validity == ref.validity.value and "正确" in response
        checks.append(CheckResult(
            name="recognizes_correct_step", passed=recognized,
            detail=f"validity={state.student_step_validity!r}，回复未明确肯定" if not recognized else ""))
    elif ref is not None and ref.validity is StepValidity.INCORRECT_STEP:
        rejected = state.student_step_validity == "incorrect_step" and (
            "不是" in response or "先停" in response or "不对" in response)
        checks.append(CheckResult(
            name="rejects_wrong_step", passed=rejected,
            detail=f"validity={state.student_step_validity!r}，回复未指出错误" if not rejected else ""))

    # 6) 是否回答学生当前问题（启发式）
    if "为什么" in student_msg:
        answered = ("两边同时" in response or "等式" in response)
        checks.append(CheckResult(
            name="answers_why_question", passed=answered,
            detail="学生问为什么，回复未解释等式性质" if not answered else ""))

    # 7) 每轮最多一个问题
    n_q = response.count("？") + response.count("?")
    checks.append(CheckResult(
        name="asks_at_most_one_question", passed=n_q <= 1, detail=f"问题数={n_q}"))
    if "must_ask_one_next_question" in behaviors:
        checks.append(CheckResult(
            name="asks_exactly_one_question", passed=n_q == 1, detail=f"问题数={n_q}"))

    # 8) 是否过早直接给最终答案
    if case.expected_final_answer and not final_answer_already_given \
            and state.student_step_validity != StepValidity.CORRECT_FINAL_ANSWER.value:
        leaked = _norm(case.expected_final_answer) in _norm(response)
        checks.append(CheckResult(
            name="no_premature_final_answer", passed=not leaked,
            detail=f"学生尚未解出，回复提前出现 {case.expected_final_answer}" if leaked else ""))

    # mastery 相关行为（只在最后一轮判定，此时学生历史已经累积完成）
    if is_final_turn and "must_adapt_low_mastery" in behaviors:
        from app.agent.workflow import BASIC_LANGUAGE_MARKERS
        ok = (state.action is TeachingAction.ASK_DIAGNOSTIC_QUESTION
              and any(m in response for m in BASIC_LANGUAGE_MARKERS))
        checks.append(CheckResult(
            name="adapts_low_mastery", passed=ok,
            detail=f"低掌握度学生期望诊断式提问+基础语言，实际 action={action!r}" if not ok else ""))
    if is_final_turn and "must_not_overexplain_high_mastery" in behaviors:
        ok = state.action is TeachingAction.GIVE_PRACTICE_QUESTION
        checks.append(CheckResult(
            name="no_overexplain_high_mastery", passed=ok,
            detail=f"高掌握度学生期望变式练习，实际 action={action!r}" if not ok else ""))
    if is_final_turn and "must_track_repeated_misconception" in behaviors:
        ok = any("变号" in m for m in state.known_misconceptions)
        checks.append(CheckResult(
            name="tracks_repeated_misconception", passed=ok,
            detail=f"重复错误应识别为同一误因，known={state.known_misconceptions}" if not ok else ""))

    if is_final_turn and "must_accept_multi_step_plan" in behaviors:
        ok = (state.student_step_validity in ("correct_step", "correct_final_answer")
              and ("思路" in response or "正确" in response))
        checks.append(CheckResult(
            name="accepts_multi_step_plan", passed=ok,
            detail=f"多步计划未被整体认可：validity={state.student_step_validity!r}" if not ok else ""))
    if is_final_turn and "must_not_force_single_step" in behaviors:
        ok = "除以" in response  # 第二步被展开确认，而不是拉回第一步
        checks.append(CheckResult(
            name="not_forcing_single_step", passed=ok,
            detail="回复没有认可计划中的后续步骤" if not ok else ""))
    if is_final_turn and "must_not_repeat_already_answered_step" in behaviors:
        ok = ("减去几" not in response and "减几" not in response
              and "第一步应该做什么" not in response)
        checks.append(CheckResult(
            name="not_repeating_answered_step", passed=ok,
            detail="回复重复追问了学生已经回答过的步骤" if not ok else ""))

    # 9) 是否 hallucinate 不存在的方程（非例题动作下，回复中的方程必须与原题同解或来自学生）
    if action not in _EXAMPLE_ALLOWED_ACTIONS:
        halluc = [e for e in _equations_excluding_student(response, student_msg)
                  if not mc.is_equivalent(e, case.original_problem)]
        checks.append(CheckResult(
            name="no_hallucinated_equations", passed=not halluc,
            detail=f"出现来源不明的方程：{halluc}" if halluc else ""))

    return checks


def run_golden_conversation(case: GoldenCase, student_id: str | None = None) -> CaseResult:
    """回放一组多轮对话并逐轮评测。"""
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
        is_final_turn = (i == len(case.turns) - 1)
        for check in evaluate_response(
            case, turn.content, state,
            final_answer_already_given=final_given, is_final_turn=is_final_turn,
        ):
            if not check.passed:
                failures.append(f"turn{i + 1}[{turn.content}] {check.name}: {check.detail}")
        if state.student_step_validity == StepValidity.CORRECT_FINAL_ANSWER.value:
            final_given = True

    # 知识点检查（以第一轮为准）
    if case.expected_concept and first_state is not None:
        if first_state.detected_concept_name != case.expected_concept:
            failures.append(
                f"turn1 concept: 期望 {case.expected_concept}，实际 {first_state.detected_concept_name}")

    return CaseResult(case_id=case.case_id, case_type="golden",
                      passed=not failures, failures=failures, num_turns=len(case.turns))


def run_single_case(case: SingleTurnCase) -> CaseResult:
    """单轮 case：诊断、动作与禁用模式。"""
    sid = f"eval_{case.case_id}_{uuid.uuid4().hex[:8]}"
    state = run_turn(sid, case.message)
    failures: list[str] = []

    if case.expected_concept and state.detected_concept_name != case.expected_concept:
        failures.append(f"concept: 期望 {case.expected_concept}，实际 {state.detected_concept_name}")
    if case.expected_intent:
        actual = state.diagnosis.intent.value if state.diagnosis else ""
        if actual not in case.expected_intent.split("|"):
            failures.append(f"intent: 期望 {case.expected_intent}，实际 {actual}")
    if case.expected_action:
        actual = state.action.value if state.action else ""
        if actual not in case.expected_action.split("|"):
            failures.append(f"action: 期望 {case.expected_action}，实际 {actual}")
    bad = [p for p in case.forbidden_patterns if _norm(p) in _norm(state.response)]
    if bad:
        failures.append(f"命中禁用模式：{bad}")
    if not state.response:
        failures.append("回复为空")

    return CaseResult(case_id=case.case_id, case_type="single_turn",
                      passed=not failures, failures=failures, num_turns=1)


def run_all() -> list[CaseResult]:
    results = [run_golden_conversation(c) for c in load_golden_cases()]
    results += [run_single_case(c) for c in load_single_turn_cases()]
    return results
