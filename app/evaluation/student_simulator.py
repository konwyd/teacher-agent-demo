"""Deterministic hidden-state student simulator for hybrid evaluation.

The simulator is evaluation-only.  Its mathematical intentions are fixed by
scenario data and verified by ``math_checker``; an optional LLM may only
reword an already selected intention.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.agent import math_checker as mc
from app.llm.deepseek_client import DeepSeekClient, DeepSeekError

EVAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"
DEFAULT_SCENARIO_PATH = EVAL_DATA_DIR / "student_simulator_scenarios.jsonl"

SUPPORTED_CONCEPTS = {
    "linear_equation",
    "equation_properties",
    "move_terms",
    "divide_by_coefficient",
    "remove_parentheses",
    "fractional_coefficient_equation",
}


class MasteryLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HelpSeekingStyle(str, Enum):
    ASKS_WHY = "asks_why"
    ASKS_HINT = "asks_hint"
    GUESSES = "guesses"
    DIRECT = "direct"


class UnderstandingStatus(str, Enum):
    CONFUSED = "confused"
    PARTIAL = "partial"
    UNDERSTOOD = "understood"
    VERIFIED = "verified"


class StudentIntentType(str, Enum):
    ASK_HELP = "ask_help"
    ASK_WHY = "ask_why"
    RESTATE_CONFUSION = "restate_confusion"
    INCORRECT_STEP = "incorrect_step"
    CORRECT_STEP = "correct_step"
    FINAL_ANSWER = "final_answer"
    VERIFICATION = "verification"
    GIVE_UP = "give_up"


class TerminationReason(str, Enum):
    SOLVED_AND_VERIFIED = "solved_and_verified"
    STUDENT_STUCK = "student_stuck"
    STUDENT_GAVE_UP = "student_gave_up"
    MAX_TURNS = "max_turns"


class StudentHiddenState(BaseModel):
    mastery: float = Field(ge=0.0, le=1.0)
    active_misconception: str | None = None
    current_solution_step: int = Field(default=0, ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    help_seeking_style: HelpSeekingStyle
    consecutive_failures: int = Field(default=0, ge=0)
    hint_exposure: int = Field(default=0, ge=0)
    understanding_status: UnderstandingStatus


class StudentMathIntent(BaseModel):
    intent_type: StudentIntentType
    canonical_content: str = Field(min_length=1)
    expected_step_validity: str | None = None
    required_math_anchors: list[str] = Field(default_factory=list)


class StudentScenario(BaseModel):
    scenario_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    mastery_level: MasteryLevel
    original_problem: str = Field(min_length=1)
    expected_final_answer: str = Field(min_length=1)
    initial_message: str = Field(min_length=1)
    initial_state: StudentHiddenState
    solution_steps: list[StudentMathIntent] = Field(min_length=1)
    verification_intent: StudentMathIntent
    misconception_intent: StudentMathIntent | None = None
    misconception_markers: list[str] = Field(default_factory=list)
    targeted_support_markers: list[str] = Field(default_factory=list)
    utterance_templates: dict[str, list[str]] = Field(default_factory=dict)


class TeacherObservation(BaseModel):
    acknowledges_correct: bool = False
    rejects_incorrect: bool = False
    gives_hint: bool = False
    addresses_misconception: bool = False
    asks_for_verification: bool = False
    confirms_verification: bool = False
    leaks_final_answer: bool = False
    asks_question: bool = False
    grounded_in_original_problem: bool = False
    matched_markers: list[str] = Field(default_factory=list)


class StudentWording(BaseModel):
    text: str = Field(min_length=1, max_length=200)


class StudentTransition(BaseModel):
    teacher_turn_index: int = Field(ge=1)
    state_before: StudentHiddenState
    observation: TeacherObservation
    selected_intent: StudentMathIntent | None = None
    generated_student_message: str | None = None
    state_after: StudentHiddenState
    termination_reason: TerminationReason | None = None
    used_real_student_wording_llm: bool = False
    wording_fallback_reason: str = ""


_DEFAULT_TEMPLATES: dict[StudentIntentType, list[str]] = {
    StudentIntentType.ASK_HELP: ["我还是不太明白，能再提示一下吗？"],
    StudentIntentType.ASK_WHY: ["我看到答案了，但为什么要这样做？"],
    StudentIntentType.RESTATE_CONFUSION: ["我还是不知道这一步是怎么来的。"],
    StudentIntentType.INCORRECT_STEP: ["我试着写成：{content}"],
    StudentIntentType.CORRECT_STEP: ["那我先写：{content}"],
    StudentIntentType.FINAL_ANSWER: ["所以我得到：{content}"],
    StudentIntentType.VERIFICATION: ["我来验算：{content}"],
    StudentIntentType.GIVE_UP: ["我还是不会，先不做了。"],
}

_ACK_MARKERS = ("正确", "对的", "很好", "没问题", "思路对", "做对")
_REJECT_MARKERS = ("不对", "错误", "不是", "先停", "有问题")
_HINT_MARKERS = (
    "提示",
    "两边同时",
    "移项",
    "去括号",
    "最小公倍数",
    "系数",
    "先把",
    "先用",
)
_VERIFY_MARKERS = ("验算", "代回", "检验", "检查")

_INITIAL_BASELINES = {
    MasteryLevel.LOW: (
        0.20,
        0.30,
        HelpSeekingStyle.ASKS_HINT,
        UnderstandingStatus.CONFUSED,
        True,
    ),
    MasteryLevel.MEDIUM: (
        0.50,
        0.55,
        HelpSeekingStyle.GUESSES,
        UnderstandingStatus.PARTIAL,
        True,
    ),
    MasteryLevel.HIGH: (
        0.80,
        0.80,
        HelpSeekingStyle.DIRECT,
        UnderstandingStatus.PARTIAL,
        False,
    ),
}


def _normalized(text: str) -> str:
    return "".join(text.split()).lower()


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def validate_math_intent(
    scenario: StudentScenario,
    intent: StudentMathIntent,
) -> bool:
    """Validate a declared mathematical intention with the authoritative checker."""
    if intent.expected_step_validity is None:
        return not mc.find_equations(intent.canonical_content)
    actual = mc.check_step(
        scenario.original_problem,
        intent.canonical_content,
    ).validity.value
    return actual == intent.expected_step_validity


def _validate_scenario(scenario: StudentScenario) -> None:
    if scenario.concept_id not in SUPPORTED_CONCEPTS:
        raise ValueError(f"unsupported concept_id: {scenario.concept_id}")
    if scenario.initial_state.current_solution_step != 0:
        raise ValueError(f"{scenario.scenario_id}: initial step must be zero")
    mastery, confidence, style, understanding, requires_misconception = (
        _INITIAL_BASELINES[scenario.mastery_level]
    )
    state = scenario.initial_state
    baseline_matches = (
        state.mastery == mastery
        and state.confidence == confidence
        and state.help_seeking_style is style
        and state.understanding_status is understanding
        and state.consecutive_failures == 0
        and state.hint_exposure == 0
        and (
            (state.active_misconception is not None)
            == requires_misconception
        )
    )
    if not baseline_matches:
        raise ValueError(
            f"{scenario.scenario_id}: initial state does not match "
            f"the {scenario.mastery_level.value} baseline"
        )
    if not scenario.solution_steps:
        raise ValueError(f"{scenario.scenario_id}: solution_steps is empty")
    if scenario.solution_steps[-1].intent_type is not StudentIntentType.FINAL_ANSWER:
        raise ValueError(f"{scenario.scenario_id}: final solution step must be FINAL_ANSWER")
    if scenario.verification_intent.intent_type is not StudentIntentType.VERIFICATION:
        raise ValueError(f"{scenario.scenario_id}: invalid verification intent")
    for intent in [
        *scenario.solution_steps,
        scenario.verification_intent,
        *([scenario.misconception_intent] if scenario.misconception_intent else []),
    ]:
        if not validate_math_intent(scenario, intent):
            raise ValueError(
                f"{scenario.scenario_id}: math intent does not match checker: "
                f"{intent.canonical_content}"
            )
    expected_final = mc.check_step(
        scenario.original_problem,
        scenario.expected_final_answer,
    ).validity.value
    if expected_final != mc.StepValidity.CORRECT_FINAL_ANSWER.value:
        raise ValueError(f"{scenario.scenario_id}: expected_final_answer is invalid")


def load_student_scenarios(
    path: Path | None = None,
    *,
    require_complete: bool | None = None,
) -> list[StudentScenario]:
    source = path or DEFAULT_SCENARIO_PATH
    rows = [
        StudentScenario.model_validate(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for scenario in rows:
        _validate_scenario(scenario)

    complete = path is None if require_complete is None else require_complete
    if complete:
        expected = {
            (concept, mastery.value)
            for concept in SUPPORTED_CONCEPTS
            for mastery in MasteryLevel
        }
        actual = {
            (scenario.concept_id, scenario.mastery_level.value)
            for scenario in rows
        }
        ids = [scenario.scenario_id for scenario in rows]
        if len(rows) != 18 or actual != expected or len(ids) != len(set(ids)):
            raise ValueError("default simulator scenarios must be the unique 6x3 grid")
        for concept in SUPPORTED_CONCEPTS:
            problems = {
                scenario.original_problem
                for scenario in rows
                if scenario.concept_id == concept
            }
            if len(problems) != 1:
                raise ValueError(f"{concept}: mastery levels must share one problem")
    return rows


def observe_teacher_response(
    scenario: StudentScenario,
    state: StudentHiddenState,
    teacher_response: str,
) -> TeacherObservation:
    matched: list[str] = []

    def contains_any(markers: tuple[str, ...] | list[str]) -> bool:
        found = [marker for marker in markers if marker and marker in teacher_response]
        matched.extend(found)
        return bool(found)

    acknowledges = contains_any(_ACK_MARKERS)
    rejects = contains_any(_REJECT_MARKERS)
    hints = contains_any(_HINT_MARKERS)
    targeted = contains_any(scenario.targeted_support_markers)
    asks_verify = contains_any(_VERIFY_MARKERS)
    confirms_verify = asks_verify and (
        acknowledges or "两边相等" in teacher_response or "等式成立" in teacher_response
    )
    answer_not_yet_given = state.current_solution_step < len(scenario.solution_steps)
    leaks = answer_not_yet_given and (
        _normalized(scenario.expected_final_answer) in _normalized(teacher_response)
    )
    equations = mc.find_equations(teacher_response)
    grounded = not equations or all(
        mc.is_equivalent(equation, scenario.original_problem)
        for equation in equations
    )
    return TeacherObservation(
        acknowledges_correct=acknowledges,
        rejects_incorrect=rejects,
        gives_hint=hints,
        addresses_misconception=targeted,
        asks_for_verification=asks_verify,
        confirms_verification=confirms_verify,
        leaks_final_answer=leaks,
        asks_question=("?" in teacher_response or "？" in teacher_response),
        grounded_in_original_problem=grounded,
        matched_markers=list(dict.fromkeys(matched)),
    )


def _non_math_intent(intent_type: StudentIntentType) -> StudentMathIntent:
    content = {
        StudentIntentType.ASK_HELP: "我还是不太明白，能再提示一下吗？",
        StudentIntentType.ASK_WHY: "为什么要这样做？",
        StudentIntentType.RESTATE_CONFUSION: "我还是不知道这一步怎么来。",
        StudentIntentType.GIVE_UP: "我还是不会，先不做了。",
    }[intent_type]
    return StudentMathIntent(intent_type=intent_type, canonical_content=content)


def _help_intent(state: StudentHiddenState) -> StudentMathIntent:
    if state.help_seeking_style is HelpSeekingStyle.ASKS_WHY:
        return _non_math_intent(StudentIntentType.ASK_WHY)
    if state.help_seeking_style is HelpSeekingStyle.ASKS_HINT:
        return _non_math_intent(StudentIntentType.ASK_HELP)
    return _non_math_intent(StudentIntentType.RESTATE_CONFUSION)


def _raise_understanding(status: UnderstandingStatus) -> UnderstandingStatus:
    if status is UnderstandingStatus.CONFUSED:
        return UnderstandingStatus.PARTIAL
    if status is UnderstandingStatus.PARTIAL:
        return UnderstandingStatus.UNDERSTOOD
    return status


def advance_student_state(
    scenario: StudentScenario,
    state: StudentHiddenState,
    teacher_response: str,
    teacher_turn_index: int,
) -> StudentTransition:
    """Apply one deterministic Teacher-response-to-student transition."""
    before = state.model_copy(deep=True)
    after = state.model_copy(deep=True)
    observation = observe_teacher_response(scenario, before, teacher_response)

    if (
        before.understanding_status is UnderstandingStatus.VERIFIED
        and (observation.confirms_verification or observation.acknowledges_correct)
    ):
        return StudentTransition(
            teacher_turn_index=teacher_turn_index,
            state_before=before,
            observation=observation,
            state_after=after,
            termination_reason=TerminationReason.SOLVED_AND_VERIFIED,
        )

    selected: StudentMathIntent
    if observation.leaks_final_answer:
        after.confidence = _clamp(after.confidence - 0.05)
        selected = _non_math_intent(
            StudentIntentType.ASK_WHY
            if after.help_seeking_style is HelpSeekingStyle.ASKS_WHY
            else StudentIntentType.RESTATE_CONFUSION
        )
    else:
        if observation.addresses_misconception:
            after.active_misconception = None
            after.mastery = _clamp(after.mastery + 0.10)
            after.confidence = _clamp(after.confidence + 0.10)
            after.understanding_status = _raise_understanding(
                after.understanding_status
            )
        elif observation.gives_hint:
            after.hint_exposure += 1
            after.mastery = _clamp(after.mastery + 0.05)
            after.confidence = _clamp(after.confidence + 0.05)

        if after.current_solution_step >= len(scenario.solution_steps):
            selected = scenario.verification_intent.model_copy(deep=True)
        elif observation.addresses_misconception:
            selected = scenario.solution_steps[
                after.current_solution_step
            ].model_copy(deep=True)
        elif observation.gives_hint:
            low_needs_more_support = (
                scenario.mastery_level is MasteryLevel.LOW
                and after.active_misconception is not None
                and after.hint_exposure < 2
            )
            if low_needs_more_support and scenario.misconception_intent:
                selected = scenario.misconception_intent.model_copy(deep=True)
            elif low_needs_more_support:
                selected = _help_intent(after)
            else:
                selected = scenario.solution_steps[
                    after.current_solution_step
                ].model_copy(deep=True)
        elif scenario.mastery_level is MasteryLevel.HIGH:
            selected = scenario.solution_steps[
                after.current_solution_step
            ].model_copy(deep=True)
        elif (
            scenario.mastery_level is MasteryLevel.MEDIUM
            and after.confidence >= 0.5
        ):
            selected = scenario.solution_steps[
                after.current_solution_step
            ].model_copy(deep=True)
        elif scenario.misconception_intent is not None:
            selected = scenario.misconception_intent.model_copy(deep=True)
        else:
            selected = _help_intent(after)

    termination: TerminationReason | None = None
    if selected.intent_type in {
        StudentIntentType.CORRECT_STEP,
        StudentIntentType.FINAL_ANSWER,
    }:
        if not validate_math_intent(scenario, selected):
            raise ValueError("selected correct intent failed math_checker validation")
        after.current_solution_step += 1
        after.consecutive_failures = 0
        after.confidence = _clamp(after.confidence + 0.05)
        if selected.intent_type is StudentIntentType.FINAL_ANSWER:
            after.understanding_status = UnderstandingStatus.UNDERSTOOD
    elif selected.intent_type is StudentIntentType.INCORRECT_STEP:
        if not validate_math_intent(scenario, selected):
            raise ValueError("selected misconception failed math_checker validation")
        after.consecutive_failures += 1
        after.confidence = _clamp(after.confidence - 0.10)
        if after.consecutive_failures >= 3:
            if after.confidence <= 0.10:
                selected = _non_math_intent(StudentIntentType.GIVE_UP)
                termination = TerminationReason.STUDENT_GAVE_UP
            else:
                termination = TerminationReason.STUDENT_STUCK
    elif selected.intent_type is StudentIntentType.VERIFICATION:
        if not validate_math_intent(scenario, selected):
            raise ValueError("verification intent failed math_checker validation")
        after.understanding_status = UnderstandingStatus.VERIFIED
        after.confidence = _clamp(after.confidence + 0.05)

    return StudentTransition(
        teacher_turn_index=teacher_turn_index,
        state_before=before,
        observation=observation,
        selected_intent=selected,
        state_after=after,
        termination_reason=termination,
    )


def _template_message(
    scenario: StudentScenario,
    intent: StudentMathIntent,
    teacher_turn_index: int,
) -> str:
    configured = scenario.utterance_templates.get(intent.intent_type.value)
    choices = configured or _DEFAULT_TEMPLATES[intent.intent_type]
    template = choices[(teacher_turn_index - 1) % len(choices)]
    if "{content}" in template:
        return template.format(content=intent.canonical_content)
    if intent.expected_step_validity and intent.canonical_content not in template:
        return f"{template}{intent.canonical_content}"
    return template


def _validate_rewording(
    scenario: StudentScenario,
    intent: StudentMathIntent,
    text: str,
) -> str:
    normalized = _normalized(text)
    missing = [
        anchor
        for anchor in intent.required_math_anchors
        if _normalized(anchor) not in normalized
    ]
    if missing:
        return f"missing required anchors: {missing}"

    allowed_equations = {
        _normalized(equation)
        for equation in mc.find_equations(intent.canonical_content)
    }
    actual_equations = {
        _normalized(equation) for equation in mc.find_equations(text)
    }
    if not actual_equations <= allowed_equations:
        return "wording introduced a new equation"

    if intent.expected_step_validity:
        actual = mc.check_step(scenario.original_problem, text).validity.value
        if actual != intent.expected_step_validity:
            return f"wording changed math validity to {actual}"
    elif actual_equations:
        return "non-math intent introduced an equation"

    intent_markers = {
        StudentIntentType.ASK_HELP: ("提示", "帮", "不会", "不明白"),
        StudentIntentType.ASK_WHY: ("为什么",),
        StudentIntentType.RESTATE_CONFUSION: ("不懂", "不知道", "不明白"),
        StudentIntentType.GIVE_UP: ("不做", "放弃", "不会"),
    }
    markers = intent_markers.get(intent.intent_type)
    if markers and not any(marker in text for marker in markers):
        return "wording changed non-math intent"
    return ""


def generate_student_utterance(
    scenario: StudentScenario,
    intent: StudentMathIntent,
    teacher_turn_index: int,
    *,
    use_real_wording_llm: bool = False,
    llm_client: Any | None = None,
) -> tuple[str, bool, str]:
    template = _template_message(scenario, intent, teacher_turn_index)
    if not use_real_wording_llm:
        return template, False, ""

    client = llm_client
    try:
        if client is None:
            client = DeepSeekClient()
        wording = client.chat_json(
            system=(
                "你只负责把一个已经确定的中国初中生发言改写得自然。"
                "不得改变数学内容、步骤对错或意图，不得增加任何方程或答案。"
            ),
            user=json.dumps(
                {
                    "intent_type": intent.intent_type.value,
                    "canonical_content": intent.canonical_content,
                    "required_math_anchors": intent.required_math_anchors,
                    "help_seeking_style": scenario.initial_state.help_seeking_style.value,
                },
                ensure_ascii=False,
            ),
            schema=StudentWording,
            temperature=0.0,
        )
        reason = _validate_rewording(scenario, intent, wording.text)
        if reason:
            return template, False, reason
        return wording.text, True, ""
    except (DeepSeekError, ValueError, TypeError) as exc:
        return template, False, f"{type(exc).__name__}: {exc}"


class StudentSimulator:
    """Stateful convenience wrapper around the pure transition function."""

    def __init__(
        self,
        scenario: StudentScenario,
        *,
        use_real_wording_llm: bool = False,
        llm_client: Any | None = None,
    ) -> None:
        self.scenario = scenario
        self.state = scenario.initial_state.model_copy(deep=True)
        self.use_real_wording_llm = use_real_wording_llm
        self.llm_client = llm_client

    def advance(
        self,
        teacher_response: str,
        teacher_turn_index: int,
    ) -> StudentTransition:
        transition = advance_student_state(
            self.scenario,
            self.state,
            teacher_response,
            teacher_turn_index,
        )
        if transition.selected_intent is not None:
            message, used_real, fallback = generate_student_utterance(
                self.scenario,
                transition.selected_intent,
                teacher_turn_index,
                use_real_wording_llm=self.use_real_wording_llm,
                llm_client=self.llm_client,
            )
            transition = transition.model_copy(
                update={
                    "generated_student_message": message,
                    "used_real_student_wording_llm": used_real,
                    "wording_fallback_reason": fallback,
                }
            )
        self.state = transition.state_after.model_copy(deep=True)
        return transition
