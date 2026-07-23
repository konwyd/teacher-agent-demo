"""Dynamic runner that couples deterministic students to Teacher modes."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

from app.agent import workflow
from app.evaluation import mode_runner
from app.evaluation.mode_runner import EvalMode
from app.evaluation.student_simulator import (
    StudentHiddenState,
    StudentMathIntent,
    StudentScenario,
    StudentSimulator,
    TeacherObservation,
    TerminationReason,
)


class TeacherTurnResult(BaseModel):
    response: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TeacherSession(Protocol):
    def respond(self, student_message: str) -> TeacherTurnResult: ...


class HybridTurnTrace(BaseModel):
    teacher_turn_index: int = Field(ge=1, le=8)
    student_state_before: StudentHiddenState
    student_message: str
    teacher_response: str
    teacher_metadata: dict[str, Any] = Field(default_factory=dict)
    teacher_observation: TeacherObservation
    selected_student_intent: StudentMathIntent | None = None
    generated_student_message: str | None = None
    student_state_after: StudentHiddenState
    used_real_student_wording_llm: bool = False
    wording_fallback_reason: str = ""


class HybridConversationTrace(BaseModel):
    schema_version: str = "1.0.0"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    run_id: str
    scenario_id: str
    concept_id: str
    mastery_level: str
    mode: EvalMode
    original_problem: str
    expected_final_answer: str
    solution_step_count: int = Field(ge=1)
    initial_student_state: StudentHiddenState
    final_student_state: StudentHiddenState
    initial_student_message: str
    turns: list[HybridTurnTrace]
    transcript: list[dict[str, str]]
    termination_reason: TerminationReason
    teacher_turn_count: int = Field(ge=1, le=8)


class ModeTeacherSession:
    """Stateful adapter for one existing Teacher evaluation mode."""

    def __init__(
        self,
        mode: EvalMode,
        scenario: StudentScenario,
        run_id: str,
        *,
        use_real_teacher_llm: bool = False,
    ) -> None:
        self.mode = EvalMode(mode)
        self.scenario = scenario
        self.use_real_teacher_llm = use_real_teacher_llm
        nonce = uuid.uuid4().hex[:8]
        self.student_id = (
            f"hybrid_{run_id}_{self.mode.value}_{scenario.scenario_id}_{nonce}"
        )
        self.conversation_id = f"hybrid_conv_{uuid.uuid4().hex[:12]}"
        self.transcript: list[dict[str, str]] = []

    def respond(self, student_message: str) -> TeacherTurnResult:
        if self.mode is EvalMode.DEEPSEEK_ONLY:
            response = mode_runner._direct_deepseek_response(  # noqa: SLF001
                student_message,
                self.transcript,
                self.use_real_teacher_llm,
            )
            self.transcript.extend(
                [
                    {"role": "student", "content": student_message},
                    {"role": "teacher", "content": response},
                ]
            )
            return TeacherTurnResult(
                response=response,
                metadata={
                    "diagnosis": {},
                    "teaching_action": "",
                    "retrieval_backend": "disabled",
                    "grounding_fields": {
                        "original_problem": self.scenario.original_problem,
                        "target_equation": self.scenario.original_problem,
                        "student_step_validity": "",
                        "whether_response_used_new_example": False,
                        "whether_response_returned_to_original_problem": False,
                    },
                },
            )

        graph_enabled = self.mode is EvalMode.AGENT_WITH_GRAPH
        with mode_runner._graph_setting(graph_enabled):  # noqa: SLF001
            state = workflow.run_turn(
                self.student_id,
                student_message,
                conversation_id=self.conversation_id,
            )
        return TeacherTurnResult(
            response=state.response,
            metadata=mode_runner._agent_metadata(state, self.mode),  # noqa: SLF001
        )


TeacherSessionFactory = Callable[
    [EvalMode, StudentScenario, str, bool],
    TeacherSession,
]


def create_mode_teacher_session(
    mode: EvalMode,
    scenario: StudentScenario,
    run_id: str,
    use_real_teacher_llm: bool,
) -> TeacherSession:
    return ModeTeacherSession(
        mode,
        scenario,
        run_id,
        use_real_teacher_llm=use_real_teacher_llm,
    )


def run_hybrid_conversation(
    scenario: StudentScenario,
    mode: EvalMode | str,
    *,
    run_id: str,
    max_turns: int = 8,
    teacher_session: TeacherSession | None = None,
    use_real_teacher_llm: bool = False,
    use_real_student_wording_llm: bool = False,
    student_wording_client: Any | None = None,
) -> HybridConversationTrace:
    if not 1 <= max_turns <= 8:
        raise ValueError("max_turns must be between 1 and 8")
    selected_mode = EvalMode(mode)
    session = teacher_session or create_mode_teacher_session(
        selected_mode,
        scenario,
        run_id,
        use_real_teacher_llm,
    )
    simulator = StudentSimulator(
        scenario,
        use_real_wording_llm=use_real_student_wording_llm,
        llm_client=student_wording_client,
    )
    initial_state = simulator.state.model_copy(deep=True)
    current_student_message = scenario.initial_message
    transcript: list[dict[str, str]] = [
        {"role": "student", "content": current_student_message}
    ]
    turns: list[HybridTurnTrace] = []
    termination: TerminationReason | None = None

    for teacher_turn_index in range(1, max_turns + 1):
        state_before = simulator.state.model_copy(deep=True)
        teacher_result = session.respond(current_student_message)
        transcript.append(
            {"role": "teacher", "content": teacher_result.response}
        )
        transition = simulator.advance(
            teacher_result.response,
            teacher_turn_index,
        )
        if transition.generated_student_message is not None:
            transcript.append(
                {
                    "role": "student",
                    "content": transition.generated_student_message,
                }
            )

        turns.append(
            HybridTurnTrace(
                teacher_turn_index=teacher_turn_index,
                student_state_before=state_before,
                student_message=current_student_message,
                teacher_response=teacher_result.response,
                teacher_metadata=teacher_result.metadata,
                teacher_observation=transition.observation,
                selected_student_intent=transition.selected_intent,
                generated_student_message=transition.generated_student_message,
                student_state_after=transition.state_after,
                used_real_student_wording_llm=(
                    transition.used_real_student_wording_llm
                ),
                wording_fallback_reason=transition.wording_fallback_reason,
            )
        )
        termination = transition.termination_reason
        if termination is not None:
            break
        if teacher_turn_index == max_turns:
            termination = TerminationReason.MAX_TURNS
            break
        if transition.generated_student_message is None:
            raise RuntimeError("non-terminal transition did not generate a student message")
        current_student_message = transition.generated_student_message

    if termination is None:  # defensive; the bounded loop always assigns it
        termination = TerminationReason.MAX_TURNS

    return HybridConversationTrace(
        run_id=run_id,
        scenario_id=scenario.scenario_id,
        concept_id=scenario.concept_id,
        mastery_level=scenario.mastery_level.value,
        mode=selected_mode,
        original_problem=scenario.original_problem,
        expected_final_answer=scenario.expected_final_answer,
        solution_step_count=len(scenario.solution_steps),
        initial_student_state=initial_state,
        final_student_state=simulator.state.model_copy(deep=True),
        initial_student_message=scenario.initial_message,
        turns=turns,
        transcript=transcript,
        termination_reason=termination,
        teacher_turn_count=len(turns),
    )


def run_hybrid_matrix(
    scenarios: list[StudentScenario],
    modes: list[EvalMode],
    *,
    run_id: str,
    max_turns: int = 8,
    use_real_teacher_llm: bool = False,
    use_real_student_wording_llm: bool = False,
    teacher_session_factory: TeacherSessionFactory | None = None,
) -> list[HybridConversationTrace]:
    factory = teacher_session_factory or create_mode_teacher_session
    traces: list[HybridConversationTrace] = []
    for scenario in scenarios:
        for mode in modes:
            session = factory(
                mode,
                scenario,
                run_id,
                use_real_teacher_llm,
            )
            traces.append(
                run_hybrid_conversation(
                    scenario,
                    mode,
                    run_id=run_id,
                    max_turns=max_turns,
                    teacher_session=session,
                    use_real_teacher_llm=use_real_teacher_llm,
                    use_real_student_wording_llm=(
                        use_real_student_wording_llm
                    ),
                )
            )
    return traces


def write_hybrid_traces(
    traces: list[HybridConversationTrace],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(
                json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)
                + "\n"
            )


def load_hybrid_traces(path: Path) -> list[HybridConversationTrace]:
    return [
        HybridConversationTrace.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
