"""Serializable contracts for the bounded multisubject teacher loop."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Subject(str, Enum):
    MATH = "数学"
    CHINESE = "语文"
    ENGLISH = "英语"
    PHYSICS = "物理"
    CHEMISTRY = "化学"
    BIOLOGY = "生物"
    HISTORY = "历史"
    GEOGRAPHY = "地理"
    POLITICS = "道法"


class SchoolStage(str, Enum):
    PRIMARY = "小学"
    MIDDLE = "初中"
    UNKNOWN = "未知"


class RequestScope(str, Enum):
    K9 = "k9"
    OFF_TOPIC = "off_topic"


class AnswerMode(str, Enum):
    GUIDED = "guided"
    COMPLETE = "complete"
    BRIEF_REDIRECT = "brief_redirect"


class TeachingResponseMode(str, Enum):
    CONCEPT_EXPLANATION = "concept_explanation"
    ONE_STEP = "one_step"
    GUIDED_SOLUTION = "guided_solution"
    FULL_SOLUTION = "full_solution"
    BRIEF_REDIRECT = "brief_redirect"


class TeachingContract(BaseModel):
    response_mode: TeachingResponseMode
    requested_target: str = Field(min_length=1, max_length=300)
    allowed_content: list[str] = Field(default_factory=list, max_length=5)
    forbidden_content: list[str] = Field(default_factory=list, max_length=5)
    may_apply_to_active_problem: bool = False
    may_reveal_final_answer: bool = False
    reason: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def validate_disclosure(self) -> "TeachingContract":
        if (
            self.may_reveal_final_answer
            and self.response_mode is not TeachingResponseMode.FULL_SOLUTION
        ):
            raise ValueError("only full_solution may reveal final answer")
        if (
            self.response_mode is TeachingResponseMode.CONCEPT_EXPLANATION
            and self.may_apply_to_active_problem
        ):
            raise ValueError("concept explanation cannot apply to active problem")
        return self


class BoundaryViolation(str, Enum):
    OFF_TARGET = "off_target"
    APPLIED_WITHOUT_PERMISSION = "applied_without_permission"
    FINAL_ANSWER_LEAK = "final_answer_leak"
    TOO_MANY_STEPS = "too_many_steps"
    IGNORED_NO_ANSWER = "ignored_no_answer"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class BoundaryAudit(BaseModel):
    passed: bool
    violations: list[BoundaryViolation] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    repair_instructions: list[str] = Field(default_factory=list)
    audit_error: str = ""


class RequestClassification(BaseModel):
    scope: RequestScope
    subject: Subject | None = None
    stage: SchoolStage = SchoolStage.UNKNOWN
    is_new_problem: bool
    answer_mode: AnswerMode
    reason: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_scope(self) -> "RequestClassification":
        if self.scope is RequestScope.K9 and self.subject is None:
            raise ValueError("k9 classification requires subject")
        if self.scope is RequestScope.OFF_TOPIC and (
            self.subject is not None
            or self.answer_mode is not AnswerMode.BRIEF_REDIRECT
        ):
            raise ValueError("off_topic requires no subject and brief_redirect")
        return self


class AgentActionKind(str, Enum):
    CALL_TOOL = "call_tool"
    FINISH = "finish"


class AgentDecision(BaseModel):
    kind: AgentActionKind
    reason: str = Field(min_length=1, max_length=400)
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    plan: str = ""
    response: str = ""
    contract: TeachingContract | None = None

    @model_validator(mode="after")
    def validate_action(self) -> "AgentDecision":
        if self.kind is AgentActionKind.CALL_TOOL and not self.tool_name:
            raise ValueError("call_tool requires tool_name")
        if self.kind is AgentActionKind.FINISH and self.tool_name:
            raise ValueError("finish cannot name a tool")
        if self.kind is AgentActionKind.FINISH and self.contract is None:
            raise ValueError("finish requires teaching contract")
        return self


class ToolResult(BaseModel):
    name: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    elapsed_seconds: float = 0.0


class AgentTraceEvent(BaseModel):
    step: int = Field(ge=0)
    event_type: str = Field(min_length=1, max_length=80)
    summary: str = Field(min_length=1, max_length=600)
    tool_name: str = ""
    ok: bool | None = None
    elapsed_seconds: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class AgentOrchestrationResult(BaseModel):
    """Loop result; rag_context stays runtime-typed to avoid adapter cycles."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    classification: RequestClassification
    teaching_contract: TeachingContract
    rag_context: Any = None
    observations: list[ToolResult] = Field(default_factory=list)
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    loaded_skills: list[str] = Field(default_factory=list)
    termination_reason: str = ""
    routing_guidance: dict[str, Any] = Field(default_factory=dict)
