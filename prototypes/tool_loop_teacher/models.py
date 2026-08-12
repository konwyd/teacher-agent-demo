"""Pydantic contracts for the isolated teaching tool loop."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.agent.state import MathAnalysis, VerificationResult
from app.agent.teaching_actions import SafetyConstraints, TeachingActionPlan
from app.agent.teaching_brief import TeachingBrief


class DecisionKind(str, Enum):
    CALL_TOOL = "call_tool"
    FINISH = "finish"


class LoopDecision(BaseModel):
    """One model-selected action in the loop.

    The prototype uses this JSON action protocol because the production
    ``DeepSeekClient`` currently exposes validated JSON output, not native
    provider tool-call messages.
    """

    kind: DecisionKind
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    proposed_plan: TeachingActionPlan | None = None
    rationale: str = ""

    @model_validator(mode="after")
    def validate_shape(self) -> LoopDecision:
        if self.kind is DecisionKind.CALL_TOOL and not self.tool_name.strip():
            raise ValueError("call_tool requires tool_name")
        if self.kind is DecisionKind.FINISH and self.proposed_plan is None:
            raise ValueError("finish requires proposed_plan")
        return self


class ToolResult(BaseModel):
    tool_name: str
    ok: bool = True
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class LoopTraceEvent(BaseModel):
    iteration: int = Field(ge=0)
    event: str
    detail: dict[str, Any] = Field(default_factory=dict)


class LoopResult(BaseModel):
    response: str
    action_plan: TeachingActionPlan
    teaching_brief: TeachingBrief
    analysis: MathAnalysis
    verification: VerificationResult
    safety_constraints: SafetyConstraints
    observations: list[ToolResult] = Field(default_factory=list)
    trace: list[LoopTraceEvent] = Field(default_factory=list)
    loaded_skills: list[str] = Field(default_factory=list)
    termination_reason: str
    fallback_used: bool = False
