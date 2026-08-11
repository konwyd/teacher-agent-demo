"""Serializable evidence for topic-agnostic teaching orchestration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.teaching_actions import TeachingActionPlan


class ModuleDecision(BaseModel):
    module: str
    called: bool
    reason: str
    llm_calls: int = Field(default=0, ge=0)


class OrchestrationTrace(BaseModel):
    decisions: list[ModuleDecision] = Field(default_factory=list)
    llm_call_count: int = Field(default=0, ge=0, le=3)
    shadow_mode: bool = True
    active_plan: TeachingActionPlan | None = None
    shadow_plan: TeachingActionPlan | None = None
    fallback_reasons: list[str] = Field(default_factory=list)
