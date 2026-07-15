"""Evidence-driven teaching planner constrained by the deterministic safety gate."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from app.agent.teaching_actions import (
    PracticeDifficulty,
    PracticeSimilarity,
    SafetyConstraints,
    TeachingAction,
    TeachingActionPlan,
)
from app.llm.deepseek_client import DeepSeekClient, DeepSeekError, is_configured


class PlannerInput(BaseModel):
    problem_context: str
    current_student_request: str
    constraints: SafetyConstraints
    verified_facts: list[str] = Field(default_factory=list)
    student_evidence: dict = Field(default_factory=dict)
    strategy_bundle: dict = Field(default_factory=dict)
    has_verified_error: bool = False
    has_confirmed_progress: bool = False


class PlannerResult(BaseModel):
    plan: TeachingActionPlan
    used_llm: bool = False
    reason: str = ""


class _PlannerClient(Protocol):
    def chat_json(
        self,
        system: str,
        user: str,
        schema,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> TeachingActionPlan:
        """Return one validated teaching action plan."""


_FALLBACK_PRIORITY = (
    TeachingAction.ELICIT,
    TeachingAction.FOCUS,
    TeachingAction.SCAFFOLD,
    TeachingAction.CHECK,
    TeachingAction.EXPLAIN,
    TeachingAction.CORRECT,
    TeachingAction.CONSOLIDATE,
    TeachingAction.PRACTICE,
    TeachingAction.MODEL,
)


def _action_is_valid(action: TeachingAction, planner_input: PlannerInput) -> bool:
    constraints = planner_input.constraints
    if action not in constraints.allowed_actions:
        return False
    if action is TeachingAction.MODEL and not constraints.worked_example_allowed:
        return False
    if action is TeachingAction.PRACTICE and not constraints.new_problem_allowed:
        return False
    if action is TeachingAction.CORRECT and not planner_input.has_verified_error:
        return False
    if action is TeachingAction.CONSOLIDATE and not planner_input.has_confirmed_progress:
        return False
    return True


def _default_plan(action: TeachingAction) -> TeachingActionPlan:
    if action is TeachingAction.SCAFFOLD:
        return TeachingActionPlan(action=action, scaffold_level=1)
    if action is TeachingAction.PRACTICE:
        return TeachingActionPlan(
            action=action,
            practice_difficulty=PracticeDifficulty.SIMILAR,
            practice_similarity=PracticeSimilarity.SAME_STRATEGY,
        )
    return TeachingActionPlan(action=action)


class TeachingPlanner:
    """Select an allowed action, with deterministic fallback on any LLM failure."""

    def __init__(self, client: _PlannerClient | None = None) -> None:
        self._client = client

    def _candidates(self, planner_input: PlannerInput) -> list[TeachingAction]:
        allowed = set(planner_input.constraints.allowed_actions)
        return [
            action
            for action in _FALLBACK_PRIORITY
            if action in allowed and _action_is_valid(action, planner_input)
        ]

    def _fallback(self, planner_input: PlannerInput) -> TeachingActionPlan:
        candidates = self._candidates(planner_input)
        if not candidates:
            raise ValueError("SafetyConstraints contain no valid allowed action")
        return _default_plan(candidates[0])

    def validate_or_fallback(
        self,
        proposed_plan: TeachingActionPlan,
        planner_input: PlannerInput,
        *,
        used_llm: bool = False,
    ) -> PlannerResult:
        if _action_is_valid(proposed_plan.action, planner_input):
            return PlannerResult(
                plan=proposed_plan,
                used_llm=used_llm,
                reason="planner_selected_action",
            )
        return PlannerResult(
            plan=self._fallback(planner_input),
            used_llm=used_llm,
            reason="planner_output_rejected",
        )

    def plan(self, planner_input: PlannerInput) -> PlannerResult:
        candidates = self._candidates(planner_input)
        if not candidates:
            raise ValueError("SafetyConstraints contain no valid allowed action")
        if len(candidates) == 1:
            return PlannerResult(
                plan=_default_plan(candidates[0]),
                used_llm=False,
                reason="single_allowed_action",
            )
        if self._client is None and not is_configured():
            return PlannerResult(
                plan=_default_plan(candidates[0]),
                used_llm=False,
                reason="planner_unavailable",
            )

        try:
            client = self._client or DeepSeekClient()
            proposed = client.chat_json(
                system=(
                    "你是教学动作规划器。只能从 allowed_actions 中选择动作，并严格遵守"
                    "披露、示范、练习和证据约束。不要解题，不要生成教师回复。"
                ),
                user=planner_input.model_dump_json(),
                schema=TeachingActionPlan,
                temperature=0.0,
                max_tokens=300,
            )
            proposed_plan = TeachingActionPlan.model_validate(proposed)
        except (DeepSeekError, ValueError, ValidationError):
            return PlannerResult(
                plan=_default_plan(candidates[0]),
                used_llm=True,
                reason="planner_call_failed",
            )
        return self.validate_or_fallback(
            proposed_plan,
            planner_input,
            used_llm=True,
        )
