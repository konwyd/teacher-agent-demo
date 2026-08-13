"""One-way state transitions for the public thinking panel."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


ThinkingPhase = Literal["running", "complete", "failed"]


@dataclass(frozen=True)
class ThinkingPanelState:
    """Server-side phase and the latest explicit expansion choice."""

    phase: ThinkingPhase
    expanded: bool


def thinking_key(conversation_id: str, turn_id: int) -> str:
    """Return a Streamlit state key isolated to one conversation turn."""

    return f"public_thinking:{conversation_id}:turn:{turn_id}"


def begin_thinking(state: ThinkingPanelState | None) -> ThinkingPanelState:
    """Start or retry generation, preserving an already-running state."""

    if state is not None and state.phase == "running":
        return state
    return ThinkingPanelState(phase="running", expanded=True)


def complete_thinking(state: ThinkingPanelState | None) -> ThinkingPanelState:
    """Collapse on the first completion and never override later choices."""

    if state is not None and state.phase == "complete":
        return state
    return ThinkingPanelState(phase="complete", expanded=False)


def fail_thinking(state: ThinkingPanelState | None) -> ThinkingPanelState:
    """Keep a failed generation open so its public progress stays visible."""

    if state is not None and state.phase == "failed" and state.expanded:
        return state
    return ThinkingPanelState(phase="failed", expanded=True)


def set_user_expanded(
    state: ThinkingPanelState,
    expanded: bool,
) -> ThinkingPanelState:
    """Record a user's explicit choice without changing the lifecycle phase."""

    if state.expanded is expanded:
        return state
    return replace(state, expanded=expanded)
