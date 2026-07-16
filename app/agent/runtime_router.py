"""Compatibility-boundary routing for legacy and topic-neutral runtimes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from app.agent.linear_equation_adapter import (
    is_legacy_concept_request,
    supports_legacy_equation_turn,
)
from app.agent.optional_verifier import LinearEquationVerifier, VerificationRequest
from app.memory.episode_state import get_episode_store
from app.memory.schemas import EpisodeState


class RuntimePath(str, Enum):
    LEGACY_EQUATION = "legacy_equation"
    GENERIC = "generic"
    UNIFIED_LINEAR_SLICE = "unified_linear_slice"


@dataclass(frozen=True)
class RuntimeDecision:
    """Serializable-friendly explanation for callers that record routing."""

    path: RuntimePath
    reason: str = ""


def unified_linear_slice_enabled() -> bool:
    return os.getenv("TEACHER_UNIFIED_LINEAR_SLICE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_runtime_path(
    student_id: str,
    message: str,
    conversation_id: str | None = None,
    *,
    episode: EpisodeState | None = None,
) -> RuntimePath:
    """Choose the one runtime that owns this turn at the compatibility edge."""

    existing_episode = episode
    if existing_episode is None:
        existing_episode = get_episode_store().find(student_id, conversation_id)

    if existing_episode and existing_episode.target_equation:
        return RuntimePath.LEGACY_EQUATION
    if existing_episode and existing_episode.orchestration_path:
        return RuntimePath(existing_episode.orchestration_path)
    if existing_episode and existing_episode.problem_context:
        return RuntimePath.GENERIC
    if is_legacy_concept_request(message):
        return RuntimePath.LEGACY_EQUATION
    if unified_linear_slice_enabled() and LinearEquationVerifier().supports(
        VerificationRequest(problem_context=message)
    ):
        return RuntimePath.UNIFIED_LINEAR_SLICE
    if supports_legacy_equation_turn(student_id, message, conversation_id):
        return RuntimePath.LEGACY_EQUATION
    return RuntimePath.GENERIC
