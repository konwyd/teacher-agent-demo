"""Typed retrieval-routing contract with deterministic safety guards."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from prototypes.llm_led_teacher.orchestration_models import RequestClassification


class RetrievalMode(str, Enum):
    NONE = "none"
    LOCAL = "local"
    LOCAL_THEN_WEB = "local_then_web"


class RetrievalDecision(BaseModel):
    mode: RetrievalMode
    reason: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_current_information: bool = False


_NO_RAG_PATTERNS = (
    "润色",
    "改写",
    "仿写",
    "续写",
    "作文开头",
    "换一种说法",
    "头脑风暴",
)
_CURRENT_PATTERNS = ("今天", "最新", "刚刚", "目前", "发布", "新闻", "通知")


def apply_retrieval_guards(
    decision: RetrievalDecision,
    *,
    message: str,
    classification: RequestClassification,
) -> RetrievalDecision:
    from prototypes.llm_led_teacher.orchestration_models import RequestScope

    if classification.scope is RequestScope.OFF_TOPIC:
        return RetrievalDecision(
            mode=RetrievalMode.NONE,
            reason="当前请求无需课程资料",
            confidence=1.0,
        )
    if any(marker in message for marker in _NO_RAG_PATTERNS):
        return RetrievalDecision(
            mode=RetrievalMode.NONE,
            reason="这是基于用户文本的创作指导，无需外部资料",
            confidence=1.0,
        )
    if decision.mode is RetrievalMode.LOCAL_THEN_WEB and not (
        decision.needs_current_information
        or any(marker in message for marker in _CURRENT_PATTERNS)
    ):
        return decision.model_copy(update={"mode": RetrievalMode.LOCAL})
    return decision
