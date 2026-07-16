import pytest

from app.agent.generic_math_reasoning import GenericMathReasoner
from app.agent.state import (
    GraphContext,
    VerificationResult,
    VerificationStatus,
)
from app.agent.teaching_harness import run_generic_teaching_turn
from app.llm.deepseek_client import DeepSeekError
from app.memory.schemas import EpisodeState


pytestmark = pytest.mark.safety_contract


class BrokenVerifier:
    name = "broken"

    def supports(self, request):
        return True

    def verify(self, request):
        raise RuntimeError("verifier unavailable")


class BrokenGraph:
    name = "broken"
    available = False

    def get_context(self, request):
        raise RuntimeError("graph unavailable")


class FailingEpisodeStore:
    def get(self, student_id, conversation_id=None):
        return EpisodeState(conversation_id=conversation_id or student_id, student_id=student_id)

    def save(self, episode):
        raise OSError("read-only store")


class EvidenceVerifier:
    name = "evidence"

    def supports(self, request):
        return True

    def verify(self, request):
        return VerificationResult(
            status=VerificationStatus.VERIFIED_CORRECT,
            summary="verified",
            evidence=["学生步骤已验证"],
            verifier_name=self.name,
        )


class UnexpectedReasoner:
    def analyze(self, message, problem_context, analysis):
        raise AssertionError("reasoner must be skipped")


class BrokenReasoner:
    def analyze(self, message, problem_context, analysis):
        raise RuntimeError("reasoner unavailable")


class FailingReasoningClient:
    def chat_json(self, **kwargs):
        raise DeepSeekError("reasoner transport failed")


def _decision(state, module):
    return next(
        decision
        for decision in state.orchestration_trace.decisions
        if decision.module == module
    )


def test_optional_failures_return_a_guarded_response() -> None:
    state = run_generic_teaching_turn(
        "degraded-student",
        "三角形内角和为什么是180度？",
        verifier=BrokenVerifier(),
        graph_context_provider=BrokenGraph(),
        analysis_allow_llm=False,
    )
    assert state.verification.status is VerificationStatus.NOT_APPLICABLE
    assert state.graph_context == GraphContext()
    assert state.response.count("？") == 1
    assert {"verifier_failed", "graph_context_failed"}.issubset(
        set(state.orchestration_trace.fallback_reasons)
    )
    assert _decision(state, "verifier").reason == "verifier_failed"
    assert _decision(state, "graph_provider").reason == "graph_context_failed"


def test_persistence_failure_does_not_block_safe_response() -> None:
    state = run_generic_teaching_turn(
        "storage-student",
        "三角形内角和为什么是180度？",
        episode_store=FailingEpisodeStore(),
        analysis_allow_llm=False,
    )
    assert state.response.count("？") == 1
    assert "persistence_failed" in state.orchestration_trace.fallback_reasons
    assert _decision(state, "persistence").reason == "persistence_failed"


def test_verifier_evidence_skips_generic_reasoning_with_exact_reason() -> None:
    state = run_generic_teaching_turn(
        "verified-student",
        "我认为可以先整理关系。",
        verifier=EvidenceVerifier(),
        reasoner=UnexpectedReasoner(),
        analysis_allow_llm=False,
    )

    decision = _decision(state, "reasoner")
    assert decision.called is False
    assert decision.reason == "verifier_evidence_sufficient"
    assert decision.llm_calls == 0
    assert state.orchestration_trace.llm_call_count <= 3


def test_reasoner_failure_degrades_to_clarification_with_exact_reason() -> None:
    state = run_generic_teaching_turn(
        "reasoner-failure-student",
        "三角形内角和为什么是180度？",
        reasoner=BrokenReasoner(),
        analysis_allow_llm=False,
    )

    assert state.math_reasoning.needs_clarification is True
    assert "reasoner_failed" in state.orchestration_trace.fallback_reasons
    assert _decision(state, "reasoner").reason == "reasoner_failed"
    assert state.response.count("？") == 1


def test_default_reasoner_error_return_is_recorded_as_reasoner_failed() -> None:
    state = run_generic_teaching_turn(
        "default-reasoner-failure-student",
        "三角形内角和为什么是180度？",
        reasoner=GenericMathReasoner(client=FailingReasoningClient()),
        analysis_allow_llm=False,
    )

    decision = _decision(state, "reasoner")
    assert state.math_reasoning.needs_clarification is True
    assert "reasoner_failed" in state.orchestration_trace.fallback_reasons
    assert decision.called is True
    assert decision.reason == "reasoner_failed"
    assert decision.llm_calls == 1
    assert state.response.count("？") == 1
