from __future__ import annotations

import json

import pytest

from app.retrieval.semantic_provider import (
    DisabledSemanticRetrievalProvider,
    RetrievalContentKind,
    RetrievalHit,
    RetrievalStatus,
    SemanticRetrievalRequest,
    SemanticRetrievalResult,
)
from prototypes.llm_led_teacher import logic as llm_logic
from prototypes.llm_led_teacher.logic import PrototypeEpisode, run_turn


class _CapturingClient:
    def __init__(self, response: str = "教材说明三角形内角和是180度。") -> None:
        self.response = response
        self.observer_payloads: list[dict] = []
        self.teacher_payloads: list[dict] = []

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.observer_payloads.append(json.loads(user))
        return schema.model_validate(
            {
                "starts_new_problem": True,
                "pacing": "complete_response",
                "reason": "new problem",
            }
        )

    def chat(self, system, user, temperature=0.0, max_tokens=None):
        self.teacher_payloads.append(json.loads(user))
        return self.response


class _ResultProvider:
    name = "result-provider"

    def __init__(self, result: SemanticRetrievalResult) -> None:
        self.result = result
        self.requests: list[SemanticRetrievalRequest] = []

    def retrieve(self, request: SemanticRetrievalRequest) -> SemanticRetrievalResult:
        self.requests.append(request)
        return self.result.model_copy(update={"query": request.problem_context})


class _RaisingProvider:
    name = "raising-provider"

    def retrieve(self, request: SemanticRetrievalRequest) -> SemanticRetrievalResult:
        raise RuntimeError("private Chroma connection detail")


def _hit(
    *,
    kind: RetrievalContentKind = RetrievalContentKind.REFERENCE,
    text: str = "有效教材片段",
    reference_id: str = "textbook:1",
) -> RetrievalHit:
    return RetrievalHit(
        reference_id=reference_id,
        source_label="七年级教材" if kind is RetrievalContentKind.REFERENCE else "题库",
        collection=("k9_textbook" if kind is RetrievalContentKind.REFERENCE else "k9_exercises"),
        content_kind=kind,
        text=text,
        similarity=0.9,
    )


def test_retrieval_enters_only_teacher_payload_after_problem_is_anchored() -> None:
    client = _CapturingClient()
    provider = _ResultProvider(
        SemanticRetrievalResult(
            status=RetrievalStatus.SUCCESS,
            provider="fake-vector",
            hits=[_hit()],
        )
    )
    episode = PrototypeEpisode()

    turn = run_turn(
        episode,
        "三角形内角和是多少",
        client=client,
        retrieval_provider=provider,
    )

    assert "retrieval_reference" not in client.observer_payloads[0]
    assert provider.requests[0].problem_context == "三角形内角和是多少"
    teacher_payload = client.teacher_payloads[0]
    assert teacher_payload["problem_context"] == "三角形内角和是多少"
    assert teacher_payload["current_student_message"] == "三角形内角和是多少"
    assert teacher_payload["retrieval_reference"]["hits"][0]["text"] == "有效教材片段"
    assert episode.problem_context == "三角形内角和是多少"
    assert turn.retrieval_status is RetrievalStatus.SUCCESS


@pytest.mark.parametrize(
    ("provider", "expected_status", "expected_error"),
    [
        (DisabledSemanticRetrievalProvider(), RetrievalStatus.DISABLED, ""),
        (
            _ResultProvider(
                SemanticRetrievalResult(
                    status=RetrievalStatus.EMPTY,
                    provider="empty-provider",
                )
            ),
            RetrievalStatus.EMPTY,
            "",
        ),
        (
            _ResultProvider(
                SemanticRetrievalResult(
                    status=RetrievalStatus.ERROR,
                    provider="error-provider",
                    error_code="backend_unavailable",
                )
            ),
            RetrievalStatus.ERROR,
            "backend_unavailable",
        ),
        (_RaisingProvider(), RetrievalStatus.ERROR, "provider_failed"),
    ],
)
def test_retrieval_degradation_never_blocks_teacher_response(
    provider,
    expected_status: RetrievalStatus,
    expected_error: str,
) -> None:
    client = _CapturingClient("三角形内角和是180度。")

    turn = run_turn(
        PrototypeEpisode(),
        "三角形内角和是多少",
        client=client,
        retrieval_provider=provider,
    )

    assert turn.teacher_output.response == "三角形内角和是180度。"
    assert turn.retrieval_status is expected_status
    assert turn.retrieval_error_code == expected_error
    assert "retrieval_reference" not in client.teacher_payloads[0]
    assert "private Chroma connection detail" not in turn.model_dump_json()


def test_provider_factory_failure_never_blocks_teacher_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_factory():
        raise ValueError("private invalid environment value")

    monkeypatch.setattr(llm_logic, "get_semantic_retrieval_provider", fail_factory)
    client = _CapturingClient("三角形内角和是180度。")

    turn = run_turn(
        PrototypeEpisode(),
        "三角形内角和是多少",
        client=client,
    )

    assert turn.teacher_output.response == "三角形内角和是180度。"
    assert turn.retrieval_status is RetrievalStatus.ERROR
    assert turn.retrieval_error_code == "provider_factory_failed"
    assert "private invalid environment value" not in turn.model_dump_json()


def test_prompt_boundary_removes_example_from_noncompliant_provider() -> None:
    client = _CapturingClient("两边同时减3，得到2x=4。你能继续吗？")
    provider = _ResultProvider(
        SemanticRetrievalResult(
            status=RetrievalStatus.SUCCESS,
            provider="noncompliant-provider",
            hits=[
                _hit(),
                _hit(
                    kind=RetrievalContentKind.EXAMPLE,
                    text="例题：解方程3x+2=8。",
                    reference_id="exercise:1",
                ),
            ],
        )
    )

    turn = run_turn(
        PrototypeEpisode(),
        "2x+3=7怎么解",
        client=client,
        retrieval_provider=provider,
    )

    payload_text = json.dumps(client.teacher_payloads[0], ensure_ascii=False)
    assert "有效教材片段" in payload_text
    assert "例题：解方程3x+2=8。" not in payload_text
    assert turn.retrieval_examples_allowed is False
    assert all(
        hit.content_kind is RetrievalContentKind.REFERENCE
        for hit in turn.retrieval_hits
    )
