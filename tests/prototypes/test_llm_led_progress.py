from __future__ import annotations

from app.retrieval.semantic_provider import DisabledSemanticRetrievalProvider
from prototypes.llm_led_teacher.live_progress import LiveProgressTrace
from prototypes.llm_led_teacher.logic import PrototypeEpisode, run_turn


class _Client:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        return schema.model_validate(
            {
                "starts_new_problem": True,
                "pacing": "complete_response",
                "reason": "direct fact",
            }
        )

    def chat(self, system, user, temperature=0.0, max_tokens=None):
        response = self.responses[self.calls]
        self.calls += 1
        return response


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class _StatusWriter:
    def __init__(self) -> None:
        self.values: list[str] = []

    def write(self, value: str) -> None:
        self.values.append(value)


def test_run_turn_reports_progress_and_stage_timings() -> None:
    stages: list[str] = []

    turn = run_turn(
        PrototypeEpisode(),
        "三角形内角和是多少",
        client=_Client("三角形内角和是180度。"),
        retrieval_provider=DisabledSemanticRetrievalProvider(),
        progress_callback=stages.append,
        clock=_Clock([0.0, 1.0, 3.0, 6.0, 10.0]),
    )

    assert stages == [
        "observing",
        "retrieving",
        "generating",
        "auditing",
        "complete",
    ]
    assert turn.stage_timings_seconds == {
        "observing": 1.0,
        "retrieving": 2.0,
        "generating": 3.0,
        "auditing": 4.0,
    }
    assert turn.backend_total_seconds == 10.0
    assert turn.repair_performed is False
    assert not hasattr(turn, "chain_of_thought")


def test_progress_callback_failure_does_not_block_teacher_response() -> None:
    def fail(stage: str) -> None:
        raise RuntimeError("UI disconnected")

    turn = run_turn(
        PrototypeEpisode(),
        "三角形内角和是多少",
        client=_Client("三角形内角和是180度。"),
        retrieval_provider=DisabledSemanticRetrievalProvider(),
        progress_callback=fail,
    )

    assert turn.teacher_output.response == "三角形内角和是180度。"


def test_live_progress_trace_exposes_only_public_stage_status() -> None:
    writer = _StatusWriter()
    trace = LiveProgressTrace(writer, clock=_Clock([0.0, 1.25, 2.0]))

    trace.update("retrieving")
    trace.update("complete")

    rendered = "\n".join(writer.values)
    assert "识别题目与教学节奏" in rendered
    assert "完成题目与节奏判断" in rendered
    assert "检索教材与题库" in rendered
    assert "完成教材与题库检索" in rendered
    assert "chain_of_thought" not in rendered
