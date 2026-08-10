from __future__ import annotations

import runpy
import sys
from pathlib import Path

from app.agent.state import VerificationResult, VerificationStatus
from app.retrieval.semantic_provider import (
    RetrievalContentKind,
    RetrievalHit,
    RetrievalStatus,
)
from prototypes.llm_led_teacher.logic import (
    PrototypeEpisode,
    PrototypeTurnRecord,
    TeacherResponse,
)


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _SessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


class _RecordingStreamlit:
    def __init__(self, episode: PrototypeEpisode) -> None:
        self.session_state = _SessionState(
            prototype_episode=episode,
            prototype_session_schema_version=5,
        )
        self.sidebar = _Context()
        self.writes: list[tuple] = []
        self.warnings: list[str] = []

    def write(self, *values, **kwargs) -> None:
        self.writes.append(values)

    def warning(self, value: str) -> None:
        self.warnings.append(value)

    def __getattr__(self, name: str):
        if name in {"chat_message", "expander", "spinner", "status"}:
            return lambda *args, **kwargs: _Context()
        if name == "button":
            return lambda *args, **kwargs: False
        if name == "chat_input":
            return lambda *args, **kwargs: None
        return lambda *args, **kwargs: None


def _turn(
    *,
    status: RetrievalStatus,
    hits: list[RetrievalHit] | None = None,
    error_code: str = "",
) -> PrototypeTurnRecord:
    return PrototypeTurnRecord(
        turn_id=1,
        student_message="三角形内角和是多少",
        teacher_output=TeacherResponse(response="三角形内角和是180度。"),
        student_verification=VerificationResult(
            status=VerificationStatus.UNABLE_TO_VERIFY,
            verifier_name="none",
        ),
        evidence_level="unverified",
        retrieval_status=status,
        retrieval_provider="chroma_vector",
        retrieval_query="三角形内角和是多少",
        retrieval_hits=hits or [],
        retrieval_examples_allowed=False,
        retrieval_error_code=error_code,
    )


def _run_app(monkeypatch, fake_streamlit: _RecordingStreamlit) -> str:
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    app_path = (
        Path(__file__).resolve().parents[2]
        / "prototypes"
        / "llm_led_teacher"
        / "streamlit_app.py"
    )
    runpy.run_path(str(app_path), run_name="llm_led_rag_ui_test")
    return repr(fake_streamlit.writes) + repr(fake_streamlit.warnings)


def test_successful_rag_turn_displays_safe_source_audit(monkeypatch) -> None:
    hit = RetrievalHit(
        reference_id="k9_textbook:t1",
        source_label="七年级数学",
        collection="k9_textbook",
        content_kind=RetrievalContentKind.REFERENCE,
        text="不会直接显示在摘要中的教材正文",
        similarity=0.91,
    )
    fake_streamlit = _RecordingStreamlit(
        PrototypeEpisode(
            problem_context="三角形内角和是多少",
            turns=[_turn(status=RetrievalStatus.SUCCESS, hits=[hit])],
        )
    )

    rendered = _run_app(monkeypatch, fake_streamlit)

    assert "success" in rendered
    assert "chroma_vector" in rendered
    assert "k9_textbook:t1" in rendered
    assert "七年级数学" in rendered
    assert "reference" in rendered
    assert "0.91" in rendered
    assert "不会直接显示在摘要中的教材正文" not in rendered


def test_rag_error_turn_displays_only_sanitized_error_code(monkeypatch) -> None:
    fake_streamlit = _RecordingStreamlit(
        PrototypeEpisode(
            problem_context="三角形内角和是多少",
            turns=[
                _turn(
                    status=RetrievalStatus.ERROR,
                    error_code="backend_unavailable",
                )
            ],
        )
    )

    rendered = _run_app(monkeypatch, fake_streamlit)

    assert "error" in rendered
    assert "backend_unavailable" in rendered
    assert "private" not in rendered
