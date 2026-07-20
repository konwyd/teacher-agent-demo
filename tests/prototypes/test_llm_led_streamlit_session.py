from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


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


class _FakeStreamlit:
    def __init__(self, episode) -> None:
        self.session_state = _SessionState(prototype_episode=episode)
        self.sidebar = _Context()

    def __getattr__(self, name: str):
        if name in {"chat_message", "expander", "spinner"}:
            return lambda *args, **kwargs: _Context()
        if name == "columns":
            return lambda *args, **kwargs: (_Context(), _Context())
        if name == "button":
            return lambda *args, **kwargs: False
        if name == "chat_input":
            return lambda *args, **kwargs: None
        return lambda *args, **kwargs: None


def _legacy_episode():
    value = SimpleNamespace(value="legacy")
    teacher_output = SimpleNamespace(
        response="先加5，再除以3，得到x=5。",
        action_label=value,
        student_assessment=value,
        expects_student_response=True,
        disclosure_level=value,
        disclosure_reason=value,
        evidence_turn_ids=[],
        confidence=0.5,
        rationale="",
        mathematical_claims=["x=5"],
    )
    turn = SimpleNamespace(
        student_message="老师，3x-5=10怎么解？",
        teacher_output=teacher_output,
        evidence_level="unverified",
        student_verification=SimpleNamespace(
            status=value,
            verifier_name="none",
            summary="",
        ),
        claim_audits=[],
        repair_reasons=[],
        remaining_warnings=[],
    )
    return SimpleNamespace(problem_context="3x-5=10", turns=[turn])


def test_streamlit_resets_a_stale_episode_after_schema_change(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit(_legacy_episode())
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    app_path = (
        Path(__file__).resolve().parents[2]
        / "prototypes"
        / "llm_led_teacher"
        / "streamlit_app.py"
    )
    runpy.run_path(str(app_path), run_name="llm_led_streamlit_session_test")

    assert fake_streamlit.session_state.prototype_episode.turns == []
    assert fake_streamlit.session_state.prototype_session_schema_version == 4
