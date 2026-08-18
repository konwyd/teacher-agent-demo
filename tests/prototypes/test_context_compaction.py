from types import SimpleNamespace

from prototypes.llm_led_teacher.context_compaction import (
    ContextCompactionSettings,
    ContextCompactionState,
    RollingContextSummary,
    estimate_tokens,
    maybe_compact_context,
    working_history_payload,
)
from prototypes.llm_led_teacher.logic import PrototypeEpisode, _teacher_payload
from app.agent.state import VerificationResult, VerificationStatus
from prototypes.llm_led_teacher.logic import PacingMode, TurnGuidance


def _turn(index: int, size: int = 220):
    return SimpleNamespace(
        turn_id=index,
        student_message=f"学生{index}-" + "甲" * size,
        teacher_output=SimpleNamespace(response=f"教师{index}-" + "乙" * size),
        evidence_level="verified" if index % 2 else "unverified",
    )


class _SummaryClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("summarizer unavailable")
        return RollingContextSummary(
            problem_and_goals=["继续原题"],
            established_facts=[f"已压缩批次{self.calls}"],
            student_progress=["保留已验证进度"],
        )


def _settings() -> ContextCompactionSettings:
    return ContextCompactionSettings(
        context_window_tokens=4_000,
        trigger_ratio=0.50,
        target_ratio=0.30,
        min_recent_turns=2,
        output_reserve_tokens=128,
        summary_max_tokens=256,
    )


def test_mixed_language_token_estimate_is_conservative_and_nonzero():
    assert estimate_tokens("") == 1
    assert estimate_tokens("数学abc") >= 4


def test_semantic_compaction_keeps_full_archive_but_slides_model_history():
    turns = [_turn(index) for index in range(1, 9)]
    client = _SummaryClient()

    result = maybe_compact_context(
        turns=turns,
        state=ContextCompactionState(),
        problem_context="原题",
        current_message="继续",
        system_prompt="system",
        client=client,
        settings=_settings(),
    )

    assert len(turns) == 8
    assert result.compacted_now == 6
    assert result.state.compacted_turns == 6
    assert result.state.compaction_count == 1
    assert result.state.last_status == "summarized"
    assert client.calls >= 1
    history = working_history_payload(turns, result.state)
    assert [item["turn_id"] for item in history] == [7, 8]
    assert result.projected_tokens_after < result.projected_tokens_before


def test_summarizer_failure_falls_back_to_bounded_extract_and_keeps_running():
    turns = [_turn(index) for index in range(1, 9)]
    result = maybe_compact_context(
        turns=turns,
        state=ContextCompactionState(),
        problem_context="原题",
        current_message="继续",
        system_prompt="system",
        client=_SummaryClient(fail=True),
        settings=_settings(),
    )

    assert result.state.last_status == "fallback_window"
    assert "summarizer unavailable" in result.state.last_error
    assert result.state.summary.continuity_notes
    assert len(working_history_payload(turns, result.state)) == 2


def test_recent_turn_preference_yields_to_hard_token_target():
    turns = [_turn(index, size=900) for index in range(1, 7)]
    settings = ContextCompactionSettings(
        context_window_tokens=4_000,
        trigger_ratio=0.50,
        target_ratio=0.25,
        min_recent_turns=4,
        output_reserve_tokens=128,
        summary_max_tokens=256,
    )

    result = maybe_compact_context(
        turns=turns,
        state=ContextCompactionState(),
        problem_context="原题",
        current_message="继续",
        system_prompt="system",
        client=_SummaryClient(),
        settings=settings,
    )

    assert result.state.compacted_turns > 2
    assert len(working_history_payload(turns, result.state)) == 1
    assert result.projected_tokens_after < result.projected_tokens_before
    assert result.projected_tokens_after < settings.context_window_tokens


def test_repeated_compaction_merges_prior_summary_hierarchically():
    client = _SummaryClient()
    settings = _settings()
    first_turns = [_turn(index) for index in range(1, 9)]

    first = maybe_compact_context(
        turns=first_turns,
        state=ContextCompactionState(),
        problem_context="原题",
        current_message="继续",
        system_prompt="system",
        client=client,
        settings=settings,
    )
    extended_turns = first_turns + [_turn(index) for index in range(9, 15)]
    second = maybe_compact_context(
        turns=extended_turns,
        state=first.state,
        problem_context="原题",
        current_message="再继续",
        system_prompt="system",
        client=client,
        settings=settings,
    )

    assert len(extended_turns) == 14
    assert second.state.compaction_count == 2
    assert second.state.compacted_turns == 12
    assert [item["turn_id"] for item in working_history_payload(extended_turns, second.state)] == [13, 14]
    assert client.calls >= 2


def test_old_episode_json_remains_compatible_without_compaction_field():
    episode = PrototypeEpisode.model_validate({"problem_context": "2x=4"})

    assert episode.context_compaction.compacted_turns == 0
    assert episode.context_compaction.summary.is_empty()


def test_teacher_payload_contains_summary_plus_recent_verbatim_turns():
    turns = [_turn(index) for index in range(1, 5)]
    episode = PrototypeEpisode.model_construct(
        problem_context="原题",
        turns=turns,
        context_compaction=ContextCompactionState(
            summary=RollingContextSummary(established_facts=["旧事实"]),
            compacted_turns=2,
            compaction_count=1,
        ),
    )
    payload = _teacher_payload(
        episode,
        "继续",
        VerificationResult(
            status=VerificationStatus.NOT_APPLICABLE,
            summary="n/a",
            verifier_name="none",
        ),
        TurnGuidance(pacing=PacingMode.COMPLETE_RESPONSE),
    )

    assert '"established_facts":["旧事实"]' in payload
    assert '"turn_id":1' not in payload
    assert '"turn_id":3' in payload


def test_logical_conversation_can_cross_524k_repeatedly_without_deleting_archive():
    settings = ContextCompactionSettings(
        context_window_tokens=524_288,
        trigger_ratio=0.80,
        target_ratio=0.55,
        min_recent_turns=12,
        output_reserve_tokens=8_192,
        summary_max_tokens=4_096,
    )
    turns = [_turn(index, size=1_000) for index in range(1, 241)]
    client = _SummaryClient()

    first = maybe_compact_context(
        turns=turns,
        state=ContextCompactionState(),
        problem_context="跨窗口原题",
        current_message="继续第一阶段",
        system_prompt="system",
        client=client,
        settings=settings,
    )

    assert first.projected_tokens_before > 524_288
    assert first.projected_tokens_after < settings.trigger_tokens
    assert len(turns) == 240
    assert len(working_history_payload(turns, first.state)) == 12

    turns.extend(_turn(index, size=1_000) for index in range(241, 481))
    second = maybe_compact_context(
        turns=turns,
        state=first.state,
        problem_context="跨窗口原题",
        current_message="继续第二阶段",
        system_prompt="system",
        client=client,
        settings=settings,
    )

    assert second.state.compaction_count == 2
    assert second.state.compacted_turns == 468
    assert second.projected_tokens_after < settings.trigger_tokens
    assert len(turns) == 480
