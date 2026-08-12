from app.agent.state import VerificationResult, VerificationStatus
from prototypes.llm_led_teacher.orchestration_models import AgentTraceEvent
from prototypes.llm_led_teacher.public_thinking import (
    PublicThinkingKind,
    PublicThinkingRecorder,
    PublicThinkingStatus,
    format_public_thinking_event,
)


def test_recorder_assigns_strict_sequence_and_emits_callback() -> None:
    emitted = []
    recorder = PublicThinkingRecorder(callback=emitted.append)

    recorder.stage("observing")
    recorder.stage("retrieving")
    recorder.complete(2.5)

    assert [event.sequence for event in recorder.events] == [1, 2, 3]
    assert emitted == recorder.events
    assert recorder.events[0].kind is PublicThinkingKind.CLASSIFICATION
    assert recorder.events[-1].status is PublicThinkingStatus.COMPLETED
    assert recorder.events[-1].elapsed_seconds == 2.5


def test_callback_failure_does_not_remove_recorded_event() -> None:
    def broken_callback(event) -> None:
        raise RuntimeError(f"UI unavailable at event {event.sequence}")

    recorder = PublicThinkingRecorder(callback=broken_callback)

    recorder.stage("observing")

    assert len(recorder.events) == 1
    assert recorder.events[0].summary == "正在识别题目和学生意图……"


def test_agent_mapper_uses_only_whitelisted_tool_result_fields() -> None:
    recorder = PublicThinkingRecorder()
    recorder.agent(
        AgentTraceEvent(
            step=2,
            event_type="tool_result",
            summary="arbitrary model reason containing SECRET",
            tool_name="search_local_knowledge",
            ok=True,
            elapsed_seconds=0.25,
            details={
                "hit_count": 4,
                "reasoning_content": "SECRET COT",
                "system_prompt": "SECRET PROMPT",
            },
        )
    )

    dumped = repr(recorder.events)
    assert "找到 4 条可用的本地参考" in dumped
    assert "SECRET" not in dumped
    assert "reasoning_content" not in dumped
    assert "system_prompt" not in dumped
    assert "Agent 2" not in format_public_thinking_event(recorder.events[0])


def test_classification_mapper_uses_typed_public_scalars_only() -> None:
    recorder = PublicThinkingRecorder()
    recorder.agent(
        AgentTraceEvent(
            step=0,
            event_type="classification",
            summary="model-authored summary must not be rendered",
            details={
                "scope": "k9",
                "subject": "物理",
                "stage": "初中",
                "is_new_problem": True,
                "reason": "SECRET ROUTING REASON",
            },
        )
    )

    assert recorder.events[0].summary == "已确定为初中物理新题。"
    assert "SECRET" not in repr(recorder.events)


def test_tool_failure_never_renders_raw_error_or_unknown_payload() -> None:
    recorder = PublicThinkingRecorder()
    recorder.agent(
        AgentTraceEvent(
            step=3,
            event_type="tool_result",
            summary="SECRET SUMMARY",
            tool_name="search_web",
            ok=False,
            details={"error": "SECRET STACK TRACE", "api_key": "SECRET KEY"},
        )
    )

    assert recorder.events[0].status is PublicThinkingStatus.WARNING
    assert recorder.events[0].summary == "联网参考暂时不可用，将使用已有证据继续。"
    assert "SECRET" not in repr(recorder.events)


def test_unknown_agent_event_is_not_made_public() -> None:
    recorder = PublicThinkingRecorder()

    recorder.agent(
        AgentTraceEvent(
            step=4,
            event_type="private_planning",
            summary="SECRET PRIVATE PLAN",
            tool_name="unregistered_tool",
            details={"reasoning_content": "SECRET COT"},
        )
    )

    assert recorder.events == []


def test_unverified_attempt_is_reported_as_warning_not_success() -> None:
    recorder = PublicThinkingRecorder()

    recorder.verification(
        VerificationResult(
            status=VerificationStatus.UNABLE_TO_VERIFY,
            summary="Internal verifier message",
            verifier_name="none",
        )
    )

    assert recorder.events[0].kind is PublicThinkingKind.VERIFICATION
    assert recorder.events[0].status is PublicThinkingStatus.WARNING
    assert recorder.events[0].summary == (
        "当前题型暂无确定性验证，回答将保留必要的不确定性说明。"
    )
    assert "Internal verifier message" not in repr(recorder.events)


def test_retrieval_and_audit_use_typed_counts_and_states() -> None:
    recorder = PublicThinkingRecorder()

    recorder.retrieval(
        source="hybrid",
        hit_count=5,
        online_triggered=True,
        online_status="used",
    )
    recorder.audit(repair_performed=True, remaining_warning_count=0)

    assert recorder.events[0].summary == "已找到 5 条可用参考，其中包含联网补充资料。"
    assert recorder.events[0].status is PublicThinkingStatus.COMPLETED
    assert recorder.events[1].summary == "修正后的回答已经通过教学内容与节奏审计。"
    assert recorder.events[1].status is PublicThinkingStatus.COMPLETED


def test_formatter_shows_only_public_status_summary_and_elapsed_time() -> None:
    recorder = PublicThinkingRecorder()
    recorder.agent(
        AgentTraceEvent(
            step=2,
            event_type="tool_result",
            summary="ignored",
            tool_name="search_local_knowledge",
            ok=True,
            elapsed_seconds=0.25,
            details={"hit_count": 4},
        )
    )

    assert format_public_thinking_event(recorder.events[0]) == (
        "✓ 找到 4 条可用的本地参考。 · 0.250 秒"
    )
