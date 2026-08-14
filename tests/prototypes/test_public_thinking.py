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


def test_update_callback_receives_incremental_stable_section_snapshots() -> None:
    updates = []
    recorder = PublicThinkingRecorder(update_callback=updates.append)

    recorder.classification(
        subject="数学",
        stage="初中",
        is_new_problem=True,
        intent="引导完成当前题目",
    )
    recorder.strategy("先验证学生步骤，再给下一步提示")

    assert [section.title for section in updates[0].sections] == ["题意"]
    assert [section.title for section in updates[1].sections] == ["题意", "写法"]
    assert updates[0].sections[0].details == ["数学新题", "引导完成当前题目"]
    assert updates[0].sections is not recorder.sections


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


def test_public_summary_contains_four_useful_sections() -> None:
    recorder = PublicThinkingRecorder()

    recorder.classification(
        subject="语文",
        stage="未知",
        is_new_problem=True,
        intent="把普通问句改成作文开头",
    )
    recorder.strategy("从画面、情绪、悬念三个方向改写")
    recorder.retrieval_decision(mode="none", reason="创作指导无需外部资料")
    recorder.composition("先解释方法，再给三个示例和迁移提示")

    assert [(section.title, section.details) for section in recorder.sections] == [
        ("题意", ["语文新题", "把普通问句改成作文开头"]),
        ("写法", ["从画面、情绪、悬念三个方向改写"]),
        ("资料", ["未使用 RAG", "创作指导无需外部资料"]),
        ("成稿", ["先解释方法，再给三个示例和迁移提示"]),
    ]


def test_public_events_explain_how_the_turn_is_being_handled() -> None:
    recorder = PublicThinkingRecorder()

    recorder.classification(
        subject="数学",
        stage="初中",
        is_new_problem=True,
        intent="引导完成当前题目",
    )
    recorder.strategy("先验证学生步骤，再给下一步提示")
    recorder.retrieval_decision(mode="none", reason="当前题可直接确定性计算")
    recorder.composition("先回应当前步骤，再给下一步提示")

    assert [event.summary for event in recorder.events] == [
        "已识别为初中数学新题；本轮目标是引导完成当前题目。",
        "将先验证学生当前步骤，再只推进一个关键步骤，避免一次给完答案。",
        "本题可直接依据题目与确定性工具处理，因此不调用 RAG。",
        "回答会先回应当前步骤，再给出下一步提示。",
    ]

def test_public_summary_rejects_private_agent_fields() -> None:
    recorder = PublicThinkingRecorder()

    recorder.agent(
        AgentTraceEvent(
            step=1,
            event_type="tool_result",
            tool_name="search_web",
            ok=True,
            summary="SECRET SUMMARY",
            details={
                "hit_count": 2,
                "details": "SECRET DETAILS",
                "system_prompt": "SECRET SYSTEM PROMPT",
                "reasoning_content": "SECRET COT",
            },
        )
    )

    dumped = repr(recorder.sections)
    assert "2 条可用的联网参考" in dumped
    assert "SECRET" not in dumped
    assert "system_prompt" not in dumped
    assert "reasoning_content" not in dumped


def test_public_summary_uses_neutral_copy_for_unsupported_values() -> None:
    recorder = PublicThinkingRecorder()

    recorder.classification(
        subject="SECRET SUBJECT",
        stage="SECRET STAGE",
        is_new_problem=True,
        intent="SECRET INTENT",
    )
    recorder.strategy("SECRET STRATEGY")
    recorder.retrieval_decision(mode="SECRET MODE", reason="SECRET REASON")
    recorder.composition("SECRET STRUCTURE")

    assert [section.details for section in recorder.sections] == [
        ["题目类型已确认"],
        ["处理方式已确定"],
        ["资料使用方式已确定"],
        ["成稿结构已确定"],
    ]
    assert "SECRET" not in repr(recorder.sections)


def test_public_reason_is_normalized_and_capped_at_120_characters() -> None:
    recorder = PublicThinkingRecorder()

    recorder.retrieval_decision(
        mode="local",
        reason="  使用\n课程资料  " + "补" * 120,
    )

    assert recorder.sections[0].details[0] == "使用本地 RAG"
    assert recorder.sections[0].details[1].startswith("使用 课程资料")
    assert "\n" not in recorder.sections[0].details[1]
    assert len(recorder.sections[0].details[1]) == 120


def test_retrieval_summary_accepts_only_integer_hit_counts() -> None:
    recorder = PublicThinkingRecorder()

    recorder.retrieval(
        source="local",
        hit_count="99",
        online_triggered=False,
        online_status="not_triggered",
    )

    assert recorder.sections[0].details == ["未找到可用参考"]
    assert "99" not in repr(recorder.sections)


def test_repeated_updates_stay_consolidated_and_keep_legacy_callback() -> None:
    emitted = []
    recorder = PublicThinkingRecorder(callback=emitted.append)

    recorder.strategy("先确认题意，再分步讲解")
    recorder.strategy("先验证学生步骤，再给下一步提示")
    recorder.retrieval(
        source="local",
        hit_count=3,
        online_triggered=False,
        online_status="not_triggered",
    )
    recorder.complete(1.25)

    assert len(recorder.sections) == 3
    assert [section.title for section in recorder.sections] == ["写法", "资料", "成稿"]
    assert recorder.sections[0].details == ["先验证学生步骤，再给下一步提示"]
    assert recorder.sections[-1].elapsed_seconds == 1.25
    assert emitted == recorder.events
    assert any(
        event.summary == "将先验证学生当前步骤，再只推进一个关键步骤，避免一次给完答案。"
        for event in emitted
    )


def test_section_subslots_preserve_safe_details_in_stable_order() -> None:
    recorder = PublicThinkingRecorder()

    recorder.strategy("先验证学生步骤，再给下一步提示")
    recorder.verification(
        VerificationResult(
            status=VerificationStatus.VERIFIED_CORRECT,
            summary="PRIVATE VERIFIER SUMMARY",
            verifier_name="linear_equation",
        )
    )
    recorder.agent(
        AgentTraceEvent(
            step=2,
            event_type="tool_result",
            tool_name="search_local_knowledge",
            ok=True,
            summary="PRIVATE TOOL SUMMARY",
            elapsed_seconds=0.25,
            details={"hit_count": 2, "reasoning_content": "PRIVATE COT"},
        )
    )
    recorder.retrieval(
        source="local",
        hit_count=2,
        online_triggered=False,
        online_status="not_triggered",
    )
    recorder.retrieval_decision(mode="local", reason="需要课程内定义")
    recorder.composition("先回应当前步骤，再给下一步提示")
    recorder.audit(repair_performed=True, remaining_warning_count=0)
    recorder.complete(1.5)

    assert [section.title for section in recorder.sections] == [
        "写法",
        "资料",
        "成稿",
    ]
    assert recorder.sections[0].details == [
        "先验证学生步骤，再给下一步提示",
        "学生当前步骤已经通过确定性验证。",
    ]
    assert recorder.sections[1].details == [
        "使用本地 RAG",
        "需要课程内定义",
        "找到 2 条可用的本地参考。 · 0.250 秒",
        "找到 2 条可用参考",
    ]
    assert recorder.sections[2].details == [
        "先回应当前步骤，再给下一步提示",
        "修正后的成稿已通过审计",
    ]
    assert recorder.sections[2].elapsed_seconds == 1.5
    assert "PRIVATE" not in repr(recorder.sections)
