"""Behavior tests for the structured 8502 student-memory contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.memory.conversation_archive import LocalConversationStore
from app.memory.learner_memory import (
    GlobalStudentProfile,
    LearnerMemoryStore,
    MemoryResolutionStatus,
    StudentMemoryIdentity,
    StudentMemoryService,
    TopicMemory,
    TopicMemoryEvidence,
)
from prototypes.llm_led_teacher.logic import (
    CrossSessionMemory,
    PrototypeEpisode,
    run_turn,
)
from prototypes.llm_led_teacher.orchestration_models import (
    AnswerMode,
    RequestClassification,
    RequestScope,
    SchoolStage,
    Subject,
)
from prototypes.llm_led_teacher.orchestrator import MultisubjectAgentLoop


@pytest.fixture
def memory_service(tmp_path: Path):
    service = StudentMemoryService(
        LearnerMemoryStore(tmp_path / "memory.sqlite3"),
    )
    try:
        yield service
    finally:
        service.shutdown()


def _seed_composition(
    service: StudentMemoryService,
    profile_id: str,
    topic: str,
    conversation_id: str,
) -> TopicMemory:
    memory = TopicMemory(
        memory_id=f"memory_{conversation_id}",
        student_profile_id=profile_id,
        subject="语文",
        subject_key="chinese",
        domain="composition",
        topic=topic,
        normalized_topic="".join(topic.split()),
        problem_context=f"怎么写好《{topic}》这篇作文？",
        summary=f"学生正在写“{topic}”。",
        progress="已经开始整理作文内容。",
        next_step="继续完善作文结构。",
        evidence=[
            TopicMemoryEvidence(
                conversation_id=conversation_id,
                turn_id=1,
                kind="learning_turn",
                summary="已经开始整理作文内容。",
                confidence=0.9,
            )
        ],
        source_conversation_ids=[conversation_id],
        confidence=0.9,
        status="active",
    )
    return service.store.upsert_topic_memory(memory)


def _classification() -> dict[str, object]:
    return {
        "scope": "k9",
        "subject": "语文",
        "stage": "小学",
        "is_new_problem": True,
        "answer_mode": "guided",
        "reason": "明确的语文作文请求",
    }


def _writeback_payload(message: str, turn_id: int = 1) -> dict[str, object]:
    return {
        "turn_id": turn_id,
        "student_message": message,
        "problem_context": "请写《我的植物朋友》这篇作文",
        "classification": _classification(),
        "verification": {
            "status": "unable_to_verify",
            "summary": "作文不适用数学验证",
            "verifier_name": "none",
        },
        "teaching_contract": {
            "response_mode": "one_step",
            "requested_target": "先确定描写对象",
            "allowed_content": ["作文构思"],
            "forbidden_content": ["代写全文"],
            "may_apply_to_active_problem": True,
            "may_reveal_final_answer": False,
            "reason": "逐步指导",
        },
        "memory_resolution_status": "none",
    }


def test_async_writeback_updates_global_and_subject_topic_namespaces(
    memory_service: StudentMemoryService,
) -> None:
    profile = memory_service.load_profile("user_a")
    identity = StudentMemoryIdentity("user_a", "conv_a")

    job_id = memory_service.enqueue_turn(
        identity,
        profile,
        _writeback_payload(
            "我是五年级，我叫小禾，请讲慢点。我在写《我的植物朋友》作文。"
        ),
    )
    memory_service.wait_for_idle()

    assert memory_service.store.job_status(job_id) == "completed"
    updated = memory_service.load_profile("user_a")
    assert updated.student_profile_id == profile.student_profile_id
    assert updated.global_profile.grade == "五年级"
    assert updated.global_profile.preferred_name == "小禾"
    assert "慢节奏讲解" in updated.global_profile.teaching_preferences

    memories = memory_service.store.list_topic_memories(
        profile.student_profile_id,
        "语文",
    )
    assert len(memories) == 1
    assert memories[0].domain == "composition"
    assert memories[0].topic == "我的植物朋友"
    assert memories[0].source_conversation_ids == ["conv_a"]


def test_unique_history_topic_is_selected_and_restores_problem_context(
    memory_service: StudentMemoryService,
) -> None:
    profile = memory_service.load_profile("user_unique")
    seeded = _seed_composition(
        memory_service,
        profile.student_profile_id,
        "我的植物朋友",
        "conv_old",
    )

    resolution = memory_service.resolve(
        profile,
        user_id="user_unique",
        conversation_id="conv_new",
        subject="语文",
        stage="小学",
        message="这个作文的开头怎么写？",
        problem_context="",
        is_new_problem=True,
    )

    assert resolution.status is MemoryResolutionStatus.UNIQUE
    assert resolution.selected is not None
    assert resolution.selected.memory_id == seeded.memory_id
    assert resolution.should_restore_context is True
    assert resolution.prompt_payload()["topic_memory"]["topic"] == "我的植物朋友"


def test_multiple_history_topics_require_one_short_clarification(
    memory_service: StudentMemoryService,
) -> None:
    profile = memory_service.load_profile("user_ambiguous")
    _seed_composition(
        memory_service,
        profile.student_profile_id,
        "我的植物朋友",
        "conv_plant",
    )
    _seed_composition(
        memory_service,
        profile.student_profile_id,
        "难忘的一天",
        "conv_day",
    )

    resolution = memory_service.resolve(
        profile,
        user_id="user_ambiguous",
        conversation_id="conv_new",
        subject="语文",
        stage="小学",
        message="继续上次的作文",
        problem_context="",
        is_new_problem=True,
    )

    assert resolution.status is MemoryResolutionStatus.AMBIGUOUS
    assert len(resolution.candidates) == 2
    assert "我的植物朋友" in resolution.clarification_question
    assert "难忘的一天" in resolution.clarification_question
    assert resolution.clarification_question.count("？") == 1


def test_unrelated_new_question_gets_global_only_without_topic_injection(
    memory_service: StudentMemoryService,
) -> None:
    profile = memory_service.load_profile("user_none")
    memory_service.store.update_global_profile(
        profile.student_profile_id,
        preferred_name="小语",
        teaching_preferences=["分步骤讲解"],
    )
    profile = memory_service.load_profile("user_none")
    _seed_composition(
        memory_service,
        profile.student_profile_id,
        "我的植物朋友",
        "conv_old",
    )

    resolution = memory_service.resolve(
        profile,
        user_id="user_none",
        conversation_id="conv_new",
        subject="语文",
        stage="小学",
        message="怎样提高语文成绩？",
        problem_context="",
        is_new_problem=True,
    )

    assert resolution.status is MemoryResolutionStatus.NONE
    assert resolution.selected is None
    assert resolution.global_profile == GlobalStudentProfile(
        preferred_name="小语",
        teaching_preferences=["分步骤讲解"],
    )
    assert "topic_memory" not in resolution.prompt_payload()


def test_profiles_and_topic_memories_are_isolated_by_user(
    memory_service: StudentMemoryService,
) -> None:
    first = memory_service.load_profile("user_first")
    second = memory_service.load_profile("user_second")
    _seed_composition(
        memory_service,
        first.student_profile_id,
        "只属于第一个学生",
        "conv_first",
    )

    first_resolution = memory_service.resolve(
        first,
        user_id="user_first",
        conversation_id="new_first",
        subject="语文",
        stage="小学",
        message="继续上次的作文",
        problem_context="",
        is_new_problem=True,
    )
    second_resolution = memory_service.resolve(
        second,
        user_id="user_second",
        conversation_id="new_second",
        subject="语文",
        stage="小学",
        message="继续上次的作文",
        problem_context="",
        is_new_problem=True,
    )

    assert first.student_profile_id != second.student_profile_id
    assert first_resolution.status is MemoryResolutionStatus.UNIQUE
    assert second_resolution.status is MemoryResolutionStatus.NONE


def test_existing_conversation_archive_is_used_before_profile_backfill(
    tmp_path: Path,
) -> None:
    conversation_db = tmp_path / "conversations.sqlite3"
    archive = LocalConversationStore(conversation_db)
    service = StudentMemoryService(
        LearnerMemoryStore(tmp_path / "memory.sqlite3"),
        archive_store=archive,
    )
    try:
        profile = service.load_profile("archive_user")
        old = archive.create(
            "archive_user",
            {
                "problem_context": "怎么写好《难忘的一天》这篇作文？",
                "active_subject": "语文",
                "active_stage": "小学",
                "turns": [
                    {
                        "student_message": "怎么写好《难忘的一天》这篇作文？",
                        "teacher_output": {"response": "先确定最难忘的事件。"},
                    }
                ],
            },
            title="难忘的一天作文",
        )

        resolution = service.resolve(
            profile,
            user_id="archive_user",
            conversation_id="conv_new",
            subject="语文",
            stage="小学",
            message="这个作文的开头怎么写？",
            problem_context="",
            is_new_problem=True,
        )

        assert resolution.status is MemoryResolutionStatus.UNIQUE
        assert resolution.selected is not None
        assert resolution.selected.memory_id == f"archive_{old.conversation_id}"
        assert resolution.selected.problem_context == "怎么写好《难忘的一天》这篇作文？"
    finally:
        service.shutdown()


class _FixedClassifier:
    def classify(self, *args, **kwargs) -> RequestClassification:
        return RequestClassification(
            scope=RequestScope.K9,
            subject=Subject.CHINESE,
            stage=SchoolStage.PRIMARY,
            is_new_problem=True,
            answer_mode=AnswerMode.GUIDED,
            reason="test",
        )


class _ForbiddenActionProvider:
    def decide(self, payload):
        raise AssertionError("ambiguous memory must stop before action planning")


class _NoModelClient:
    usage_history: list[dict] = []
    last_chat_metadata: dict = {}

    def chat(self, *args, **kwargs):
        raise AssertionError("memory clarification must not call teacher model")

    def chat_json(self, *args, **kwargs):
        raise AssertionError("memory clarification must not call audit model")


def test_ambiguous_memory_stops_agent_before_planning_or_model_calls(
    memory_service: StudentMemoryService,
) -> None:
    profile = memory_service.load_profile("user_loop")
    _seed_composition(
        memory_service,
        profile.student_profile_id,
        "我的植物朋友",
        "conv_plant",
    )
    _seed_composition(
        memory_service,
        profile.student_profile_id,
        "难忘的一天",
        "conv_day",
    )

    def resolve_memory(classification: RequestClassification):
        assert classification.subject is Subject.CHINESE
        return memory_service.resolve(
            profile,
            user_id="user_loop",
            conversation_id="conv_new",
            subject=classification.subject.value,
            stage=classification.stage.value,
            message="继续上次的作文",
            problem_context="",
            is_new_problem=classification.is_new_problem,
        )

    orchestrator = MultisubjectAgentLoop(
        classifier=_FixedClassifier(),
        action_provider=_ForbiddenActionProvider(),
        memory_resolver=resolve_memory,
        enable_fast_routing=False,
    )
    turn = run_turn(
        PrototypeEpisode(),
        "继续上次的作文",
        client=_NoModelClient(),
        orchestrator=orchestrator,
    )

    assert turn.agent_termination_reason == "memory_clarification"
    assert turn.memory_resolution is not None
    assert turn.memory_resolution.status is MemoryResolutionStatus.AMBIGUOUS
    assert "我的植物朋友" in turn.teacher_output.response
    assert "难忘的一天" in turn.teacher_output.response
    assert turn.student_verification.verifier_name == "memory_resolver"
    assert turn.model_generation_metadata.get("llm_call_count", 0) == 0


class _TeachingOnlyClient:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.json_schemas: list[type] = []
        self.user_payloads: list[dict] = []
        self.usage_history: list[dict] = []
        self.last_chat_metadata: dict = {}

    def chat(self, system, user, *args, **kwargs):
        self.chat_calls += 1
        self.user_payloads.append(json.loads(user))
        return "我们继续《难忘的一天》，先确定开头要交代的时间和地点。"

    def chat_json(self, *args, schema, **kwargs):
        self.json_schemas.append(schema)
        from prototypes.llm_led_teacher.orchestration_models import BoundaryAudit

        assert schema is BoundaryAudit
        return BoundaryAudit(passed=True)


def test_student_choice_after_ambiguity_restores_topic_without_reclassification(
    memory_service: StudentMemoryService,
) -> None:
    profile = memory_service.load_profile("user_choice")
    _seed_composition(
        memory_service,
        profile.student_profile_id,
        "我的植物朋友",
        "conv_plant",
    )
    _seed_composition(
        memory_service,
        profile.student_profile_id,
        "难忘的一天",
        "conv_day",
    )

    def resolve_memory(classification: RequestClassification):
        return memory_service.resolve(
            profile,
            user_id="user_choice",
            conversation_id="conv_new",
            subject=classification.subject.value,
            stage=classification.stage.value,
            message="继续上次的作文",
            problem_context="",
            is_new_problem=classification.is_new_problem,
        )

    episode = PrototypeEpisode()
    first_orchestrator = MultisubjectAgentLoop(
        classifier=_FixedClassifier(),
        action_provider=_ForbiddenActionProvider(),
        memory_resolver=resolve_memory,
        enable_fast_routing=False,
    )
    first = run_turn(
        episode,
        "继续上次的作文",
        client=_NoModelClient(),
        orchestrator=first_orchestrator,
    )
    assert first.agent_termination_reason == "memory_clarification"
    assert episode.pending_memory_clarification is not None

    client = _TeachingOnlyClient()
    episode.cross_session_memory = CrossSessionMemory(
        source_conversation_id="legacy_conversation",
        problem_context="旧机制不应注入的主题",
        active_subject="语文",
        active_stage="小学",
        recent_student_messages=["继续旧主题"],
        recent_teacher_response="旧回复",
    )
    second = run_turn(
        episode,
        "难忘的一天",
        client=client,
        enable_agent_loop=True,
        memory_identity=StudentMemoryIdentity("user_choice", "conv_new"),
        memory_service=memory_service,
    )
    memory_service.wait_for_idle()

    assert second.classification is not None
    assert second.classification.subject is Subject.CHINESE
    assert second.classification.is_new_problem is False
    assert second.memory_resolution is not None
    assert second.memory_resolution.status is MemoryResolutionStatus.UNIQUE
    assert second.memory_resolution.selected is not None
    assert second.memory_resolution.selected.topic == "难忘的一天"
    assert episode.problem_context == "怎么写好《难忘的一天》这篇作文？"
    assert episode.pending_memory_clarification is None
    assert any(
        event.details.get("router") == "memory_clarification_reply"
        for event in second.agent_trace
        if event.event_type == "classification"
    )
    assert client.chat_calls == 1
    teacher_payload = client.user_payloads[-1]
    assert "cross_session_memory" not in teacher_payload
    assert teacher_payload["student_memory_context"]["topic_memory"]["topic"] == (
        "难忘的一天"
    )
