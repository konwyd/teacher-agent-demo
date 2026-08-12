import json

import pytest

from prototypes.llm_led_teacher.logic import (
    PacingMode,
    PrototypeEpisode,
    PrototypeTurnRecord,
    TurnGuidance,
    build_orchestration_timings,
    run_turn,
)
from prototypes.llm_led_teacher.rag_hook import RagContext
from prototypes.llm_led_teacher.orchestration_models import (
    AgentOrchestrationResult,
    AgentTraceEvent,
    RequestClassification,
    ToolResult,
)
from prototypes.llm_led_teacher.retrieval_policy import RetrievalDecision


class _ObservedClient:
    def __init__(self) -> None:
        self.teacher_payloads: list[dict] = []
        self.observer_payloads: list[dict] = []
        self.chat_calls = 0

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.observer_payloads.append(json.loads(user))
        return TurnGuidance(
            starts_new_problem=True,
            pacing=PacingMode.COMPLETE_RESPONSE,
            reason="这是多步骤方程",
        )

    def chat(self, system, user, temperature=0.0, max_tokens=None):
        self.chat_calls += 1
        payload = json.loads(user)
        if "turn_context" not in payload:
            self.teacher_payloads.append(payload)
            return "先把等式两边同时减去9，得到2x=5，再除以2，得到x=5/2。你能验算一下吗？"
        return "先把等式两边同时减去9，得到2x=5。你能说说这一步为什么两边都要减9吗？"


def test_wrong_observer_complete_response_is_normalized_and_final_answer_repaired(
    monkeypatch,
):
    monkeypatch.setattr(
        "prototypes.llm_led_teacher.logic.retrieve_context",
        lambda problem, online_mode: RagContext(
            source="local",
            text="无关的小学加法材料",
            refs=["irrelevant"],
        ),
    )
    client = _ObservedClient()

    turn = run_turn(PrototypeEpisode(), "2x+9=14，这道题怎么解", client=client)

    assert turn.turn_guidance.pacing is PacingMode.SINGLE_MOVE
    assert client.teacher_payloads[0]["turn_guidance"]["pacing"] == "single_move"
    assert client.chat_calls == 2
    assert turn.repair_reasons
    assert "2x=5" in turn.teacher_output.response
    assert "x=5/2" not in turn.teacher_output.response
    assert turn.teacher_output.response.count("？") + turn.teacher_output.response.count("?") <= 1


class _ScriptedClient(_ObservedClient):
    def __init__(self, guidance, drafts):
        super().__init__()
        self.guidance = guidance
        self.drafts = iter(drafts)

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.observer_payloads.append(json.loads(user))
        if isinstance(self.guidance, Exception):
            raise self.guidance
        return self.guidance

    def chat(self, system, user, temperature=0.0, max_tokens=None):
        self.chat_calls += 1
        payload = json.loads(user)
        if "turn_context" not in payload:
            self.teacher_payloads.append(payload)
        return next(self.drafts)


def _no_rag(monkeypatch):
    monkeypatch.setattr(
        "prototypes.llm_led_teacher.logic.retrieve_context",
        lambda problem, online_mode: RagContext(source="none"),
    )


def test_kv_persistence_restores_after_observer_and_saves_final_teacher_state(
    monkeypatch,
):
    from app.llm.kv_cache import KvCacheIdentity

    _no_rag(monkeypatch)
    events = []

    class EventClient(_ScriptedClient):
        def chat_json(self, *args, **kwargs):
            events.append("observer")
            return super().chat_json(*args, **kwargs)

        def chat(self, *args, **kwargs):
            events.append("teacher")
            return super().chat(*args, **kwargs)

    class Status:
        def to_metadata(self):
            return {
                "kv_persistence_enabled": True,
                "kv_persistence_slot": 2,
                "kv_persistence_restore_status": "restored",
                "kv_persistence_save_status": "saved",
            }

    class Lease:
        slot_id = 2
        status = Status()

        def restore(self):
            events.append("restore")
            return True

        def save(self):
            events.append("save")
            return True

    class Manager:
        class Context:
            def __enter__(self):
                events.append("lease-enter")
                return Lease()

            def __exit__(self, *_args):
                events.append("lease-exit")

        def lease(self, identity):
            assert identity == KvCacheIdentity("usr_1", "conv_1")
            return self.Context()

    client = EventClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
    )
    turn = run_turn(
        PrototypeEpisode(),
        "2x+9=14，这道题怎么解",
        client=client,
        kv_identity=KvCacheIdentity("usr_1", "conv_1"),
        kv_manager=Manager(),
    )

    assert events == [
        "lease-enter",
        "observer",
        "restore",
        "teacher",
        "save",
        "lease-exit",
    ]
    assert turn.model_generation_metadata["kv_persistence_slot"] == 2
    assert turn.model_generation_metadata["kv_persistence_save_status"] == "saved"


def test_kv_persistence_does_not_save_when_teacher_generation_fails(monkeypatch):
    from app.llm.kv_cache import KvCacheIdentity

    _no_rag(monkeypatch)
    events = []

    class Lease:
        slot_id = 0
        status = type("Status", (), {"to_metadata": lambda self: {}})()

        def restore(self):
            events.append("restore")

        def save(self):
            events.append("save")

    class Manager:
        @staticmethod
        def lease(_identity):
            class Context:
                def __enter__(self):
                    return Lease()

                def __exit__(self, *_args):
                    events.append("lease-exit")
            return Context()

    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        [],
    )
    with pytest.raises(StopIteration):
        run_turn(
            PrototypeEpisode(),
            "2x+9=14，这道题怎么解",
            client=client,
            kv_identity=KvCacheIdentity("usr_1", "conv_1"),
            kv_manager=Manager(),
        )

    assert events == ["restore", "lease-exit"]


def test_correct_single_move_observer_keeps_single_move(monkeypatch):
    _no_rag(monkeypatch)
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
    )

    turn = run_turn(PrototypeEpisode(), "2x+9=14，这道题怎么解", client=client)

    assert turn.observer_raw_guidance.pacing is PacingMode.SINGLE_MOVE
    assert turn.turn_guidance.pacing is PacingMode.SINGLE_MOVE
    assert client.chat_calls == 1
    assert "x=5/2" not in turn.teacher_output.response


def test_run_turn_reports_safe_progress_stages(monkeypatch):
    _no_rag(monkeypatch)
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
    )
    stages = []

    turn = run_turn(
        PrototypeEpisode(),
        "2x+9=14，这道题怎么解",
        client=client,
        progress_callback=stages.append,
    )

    assert stages == [
        "observing",
        "retrieving",
        "generating",
        "auditing",
        "complete",
    ]
    assert "2x=5" in turn.teacher_output.response


def test_run_turn_reports_repair_before_complete(monkeypatch):
    _no_rag(monkeypatch)
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        [
            "两边减9得到2x=5，再除以2得到x=5/2。你会验算吗？",
            "先在等式两边同时减去9，得到2x=5。下一步该怎么做？",
        ],
    )
    stages = []

    turn = run_turn(
        PrototypeEpisode(),
        "2x+9=14，这道题怎么解",
        client=client,
        progress_callback=stages.append,
    )

    assert stages == [
        "observing",
        "retrieving",
        "generating",
        "auditing",
        "repairing",
        "complete",
    ]
    assert "x=5/2" not in turn.teacher_output.response
    assert any(
        event.kind.value == "repair" for event in turn.public_thinking_events
    )


def test_turn_persists_public_thinking_and_emits_the_same_events(monkeypatch):
    _no_rag(monkeypatch)
    emitted = []
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
    )

    turn = run_turn(
        PrototypeEpisode(),
        "2x+9=14，这道题怎么解",
        client=client,
        thinking_callback=emitted.append,
    )

    assert emitted == turn.public_thinking_events
    assert [event.sequence for event in emitted] == list(range(1, len(emitted) + 1))
    assert any(
        event.summary == "最终回答已经通过教学内容与节奏审计。"
        for event in emitted
    )
    assert emitted[-1].kind.value == "complete"
    assert "reasoning_content" not in turn.model_dump_json()


def test_old_turn_without_public_thinking_events_still_loads() -> None:
    payload = {
        "turn_id": 1,
        "student_message": "旧问题",
        "teacher_output": {"response": "旧回答"},
        "student_verification": {
            "status": "unable_to_verify",
            "summary": "旧记录没有公开思考字段。",
            "verifier_name": "none",
        },
        "evidence_level": "unverified",
    }

    restored = PrototypeTurnRecord.model_validate(payload)

    assert restored.public_thinking_events == []
    assert restored.retrieval_decision.mode.value == "none"


def test_run_turn_records_stage_timings_and_safe_model_summary(monkeypatch):
    _no_rag(monkeypatch)
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        [
            "两边减9得到2x=5，再除以2得到x=5/2。你会验算吗？",
            "先在等式两边同时减去9，得到2x=5。下一步该怎么做？",
        ],
    )
    clock = iter([0.0, 1.0, 3.0, 7.0, 8.0, 13.0]).__next__

    turn = run_turn(
        PrototypeEpisode(),
        "2x+9=14，这道题怎么解",
        client=client,
        clock=clock,
    )

    assert turn.stage_timings_seconds == {
        "observing": 1.0,
        "local_retrieval": 2.0,
        "generating": 4.0,
        "auditing": 1.0,
        "repairing": 5.0,
    }
    assert turn.backend_total_seconds == 13.0
    assert turn.repair_performed is True
    assert turn.draft_character_count == len(
        "两边减9得到2x=5，再除以2得到x=5/2。你会验算吗？"
    )
    assert turn.final_character_count == len(turn.teacher_output.response)
    assert turn.display_seconds == 0.0
    assert not hasattr(turn, "chain_of_thought")


def test_teacher_generation_and_repair_have_no_hard_token_cap(monkeypatch):
    _no_rag(monkeypatch)

    class RecordingClient(_ScriptedClient):
        def __init__(self):
            super().__init__(
                TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
                [
                    "两边减9得到2x=5，再除以2得到x=5/2。你会验算吗？",
                    "先在等式两边同时减去9，得到2x=5。下一步该怎么做？",
                ],
            )
            self.teacher_token_limits = []

        def chat(self, system, user, temperature=0.0, max_tokens=None):
            self.teacher_token_limits.append(max_tokens)
            return super().chat(system, user, temperature, max_tokens)

    client = RecordingClient()

    turn = run_turn(
        PrototypeEpisode(),
        "2x+9=14，这道题怎么解",
        client=client,
    )

    assert turn.repair_performed is True
    assert client.teacher_token_limits == [None, None]


def test_turn_records_safe_generation_metadata_without_raw_reasoning(monkeypatch):
    _no_rag(monkeypatch)

    class MetadataClient(_ScriptedClient):
        last_chat_metadata = {
            "finish_reason": "stop",
            "completion_tokens": 321,
            "reasoning_tokens": 180,
            "reasoning_characters": 760,
            "reasoning_available": True,
        }

    client = MetadataClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
    )

    turn = run_turn(PrototypeEpisode(), "2x+9=14，这道题怎么解", client=client)

    assert turn.model_generation_metadata == MetadataClient.last_chat_metadata
    assert "reasoning_content" not in turn.model_generation_metadata
    assert "chain_of_thought" not in turn.model_dump()


def test_turn_aggregates_observer_and_teacher_token_usage(monkeypatch):
    _no_rag(monkeypatch)

    class UsageClient(_ScriptedClient):
        def __init__(self):
            super().__init__(
                TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
                ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
            )
            self.usage_history = []

        def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
            result = super().chat_json(system, user, schema, temperature, max_tokens)
            self.usage_history.append(
                {
                    "prompt_tokens": 100,
                    "prompt_cache_hit_tokens": 20,
                    "prompt_cache_miss_tokens": 80,
                    "completion_tokens": 10,
                    "response_tokens": 10,
                    "reasoning_tokens": 0,
                    "total_tokens": 110,
                    "context_used_tokens": 110,
                    "context_window_tokens": 524288,
                    "cache_usage_available": True,
                    "finish_reason": "stop",
                }
            )
            return result

        def chat(self, system, user, temperature=0.0, max_tokens=None):
            result = super().chat(system, user, temperature, max_tokens)
            self.usage_history.append(
                {
                    "prompt_tokens": 900,
                    "prompt_cache_hit_tokens": 580,
                    "prompt_cache_miss_tokens": 320,
                    "completion_tokens": 190,
                    "response_tokens": 140,
                    "reasoning_tokens": 50,
                    "total_tokens": 1090,
                    "context_used_tokens": 1090,
                    "context_window_tokens": 524288,
                    "cache_usage_available": True,
                    "finish_reason": "stop",
                }
            )
            return result

    turn = run_turn(PrototypeEpisode(), "2x+9=14，这道题怎么解", client=UsageClient())
    usage = turn.model_generation_metadata

    assert usage["llm_call_count"] == 2
    assert usage["prompt_tokens"] == 1000
    assert usage["prompt_cache_hit_tokens"] == 600
    assert usage["prompt_cache_miss_tokens"] == 400
    assert usage["cache_hit_rate"] == 0.6
    assert usage["completion_tokens"] == 200
    assert usage["reasoning_tokens"] == 50
    assert usage["response_tokens"] == 150
    assert usage["total_tokens"] == 1200
    assert usage["context_used_tokens"] == 1090
    assert usage["context_window_tokens"] == 524288


def test_episode_preserves_pending_student_message_across_serialization():
    episode = PrototypeEpisode(pending_student_message="上一条还没有生成完")

    restored = PrototypeEpisode.model_validate(episode.model_dump(mode="json"))

    assert restored.pending_student_message == "上一条还没有生成完"


def test_progress_callback_failure_does_not_block_teacher_response(monkeypatch):
    _no_rag(monkeypatch)
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
    )

    def broken_callback(stage):
        raise RuntimeError(f"UI unavailable at {stage}")

    turn = run_turn(
        PrototypeEpisode(),
        "2x+9=14，这道题怎么解",
        client=client,
        progress_callback=broken_callback,
    )

    assert "2x=5" in turn.teacher_output.response


def test_thinking_callback_failure_does_not_block_or_truncate_record(monkeypatch):
    _no_rag(monkeypatch)
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
    )

    def broken_callback(event):
        raise RuntimeError(f"UI unavailable at {event.sequence}")

    turn = run_turn(
        PrototypeEpisode(),
        "2x+9=14，这道题怎么解",
        client=client,
        thinking_callback=broken_callback,
    )

    assert "2x=5" in turn.teacher_output.response
    assert turn.public_thinking_events[-1].kind.value == "complete"
    assert any(
        event.kind.value == "verification"
        and event.status.value == "warning"
        for event in turn.public_thinking_events
    )


def test_explicit_full_solution_allows_complete_response_and_final_answer(monkeypatch):
    _no_rag(monkeypatch)
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.COMPLETE_RESPONSE),
        ["两边减9得到2x=5，再同时除以2得到x=5/2。"],
    )

    turn = run_turn(
        PrototypeEpisode(), "请给出完整步骤和答案：2x+9=14", client=client
    )

    assert turn.turn_guidance.pacing is PacingMode.COMPLETE_RESPONSE
    assert "x=5/2" in turn.teacher_output.response
    assert turn.repair_reasons == []
    assert client.chat_calls == 1


def test_direct_fact_allows_complete_response(monkeypatch):
    _no_rag(monkeypatch)
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.COMPLETE_RESPONSE),
        ["三角形的内角和是180度。"],
    )

    turn = run_turn(PrototypeEpisode(), "三角形内角和是多少", client=client)

    assert turn.turn_guidance.pacing is PacingMode.COMPLETE_RESPONSE
    assert turn.teacher_output.response == "三角形的内角和是180度。"
    assert client.chat_calls == 1


def test_observer_failure_uses_fallback_and_teacher_response_is_shown(monkeypatch):
    _no_rag(monkeypatch)
    client = _ScriptedClient(
        RuntimeError("network down"),
        ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
    )

    turn = run_turn(PrototypeEpisode(), "2x+9=14，这道题怎么解", client=client)

    assert turn.turn_guidance.pacing is PacingMode.SINGLE_MOVE
    assert "RuntimeError" in turn.observer_error
    assert "2x=5" in turn.teacher_output.response


def test_unsupported_problem_does_not_crash_or_block_response(monkeypatch):
    _no_rag(monkeypatch)
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.COMPLETE_RESPONSE),
        ["因为平行线的同旁内角互补，所以三角形三个内角之和是180度。"],
    )

    turn = run_turn(PrototypeEpisode(), "为什么三角形内角和是180度", client=client)

    assert turn.student_verification.verifier_name == "none"
    assert turn.teacher_output.response
    assert turn.repair_reasons == []


def test_rag_is_absent_from_observer_and_cannot_replace_active_problem(monkeypatch):
    irrelevant = "一年级材料：2+3=5"
    monkeypatch.setattr(
        "prototypes.llm_led_teacher.logic.retrieve_context",
        lambda problem, online_mode: RagContext(
            source="local", text=irrelevant, refs=["primary-arithmetic"]
        ),
    )
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.SINGLE_MOVE),
        ["先在等式两边同时减去9，得到2x=5。下一步该怎么做？"],
    )
    episode = PrototypeEpisode()

    run_turn(episode, "2x+9=14，这道题怎么解", client=client)

    observer_payload = client.observer_payloads[0]
    teacher_payload = client.teacher_payloads[0]
    assert irrelevant not in json.dumps(observer_payload, ensure_ascii=False)
    assert "retrieval_reference" not in observer_payload
    assert teacher_payload["problem_context"] == "2x+9=14，这道题怎么解"
    assert teacher_payload["turn_guidance"]["pacing"] == "single_move"
    assert teacher_payload["retrieval_reference"]["content"] == irrelevant
    assert episode.problem_context == "2x+9=14，这道题怎么解"


def test_post_audit_blocks_final_answer_even_if_effective_guidance_is_wrong(
    monkeypatch,
):
    _no_rag(monkeypatch)
    monkeypatch.setattr(
        "prototypes.llm_led_teacher.logic._normalize_guidance",
        lambda episode, message, guidance: guidance,
    )
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.COMPLETE_RESPONSE),
        [
            "两边减9得到2x=5，再除以2得到x=5/2。你会验算吗？",
            "先在等式两边同时减去9，得到2x=5。下一步该怎么做？",
        ],
    )

    turn = run_turn(PrototypeEpisode(), "2x+9=14，这道题怎么解", client=client)

    assert turn.turn_guidance.pacing is PacingMode.COMPLETE_RESPONSE
    assert turn.repair_reasons
    assert client.chat_calls == 2
    assert "x=5/2" not in turn.teacher_output.response


class _FakeEmbedding:
    def tolist(self):
        return [0.1, 0.2]


class _FakeEmbedder:
    def encode(self, query, normalize_embeddings=True):
        return _FakeEmbedding()


def test_rag_auto_uses_original_retriever_auto_mode(monkeypatch):
    from prototypes.llm_led_teacher import rag_hook

    calls = []

    def fake_original(problem, subject, topk, online_mode, verbose, diagnostics):
        calls.append((problem, subject, topk, online_mode, verbose))
        diagnostics.update(
            {
                "online_triggered": True,
                "online_status": "used",
                "online_reason": "本地最高相似度 0.56 低于阈值 0.65",
                "local_threshold": 0.65,
                "local_top_score": 0.56,
            }
        )
        return (
            "[🌐 online] 欧拉-麦克劳林公式联网材料",
            ["🌐 数学参考 (相似度 0.82)"],
            False,
        )

    monkeypatch.setattr(rag_hook, "_original_retrieve_for_exam", fake_original)
    result = rag_hook.retrieve_context("欧拉-麦克劳林公式是什么", online_mode="auto")

    assert calls == [("欧拉-麦克劳林公式是什么", "初中数学", 5, "auto", False)]
    assert result.source == "hybrid"
    assert result.hits[0]["similarity"] == 0.82

    assert result.online_triggered is True
    assert result.online_status == "used"
    assert result.local_top_score == 0.56


class _FakeOrchestrator:
    def __init__(self, result: AgentOrchestrationResult) -> None:
        self.result = result
        self.calls = []

    def run(self, problem_context, student_message, active_subject, active_stage, progress_callback=None):
        self.calls.append((problem_context, student_message, active_subject, active_stage))
        return self.result


class _FixedOrchestrator:
    def __init__(self, result: AgentOrchestrationResult) -> None:
        self.result = result

    def run(self, *args, **kwargs) -> AgentOrchestrationResult:
        return self.result


def test_creative_turn_records_no_rag_and_no_repair_call(monkeypatch) -> None:
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.COMPLETE_RESPONSE),
        ["可以从画面、情绪和悬念三个方向来写。你更喜欢哪一种？"],
    )
    orchestration = AgentOrchestrationResult(
        classification=RequestClassification(
            scope="k9",
            subject="语文",
            stage="初中",
            is_new_problem=True,
            answer_mode="guided",
            reason="作文创作指导",
        ),
        rag_context=RagContext(source="none"),
        retrieval_decision=RetrievalDecision(
            mode="none",
            reason="创作指导无需外部资料",
            confidence=1.0,
        ),
    )

    turn = run_turn(
        PrototypeEpisode(),
        "把这句话润色成作文开头",
        client=client,
        orchestrator=_FixedOrchestrator(orchestration),
    )

    assert turn.retrieval_decision.mode.value == "none"
    assert turn.rag_hits == []
    assert "local_retrieval" not in turn.stage_timings_seconds
    assert "web_retrieval" not in turn.stage_timings_seconds
    assert turn.repair_performed is False


def test_creative_advisory_alone_does_not_trigger_repair() -> None:
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.COMPLETE_RESPONSE),
        ["可以从画面、情绪和悬念三个方向来写。你喜欢哪一种？要试试悬念吗？"],
    )
    orchestration = AgentOrchestrationResult(
        classification=RequestClassification(
            scope="k9",
            subject="语文",
            stage="初中",
            is_new_problem=True,
            answer_mode="guided",
            reason="作文创作指导",
        ),
        rag_context=RagContext(source="none"),
        retrieval_decision=RetrievalDecision(
            mode="none",
            reason="创作指导无需外部资料",
            confidence=1.0,
        ),
    )

    turn = run_turn(
        PrototypeEpisode(),
        "把这句话润色成作文开头",
        client=client,
        orchestrator=_FixedOrchestrator(orchestration),
    )

    assert turn.repair_performed is False
    assert turn.repair_reasons == []
    assert client.chat_calls == 1


def test_tool_timings_are_not_hidden_inside_routing() -> None:
    observations = [
        ToolResult(name="search_local_knowledge", ok=True, elapsed_seconds=1.25),
        ToolResult(name="search_web", ok=True, elapsed_seconds=2.5),
    ]

    timings = build_orchestration_timings(
        total_seconds=4.5,
        observations=observations,
    )

    assert timings["routing"] == 0.75
    assert timings["local_retrieval"] == 1.25
    assert timings["web_retrieval"] == 2.5


def test_run_turn_persists_multisubject_classification_and_trace(monkeypatch):
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.COMPLETE_RESPONSE),
        ["力可以改变物体速度的大小或方向。你想看一个生活中的例子吗？"],
    )
    classification = RequestClassification(
        scope="k9",
        subject="物理",
        stage="初中",
        is_new_problem=True,
        answer_mode="complete",
        reason="new physics question",
    )
    trace = [
        AgentTraceEvent(
            step=0,
            event_type="classification",
            summary="识别为初中物理新题",
        )
    ]
    orchestrator = _FakeOrchestrator(
        AgentOrchestrationResult(
            classification=classification,
            rag_context=RagContext(
                source="local",
                text="力可以改变运动状态",
                refs=["物理教材"],
            ),
            trace=trace,
            loaded_skills=["concept_explanation"],
            termination_reason="model_finished",
            routing_guidance={"answer_mode": "complete"},
        )
    )
    episode = PrototypeEpisode(active_subject="数学", active_stage="初中")

    turn = run_turn(
        episode,
        "物体受力后运动状态为什么会改变？",
        client=client,
        orchestrator=orchestrator,
    )

    assert episode.active_subject == "物理"
    assert episode.active_stage == "初中"
    assert episode.problem_context == "物体受力后运动状态为什么会改变？"
    assert any(
        event.summary == "已确定为初中物理新题。"
        for event in turn.public_thinking_events
    )
    assert turn.classification == classification
    assert turn.agent_trace == trace
    assert turn.loaded_skills == ["concept_explanation"]
    assert turn.agent_termination_reason == "model_finished"
    assert turn.rag_source == "local"
    assert client.teacher_payloads[0]["request_classification"]["subject"] == "物理"
    assert client.teacher_payloads[0]["loaded_skills"] == ["concept_explanation"]


def test_agent_loop_keeps_existing_math_verifier_and_audit(monkeypatch):
    classification = RequestClassification(
        scope="k9",
        subject="数学",
        stage="初中",
        is_new_problem=False,
        answer_mode="guided",
        reason="math follow-up",
    )
    orchestrator = _FakeOrchestrator(
        AgentOrchestrationResult(
            classification=classification,
            rag_context=RagContext(source="local_empty"),
            termination_reason="model_finished",
            routing_guidance={"answer_mode": "guided"},
        )
    )
    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=False, pacing=PacingMode.SINGLE_MOVE),
        ["两边减3以后得到2x=4。下一步你会怎么做？"],
    )
    episode = PrototypeEpisode(
        problem_context="解方程 2x+3=7",
        active_subject="数学",
        active_stage="初中",
    )

    turn = run_turn(episode, "2x=4", client=client, orchestrator=orchestrator)

    assert turn.student_verification.status.value == "verified_correct"
    assert episode.problem_context == "解方程 2x+3=7"
    assert episode.active_subject == "数学"
    assert turn.claim_audits
