import json

from prototypes.llm_led_teacher.logic import (
    PacingMode,
    PrototypeEpisode,
    TurnGuidance,
    run_turn,
)
from prototypes.llm_led_teacher.rag_hook import RagContext


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
        "retrieving": 2.0,
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
