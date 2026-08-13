import json

from prototypes.llm_led_teacher.logic import (
    PacingMode,
    PrototypeEpisode,
    TurnGuidance,
    run_turn,
)
from prototypes.llm_led_teacher.rag_hook import RagContext
from prototypes.llm_led_teacher.orchestration_models import (
    AgentOrchestrationResult,
    AgentTraceEvent,
    BoundaryAudit,
    RequestClassification,
    TeachingContract,
)


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
        if schema is BoundaryAudit:
            return BoundaryAudit(passed=True)
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


class _ContractTurnClient:
    def __init__(self, drafts, audits, *, record_usage=False):
        self.drafts = iter(drafts)
        self.audits = iter(audits)
        self.record_usage = record_usage
        self.teacher_payloads = []
        self.repair_payloads = []
        self.requested_schemas = []
        self.chat_calls = 0
        self.usage_history = []

    def _record_usage(self, kind):
        if not self.record_usage:
            return
        self.usage_history.append(
            {
                "prompt_tokens": 10,
                "prompt_cache_hit_tokens": 2,
                "prompt_cache_miss_tokens": 8,
                "completion_tokens": 5,
                "response_tokens": 4,
                "reasoning_tokens": 1,
                "total_tokens": 15,
                "context_used_tokens": 15,
                "context_window_tokens": 524288,
                "cache_usage_available": True,
                "finish_reason": "stop",
                "kind": kind,
            }
        )

    def chat_json(self, system, user, schema, temperature=0.0, max_tokens=None):
        self.requested_schemas.append(schema)
        result = next(self.audits)
        self._record_usage("audit")
        if isinstance(result, Exception):
            raise result
        return result

    def chat(self, system, user, temperature=0.0, max_tokens=None):
        self.chat_calls += 1
        payload = json.loads(user)
        if "turn_context" in payload:
            self.repair_payloads.append(payload)
        else:
            self.teacher_payloads.append(payload)
        result = next(self.drafts)
        self._record_usage("generation")
        if isinstance(result, Exception):
            raise result
        return result


class _UsageOrchestrator(_FakeOrchestrator):
    def __init__(self, result, client):
        super().__init__(result)
        self.client = client

    def run(self, *args, **kwargs):
        self.client._record_usage("planning")
        return super().run(*args, **kwargs)


def _concept_formula_result():
    return AgentOrchestrationResult(
        classification=RequestClassification(
            scope="k9",
            subject="数学",
            stage="初中",
            is_new_problem=False,
            answer_mode="complete",
            reason="student asks for a formula concept",
        ),
        teaching_contract=TeachingContract(
            response_mode="concept_explanation",
            requested_target="解释求根公式本身",
            allowed_content=["通用公式", "符号含义"],
            forbidden_content=["代入本题", "最终答案"],
            may_apply_to_active_problem=False,
            may_reveal_final_answer=False,
            reason="student requested a concept explanation only",
        ),
        rag_context=RagContext(source="none"),
        termination_reason="model_finished",
    )


def test_agent_contract_audits_repairs_and_normalizes_formula():
    audit = BoundaryAudit(
        passed=False,
        violations=["applied_without_permission", "final_answer_leak"],
        evidence=["草稿把公式代入了当前题"],
        repair_instructions=["删除本题代入和答案"],
    )
    client = _ContractTurnClient(
        drafts=[
            "通用公式是：\n\n[ x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a} ]\n\n代入本题得到 x=...",
            "通用公式是：\n\n[ x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a} ]\n\n其中 a、b、c 是系数。",
        ],
        audits=[audit, BoundaryAudit(passed=True)],
    )
    episode = PrototypeEpisode(
        problem_context="3x^2+2x-11=0",
        active_subject="数学",
        active_stage="初中",
    )

    turn = run_turn(
        episode,
        "求根公式是什么？",
        client=client,
        orchestrator=_FakeOrchestrator(_concept_formula_result()),
    )

    assert "代入本题" not in turn.teacher_output.response
    assert "$$x=\\frac" in turn.teacher_output.response
    assert turn.repair_performed is True
    assert turn.teaching_contract == _concept_formula_result().teaching_contract
    assert turn.boundary_audit == audit.model_copy(
        update={"evidence": [], "repair_instructions": []}
    )
    assert turn.post_repair_boundary_audit == BoundaryAudit(passed=True)
    assert turn.boundary_fallback_reason == ""
    assert turn.formula_normalized is True
    assert client.teacher_payloads[0]["teaching_contract"]["response_mode"] == "concept_explanation"
    assert client.repair_payloads[0]["turn_context"]["teaching_contract"] == client.teacher_payloads[0]["teaching_contract"]
    assert client.repair_payloads[0]["must_fix"] == ["删除本题代入和答案"]


def test_agent_turn_never_requests_observer_and_aggregates_all_llm_usage():
    result = AgentOrchestrationResult(
        classification=RequestClassification(
            scope="k9",
            subject="数学",
            stage="初中",
            is_new_problem=False,
            answer_mode="guided",
            reason="continue active problem",
        ),
        teaching_contract=TeachingContract(
            response_mode="one_step",
            requested_target="只推进下一步",
            allowed_content=["一个步骤"],
            forbidden_content=["最终答案"],
            may_apply_to_active_problem=True,
            may_reveal_final_answer=False,
            reason="student did not request the full solution",
        ),
        rag_context=RagContext(source="none"),
        termination_reason="model_finished",
    )
    audit = BoundaryAudit(
        passed=False,
        violations=["final_answer_leak"],
        repair_instructions=["删除最终答案"],
    )
    client = _ContractTurnClient(
        drafts=["先移项，最终答案是 x=2。", "先移项得到2x=4。"],
        audits=[audit, BoundaryAudit(passed=True)],
        record_usage=True,
    )

    turn = run_turn(
        PrototypeEpisode(problem_context="2x+3=7"),
        "下一步呢",
        client=client,
        orchestrator=_UsageOrchestrator(result, client),
    )

    assert client.requested_schemas == [BoundaryAudit, BoundaryAudit]
    assert turn.observer_raw_guidance.pacing is PacingMode.SINGLE_MOVE
    assert turn.turn_guidance.pacing is PacingMode.SINGLE_MOVE
    assert turn.observer_error == "observer_replaced_by_teaching_contract"
    assert turn.model_generation_metadata["llm_call_count"] == 5
    assert turn.model_generation_metadata["prompt_tokens"] == 50
    assert turn.model_generation_metadata["completion_tokens"] == 25
    assert turn.model_generation_metadata["reasoning_tokens"] == 5
    assert turn.model_generation_metadata["response_tokens"] == 20


def test_agent_advisory_repair_always_runs_post_repair_audit_and_counts_usage():
    client = _ContractTurnClient(
        drafts=[
            "求根公式里的 a 是什么？b 又是什么？",
            "求根公式中的 a、b、c 表示方程的三个系数。",
        ],
        audits=[BoundaryAudit(passed=True), BoundaryAudit(passed=True)],
        record_usage=True,
    )

    turn = run_turn(
        PrototypeEpisode(problem_context="3x^2+2x-11=0"),
        "求根公式是什么？",
        client=client,
        orchestrator=_UsageOrchestrator(_concept_formula_result(), client),
    )

    assert turn.repair_performed is True
    assert client.requested_schemas == [BoundaryAudit, BoundaryAudit]
    assert turn.post_repair_boundary_audit == BoundaryAudit(passed=True)
    assert turn.model_generation_metadata["llm_call_count"] == 5


def test_advisory_repair_failed_second_audit_uses_redacted_fixed_fallback():
    initial_audit = BoundaryAudit(
        passed=True,
        evidence=["INITIAL_REJECTED_EVIDENCE"],
    )
    post_repair_audit = BoundaryAudit(
        passed=False,
        violations=["applied_without_permission"],
        evidence=["POST_REJECTED_EVIDENCE"],
        repair_instructions=["POST_REPAIR_RAW_TEXT"],
    )
    client = _ContractTurnClient(
        drafts=[
            "求根公式里的 a 是什么？b 又是什么？",
            "令三个系数分别取当前题中的数值即可得到结果。",
        ],
        audits=[initial_audit, post_repair_audit],
    )

    turn = run_turn(
        PrototypeEpisode(problem_context="3x^2+2x-11=0"),
        "求根公式是什么？",
        client=client,
        orchestrator=_FakeOrchestrator(_concept_formula_result()),
    )

    serialized = turn.model_dump_json()
    assert turn.boundary_fallback_reason == (
        "post_repair_audit_failed:applied_without_permission"
    )
    assert "三个系数分别取当前题" not in turn.teacher_output.response
    assert turn.boundary_audit == BoundaryAudit(passed=True)
    assert turn.post_repair_boundary_audit == BoundaryAudit(
        passed=False,
        violations=["applied_without_permission"],
    )
    assert "INITIAL_REJECTED_EVIDENCE" not in serialized
    assert "POST_REJECTED_EVIDENCE" not in serialized
    assert "POST_REPAIR_RAW_TEXT" not in serialized


def test_boundary_repair_raw_audit_text_is_never_persisted():
    client = _ContractTurnClient(
        drafts=["代入本题计算。", "仍把当前题的数值带入公式。"],
        audits=[
            BoundaryAudit(
                passed=False,
                violations=["applied_without_permission"],
                evidence=["INITIAL_REJECTED_EVIDENCE"],
                repair_instructions=["INITIAL_REPAIR_RAW_TEXT"],
            ),
            BoundaryAudit(
                passed=False,
                violations=["applied_without_permission"],
                evidence=["POST_REJECTED_EVIDENCE"],
                repair_instructions=["POST_REPAIR_RAW_TEXT"],
            ),
        ],
    )

    episode = PrototypeEpisode(problem_context="3x^2+2x-11=0")
    turn = run_turn(
        episode,
        "求根公式是什么？",
        client=client,
        orchestrator=_FakeOrchestrator(_concept_formula_result()),
    )

    serialized = episode.model_dump_json()
    assert turn.boundary_audit is not None
    assert turn.boundary_audit.violations == ["applied_without_permission"]
    assert turn.post_repair_boundary_audit is not None
    assert turn.post_repair_boundary_audit.violations == [
        "applied_without_permission"
    ]
    assert turn.boundary_fallback_reason == (
        "post_repair_audit_failed:applied_without_permission"
    )
    for sentinel in (
        "INITIAL_REJECTED_EVIDENCE",
        "INITIAL_REPAIR_RAW_TEXT",
        "POST_REJECTED_EVIDENCE",
        "POST_REPAIR_RAW_TEXT",
    ):
        assert sentinel not in serialized


def test_agent_hard_recheck_replaces_still_leaking_repair_with_safe_fallback():
    result = _concept_formula_result()
    contract = result.teaching_contract.model_copy(
        update={
            "forbidden_content": [
                "代入本题",
                "最终答案",
                "不展开当前题的计算",
            ]
        }
    )
    audit = BoundaryAudit(
        passed=False,
        violations=["applied_without_permission"],
        repair_instructions=["不要应用到当前题"],
    )
    client = _ContractTurnClient(
        drafts=["代入本题可得结果。", "我仍然代入本题计算。"],
        audits=[audit, BoundaryAudit(passed=True)],
    )

    turn = run_turn(
        PrototypeEpisode(problem_context="3x^2+2x-11=0"),
        "求根公式是什么？",
        client=client,
        orchestrator=_FakeOrchestrator(
            result.model_copy(update={"teaching_contract": contract})
        ),
    )

    assert client.chat_calls == 2
    assert "代入本题" not in turn.teacher_output.response
    assert "我仍然代入本题计算" not in turn.teacher_output.response
    assert "不展开当前题的计算" not in turn.teacher_output.response
    assert turn.repair_performed is True
    assert turn.post_repair_boundary_audit == BoundaryAudit(passed=True)
    assert turn.boundary_fallback_reason == (
        "hard_boundary_recheck:applied_without_permission"
    )


def test_agent_audit_transport_failure_repairs_once_and_never_shows_draft():
    client = _ContractTurnClient(
        drafts=["代入本题可得答案。", "这里只说明求根公式中各符号的含义。"],
        audits=[RuntimeError("audit unavailable"), BoundaryAudit(passed=True)],
    )

    turn = run_turn(
        PrototypeEpisode(problem_context="3x^2+2x-11=0"),
        "求根公式是什么？",
        client=client,
        orchestrator=_FakeOrchestrator(_concept_formula_result()),
    )

    assert turn.boundary_audit is not None
    assert turn.boundary_audit.audit_error == "RuntimeError"
    assert turn.post_repair_boundary_audit == BoundaryAudit(passed=True)
    assert turn.boundary_fallback_reason == ""
    assert turn.repair_performed is True
    assert client.chat_calls == 2
    assert turn.teacher_output.response == "这里只说明求根公式中各符号的含义。"


def test_failed_post_repair_audit_replaces_semantic_leak_with_safe_fallback():
    initial_audit = BoundaryAudit(
        passed=False,
        violations=["applied_without_permission"],
        repair_instructions=["不要代入当前题"],
    )
    post_repair_audit = BoundaryAudit(
        passed=False,
        violations=["applied_without_permission", "final_answer_leak"],
        evidence=["修正版仍写出了当前题系数和判别式"],
        repair_instructions=["改用不含当前题数据的安全回复"],
    )
    client = _ContractTurnClient(
        drafts=[
            "代入本题计算。",
            "令 a=3，b=2，c=-11，因此判别式等于136。",
        ],
        audits=[initial_audit, post_repair_audit],
    )

    turn = run_turn(
        PrototypeEpisode(problem_context="3x^2+2x-11=0"),
        "求根公式是什么？",
        client=client,
        orchestrator=_FakeOrchestrator(_concept_formula_result()),
    )

    assert turn.boundary_audit == initial_audit.model_copy(
        update={"evidence": [], "repair_instructions": []}
    )
    assert turn.post_repair_boundary_audit == post_repair_audit.model_copy(
        update={"evidence": [], "repair_instructions": []}
    )
    assert turn.boundary_fallback_reason == (
        "post_repair_audit_failed:applied_without_permission,final_answer_leak"
    )
    assert "a=3" not in turn.teacher_output.response
    assert "b=2" not in turn.teacher_output.response
    assert "c=-11" not in turn.teacher_output.response
    assert "136" not in turn.teacher_output.response


def test_post_repair_audit_error_fails_closed_to_safe_fallback():
    client = _ContractTurnClient(
        drafts=["代入本题可得答案。", "令 a=3，b=2，c=-11。"],
        audits=[
            BoundaryAudit(
                passed=False,
                violations=["applied_without_permission"],
                repair_instructions=["不要代入当前题"],
            ),
            RuntimeError("post-repair audit unavailable"),
        ],
    )

    turn = run_turn(
        PrototypeEpisode(problem_context="3x^2+2x-11=0"),
        "求根公式是什么？",
        client=client,
        orchestrator=_FakeOrchestrator(_concept_formula_result()),
    )

    assert turn.post_repair_boundary_audit is not None
    assert turn.post_repair_boundary_audit.audit_error == "RuntimeError"
    assert turn.boundary_fallback_reason == (
        "post_repair_audit_error:RuntimeError"
    )
    assert "a=3" not in turn.teacher_output.response


def test_agent_repair_failure_raises_without_consuming_pending_message():
    result = AgentOrchestrationResult(
        classification=RequestClassification(
            scope="k9",
            subject="物理",
            stage="初中",
            is_new_problem=True,
            answer_mode="complete",
            reason="new physics question",
        ),
        teaching_contract=TeachingContract(
            response_mode="concept_explanation",
            requested_target="解释惯性",
            allowed_content=["概念定义"],
            forbidden_content=["套用旧数学题"],
            may_apply_to_active_problem=False,
            may_reveal_final_answer=False,
            reason="student requested a concept explanation",
        ),
        rag_context=RagContext(source="none"),
        termination_reason="model_finished",
    )
    client = _ContractTurnClient(
        drafts=["未经审计的草稿", RuntimeError("repair unavailable")],
        audits=[RuntimeError("audit unavailable")],
    )
    episode = PrototypeEpisode(
        problem_context="2x+3=7",
        previous_verified_step="2x=4",
        active_subject="数学",
        active_stage="小学",
        pending_student_message="惯性是什么？",
    )
    orchestrator = _FakeOrchestrator(result)

    try:
        run_turn(
            episode,
            "惯性是什么？",
            client=client,
            orchestrator=orchestrator,
        )
    except RuntimeError as exc:
        assert str(exc) == "repair unavailable"
    else:
        raise AssertionError("repair failure must reach the pending-message recovery path")

    assert orchestrator.calls[0][:4] == (
        "2x+3=7",
        "惯性是什么？",
        "数学",
        "小学",
    )
    assert episode.problem_context == "2x+3=7"
    assert episode.previous_verified_step == "2x=4"
    assert episode.active_subject == "数学"
    assert episode.active_stage == "小学"
    assert episode.pending_student_message == "惯性是什么？"
    assert episode.turns == []


def test_normalized_teacher_text_is_reused_in_next_turn_history():
    result = _concept_formula_result()
    client = _ContractTurnClient(
        drafts=[
            "公式是：\n\n[ x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a} ]",
            "其中 a、b、c 分别是二次项、一次项和常数项的系数。",
        ],
        audits=[BoundaryAudit(passed=True), BoundaryAudit(passed=True)],
    )
    orchestrator = _FakeOrchestrator(result)
    episode = PrototypeEpisode(
        problem_context="3x^2+2x-11=0",
        active_subject="数学",
        active_stage="初中",
    )

    first = run_turn(
        episode,
        "求根公式是什么？",
        client=client,
        orchestrator=orchestrator,
    )
    run_turn(
        episode,
        "a、b、c 分别是什么？",
        client=client,
        orchestrator=orchestrator,
    )

    persisted = first.teacher_output.response
    history_text = client.teacher_payloads[1]["conversation_history"][0]["teacher"]
    assert persisted == history_text
    assert "$$x=\\frac" in persisted


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
            teaching_contract=TeachingContract(
                response_mode="concept_explanation",
                requested_target="解释受力改变运动状态的原因",
                allowed_content=["物理概念", "原因说明"],
                forbidden_content=["套用无关数学题"],
                may_apply_to_active_problem=False,
                may_reveal_final_answer=False,
                reason="student requested a concept explanation",
            ),
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
    assert turn.classification == classification
    assert turn.agent_trace == trace
    assert turn.loaded_skills == ["concept_explanation"]
    assert turn.agent_termination_reason == "model_finished"
    assert turn.rag_source == "local"
    assert client.teacher_payloads[0]["request_classification"]["subject"] == "物理"
    assert client.teacher_payloads[0]["loaded_skills"] == ["concept_explanation"]


def test_agent_turn_records_classification_and_planning_as_distinct_stages() -> None:
    classification = RequestClassification(
        scope="k9",
        subject="物理",
        stage="初中",
        is_new_problem=True,
        answer_mode="complete",
        reason="new physics question",
    )
    classification_event = AgentTraceEvent(
        step=0,
        event_type="classification",
        summary="识别为初中物理新题",
    )
    result = AgentOrchestrationResult(
        classification=classification,
        teaching_contract=TeachingContract(
            response_mode="concept_explanation",
            requested_target="解释力与运动状态",
            allowed_content=["物理概念"],
            forbidden_content=["无关数学题"],
            may_apply_to_active_problem=False,
            may_reveal_final_answer=False,
            reason="student requested a concept explanation",
        ),
        rag_context=RagContext(source="none"),
        trace=[classification_event],
        termination_reason="model_finished",
    )

    class ProgressOrchestrator(_FakeOrchestrator):
        def run(self, *args, progress_callback=None, **kwargs):
            progress_callback("classification")
            progress_callback(classification_event)
            progress_callback("planning")
            return super().run(*args, progress_callback=progress_callback, **kwargs)

    client = _ScriptedClient(
        TurnGuidance(starts_new_problem=True, pacing=PacingMode.COMPLETE_RESPONSE),
        ["力可以改变物体的运动状态。"],
    )
    progress: list[str | AgentTraceEvent] = []

    turn = run_turn(
        PrototypeEpisode(),
        "力为什么能改变物体的运动状态？",
        client=client,
        orchestrator=ProgressOrchestrator(result),
        progress_callback=progress.append,
    )

    assert progress == [
        "classification",
        classification_event,
        "planning",
        "generating",
        "auditing",
        "complete",
    ]
    assert "classification" in turn.stage_timings_seconds
    assert "planning" in turn.stage_timings_seconds
    assert "observing" not in turn.stage_timings_seconds
    assert "retrieving" not in turn.stage_timings_seconds


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
            teaching_contract=TeachingContract(
                response_mode="one_step",
                requested_target="继续当前方程的下一步",
                allowed_content=["一个有意义的步骤"],
                forbidden_content=["完整求解", "最终答案"],
                may_apply_to_active_problem=True,
                may_reveal_final_answer=False,
                reason="student is continuing the active problem",
            ),
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


def test_quadratic_turn_keeps_optional_linear_verifier_neutral() -> None:
    classification = RequestClassification(
        scope="k9",
        subject="数学",
        stage="初中",
        is_new_problem=False,
        answer_mode="complete",
        reason="quadratic follow-up",
    )
    result = _concept_formula_result().model_copy(
        update={"classification": classification}
    )
    client = _ContractTurnClient(
        drafts=[
            "令 a=3，b=2，c=-11，先计算判别式："
            "$$\\Delta=b^2-4ac=136$$"
        ],
        audits=[BoundaryAudit(passed=True)],
    )
    episode = PrototypeEpisode(
        problem_context="3x^2+2x-11=0",
        previous_verified_step="旧的线性证据",
        active_subject="数学",
        active_stage="初中",
    )

    turn = run_turn(
        episode,
        "请完整解答并解释为什么",
        client=client,
        orchestrator=_FakeOrchestrator(result),
    )

    assert turn.student_verification.status.value == "not_applicable"
    assert turn.student_verification.verifier_name == "none"
    assert turn.extracted_equations == []
    assert turn.claim_audits == []
    assert turn.evidence_level == "unverified"
    assert episode.previous_verified_step == "旧的线性证据"
