from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.memory.admission_controller import MemoryAdmissionController
from app.memory.admission_ledger import ShadowEvidenceLedger
from app.memory.admission_normalization import CandidateNormalizer
from app.memory.contracts import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionReasonCode,
    BaselineMemoryAction,
    MemoryRelation,
    MemoryType,
)
from app.memory.foundation import (
    MemoryV2FeatureFlags,
    finalize_shadow_admission,
    prepare_shadow_admission,
)
from app.memory.learner_memory import (
    LearnerMemoryStore,
    StudentMemoryIdentity,
    StudentMemoryService,
)


def _payload(
    *,
    message: str = "以后请讲慢点",
    scope: str = "k9",
    verification_status: str = "unable_to_verify",
    verification_summary: str = "不适用确定性验证",
) -> dict[str, object]:
    return {
        "turn_id": 1,
        "student_message": message,
        "problem_context": "解方程 2x+3=9",
        "classification": {
            "scope": scope,
            "subject": "数学",
            "stage": "初中",
            "is_new_problem": True,
            "answer_mode": "guided",
            "reason": "integration fixture",
        },
        "verification": {
            "status": verification_status,
            "summary": verification_summary,
            "verifier_name": "integration_checker",
        },
        "teaching_contract": {
            "response_mode": "one_step",
            "requested_target": "理解移项",
            "allowed_content": ["一步提示"],
            "forbidden_content": ["最终答案"],
            "may_apply_to_active_problem": True,
            "may_reveal_final_answer": False,
            "reason": "逐步指导",
        },
        "memory_resolution_status": "none",
    }


@pytest.fixture
def services(tmp_path: Path):
    service = StudentMemoryService(LearnerMemoryStore(tmp_path / "learner.sqlite3"))
    ledger = ShadowEvidenceLedger(tmp_path / "shadow.sqlite3")
    try:
        yield service, ledger
    finally:
        service.shutdown()


def _flags() -> MemoryV2FeatureFlags:
    return MemoryV2FeatureFlags(memory_admission_shadow_mode=True)


def test_decision_uses_prewrite_memory_snapshot_after_baseline_profile_write(
    services,
) -> None:
    service, ledger = services
    identity = StudentMemoryIdentity("user-1", "conv-1")
    profile = service.load_profile(identity.user_id)
    payload = _payload(message="以后请讲慢点")

    prepared = prepare_shadow_admission(
        payload,
        identity=identity,
        profile=profile,
        memory_service=service,
        ledger=ledger,
        flags=_flags(),
    )
    preference_item = next(
        item
        for item in prepared.items
        if item.candidate.memory_type is MemoryType.PREFERENCE
    )
    assert preference_item.context.existing_memories == ()

    job_id = service.enqueue_turn(identity, profile, payload)
    service.wait_for_idle()
    assert "慢节奏讲解" in service.load_profile(identity.user_id).global_profile.teaching_preferences

    records = finalize_shadow_admission(
        prepared,
        baseline_job_id=job_id,
        ledger=ledger,
    )
    preference = next(
        record
        for record in records
        if record.candidate.memory_type is MemoryType.PREFERENCE
    )

    assert preference.relation is MemoryRelation.NEW
    assert preference.proposed_decision is not None
    assert preference.proposed_decision.decision is AdmissionAction.ADMIT
    assert preference.matched_existing_memory_id == ""


def test_decision_uses_prewrite_ledger_snapshot_when_ledger_changes_before_finalize(
    services,
) -> None:
    service, ledger = services
    identity = StudentMemoryIdentity("user-1", "conv-1")
    profile = service.load_profile(identity.user_id)
    payload = _payload(
        message="我算出了 2x=12",
        verification_status="verified_incorrect",
        verification_summary="移项符号错误",
    )
    prepared = prepare_shadow_admission(
        payload,
        identity=identity,
        profile=profile,
        memory_service=service,
        ledger=ledger,
        flags=_flags(),
    )
    evidence_item = next(
        item
        for item in prepared.items
        if item.candidate.memory_type is MemoryType.BEHAVIOR_EVIDENCE
    )
    assert evidence_item.context.evidence_clusters == ()

    candidate = evidence_item.candidate
    ledger.append(
        student_key=identity.user_id,
        candidate=candidate,
        normalized=CandidateNormalizer().normalize(candidate),
        decision=AdmissionDecision(
            candidate_id=candidate.candidate_id,
            decision=AdmissionAction.PENDING,
            reason_codes=[AdmissionReasonCode.SINGLE_OBSERVATION],
            policy_name="teacher-memory-admission",
            policy_version="v1",
        ),
        evidence_maturity=0.35,
    )

    job_id = service.enqueue_turn(identity, profile, payload)
    service.wait_for_idle()
    records = finalize_shadow_admission(
        prepared,
        baseline_job_id=job_id,
        ledger=ledger,
    )
    evidence = next(
        record
        for record in records
        if record.candidate.memory_type is MemoryType.BEHAVIOR_EVIDENCE
    )

    assert evidence.relation is MemoryRelation.NEW
    assert evidence.evidence_summary is not None
    assert evidence.evidence_summary.support_count == 1
    assert evidence.proposed_decision is not None
    assert evidence.proposed_decision.decision is AdmissionAction.PENDING


def test_controller_reject_does_not_stop_baseline_enqueue(services) -> None:
    service, ledger = services
    identity = StudentMemoryIdentity("user-1", "conv-1")
    profile = service.load_profile(identity.user_id)
    payload = _payload(message="现在继续完成第12题")
    prepared = prepare_shadow_admission(
        payload,
        identity=identity,
        profile=profile,
        memory_service=service,
        ledger=ledger,
        flags=_flags(),
    )

    job_id = service.enqueue_turn(identity, profile, payload)
    service.wait_for_idle()
    records = finalize_shadow_admission(
        prepared,
        baseline_job_id=job_id,
        ledger=ledger,
    )
    task = next(
        record for record in records if record.candidate.memory_type is MemoryType.TASK_STATE
    )

    assert job_id.startswith("memory_job_")
    assert service.store.job_status(job_id) == "completed"
    assert task.baseline_action is BaselineMemoryAction.ENQUEUE
    assert task.proposed_decision is not None
    assert task.proposed_decision.decision is AdmissionAction.REJECT
    assert task.divergence is True


def test_controller_admit_does_not_force_baseline_skip(services) -> None:
    service, ledger = services
    identity = StudentMemoryIdentity("user-1", "conv-1")
    profile = service.load_profile(identity.user_id)
    payload = _payload(message="以后请讲慢点", scope="social")
    prepared = prepare_shadow_admission(
        payload,
        identity=identity,
        profile=profile,
        memory_service=service,
        ledger=ledger,
        flags=_flags(),
    )

    job_id = service.enqueue_turn(identity, profile, payload)
    records = finalize_shadow_admission(
        prepared,
        baseline_job_id=job_id,
        ledger=ledger,
    )
    preference = next(
        record
        for record in records
        if record.candidate.memory_type is MemoryType.PREFERENCE
    )

    assert job_id == ""
    assert preference.baseline_action is BaselineMemoryAction.SKIP
    assert preference.proposed_decision is not None
    assert preference.proposed_decision.decision is AdmissionAction.ADMIT
    assert preference.divergence is True


class _FailingController(MemoryAdmissionController):
    def evaluate(self, candidate, context):
        raise RuntimeError("controller unavailable")


class _AppendFailingLedger:
    def snapshot(self, student_key, *, semantic_keys=None):
        return ()

    def append(self, **kwargs):
        raise sqlite3.OperationalError("ledger unavailable")


def test_controller_exception_is_recorded_without_changing_baseline(services) -> None:
    service, ledger = services
    identity = StudentMemoryIdentity("user-1", "conv-1")
    profile = service.load_profile(identity.user_id)
    payload = _payload()
    prepared = prepare_shadow_admission(
        payload,
        identity=identity,
        profile=profile,
        memory_service=service,
        ledger=ledger,
        flags=_flags(),
    )

    job_id = service.enqueue_turn(identity, profile, payload)
    service.wait_for_idle()
    records = finalize_shadow_admission(
        prepared,
        baseline_job_id=job_id,
        ledger=ledger,
        controller=_FailingController(),
    )

    assert service.store.job_status(job_id) == "completed"
    assert records
    assert all(record.proposed_decision is None for record in records)
    assert all(record.controller_error == "controller:RuntimeError" for record in records)


def test_ledger_append_failure_keeps_decision_and_baseline(services) -> None:
    service, _ = services
    failing_ledger = _AppendFailingLedger()
    identity = StudentMemoryIdentity("user-1", "conv-1")
    profile = service.load_profile(identity.user_id)
    payload = _payload()
    prepared = prepare_shadow_admission(
        payload,
        identity=identity,
        profile=profile,
        memory_service=service,
        ledger=failing_ledger,
        flags=_flags(),
    )

    job_id = service.enqueue_turn(identity, profile, payload)
    service.wait_for_idle()
    records = finalize_shadow_admission(
        prepared,
        baseline_job_id=job_id,
        ledger=failing_ledger,
    )

    assert service.store.job_status(job_id) == "completed"
    assert any(record.proposed_decision is not None for record in records)
    assert all(record.controller_error == "ledger:OperationalError" for record in records)


def test_flags_off_do_not_snapshot_or_emit(services) -> None:
    service, ledger = services
    identity = StudentMemoryIdentity("user-1", "conv-1")
    profile = service.load_profile(identity.user_id)

    prepared = prepare_shadow_admission(
        _payload(),
        identity=identity,
        profile=profile,
        memory_service=service,
        ledger=ledger,
        flags=MemoryV2FeatureFlags(),
    )

    assert prepared.enabled is False
    assert prepared.items == ()
    assert finalize_shadow_admission(
        prepared,
        baseline_job_id="memory_job_unused",
        ledger=ledger,
    ) == []

def test_shadow_mode_preserves_profile_topic_archive_and_retrieval_parity(
    tmp_path: Path,
) -> None:
    from app.memory.conversation_archive import LocalConversationStore

    archive = LocalConversationStore(tmp_path / "archive.sqlite3")
    archived = archive.create(
        "user-parity",
        {"turns": [], "problem_context": "旧题"},
        title="已有对话",
    )
    off = StudentMemoryService(
        LearnerMemoryStore(tmp_path / "off.sqlite3"),
        archive_store=archive,
    )
    on = StudentMemoryService(
        LearnerMemoryStore(tmp_path / "on.sqlite3"),
        archive_store=archive,
    )
    identity = StudentMemoryIdentity("user-parity", "conv-parity")
    payload = _payload(
        message="我是五年级，请讲慢点，我算出 2x=12",
        verification_status="verified_incorrect",
        verification_summary="移项符号错误",
    )
    try:
        off_profile = off.load_profile(identity.user_id)
        off_job = off.enqueue_turn(identity, off_profile, payload)
        off.wait_for_idle()

        on_profile = on.load_profile(identity.user_id)
        ledger = ShadowEvidenceLedger(tmp_path / "parity-shadow.sqlite3")
        prepared = prepare_shadow_admission(
            payload,
            identity=identity,
            profile=on_profile,
            memory_service=on,
            ledger=ledger,
            flags=_flags(),
        )
        on_job = on.enqueue_turn(identity, on_profile, payload)
        on.wait_for_idle()
        records = finalize_shadow_admission(
            prepared,
            baseline_job_id=on_job,
            ledger=ledger,
        )

        off_loaded = off.load_profile(identity.user_id)
        on_loaded = on.load_profile(identity.user_id)
        assert off_loaded.global_profile == on_loaded.global_profile

        def scrub(memory):
            value = memory.model_dump(mode="json")
            for key in (
                "memory_id",
                "student_profile_id",
                "created_at",
                "updated_at",
                "last_accessed_at",
            ):
                value.pop(key)
            for evidence in value["evidence"]:
                evidence.pop("created_at")
            return value

        off_memories = off.store.list_topic_memories(
            off_loaded.student_profile_id,
            "数学",
        )
        on_memories = on.store.list_topic_memories(
            on_loaded.student_profile_id,
            "数学",
        )
        assert [scrub(item) for item in off_memories] == [
            scrub(item) for item in on_memories
        ]

        resolve_kwargs = {
            "user_id": identity.user_id,
            "conversation_id": identity.conversation_id,
            "subject": "数学",
            "stage": "初中",
            "message": "继续这道方程题",
            "problem_context": "解方程 2x+3=9",
            "is_new_problem": False,
        }
        off_resolution = off.resolve(off_loaded, **resolve_kwargs)
        on_resolution = on.resolve(on_loaded, **resolve_kwargs)
        assert off_resolution.status == on_resolution.status
        off_prompt = off_resolution.prompt_payload()
        on_prompt = on_resolution.prompt_payload()
        off_prompt.pop("student_profile_id")
        on_prompt.pop("student_profile_id")
        assert off_prompt == on_prompt

        assert archive.load("user-parity", archived.conversation_id).episode == {
            "turns": [],
            "problem_context": "旧题",
        }
        assert len(archive.list_for_user("user-parity")) == 1
        assert records
        assert off.store.job_status(off_job) == on.store.job_status(on_job) == "completed"
    finally:
        off.shutdown()
        on.shutdown()
