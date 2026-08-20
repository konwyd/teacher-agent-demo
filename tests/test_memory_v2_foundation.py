from __future__ import annotations

from app.memory.contracts import (
    BaselineMemoryAction,
    MemorySourceKind,
    MemoryType,
    TemporalScope,
)
from app.memory.foundation import (
    MemoryV2FeatureFlags,
    observe_baseline_memory_action,
)


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "turn_id": 2,
        "student_message": "我是五年级，请讲慢点。我不懂移项。",
        "problem_context": "解方程 2x+3=9",
        "classification": {"scope": "k9", "subject": "数学"},
        "verification": {"status": "verified_incorrect", "summary": "移项符号错误"},
        "teaching_contract": {"requested_target": "理解移项为什么变号"},
        "memory_resolution_status": "none",
    }
    payload.update(overrides)
    return payload


def test_feature_flags_default_off_and_enabled_is_non_authoritative(monkeypatch) -> None:
    for name in (
        "TEACHER_MEMORY_ADMISSION_ENABLED",
        "TEACHER_MEMORY_ADMISSION_SHADOW_MODE",
        "TEACHER_MEMORY_CONTROLLER_ENABLED",
        "TEACHER_MEMORY_CONTROLLER_SHADOW_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    assert MemoryV2FeatureFlags.from_env().model_dump() == {
        "memory_admission_enabled": False,
        "memory_admission_shadow_mode": False,
        "memory_controller_enabled": False,
        "memory_controller_shadow_mode": False,
    }

    monkeypatch.setenv("TEACHER_MEMORY_ADMISSION_ENABLED", "true")
    flags = MemoryV2FeatureFlags.from_env()
    assert flags.memory_admission_enabled is True
    assert flags.authoritative_admission is False


def test_disabled_observer_emits_nothing() -> None:
    records = observe_baseline_memory_action(
        _payload(),
        conversation_id="conv-1",
        baseline_job_id="memory_job_1",
        flags=MemoryV2FeatureFlags(),
    )
    assert records == []


def test_shadow_observer_records_candidates_without_proposed_decision() -> None:
    records = observe_baseline_memory_action(
        _payload(),
        conversation_id="conv-1",
        baseline_job_id="memory_job_1",
        flags=MemoryV2FeatureFlags(memory_admission_shadow_mode=True),
    )

    assert records
    assert {record.baseline_action for record in records} == {
        BaselineMemoryAction.ENQUEUE
    }
    assert all(record.proposed_decision is None for record in records)
    assert all(record.policy_name == "none" for record in records)
    by_type = {record.candidate.memory_type: record.candidate for record in records}
    assert by_type[MemoryType.PREFERENCE].source_kind is MemorySourceKind.USER_STATED
    assert by_type[MemoryType.PREFERENCE].temporal_scope is TemporalScope.PERSISTENT
    assert (
        by_type[MemoryType.BEHAVIOR_EVIDENCE].source_kind
        is MemorySourceKind.VERIFIED_RESULT
    )
    assert by_type[MemoryType.DIFFICULTY].source_kind is MemorySourceKind.USER_STATED
    assert by_type[MemoryType.TASK_STATE].temporal_scope is TemporalScope.SESSION


def test_shadow_observer_classifies_baseline_skip_without_changing_it() -> None:
    records = observe_baseline_memory_action(
        _payload(classification={"scope": "social", "subject": ""}),
        conversation_id="conv-skip",
        baseline_job_id="",
        flags=MemoryV2FeatureFlags(memory_controller_shadow_mode=True),
    )
    assert records
    assert all(record.baseline_action is BaselineMemoryAction.SKIP for record in records)
    assert all(record.baseline_skip_reason == "baseline_skipped_scope" for record in records)


def test_candidate_ids_are_deterministic_and_payloads_are_bounded() -> None:
    flags = MemoryV2FeatureFlags(memory_admission_shadow_mode=True)
    first = observe_baseline_memory_action(
        _payload(), conversation_id="conv-1", baseline_job_id="job-1", flags=flags
    )
    second = observe_baseline_memory_action(
        _payload(), conversation_id="conv-1", baseline_job_id="job-2", flags=flags
    )
    assert [item.candidate.candidate_id for item in first] == [
        item.candidate.candidate_id for item in second
    ]
    assert all("student_message" not in item.candidate.payload for item in first)
