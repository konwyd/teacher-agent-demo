import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _settings(tmp_path: Path, **overrides):
    from app.llm.kv_cache import KvCacheSettings

    values = {
        "enabled": True,
        "cache_dir": tmp_path,
        "server_url": "http://127.0.0.1:8080",
        "parallel": 4,
        "http_timeout_seconds": 30.0,
        "keep_versions": 2,
        "max_bytes": 100,
        "high_water_bytes": 80,
        "low_water_bytes": 70,
    }
    values.update(overrides)
    return KvCacheSettings(**values)


def test_identity_hash_hides_student_and_conversation_and_maps_stably(tmp_path: Path):
    from app.llm.kv_cache import KvCacheIdentity, KvCacheManager

    identity = KvCacheIdentity("usr_secret", "conv_secret")
    manager = KvCacheManager(_settings(tmp_path), opener=object())

    prefix = manager.cache_prefix(identity)
    assert prefix.startswith("kv_")
    assert "usr_secret" not in prefix
    assert "conv_secret" not in prefix
    assert manager.slot_for(identity) in range(4)
    assert manager.slot_for(identity) == manager.slot_for(identity)


def test_settings_derive_existing_server_address_without_replacing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from app.llm.kv_cache import KvCacheSettings

    monkeypatch.setenv("TEACHER_KV_PERSIST", "1")
    monkeypatch.setenv("TEACHER_KV_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TEACHER_KV_SERVER_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://10.0.0.8:9099/v1")
    monkeypatch.setenv("TEACHER_LLAMA_PARALLEL", "4")

    settings = KvCacheSettings.from_env()

    assert settings.enabled is True
    assert settings.server_url == "http://10.0.0.8:9099"
    assert settings.cache_dir == tmp_path
    assert settings.parallel == 4


def test_settings_are_disabled_by_default(monkeypatch: pytest.MonkeyPatch):
    from app.llm.kv_cache import KvCacheSettings

    monkeypatch.delenv("TEACHER_KV_PERSIST", raising=False)
    assert KvCacheSettings.from_env().enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parallel", 0),
        ("http_timeout_seconds", 0),
        ("keep_versions", 0),
        ("max_bytes", 0),
    ],
)
def test_settings_reject_invalid_runtime_limits(tmp_path: Path, field: str, value):
    values = {field: value}
    with pytest.raises(ValueError):
        _settings(tmp_path, **values)


class _Response:
    status = 200

    def __init__(self, body: dict):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")


class _RecordingOpener:
    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = iter(outcomes or [])

    def open(self, request, timeout):
        body = json.loads((request.data or b"{}").decode("utf-8"))
        self.calls.append((request.full_url, body, timeout))
        outcome = next(
            self.outcomes,
            {"id_slot": 0, "filename": body.get("filename", ""), "n_saved_tokens": 8, "n_written_bytes": 64},
        )
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)


def test_restore_tries_newest_then_older_version(tmp_path: Path):
    from app.llm.kv_cache import KvCacheIdentity, KvCacheManager

    identity = KvCacheIdentity("usr_a", "conv_a")
    opener = _RecordingOpener(
        [OSError("new file incompatible"), {"id_slot": 0, "n_restored_tokens": 21, "n_read_bytes": 99}]
    )
    manager = KvCacheManager(_settings(tmp_path, parallel=1), opener=opener)
    prefix = manager.cache_prefix(identity)
    (tmp_path / f"{prefix}_100.bin").write_bytes(b"old")
    (tmp_path / f"{prefix}_200.bin").write_bytes(b"new")

    with manager.lease(identity) as lease:
        assert lease.restore() is True

    assert [call[1]["filename"] for call in opener.calls] == [
        f"{prefix}_200.bin",
        f"{prefix}_100.bin",
    ]
    assert lease.status.restore_status == "restored"
    assert lease.status.filename == f"{prefix}_100.bin"


def test_restore_miss_erases_slot_and_does_not_fail_turn(tmp_path: Path):
    from app.llm.kv_cache import KvCacheIdentity, KvCacheManager

    opener = _RecordingOpener([{"id_slot": 0, "n_erased": 4}])
    manager = KvCacheManager(_settings(tmp_path, parallel=1), opener=opener)

    with manager.lease(KvCacheIdentity("usr_a", "conv_new")) as lease:
        assert lease.restore() is False

    assert opener.calls == [
        ("http://127.0.0.1:8080/slots/0?action=erase", {}, 30.0)
    ]
    assert lease.status.restore_status == "miss"


def test_save_prunes_only_after_success_and_keeps_two_versions(tmp_path: Path):
    from app.llm.kv_cache import KvCacheIdentity, KvCacheManager

    identity = KvCacheIdentity("usr_a", "conv_a")
    opener = _RecordingOpener()
    manager = KvCacheManager(_settings(tmp_path, parallel=1), opener=opener)
    prefix = manager.cache_prefix(identity)
    old = tmp_path / f"{prefix}_100.bin"
    middle = tmp_path / f"{prefix}_200.bin"
    old.write_bytes(b"old")
    middle.write_bytes(b"middle")

    with manager.lease(identity) as lease:
        assert lease.save() is True
        saved = tmp_path / lease.status.filename
        saved.write_bytes(b"new")
        lease.prune_versions()

    assert old.exists() is False
    assert middle.exists() is True
    assert saved.exists() is True
    save_url, save_body, _ = opener.calls[0]
    assert save_url == "http://127.0.0.1:8080/slots/0?action=save"
    assert save_body == {"filename": lease.status.filename}
    assert lease.status.save_status == "saved"


def test_failed_save_preserves_existing_versions(tmp_path: Path):
    from app.llm.kv_cache import KvCacheIdentity, KvCacheManager

    identity = KvCacheIdentity("usr_a", "conv_a")
    opener = _RecordingOpener([OSError("server unavailable")])
    manager = KvCacheManager(_settings(tmp_path, parallel=1), opener=opener)
    prefix = manager.cache_prefix(identity)
    existing = tmp_path / f"{prefix}_100.bin"
    existing.write_bytes(b"last good state")

    with manager.lease(identity) as lease:
        assert lease.save() is False

    assert existing.read_bytes() == b"last good state"
    assert lease.status.save_status == "failed"
    assert "server unavailable" in lease.status.error


def test_status_metadata_contains_no_raw_student_identifiers(tmp_path: Path):
    from app.llm.kv_cache import KvCacheIdentity, KvCacheManager

    identity = KvCacheIdentity("usr_private", "conv_private")
    manager = KvCacheManager(_settings(tmp_path, parallel=1), opener=_RecordingOpener())
    with manager.lease(identity) as lease:
        metadata = lease.status.to_metadata()

    assert metadata["kv_persistence_enabled"] is True
    assert metadata["kv_persistence_slot"] == 0
    assert "usr_private" not in repr(metadata)
    assert "conv_private" not in repr(metadata)


def test_current_llama_server_counter_names_are_recorded(tmp_path: Path):
    from app.llm.kv_cache import KvCacheIdentity, KvCacheManager

    identity = KvCacheIdentity("usr_a", "conv_a")
    opener = _RecordingOpener(
        [
            {"id_slot": 0, "n_restored": 21, "n_read": 99},
            {"id_slot": 0, "n_saved": 34, "n_written": 144},
        ]
    )
    manager = KvCacheManager(_settings(tmp_path, parallel=1), opener=opener)
    prefix = manager.cache_prefix(identity)
    (tmp_path / f"{prefix}_100.bin").write_bytes(b"state")

    with manager.lease(identity) as lease:
        assert lease.restore() is True
        assert lease.save() is True

    assert lease.status.restored_tokens == 21
    assert lease.status.restored_bytes == 99
    assert lease.status.saved_tokens == 34
    assert lease.status.saved_bytes == 144


def test_budget_cleanup_starts_above_high_water_and_stops_at_low_water(tmp_path: Path):
    from app.llm.kv_cache import KvCacheManager

    manager = KvCacheManager(_settings(tmp_path), opener=object())
    oldest = tmp_path / "kv_a_100.bin"
    middle = tmp_path / "kv_b_200.bin"
    protected = tmp_path / "kv_c_300.bin"
    oldest.write_bytes(b"a" * 30)
    middle.write_bytes(b"b" * 30)
    protected.write_bytes(b"c" * 30)

    result = manager.cleanup_cache(protected={protected})

    assert result.before_bytes == 90
    assert result.after_bytes == 60
    assert oldest.exists() is False
    assert middle.exists() is True
    assert protected.exists() is True


def test_budget_cleanup_is_noop_at_or_below_high_water(tmp_path: Path):
    from app.llm.kv_cache import KvCacheManager

    manager = KvCacheManager(_settings(tmp_path), opener=object())
    snapshot = tmp_path / "kv_a_100.bin"
    snapshot.write_bytes(b"a" * 80)

    result = manager.cleanup_cache()

    assert result.before_bytes == result.after_bytes == 80
    assert snapshot.exists() is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "another-model"),
        ("cache_format", "llama-server-slot-v2"),
        ("context_tokens", 262_144),
        ("parallel", 2),
        ("unified_kv", True),
        ("cache_type_k", "f16"),
        ("cache_type_v", "f16"),
    ],
)
def test_snapshot_prefix_changes_for_every_compatibility_field(
    tmp_path: Path, field: str, value
):
    from app.llm.kv_cache import KvCacheIdentity, KvCacheManager

    identity = KvCacheIdentity("usr_a", "conv_a")
    baseline = KvCacheManager(_settings(tmp_path), opener=object())
    changed = KvCacheManager(_settings(tmp_path, **{field: value}), opener=object())

    assert baseline.cache_prefix(identity) != changed.cache_prefix(identity)


def test_budget_cleanup_never_deletes_snapshot_from_active_slot(tmp_path: Path):
    from app.llm.kv_cache import KvCacheIdentity, KvCacheManager

    manager = KvCacheManager(_settings(tmp_path, parallel=1), opener=object())
    active_identity = KvCacheIdentity("usr_active", "conv_active")
    same_slot_identity = KvCacheIdentity("usr_other", "conv_other")
    snapshot = tmp_path / f"{manager.cache_prefix(same_slot_identity)}_100.bin"
    snapshot.write_bytes(b"x" * 90)

    with manager.lease(active_identity):
        result = manager.cleanup_cache()

    assert result.before_bytes == result.after_bytes == 90
    assert snapshot.exists() is True


def test_hard_budget_discards_replacement_and_keeps_last_known_good(tmp_path: Path):
    from app.llm.kv_cache import KvCacheIdentity, KvCacheManager

    class SavingOpener(_RecordingOpener):
        def open(self, request, timeout):
            body = json.loads((request.data or b"{}").decode("utf-8"))
            response = super().open(request, timeout)
            if request.full_url.endswith("action=save"):
                (tmp_path / body["filename"]).write_bytes(b"n" * 60)
            return response

    identity = KvCacheIdentity("usr_a", "conv_a")
    opener = SavingOpener(
        [
            {"id_slot": 0, "n_restored": 10},
            {"id_slot": 0, "n_saved": 20},
        ]
    )
    manager = KvCacheManager(
        _settings(tmp_path, parallel=1, keep_versions=1),
        opener=opener,
    )
    previous = tmp_path / f"{manager.cache_prefix(identity)}_100.bin"
    previous.write_bytes(b"o" * 50)

    with manager.lease(identity) as lease:
        assert lease.restore() is True
        assert lease.save() is False

    assert lease.status.save_status == "discarded_budget"
    assert previous.read_bytes() == b"o" * 50
    assert sorted(tmp_path.glob(f"{manager.cache_prefix(identity)}_*.bin")) == [
        previous
    ]
