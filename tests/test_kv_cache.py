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
    [("parallel", 0), ("http_timeout_seconds", 0), ("keep_versions", 0)],
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
