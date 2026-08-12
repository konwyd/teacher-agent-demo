"""Best-effort, per-conversation llama-server KV slot persistence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


DEFAULT_CACHE_DIR = "data/runtime/kv-states"


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _server_url_from_existing_base_url(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    path = parts.path
    if path == "/v1":
        path = ""
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


@dataclass(frozen=True)
class KvCacheIdentity:
    user_id: str
    conversation_id: str

    def __post_init__(self) -> None:
        if not self.user_id.strip() or not self.conversation_id.strip():
            raise ValueError("KV cache identity fields must not be blank")


@dataclass(frozen=True)
class KvCacheSettings:
    enabled: bool = False
    cache_dir: Path = Path(DEFAULT_CACHE_DIR)
    server_url: str = "http://127.0.0.1:8080"
    parallel: int = 1
    http_timeout_seconds: float = 30.0
    keep_versions: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "server_url", self.server_url.rstrip("/"))
        if self.parallel < 1:
            raise ValueError("parallel must be at least 1")
        if self.http_timeout_seconds <= 0:
            raise ValueError("http_timeout_seconds must be positive")
        if self.keep_versions < 1:
            raise ValueError("keep_versions must be at least 1")

    @classmethod
    def from_env(cls) -> "KvCacheSettings":
        existing_base_url = os.getenv(
            "DEEPSEEK_BASE_URL", "http://127.0.0.1:8080/v1"
        )
        server_url = os.getenv("TEACHER_KV_SERVER_URL") or (
            _server_url_from_existing_base_url(existing_base_url)
        )
        return cls(
            enabled=_enabled(os.getenv("TEACHER_KV_PERSIST")),
            cache_dir=Path(os.getenv("TEACHER_KV_CACHE_DIR", DEFAULT_CACHE_DIR)),
            server_url=server_url,
            parallel=int(os.getenv("TEACHER_LLAMA_PARALLEL", "1")),
            http_timeout_seconds=float(
                os.getenv("TEACHER_KV_HTTP_TIMEOUT_SECONDS", "30")
            ),
            keep_versions=int(os.getenv("TEACHER_KV_KEEP_VERSIONS", "2")),
        )


@dataclass
class KvTurnStatus:
    enabled: bool
    slot_id: int
    cache_dir: str
    restore_status: str = "not_attempted"
    save_status: str = "not_attempted"
    filename: str = ""
    restored_tokens: int = 0
    restored_bytes: int = 0
    saved_tokens: int = 0
    saved_bytes: int = 0
    error: str = ""

    def to_metadata(self) -> dict[str, object]:
        return {
            "kv_persistence_enabled": self.enabled,
            "kv_persistence_slot": self.slot_id,
            "kv_persistence_restore_status": self.restore_status,
            "kv_persistence_save_status": self.save_status,
            "kv_persistence_filename": self.filename,
            "kv_persistence_directory": self.cache_dir,
            "kv_persistence_restored_tokens": self.restored_tokens,
            "kv_persistence_restored_bytes": self.restored_bytes,
            "kv_persistence_saved_tokens": self.saved_tokens,
            "kv_persistence_saved_bytes": self.saved_bytes,
            "kv_persistence_error": self.error,
        }


class KvCacheManager:
    """Coordinates slot ownership and versioned state files."""

    def __init__(self, settings: KvCacheSettings, *, opener=None) -> None:
        self.settings = settings
        self._opener = opener or build_opener(ProxyHandler({}))
        self._slot_locks = [threading.Lock() for _ in range(settings.parallel)]
        if settings.enabled:
            settings.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _digest(identity: KvCacheIdentity) -> str:
        raw = f"{identity.user_id}\0{identity.conversation_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def cache_prefix(self, identity: KvCacheIdentity) -> str:
        return f"kv_{self._digest(identity)[:32]}"

    def slot_for(self, identity: KvCacheIdentity) -> int:
        return int(self._digest(identity), 16) % self.settings.parallel

    @contextmanager
    def lease(self, identity: KvCacheIdentity) -> Iterator["KvCacheLease"]:
        slot_id = self.slot_for(identity)
        lock = self._slot_locks[slot_id]
        lock.acquire()
        lock_file = None
        try:
            if self.settings.enabled:
                self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
                lock_path = self.settings.cache_dir / f".slot-{slot_id}.lock"
                lock_file = lock_path.open("a+b")
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield KvCacheLease(self, identity, slot_id)
        finally:
            if lock_file is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            lock.release()

    def _post(self, slot_id: int, action: str, body: dict) -> dict:
        request = Request(
            f"{self.settings.server_url}/slots/{slot_id}?action={action}",
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener.open(
            request, timeout=self.settings.http_timeout_seconds
        ) as response:
            status = int(getattr(response, "status", 200))
            if not 200 <= status < 300:
                raise OSError(f"llama-server returned HTTP {status}")
            raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(payload, dict):
            raise ValueError("llama-server returned a non-object response")
        return payload


class KvCacheLease:
    def __init__(
        self, manager: KvCacheManager, identity: KvCacheIdentity, slot_id: int
    ) -> None:
        self.manager = manager
        self.identity = identity
        self.slot_id = slot_id
        self.status = KvTurnStatus(
            enabled=manager.settings.enabled,
            slot_id=slot_id,
            cache_dir=str(manager.settings.cache_dir.resolve()),
        )

    def _versions(self) -> list[Path]:
        prefix = self.manager.cache_prefix(self.identity)
        return sorted(
            self.manager.settings.cache_dir.glob(f"{prefix}_*.bin"),
            key=lambda path: path.name,
            reverse=True,
        )

    def _remember_error(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        self.status.error = message[:300]

    def restore(self) -> bool:
        if not self.manager.settings.enabled:
            self.status.restore_status = "disabled"
            return False
        versions = self._versions()
        for version in versions:
            try:
                payload = self.manager._post(
                    self.slot_id, "restore", {"filename": version.name}
                )
            except Exception as exc:  # noqa: BLE001 - cache failure is recoverable
                self._remember_error(exc)
                continue
            self.status.restore_status = "restored"
            self.status.filename = version.name
            self.status.restored_tokens = int(
                payload.get(
                    "n_restored",
                    payload.get("n_restored_tokens", payload.get("n_tokens", 0)),
                )
                or 0
            )
            self.status.restored_bytes = int(
                payload.get(
                    "n_read",
                    payload.get("n_read_bytes", payload.get("n_bytes", 0)),
                )
                or 0
            )
            self.status.error = ""
            return True
        try:
            self.manager._post(self.slot_id, "erase", {})
        except Exception as exc:  # noqa: BLE001
            self._remember_error(exc)
        self.status.restore_status = "miss" if not versions else "fallback"
        return False

    def save(self) -> bool:
        if not self.manager.settings.enabled:
            self.status.save_status = "disabled"
            return False
        filename = (
            f"{self.manager.cache_prefix(self.identity)}_{time.time_ns()}.bin"
        )
        try:
            payload = self.manager._post(
                self.slot_id, "save", {"filename": filename}
            )
        except Exception as exc:  # noqa: BLE001 - reply remains usable
            self.status.save_status = "failed"
            self._remember_error(exc)
            return False
        self.status.save_status = "saved"
        self.status.filename = filename
        self.status.saved_tokens = int(
            payload.get(
                "n_saved",
                payload.get("n_saved_tokens", payload.get("n_tokens", 0)),
            )
            or 0
        )
        self.status.saved_bytes = int(
            payload.get(
                "n_written",
                payload.get("n_written_bytes", payload.get("n_bytes", 0)),
            )
            or 0
        )
        self.status.error = ""
        self.prune_versions()
        return True

    def prune_versions(self) -> None:
        for old_version in self._versions()[self.manager.settings.keep_versions :]:
            try:
                old_version.unlink()
            except OSError as exc:
                self._remember_error(exc)
