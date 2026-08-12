"""Small SQLite-backed account store for the local research demo.

This module deliberately owns only identity and password verification. It does
not grant access to model APIs or make teaching decisions.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import sqlite3
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


DEFAULT_ACCOUNT_DB = "data/runtime/users.sqlite3"
_PASSWORD_SCHEME = "pbkdf2_sha256"
_PASSWORD_ITERATIONS = 600_000
_DUMMY_PASSWORD_HASH = (
    "pbkdf2_sha256$600000$MDAwMDAwMDAwMDAwMDAwMA==$"
    "2OmEhAd5S9V5qA8VdVQ3OKkK6L78zJmb6QY0hZzL4lU="
)


class AccountError(ValueError):
    """Base class for user-facing account errors."""


class InvalidUsernameError(AccountError):
    """The requested username does not satisfy the local policy."""


class InvalidPasswordError(AccountError):
    """The requested password does not satisfy the local policy."""


class DuplicateUsernameError(AccountError):
    """The normalized username already exists."""


class AuthenticationError(AccountError):
    """Authentication failed without revealing which credential was wrong."""


@dataclass(frozen=True)
class Account:
    user_id: str
    username: str
    created_at: str


def _normalize_username(username: str) -> tuple[str, str]:
    display = unicodedata.normalize("NFKC", username).strip()
    if not 2 <= len(display) <= 32:
        raise InvalidUsernameError("用户名需要 2 到 32 个字符。")
    if not all(character.isalnum() or character in "_-." for character in display):
        raise InvalidUsernameError("用户名只能包含中文、字母、数字、下划线、短横线或点。")
    return display, display.casefold()


def _validate_password(password: str) -> None:
    if not 8 <= len(password) <= 128:
        raise InvalidPasswordError("密码需要 8 到 128 个字符。")


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PASSWORD_ITERATIONS,
    )
    return "$".join(
        (
            _PASSWORD_SCHEME,
            str(_PASSWORD_ITERATIONS),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        )
    )


def _verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations_text, salt_text, digest_text = encoded.split("$", 3)
        if scheme != _PASSWORD_SCHEME:
            return False
        iterations = int(iterations_text)
        salt = base64.b64decode(salt_text, validate=True)
        expected = base64.b64decode(digest_text, validate=True)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


class LocalAccountStore:
    """Register and authenticate users in a single local SQLite database."""

    def __init__(self, path: str | Path = DEFAULT_ACCOUNT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    username_key TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def register(self, username: str, password: str) -> Account:
        display_name, username_key = _normalize_username(username)
        _validate_password(password)
        account = Account(
            user_id=f"usr_{uuid.uuid4().hex}",
            username=display_name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        password_hash = _hash_password(password)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users (user_id, username, username_key, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        account.user_id,
                        account.username,
                        username_key,
                        password_hash,
                        account.created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateUsernameError("该用户名已被注册，请直接登录或更换用户名。") from exc
        return account

    def authenticate(self, username: str, password: str) -> Account:
        try:
            _, username_key = _normalize_username(username)
        except InvalidUsernameError:
            username_key = ""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, username, password_hash, created_at
                FROM users
                WHERE username_key = ?
                """,
                (username_key,),
            ).fetchone()
        encoded_hash = row["password_hash"] if row is not None else _DUMMY_PASSWORD_HASH
        password_matches = _verify_password(password, encoded_hash)
        if row is None or not password_matches:
            raise AuthenticationError("用户名或密码不正确。")
        return Account(
            user_id=row["user_id"],
            username=row["username"],
            created_at=row["created_at"],
        )


@lru_cache
def get_local_account_store() -> LocalAccountStore:
    return LocalAccountStore(os.getenv("TEACHER_ACCOUNT_DB", DEFAULT_ACCOUNT_DB))
