from __future__ import annotations

import sqlite3

import pytest

from app.auth.local_accounts import (
    AuthenticationError,
    DuplicateUsernameError,
    InvalidPasswordError,
    InvalidUsernameError,
    LocalAccountStore,
)


def test_register_and_authenticate_round_trip(tmp_path) -> None:
    db_path = tmp_path / "users.sqlite3"
    store = LocalAccountStore(db_path)

    registered = store.register("小明_01", "correct-horse-2026")
    authenticated = store.authenticate("小明_01", "correct-horse-2026")

    assert authenticated == registered
    assert registered.user_id
    assert registered.username == "小明_01"


def test_password_is_not_stored_as_plaintext(tmp_path) -> None:
    db_path = tmp_path / "users.sqlite3"
    store = LocalAccountStore(db_path)
    password = "private-password-2026"

    store.register("student_a", password)

    with sqlite3.connect(db_path) as connection:
        stored = connection.execute(
            "SELECT password_hash FROM users WHERE username_key = ?",
            ("student_a",),
        ).fetchone()[0]
    assert password not in stored
    assert stored.startswith("pbkdf2_sha256$")


def test_username_is_unique_case_insensitively(tmp_path) -> None:
    store = LocalAccountStore(tmp_path / "users.sqlite3")
    first = store.register("Student-One", "first-password")

    with pytest.raises(DuplicateUsernameError):
        store.register("student-one", "second-password")

    assert store.authenticate("STUDENT-ONE", "first-password").user_id == first.user_id


def test_wrong_password_does_not_reveal_whether_account_exists(tmp_path) -> None:
    store = LocalAccountStore(tmp_path / "users.sqlite3")
    store.register("known_student", "right-password")

    with pytest.raises(AuthenticationError) as wrong_password:
        store.authenticate("known_student", "wrong-password")
    with pytest.raises(AuthenticationError) as missing_user:
        store.authenticate("missing_student", "wrong-password")

    assert str(wrong_password.value) == str(missing_user.value)


@pytest.mark.parametrize("username", ["", "a", "has space", "bad/name", "bad\\name"])
def test_invalid_usernames_are_rejected(tmp_path, username: str) -> None:
    store = LocalAccountStore(tmp_path / "users.sqlite3")

    with pytest.raises(InvalidUsernameError):
        store.register(username, "valid-password")


def test_short_password_is_rejected(tmp_path) -> None:
    store = LocalAccountStore(tmp_path / "users.sqlite3")

    with pytest.raises(InvalidPasswordError):
        store.register("student_a", "short")
