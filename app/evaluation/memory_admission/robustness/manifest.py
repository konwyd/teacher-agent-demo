"""Immutable PolicyV1 and robustness-label manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict


class FrozenManifestError(RuntimeError):
    """Raised when frozen policy or ground-truth state has drifted."""


class FrozenFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    dirty: bool


class PolicyV1Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.1.0"
    git_head: str
    config_sha256: str
    policy_identity: dict[str, str]
    config: dict[str, Any]
    policy_v1_path: str
    policy_v1_sha256: str
    files: tuple[FrozenFile, ...]
    dataset_paths: tuple[str, ...]
    dataset_label_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _dirty(repo_root: Path, relative_path: Path) -> bool:
    return bool(_git(repo_root, "status", "--porcelain", "--", relative_path.as_posix()))


def _canonical_label_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            item = json.loads(raw)
            rows.append({
                "case_id": item["case_id"],
                "expected_action": item.get("expected_action"),
                "expected_relation": item.get("expected_relation"),
                "labels": sorted(item.get("labels", [])),
                "trajectory_expectations": item.get("trajectory_expectations", {}),
                "expected_candidates": item.get("expected_candidates", []),
            })
    return sorted(rows, key=lambda item: item["case_id"])


def dataset_label_digest(paths: Iterable[Path]) -> str:
    encoded = json.dumps(
        _canonical_label_rows(paths),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_manifest(
    repo_root: Path,
    *,
    relevant_paths: tuple[Path, ...],
    policy_v1_path: Path,
    dataset_paths: tuple[Path, ...],
    policy_identity: dict[str, str],
    config: dict[str, Any],
) -> PolicyV1Manifest:
    root = repo_root.resolve()
    relative_datasets = tuple(
        path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
        for path in dataset_paths
    )
    absolute_datasets = tuple(root / path for path in relative_datasets)
    return PolicyV1Manifest(
        git_head=_git(root, "rev-parse", "HEAD"),
        config_sha256=_json_sha256(config),
        policy_identity=policy_identity,
        config=json.loads(json.dumps(config, sort_keys=True)),
        policy_v1_path=policy_v1_path.as_posix(),
        policy_v1_sha256=_sha256(root / policy_v1_path),
        files=tuple(
            FrozenFile(
                path=path.as_posix(),
                sha256=_sha256(root / path),
                dirty=_dirty(root, path),
            )
            for path in relevant_paths
        ),
        dataset_paths=relative_datasets,
        dataset_label_sha256=dataset_label_digest(absolute_datasets),
    )


def verify_manifest(
    repo_root: Path,
    manifest_path: Path,
    *,
    current_config: dict[str, Any],
) -> PolicyV1Manifest:
    root = repo_root.resolve()
    manifest = PolicyV1Manifest.model_validate_json(manifest_path.read_text())
    errors: list[str] = []
    if _json_sha256(manifest.config) != manifest.config_sha256:
        errors.append("manifest_config_hash")
    if _json_sha256(current_config) != manifest.config_sha256:
        errors.append("config")
    policy_v1_path = root / manifest.policy_v1_path
    if (
        not policy_v1_path.exists()
        or _sha256(policy_v1_path) != manifest.policy_v1_sha256
    ):
        errors.append("policy_v1_sha256")
    for frozen in manifest.files:
        path = Path(frozen.path)
        absolute = root / path
        if not absolute.exists() or _sha256(absolute) != frozen.sha256:
            errors.append(f"sha256:{frozen.path}")
    datasets = tuple(root / path for path in manifest.dataset_paths)
    if dataset_label_digest(datasets) != manifest.dataset_label_sha256:
        errors.append("dataset_labels")
    if errors:
        raise FrozenManifestError("frozen manifest drift: " + ", ".join(errors))
    return manifest
