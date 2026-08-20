from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.evaluation.memory_admission.robustness.manifest import (
    FrozenManifestError,
    capture_manifest,
    dataset_label_digest,
    verify_manifest,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    policy = repo / "policy.py"
    policy.write_text("POLICY = 'v1'\n")
    unrelated = repo / "notes.txt"
    unrelated.write_text("clean\n")
    dataset = repo / "attacks.jsonl"
    dataset.write_text(
        json.dumps({
            "case_id": "attack-1",
            "expected_action": "pending",
            "expected_relation": "new",
            "labels": ["temporal"],
            "payload": {"ignored_by_seal": "x"},
        }) + "\n"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo, policy, dataset


def test_manifest_records_head_paths_hashes_and_relevant_dirty_state(
    tmp_path: Path,
) -> None:
    repo, policy, dataset = _repo(tmp_path)
    policy.write_text("POLICY = 'v1'  # frozen dirty\n")

    manifest = capture_manifest(
        repo,
        relevant_paths=(Path("policy.py"),),
        dataset_paths=(Path("attacks.jsonl"),),
        policy_identity={"policy_name": "test", "policy_version": "v1"},
        config={"threshold": 0.75},
    )

    assert manifest.git_head == _git(repo, "rev-parse", "HEAD")
    assert manifest.files[0].path == "policy.py"
    assert manifest.files[0].dirty is True
    assert len(manifest.files[0].sha256) == 64
    assert manifest.dataset_label_sha256 == dataset_label_digest((dataset,))


def test_unrelated_dirty_file_does_not_fail_manifest(tmp_path: Path) -> None:
    repo, _, dataset = _repo(tmp_path)
    manifest = capture_manifest(
        repo,
        relevant_paths=(Path("policy.py"),),
        dataset_paths=(Path("attacks.jsonl"),),
        policy_identity={"policy_name": "test", "policy_version": "v1"},
        config={"threshold": 0.75},
    )
    path = repo / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))
    (repo / "notes.txt").write_text("unrelated dirty\n")

    verify_manifest(repo, path, current_config={"threshold": 0.75})


@pytest.mark.parametrize("mutation", ["relevant_file", "head", "config", "label"])
def test_relevant_policy_head_config_or_dataset_label_drift_fails(
    tmp_path: Path, mutation: str
) -> None:
    repo, policy, dataset = _repo(tmp_path)
    manifest = capture_manifest(
        repo,
        relevant_paths=(Path("policy.py"),),
        dataset_paths=(Path("attacks.jsonl"),),
        policy_identity={"policy_name": "test", "policy_version": "v1"},
        config={"threshold": 0.75},
    )
    path = repo / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))
    config = {"threshold": 0.75}
    if mutation == "relevant_file":
        policy.write_text("POLICY = 'v2'\n")
    elif mutation == "head":
        (repo / "new.txt").write_text("new\n")
        _git(repo, "add", "new.txt")
        _git(repo, "commit", "-qm", "new head")
    elif mutation == "config":
        config = {"threshold": 0.76}
    else:
        row = json.loads(dataset.read_text())
        row["expected_action"] = "admit"
        dataset.write_text(json.dumps(row) + "\n")

    with pytest.raises(FrozenManifestError):
        verify_manifest(repo, path, current_config=config)


def test_dataset_seal_ignores_attack_payload_but_freezes_labels(tmp_path: Path) -> None:
    repo, _, dataset = _repo(tmp_path)
    original = dataset_label_digest((dataset,))
    row = json.loads(dataset.read_text())
    row["payload"] = {"ignored_by_seal": "changed"}
    dataset.write_text(json.dumps(row) + "\n")
    assert dataset_label_digest((dataset,)) == original
    row["labels"].append("correction")
    dataset.write_text(json.dumps(row) + "\n")
    assert dataset_label_digest((dataset,)) != original
