from __future__ import annotations

import pytest

from scripts.run_candidate_extractor_v2_route_a import run_benchmark


def test_route_a_runner_reports_development_and_layer_b_without_holdout() -> None:
    result = run_benchmark(("development", "layer_b"))

    assert result["route"] == "A"
    assert result["extractor"] == "candidate_extractor_v2_deterministic"
    assert set(result["reports"]) == {"development", "layer_b"}
    for report in result["reports"].values():
        assert report["exact_candidate_set_accuracy"]["value"] == 1.0
        assert report["operational"]["additional_model_calls"] == 0
        assert report["operational"]["peak_vram_mb"] == 0.0
        assert report["operational"]["offline_capability"] is True
        assert report["operational"]["deterministic_replay"] is True
        assert report["operational"]["peak_ram_mb"] > 0
        assert report["operational"]["dependency_footprint"] == [
            "python-stdlib",
            "pydantic",
        ]


def test_route_a_runner_rejects_non_development_splits() -> None:
    with pytest.raises(ValueError, match="unsupported Route A split"):
        run_benchmark(("holdout",))
