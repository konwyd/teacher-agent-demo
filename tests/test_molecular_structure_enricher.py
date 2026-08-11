"""Offline tests for MolScribe/RDKit chunk enrichment orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data_pipeline.molecular_structure_enricher import (
    ENRICHMENT_START,
    ImageResolver,
    MolecularStructureEnricher,
    has_molecular_context,
    normalize_image_references,
    parse_path_maps,
)


class FakePredictor:
    def __init__(self, predictions: dict[str, dict]) -> None:
        self.predictions = predictions
        self.calls: list[str] = []

    def predict(self, image_path: Path) -> dict:
        self.calls.append(image_path.name)
        value = self.predictions[image_path.name]
        if isinstance(value, Exception):
            raise value
        return value


class FakeAnalyzer:
    def analyze(self, smiles: str):
        if smiles == "invalid":
            return None
        return {
            "canonical_smiles": "CC(=O)O",
            "molecular_formula": "C2H4O2",
            "atom_count": 4,
            "bond_count": 3,
            "ring_count": 0,
            "functional_groups": ["羧基"],
            "search_text": "分子结构识别：规范SMILES CC(=O)O；分子式 C2H4O2；官能团或结构特征 羧基。",
        }


def _chunk(image_name: str, *, subject: str = "化学", text: str = "该有机物的结构式如图") -> dict:
    return {
        "_id": "chunk-1",
        "text": text,
        "metadata": {"subject": subject, "images": [image_name]},
    }


def test_normalize_image_references_accepts_existing_shapes() -> None:
    assert normalize_image_references(["a.jpg", "b.png"]) == ["a.jpg", "b.png"]
    assert normalize_image_references('["a.jpg", "b.png"]') == ["a.jpg", "b.png"]
    assert normalize_image_references("a.jpg,b.png") == ["a.jpg", "b.png"]
    assert normalize_image_references("") == []


def test_keyword_gate_is_deliberately_narrow() -> None:
    assert has_molecular_context("观察乙酸的结构简式并回答")
    assert has_molecular_context("写出水的化学式和分子式")
    assert has_molecular_context("观察该物质的球棍模型")
    assert not has_molecular_context("观察过滤实验装置并回答")


def test_enriches_chemistry_chunk_and_preserves_image_id(tmp_path: Path) -> None:
    image = tmp_path / "abc.jpg"
    image.write_bytes(b"fake-image")
    predictor = FakePredictor({"abc.jpg": {"smiles": "CC(=O)O", "confidence": 0.95}})
    enricher = MolecularStructureEnricher(
        predictor,
        FakeAnalyzer(),
        ImageResolver([tmp_path]),
    )

    result = enricher.enrich_chunk(_chunk("abc.jpg"))

    structures = result["metadata"]["molecular_structures"]
    assert structures[0]["image_id"] == "abc.jpg"
    assert structures[0]["canonical_smiles"] == "CC(=O)O"
    assert structures[0]["functional_groups"] == ["羧基"]
    assert "分子式 C2H4O2" in result["text"]
    assert ENRICHMENT_START in result["text"]
    assert predictor.calls == ["abc.jpg"]


def test_non_chemistry_or_non_structure_context_is_not_predicted(tmp_path: Path) -> None:
    image = tmp_path / "abc.jpg"
    image.write_bytes(b"fake-image")
    predictor = FakePredictor({"abc.jpg": {"smiles": "CC", "confidence": 0.99}})
    enricher = MolecularStructureEnricher(
        predictor,
        FakeAnalyzer(),
        ImageResolver([tmp_path]),
    )

    assert "molecular_structures" not in enricher.enrich_chunk(
        _chunk("abc.jpg", subject="物理")
    )["metadata"]
    assert "molecular_structures" not in enricher.enrich_chunk(
        _chunk("abc.jpg", text="观察过滤实验装置并回答")
    )["metadata"]
    assert predictor.calls == []


def test_rejects_low_confidence_and_invalid_smiles(tmp_path: Path) -> None:
    low = tmp_path / "low.jpg"
    invalid = tmp_path / "invalid.jpg"
    low.write_bytes(b"low")
    invalid.write_bytes(b"invalid")
    predictor = FakePredictor(
        {
            "low.jpg": {"smiles": "CC", "confidence": 0.4},
            "invalid.jpg": {"smiles": "invalid", "confidence": 0.99},
        }
    )
    enricher = MolecularStructureEnricher(
        predictor,
        FakeAnalyzer(),
        ImageResolver([tmp_path]),
        candidate_mode="all",
    )
    chunk = _chunk("low.jpg")
    chunk["metadata"]["images"].append("invalid.jpg")

    result = enricher.enrich_chunk(chunk)

    assert "molecular_structures" not in result["metadata"]
    assert enricher.stats.rejected_low_confidence == 1
    assert enricher.stats.rejected_invalid_smiles == 1


def test_jsonl_output_is_non_destructive_and_idempotent(tmp_path: Path) -> None:
    image = tmp_path / "abc.jpg"
    image.write_bytes(b"fake-image")
    source = tmp_path / "chunks.jsonl"
    first_output = tmp_path / "enriched.jsonl"
    second_output = tmp_path / "enriched-again.jsonl"
    source.write_text(json.dumps(_chunk("abc.jpg"), ensure_ascii=False) + "\n", encoding="utf-8")

    def make_enricher() -> MolecularStructureEnricher:
        return MolecularStructureEnricher(
            FakePredictor({"abc.jpg": {"smiles": "CC(=O)O", "confidence": 0.95}}),
            FakeAnalyzer(),
            ImageResolver([tmp_path]),
        )

    make_enricher().enrich_jsonl(source, first_output)
    make_enricher().enrich_jsonl(first_output, second_output)
    first = json.loads(first_output.read_text(encoding="utf-8"))
    second = json.loads(second_output.read_text(encoding="utf-8"))

    assert source.read_text(encoding="utf-8").count(ENRICHMENT_START) == 0
    assert first["text"].count(ENRICHMENT_START) == 1
    assert second["text"].count(ENRICHMENT_START) == 1


def test_image_resolver_refuses_ambiguous_basename(tmp_path: Path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "same.jpg").write_bytes(b"one")
    (two / "same.jpg").write_bytes(b"two")

    assert ImageResolver([tmp_path]).resolve("same.jpg", {}) is None


def test_parse_path_maps_validates_shape() -> None:
    assert parse_path_maps(["/old=/new"]) == [("/old", "/new")]
    with pytest.raises(ValueError):
        parse_path_maps(["missing-separator"])
