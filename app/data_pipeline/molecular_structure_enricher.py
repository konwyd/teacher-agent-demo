"""Add MolScribe molecular-structure semantics to OCR/RAG JSONL chunks.

The existing OCR text and ``metadata.images`` links remain authoritative.  This
module only appends a conservative, machine-generated enrichment when all of
the following are true:

* the chunk is a chemistry chunk;
* the image is selected by keyword context or an explicit allow-list;
* MolScribe returns a SMILES string above the configured confidence threshold;
* RDKit can parse and sanitize the SMILES string.

Heavy optional dependencies are imported lazily so the normal Teacher Agent
runtime and its offline test suite do not need MolScribe or RDKit installed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlparse


ENRICHMENT_START = "<!-- molecular-structure-enrichment:start -->"
ENRICHMENT_END = "<!-- molecular-structure-enrichment:end -->"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
CHEMISTRY_SUBJECTS = {"化学", "chemistry", "chemical science"}

# A deliberately narrow gate. MolScribe is not an out-of-domain image
# classifier and may otherwise produce plausible-looking molecules for charts
# or laboratory apparatus.
MOLECULAR_CONTEXT_KEYWORDS = (
    "分子结构",
    "化学结构",
    "结构式",
    "结构简式",
    "化学式",
    "分子式",
    "键线式",
    "电子式",
    "路易斯结构",
    "分子模型",
    "结构模型",
    "球棍模型",
    "空间填充模型",
    "有机物结构",
    "有机化合物结构",
    "官能团",
    "碳骨架",
    "同分异构体",
    "结构如下",
    "结构如图",
)


class PredictionBackend(Protocol):
    """Minimal interface implemented by the MolScribe backend."""

    def predict(self, image_path: Path) -> Mapping[str, Any]:
        """Return MolScribe-compatible output for one image."""


class StructureAnalyzer(Protocol):
    """Validate and enrich a predicted SMILES string."""

    def analyze(self, smiles: str) -> Mapping[str, Any] | None:
        """Return normalized chemistry fields, or ``None`` when invalid."""


@dataclass(slots=True)
class MolecularStructureRecord:
    """Compact structure semantics attached to one existing image identifier."""

    image_id: str
    smiles: str
    canonical_smiles: str
    molecular_formula: str
    confidence: float | None
    atom_count: int
    bond_count: int
    ring_count: int
    functional_groups: list[str] = field(default_factory=list)
    search_text: str = ""
    model: str = "MolScribe"
    image_path: str = ""
    molfile: str = ""

    def compact_metadata(self) -> dict[str, Any]:
        """Return JSON-friendly fields suitable for chunk metadata."""

        data = asdict(self)
        # The molfile is too verbose for Chroma metadata and does not help text
        # retrieval; the source image and SMILES remain available for review.
        data.pop("molfile", None)
        return data


@dataclass(slots=True)
class ChunkEnrichmentStats:
    chunks_read: int = 0
    chemistry_chunks: int = 0
    chunks_enriched: int = 0
    image_references: int = 0
    candidate_images: int = 0
    images_resolved: int = 0
    predictions_run: int = 0
    predictions_from_cache: int = 0
    structures_accepted: int = 0
    rejected_low_confidence: int = 0
    rejected_invalid_smiles: int = 0
    missing_images: int = 0
    prediction_errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class MolScribePredictionBackend:
    """Lazy MolScribe loader with an official Hugging Face checkpoint."""

    DEFAULT_REPO_ID = "yujieq/MolScribe"
    DEFAULT_FILENAME = "swin_base_char_aux_1m.pth"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str = "auto",
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser() if checkpoint else None
        self.device_name = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from molscribe import MolScribe
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "MolScribe is not installed. Use scripts/install_molscribe_env.sh "
                "and run this command with .venv-molscribe/bin/python."
            ) from exc

        checkpoint = self.checkpoint
        if checkpoint is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover - optional runtime
                raise RuntimeError("huggingface_hub is required to download the MolScribe checkpoint") from exc
            checkpoint = Path(
                hf_hub_download(repo_id=self.DEFAULT_REPO_ID, filename=self.DEFAULT_FILENAME)
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MolScribe checkpoint not found: {checkpoint}")

        if self.device_name == "auto":
            selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = self.device_name
        self._model = MolScribe(checkpoint, device=torch.device(selected_device))
        return self._model

    def predict(self, image_path: Path) -> Mapping[str, Any]:
        output = self._load().predict_image_file(
            str(image_path),
            return_atoms_bonds=True,
            return_confidence=True,
        )
        if not isinstance(output, Mapping):
            raise RuntimeError(f"MolScribe returned an unexpected value for {image_path}")
        return output


class RDKitStructureAnalyzer:
    """Sanitize SMILES and derive conservative retrieval-friendly chemistry."""

    # Ordered from specific to broad; SMARTS compilation is delayed until use.
    FUNCTIONAL_GROUP_SMARTS: tuple[tuple[str, str], ...] = (
        ("羧基", "[CX3](=O)[OX2H1]"),
        ("酯基", "[CX3](=O)[OX2][#6]"),
        ("醛基", "[CX3H1](=O)[#6,H]"),
        ("酮羰基", "[#6][CX3](=O)[#6]"),
        ("醇羟基", "[OX2H][CX4]"),
        ("醚键", "[OD2]([#6])[#6]"),
        ("氨基", "[NX3;H2,H1;!$(NC=O)]"),
        ("碳碳双键", "[CX3]=[CX3]"),
        ("碳碳三键", "[CX2]#[CX2]"),
        ("苯环", "c1ccccc1"),
    )

    def __init__(self) -> None:
        self._patterns: list[tuple[str, Any]] | None = None

    @staticmethod
    def _imports() -> tuple[Any, Any]:
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "RDKit is not installed. Run scripts/install_molscribe_env.sh first."
            ) from exc
        return Chem, rdMolDescriptors

    def _compiled_patterns(self, chem: Any) -> list[tuple[str, Any]]:
        if self._patterns is None:
            patterns: list[tuple[str, Any]] = []
            for label, smarts in self.FUNCTIONAL_GROUP_SMARTS:
                pattern = chem.MolFromSmarts(smarts)
                if pattern is not None:
                    patterns.append((label, pattern))
            self._patterns = patterns
        return self._patterns

    def analyze(self, smiles: str) -> Mapping[str, Any] | None:
        chem, descriptors = self._imports()
        value = (smiles or "").strip()
        if not value:
            return None
        try:
            molecule = chem.MolFromSmiles(value, sanitize=True)
        except Exception:
            return None
        if molecule is None:
            return None

        canonical = chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        formula = descriptors.CalcMolFormula(molecule)
        groups = [
            label
            for label, pattern in self._compiled_patterns(chem)
            if molecule.HasSubstructMatch(pattern)
        ]
        atom_count = int(molecule.GetNumAtoms())
        bond_count = int(molecule.GetNumBonds())
        ring_count = int(descriptors.CalcNumRings(molecule))

        details = [f"规范SMILES {canonical}", f"分子式 {formula}"]
        if groups:
            details.append("官能团或结构特征 " + "、".join(groups))
        if ring_count:
            details.append(f"环数量 {ring_count}")
        search_text = "分子结构识别：" + "；".join(details) + "。"
        return {
            "canonical_smiles": canonical,
            "molecular_formula": formula,
            "atom_count": atom_count,
            "bond_count": bond_count,
            "ring_count": ring_count,
            "functional_groups": groups,
            "search_text": search_text,
        }


class ImageResolver:
    """Resolve existing ``metadata.images`` values without changing their IDs."""

    def __init__(
        self,
        image_roots: Sequence[str | Path] = (),
        path_maps: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.image_roots = [Path(root).expanduser().resolve() for root in image_roots]
        self.path_maps = [(old.rstrip("/"), new.rstrip("/")) for old, new in path_maps]
        self._basename_index: dict[str, list[Path]] | None = None

    def _mapped_path(self, value: str) -> Path:
        mapped = value
        for old, new in self.path_maps:
            if mapped == old or mapped.startswith(old + "/"):
                mapped = new + mapped[len(old) :]
                break
        return Path(mapped).expanduser()

    def _build_index(self) -> dict[str, list[Path]]:
        if self._basename_index is not None:
            return self._basename_index
        index: dict[str, list[Path]] = {}
        for root in self.image_roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                    index.setdefault(path.name, []).append(path)
        self._basename_index = index
        return index

    def resolve(self, image_ref: str, metadata: Mapping[str, Any]) -> Path | None:
        ref = (image_ref or "").strip()
        if not ref:
            return None
        parsed = urlparse(ref)
        if parsed.scheme in {"http", "https"}:
            return None

        direct = self._mapped_path(ref)
        if direct.is_file():
            return direct.resolve()

        source_md = str(metadata.get("source_md", "") or "")
        if source_md:
            source_path = self._mapped_path(source_md)
            candidates = (
                source_path.parent / ref,
                source_path.parent / "images" / Path(ref).name,
                source_path.parent.parent / "images" / Path(ref).name,
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.resolve()

        for root in self.image_roots:
            candidate = root / ref
            if candidate.is_file():
                return candidate.resolve()

        matches = self._build_index().get(Path(ref).name, [])
        # Hash-like MinerU image IDs should be globally unique. Refuse an
        # ambiguous basename instead of attaching the wrong structure.
        if len(matches) == 1:
            return matches[0].resolve()
        return None


class JsonlPredictionCache:
    """Append-only cache keyed by path, size, mtime and model name."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path).expanduser() if path else None
        self._records: dict[str, Mapping[str, Any]] = {}
        if self.path and self.path.is_file():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                        self._records[str(item["cache_key"])] = item["prediction"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue

    @staticmethod
    def key(image_path: Path) -> str:
        stat = image_path.stat()
        value = f"MolScribe|{image_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def get(self, image_path: Path) -> Mapping[str, Any] | None:
        return self._records.get(self.key(image_path))

    def put(self, image_path: Path, prediction: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        cache_key = self.key(image_path)
        serializable = _json_safe(dict(prediction))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"cache_key": cache_key, "image_path": str(image_path), "prediction": serializable},
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._records[cache_key] = serializable


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def normalize_image_references(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in stripped.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def is_chemistry_chunk(metadata: Mapping[str, Any]) -> bool:
    subject = str(metadata.get("subject", "") or "").strip().lower()
    return subject in CHEMISTRY_SUBJECTS


def has_molecular_context(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    return any(keyword in compact for keyword in MOLECULAR_CONTEXT_KEYWORDS)


def remove_existing_enrichment(text: str) -> str:
    pattern = re.compile(
        re.escape(ENRICHMENT_START) + r".*?" + re.escape(ENRICHMENT_END),
        flags=re.DOTALL,
    )
    return pattern.sub("", text or "").rstrip()


def render_enrichment(records: Sequence[MolecularStructureRecord]) -> str:
    lines = [ENRICHMENT_START, "## 分子结构图机器识别"]
    for record in records:
        confidence = "未知" if record.confidence is None else f"{record.confidence:.3f}"
        lines.extend(
            [
                f"- 图片索引：`{record.image_id}`",
                f"  - 规范 SMILES：`{record.canonical_smiles}`",
                f"  - 分子式：`{record.molecular_formula}`",
                "  - 官能团或结构特征："
                + ("、".join(record.functional_groups) if record.functional_groups else "未标注"),
                f"  - MolScribe 置信度：{confidence}",
                f"  - 可检索描述：{record.search_text}",
            ]
        )
    lines.append(ENRICHMENT_END)
    return "\n".join(lines)


class MolecularStructureEnricher:
    """Orchestrate conservative MolScribe enrichment for existing chunks."""

    def __init__(
        self,
        predictor: PredictionBackend,
        analyzer: StructureAnalyzer,
        resolver: ImageResolver,
        *,
        min_confidence: float = 0.80,
        min_atoms: int = 2,
        candidate_mode: str = "keyword",
        allowed_image_ids: Iterable[str] = (),
        cache: JsonlPredictionCache | None = None,
    ) -> None:
        if candidate_mode not in {"keyword", "all", "allowlist"}:
            raise ValueError("candidate_mode must be keyword, all, or allowlist")
        self.predictor = predictor
        self.analyzer = analyzer
        self.resolver = resolver
        self.min_confidence = min_confidence
        self.min_atoms = min_atoms
        self.candidate_mode = candidate_mode
        self.allowed_image_ids = {Path(item.strip()).name for item in allowed_image_ids if item.strip()}
        self.cache = cache or JsonlPredictionCache(None)
        self.stats = ChunkEnrichmentStats()
        self._accepted_by_path: dict[Path, MolecularStructureRecord | None] = {}

    def _is_candidate(self, image_id: str, text: str) -> bool:
        basename = Path(image_id).name
        if basename in self.allowed_image_ids:
            return True
        if self.candidate_mode == "allowlist":
            return False
        if self.candidate_mode == "all":
            return True
        return has_molecular_context(text)

    @staticmethod
    def _confidence(prediction: Mapping[str, Any]) -> float | None:
        value = prediction.get("confidence")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _recognize(self, image_id: str, image_path: Path) -> MolecularStructureRecord | None:
        if image_path in self._accepted_by_path:
            cached_record = self._accepted_by_path[image_path]
            if cached_record is None:
                return None
            data = asdict(cached_record)
            data["image_id"] = image_id
            return MolecularStructureRecord(**data)

        prediction = self.cache.get(image_path)
        if prediction is None:
            try:
                prediction = self.predictor.predict(image_path)
                self.stats.predictions_run += 1
                self.cache.put(image_path, prediction)
            except Exception:
                self.stats.prediction_errors += 1
                self._accepted_by_path[image_path] = None
                return None
        else:
            self.stats.predictions_from_cache += 1

        confidence = self._confidence(prediction)
        if confidence is not None and confidence < self.min_confidence:
            self.stats.rejected_low_confidence += 1
            self._accepted_by_path[image_path] = None
            return None

        smiles = str(prediction.get("smiles", "") or "").strip()
        analysis = self.analyzer.analyze(smiles)
        if analysis is None or int(analysis.get("atom_count", 0)) < self.min_atoms:
            self.stats.rejected_invalid_smiles += 1
            self._accepted_by_path[image_path] = None
            return None

        record = MolecularStructureRecord(
            image_id=image_id,
            image_path=str(image_path),
            smiles=smiles,
            canonical_smiles=str(analysis["canonical_smiles"]),
            molecular_formula=str(analysis["molecular_formula"]),
            confidence=confidence,
            atom_count=int(analysis["atom_count"]),
            bond_count=int(analysis["bond_count"]),
            ring_count=int(analysis["ring_count"]),
            functional_groups=list(analysis.get("functional_groups", [])),
            search_text=str(analysis.get("search_text", "")),
            molfile=str(prediction.get("molfile", "") or ""),
        )
        self._accepted_by_path[image_path] = record
        self.stats.structures_accepted += 1
        return record

    def enrich_chunk(self, chunk: Mapping[str, Any]) -> dict[str, Any]:
        self.stats.chunks_read += 1
        output = dict(chunk)
        metadata = dict(chunk.get("metadata", {}) or {})
        output["metadata"] = metadata
        if not is_chemistry_chunk(metadata):
            return output
        self.stats.chemistry_chunks += 1

        text = remove_existing_enrichment(str(chunk.get("text", "") or ""))
        image_refs = normalize_image_references(metadata.get("images"))
        self.stats.image_references += len(image_refs)
        records: list[MolecularStructureRecord] = []
        seen_ids: set[str] = set()
        for image_ref in image_refs:
            image_id = Path(image_ref).name
            if not image_id or image_id in seen_ids or not self._is_candidate(image_id, text):
                continue
            seen_ids.add(image_id)
            self.stats.candidate_images += 1
            image_path = self.resolver.resolve(image_ref, metadata)
            if image_path is None:
                self.stats.missing_images += 1
                continue
            self.stats.images_resolved += 1
            record = self._recognize(image_id, image_path)
            if record is not None:
                records.append(record)

        if records:
            output["text"] = text + "\n\n" + render_enrichment(records)
            metadata["molecular_structures"] = [record.compact_metadata() for record in records]
            # Chroma supports scalar metadata reliably; this compact JSON field
            # survives ingestion even when nested metadata is flattened.
            metadata["molecular_structures_json"] = json.dumps(
                metadata["molecular_structures"], ensure_ascii=False, separators=(",", ":")
            )
            self.stats.chunks_enriched += 1
        else:
            output["text"] = text
            metadata.pop("molecular_structures", None)
            metadata.pop("molecular_structures_json", None)
        return output

    def enrich_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        structures_path: str | Path | None = None,
    ) -> ChunkEnrichmentStats:
        source = Path(input_path)
        destination = Path(output_path)
        if source.resolve() == destination.resolve():
            raise ValueError("Refusing to overwrite the source JSONL; choose a separate output path")
        destination.parent.mkdir(parents=True, exist_ok=True)

        sidecar = Path(structures_path) if structures_path else None
        sidecar_handle = None
        if sidecar:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar_handle = sidecar.open("w", encoding="utf-8")
        try:
            with source.open("r", encoding="utf-8") as reader, destination.open(
                "w", encoding="utf-8"
            ) as writer:
                for line_number, line in enumerate(reader, start=1):
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
                    enriched = self.enrich_chunk(chunk)
                    writer.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    if sidecar_handle:
                        for structure in enriched.get("metadata", {}).get("molecular_structures", []):
                            sidecar_handle.write(
                                json.dumps(
                                    {"chunk_id": enriched.get("_id"), **structure},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
        finally:
            if sidecar_handle:
                sidecar_handle.close()
        return self.stats


def parse_path_maps(values: Sequence[str]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid path map {value!r}; expected OLD_PREFIX=NEW_PREFIX")
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError(f"Invalid path map {value!r}; both prefixes are required")
        mappings.append((old, new))
    return mappings


def read_allowlist(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
