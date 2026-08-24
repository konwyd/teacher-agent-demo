"""Offline semantic-family ranking for isolated CandidateExtractorV2 Route B."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Mapping, Protocol, Sequence


MODEL_ID = "BAAI/bge-m3"
MODEL_LICENSE = "MIT"
EMBEDDING_DIMENSION = 1024
DEFAULT_MODEL_PATH = Path("/home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3")


class RouteBEmbeddingError(ValueError):
    """Raised when Route B embeddings are unavailable or invalid."""


class SemanticFamily(str, Enum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    LEARNING_GOAL = "learning_goal"
    DIFFICULTY = "difficulty"
    TOPIC_PROGRESS = "topic_progress"
    TASK_STATE = "task_state"
    LEARNING_EPISODE = "learning_episode"
    BEHAVIOR_EVIDENCE = "behavior_evidence"


class EmbeddingBackend(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise NotImplementedError


@dataclass(frozen=True)
class FamilyMatch:
    family: SemanticFamily
    similarity: float


DEFAULT_PROTOTYPES: Mapping[SemanticFamily, tuple[str, ...]] = {
    SemanticFamily.PROFILE: (
        "学生陈述自己的姓名年级或身份信息",
        "学生说明希望使用的称呼",
    ),
    SemanticFamily.PREFERENCE: (
        "学生表达偏好的讲解方式或学习节奏",
        "学生希望教师调整说明形式",
    ),
    SemanticFamily.LEARNING_GOAL: (
        "学生说明想要达到的学习目标",
        "学生提出需要完成的学习计划",
    ),
    SemanticFamily.DIFFICULTY: (
        "学生陈述正在遇到的学习困难",
        "学生说明某个知识点难以理解",
    ),
    SemanticFamily.TOPIC_PROGRESS: (
        "学生说明某个主题的学习进展",
        "学生陈述已经掌握或尚未掌握的内容",
    ),
    SemanticFamily.TASK_STATE: (
        "学生说明当前正在处理的学习任务",
        "学生提出当前需要完成的题目目标",
    ),
    SemanticFamily.LEARNING_EPISODE: (
        "学生描述一次具体的学习经历",
        "学生陈述某次课堂或练习中的情况",
    ),
    SemanticFamily.BEHAVIOR_EVIDENCE: (
        "学生表现出可观察的学习行为或结果",
        "学生陈述经过验证的学习表现",
    ),
}
DEFAULT_THRESHOLDS: Mapping[SemanticFamily, float] = {
    family: 0.45 for family in SemanticFamily
}


class BgeM3Embedder:
    """Lazy wrapper around an existing local BGE-M3 snapshot."""

    def __init__(
        self, model_path: str | Path = DEFAULT_MODEL_PATH, device: str = "cpu"
    ):
        path = Path(model_path)
        if not path.is_dir():
            raise RouteBEmbeddingError(
                f"local BGE-M3 model directory not found: {path}"
            )
        self.model_path = path
        self.device = device
        self._model = None

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                str(self.model_path), device=self.device, local_files_only=True
            )
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()


class SemanticFamilyRanker:
    """Rank normalized clauses against cached generic semantic prototypes."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        prototypes: Mapping[SemanticFamily, Sequence[str]] | None = None,
        thresholds: Mapping[SemanticFamily, float] | None = None,
    ):
        self._backend = backend
        prototype_source = DEFAULT_PROTOTYPES if prototypes is None else prototypes
        self._prototypes = self._validate_prototypes(prototype_source)
        self._thresholds = self._build_thresholds(thresholds)
        self._prototype_vectors: (
            Mapping[SemanticFamily, tuple[tuple[float, ...], ...]] | None
        ) = None
        self._dimension: int | None = None
        self.encode_call_count = 0
        self.encoded_text_count = 0

    @staticmethod
    def _validate_prototypes(
        prototypes: Mapping[SemanticFamily, Sequence[str]],
    ) -> Mapping[SemanticFamily, tuple[str, ...]]:
        unknown_families = set(prototypes).difference(SemanticFamily)
        if unknown_families:
            raise RouteBEmbeddingError(
                "semantic-family prototypes contain an unknown family"
            )

        validated: dict[SemanticFamily, tuple[str, ...]] = {}
        for family in SemanticFamily:
            if family not in prototypes:
                continue
            texts = tuple(prototypes[family])
            if not texts or any(
                not isinstance(text, str) or not text.strip() for text in texts
            ):
                raise RouteBEmbeddingError(
                    f"semantic-family {family.value} needs non-empty prototypes"
                )
            validated[family] = texts
        if not validated:
            raise RouteBEmbeddingError(
                "at least one semantic-family prototype is required"
            )
        return validated

    @staticmethod
    def _build_thresholds(
        thresholds: Mapping[SemanticFamily, float] | None,
    ) -> Mapping[SemanticFamily, float]:
        values = dict(DEFAULT_THRESHOLDS)
        if thresholds is not None:
            unknown_families = set(thresholds).difference(SemanticFamily)
            if unknown_families:
                raise RouteBEmbeddingError(
                    "semantic-family thresholds contain an unknown family"
                )
            values.update(thresholds)
        for family, value in values.items():
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise RouteBEmbeddingError(
                    f"semantic-family threshold for {family.value} must be finite and between -1 and 1"
                )
        return values

    def rank(self, texts: Sequence[str]) -> tuple[tuple[FamilyMatch, ...], ...]:
        clauses = tuple(texts)
        if any(not isinstance(text, str) for text in clauses):
            raise RouteBEmbeddingError("clause text must be a string")
        if not clauses:
            return ()

        prototype_vectors = self._load_prototype_vectors()
        clause_vectors = self._encode_and_normalize(clauses)
        family_order = {family: index for index, family in enumerate(SemanticFamily)}
        rows: list[tuple[FamilyMatch, ...]] = []
        for clause_vector in clause_vectors:
            matches = [
                FamilyMatch(
                    family=family,
                    similarity=max(
                        self._dot(clause_vector, prototype) for prototype in vectors
                    ),
                )
                for family, vectors in prototype_vectors.items()
            ]
            admitted = [
                match
                for match in matches
                if match.similarity >= self._thresholds[match.family]
            ]
            admitted.sort(
                key=lambda match: (-match.similarity, family_order[match.family])
            )
            rows.append(tuple(admitted))
        return tuple(rows)

    def _load_prototype_vectors(
        self,
    ) -> Mapping[SemanticFamily, tuple[tuple[float, ...], ...]]:
        if self._prototype_vectors is not None:
            return self._prototype_vectors

        indexed_prototypes = [
            (family, text)
            for family in SemanticFamily
            for text in self._prototypes.get(family, ())
        ]
        vectors = self._encode_and_normalize(
            tuple(text for _, text in indexed_prototypes)
        )
        grouped: dict[SemanticFamily, list[tuple[float, ...]]] = {
            family: [] for family in self._prototypes
        }
        for (family, _), vector in zip(indexed_prototypes, vectors, strict=True):
            grouped[family].append(vector)
        self._prototype_vectors = {
            family: tuple(grouped[family])
            for family in SemanticFamily
            if family in grouped
        }
        return self._prototype_vectors

    def _encode_and_normalize(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        self.encode_call_count += 1
        self.encoded_text_count += len(texts)
        vectors = self._backend.encode(texts)
        if len(vectors) != len(texts):
            raise RouteBEmbeddingError(
                "embedding backend returned an unexpected vector count"
            )
        return tuple(self._normalize_vector(vector) for vector in vectors)

    def _normalize_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as exc:
            raise RouteBEmbeddingError(
                "embedding vectors must contain numeric values"
            ) from exc
        if not values:
            raise RouteBEmbeddingError("embedding vectors must not be empty")
        if not all(math.isfinite(value) for value in values):
            raise RouteBEmbeddingError(
                "embedding vectors must contain only finite values"
            )
        if self._dimension is None:
            self._dimension = len(values)
        elif len(values) != self._dimension:
            raise RouteBEmbeddingError("embedding vector dimension mismatch")
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isfinite(norm) or norm == 0.0:
            raise RouteBEmbeddingError(
                "embedding vectors must have a finite non-zero norm"
            )
        return tuple(value / norm for value in values)

    @staticmethod
    def _dot(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(first * second for first, second in zip(left, right, strict=True))
