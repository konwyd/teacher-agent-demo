"""Pedagogical Graph RAG mock backed by data/sample/pedagogical_graph.json.

This retriever returns teaching-decision evidence, not answer knowledge.
It is intentionally read-only and local so the workflow can later swap the
backend for Neo4j/Qdrant without changing the upper-level contract.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"

OLD_CONCEPT_ALIASES = {
    "c_linear": "linear_equation",
    "c_eq_prop": "equation_properties",
    "c_move": "move_terms",
    "c_bracket": "remove_parentheses",
    "c_frac": "fractional_coefficient_equation",
}


class PedagogicalMisconception(BaseModel):
    misconception_id: str
    description: str
    example_wrong_step: str = ""
    diagnostic_signal: str = ""


class PedagogicalDiagnosticQuestion(BaseModel):
    question_id: str
    prompt: str
    checks: list[str] = Field(default_factory=list)
    expected_signal: str = ""


class PedagogicalHintLevel(BaseModel):
    level: int
    hint: str


class PedagogicalExercise(BaseModel):
    exercise_id: str
    stem: str
    difficulty: int = 1
    teaching_focus: str = ""


class PedagogicalConceptContext(BaseModel):
    concept_id: str
    name: str
    prerequisites: list[str] = Field(default_factory=list)
    common_misconceptions: list[PedagogicalMisconception] = Field(default_factory=list)
    diagnostic_questions: list[PedagogicalDiagnosticQuestion] = Field(default_factory=list)
    hint_levels: list[PedagogicalHintLevel] = Field(default_factory=list)
    recommended_exercises: list[PedagogicalExercise] = Field(default_factory=list)
    mastery_based_strategy: dict[str, str] = Field(default_factory=dict)


class JsonPedagogicalRetriever:
    """Local JSON retriever for pedagogical teaching signals."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        raw = json.loads((data_dir / "pedagogical_graph.json").read_text(encoding="utf-8"))
        self._concepts: dict[str, PedagogicalConceptContext] = {
            c["concept_id"]: PedagogicalConceptContext.model_validate(c)
            for c in raw.get("concepts", [])
        }

    def _normalize_id(self, concept_id_or_name: str) -> str:
        value = concept_id_or_name.strip()
        if value in OLD_CONCEPT_ALIASES:
            return OLD_CONCEPT_ALIASES[value]
        return value

    def get_concept_context(self, concept_id_or_name: str) -> PedagogicalConceptContext | None:
        """Find a concept by new id, legacy concept id, exact name, or name inside a message."""
        value = self._normalize_id(concept_id_or_name)
        if value in self._concepts:
            return self._concepts[value]
        for concept in self._concepts.values():
            if concept.name == value or concept.name in value:
                return concept
        return None

    def get_prerequisites(self, concept_id_or_name: str) -> list[str]:
        context = self.get_concept_context(concept_id_or_name)
        return list(context.prerequisites) if context else []

    def get_misconceptions(self, concept_id_or_name: str) -> list[str]:
        context = self.get_concept_context(concept_id_or_name)
        if not context:
            return []
        return [m.description for m in context.common_misconceptions]

    def get_diagnostic_questions(self, concept_id_or_name: str) -> list[str]:
        context = self.get_concept_context(concept_id_or_name)
        if not context:
            return []
        return [q.prompt for q in context.diagnostic_questions]

    def get_hint_levels(self, concept_id_or_name: str) -> list[str]:
        context = self.get_concept_context(concept_id_or_name)
        if not context:
            return []
        return [h.hint for h in sorted(context.hint_levels, key=lambda item: item.level)]

    def get_recommended_exercises(self, concept_id_or_name: str) -> list[str]:
        context = self.get_concept_context(concept_id_or_name)
        if not context:
            return []
        return [e.stem for e in sorted(context.recommended_exercises, key=lambda item: item.difficulty)]

    def get_mastery_based_strategy(self, concept_id_or_name: str, mastery_score: float) -> str:
        context = self.get_concept_context(concept_id_or_name)
        if not context:
            return ""
        if mastery_score < 0.3:
            key = "low"
        elif mastery_score < 0.7:
            key = "medium"
        else:
            key = "high"
        return context.mastery_based_strategy.get(key, "")


@lru_cache
def get_pedagogical_retriever() -> JsonPedagogicalRetriever:
    """Process-level singleton."""
    return JsonPedagogicalRetriever()
