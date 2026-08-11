"""Graph RAG mock: concept graph backed by local JSON files.

第一版用 data/sample/*.json 模拟 Neo4j 概念图检索。
接口保持稳定，后续换 Neo4j 实现时上层不需要改动。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


class ConceptNode(BaseModel):
    id: str
    name: str
    grade: str = ""
    subject: str = ""
    description: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class Exercise(BaseModel):
    id: str
    stem: str
    answer: str
    solution_steps: list[str] = Field(default_factory=list)
    concept_ids: list[str] = Field(default_factory=list)
    difficulty: int = 1
    common_misconceptions: list[str] = Field(default_factory=list)


class Misconception(BaseModel):
    id: str
    concept_id: str
    description: str
    example: str = ""


class JsonGraphRetriever:
    """本地 JSON 模拟的概念图检索器（Graph RAG mock）。"""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self._concepts: dict[str, ConceptNode] = {
            c["id"]: ConceptNode.model_validate(c)
            for c in json.loads((data_dir / "concepts.json").read_text(encoding="utf-8"))
        }
        self._exercises: list[Exercise] = [
            Exercise.model_validate(e)
            for e in json.loads((data_dir / "exercises.json").read_text(encoding="utf-8"))
        ]
        self._misconceptions: list[Misconception] = [
            Misconception.model_validate(m)
            for m in json.loads((data_dir / "misconceptions.json").read_text(encoding="utf-8"))
        ]

    def get_concept(self, name_or_id: str) -> ConceptNode | None:
        """按 id 精确查找，或按名称（含子串）查找。"""
        if name_or_id in self._concepts:
            return self._concepts[name_or_id]
        for node in self._concepts.values():
            if node.name == name_or_id or node.name in name_or_id:
                return node
        return None

    def get_prerequisites(self, concept_id: str, depth: int = 1) -> list[ConceptNode]:
        """返回 depth 跳以内的前置知识点（去重，近的在前）。"""
        result: list[ConceptNode] = []
        seen: set[str] = set()
        frontier = [concept_id]
        for _ in range(depth):
            next_frontier: list[str] = []
            for cid in frontier:
                node = self._concepts.get(cid)
                if node is None:
                    continue
                for pid in node.prerequisites:
                    if pid not in seen and pid in self._concepts:
                        seen.add(pid)
                        result.append(self._concepts[pid])
                        next_frontier.append(pid)
            frontier = next_frontier
        return result

    def get_related_exercises(self, concept_id: str) -> list[Exercise]:
        """返回涉及该知识点的练习，按难度升序。"""
        related = [e for e in self._exercises if concept_id in e.concept_ids]
        return sorted(related, key=lambda e: e.difficulty)

    def get_misconceptions(self, concept_id: str) -> list[Misconception]:
        return [m for m in self._misconceptions if m.concept_id == concept_id]


class RetrievalEvidence(BaseModel):
    """One scored piece of teaching-decision evidence."""

    node_id: str
    node_type: str
    relation_path: list[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    content: dict[str, Any] = Field(default_factory=dict)


class PedagogicalEvidenceResult(BaseModel):
    """Ranked evidence returned by the local Pedagogical Graph."""

    retrieval_backend: str = "local_json_graph"
    concept_id: str = ""
    evidence: list[RetrievalEvidence] = Field(default_factory=list)
    mastery_strategy: str = ""


_PEDAGOGICAL_ID_FIELDS = {
    "concepts": ("concept_id", "Concept"),
    "skills": ("skill_id", "Skill"),
    "misconceptions": ("misconception_id", "Misconception"),
    "diagnostic_questions": ("question_id", "DiagnosticQuestion"),
    "hints": ("hint_id", "Hint"),
    "worked_examples": ("example_id", "WorkedExample"),
    "practice_problems": ("problem_id", "PracticeProblem"),
    "teaching_strategies": ("strategy_id", "TeachingStrategy"),
    "mastery_thresholds": ("threshold_id", "MasteryThreshold"),
}

_PEDAGOGICAL_CONCEPT_ALIASES = {
    "c_linear": "linear_equation",
    "linear_equation_basic": "linear_equation",
    "c_eq_prop": "equation_properties",
    "c_move": "move_terms",
    "c_divide": "divide_by_coefficient",
    "c_bracket": "remove_parentheses",
    "c_frac": "fractional_coefficient_equation",
}

_MISCONCEPTION_MARKERS = ("变号", "两边", "每一项", "分母", "系数", "符号", "括号", "漏乘")


class LocalPedagogicalGraphRetriever:
    """Traverse and score the normalized local Pedagogical Graph."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        raw = json.loads((data_dir / "pedagogical_graph.json").read_text(encoding="utf-8"))
        nodes = raw.get("nodes", {})
        self._nodes: dict[str, dict[str, Any]] = {}
        self._node_types: dict[str, str] = {}
        for collection, (id_field, node_type) in _PEDAGOGICAL_ID_FIELDS.items():
            for node in nodes.get(collection, []):
                node_id = str(node[id_field])
                self._nodes[node_id] = node
                self._node_types[node_id] = node_type
        self._edges: list[dict[str, Any]] = list(raw.get("edges", []))
        self._outgoing: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._incoming: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for edge in self._edges:
            self._outgoing.setdefault((str(edge["from"]), str(edge["type"])), []).append(edge)
            self._incoming.setdefault((str(edge["to"]), str(edge["type"])), []).append(edge)

    def _resolve_concept(self, concept_id_or_name: str) -> tuple[dict[str, Any] | None, float, str]:
        value = concept_id_or_name.strip()
        normalized = _PEDAGOGICAL_CONCEPT_ALIASES.get(value, value)
        node = self._nodes.get(normalized)
        if node and self._node_types.get(normalized) == "Concept":
            if value == normalized:
                return node, 1.0, "exact concept match"
            return node, 0.97, f"concept alias match: {value} -> {normalized}"
        for node_id, candidate in self._nodes.items():
            if self._node_types.get(node_id) != "Concept":
                continue
            name = str(candidate.get("name", ""))
            if name and (name == value or name in value):
                return candidate, 0.94, "exact concept name match"
        return None, 0.0, "concept not found"

    @staticmethod
    def _evidence(
        node: dict[str, Any],
        node_id: str,
        node_type: str,
        relation_path: list[str],
        score: float,
        reason: str,
    ) -> RetrievalEvidence:
        return RetrievalEvidence(
            node_id=node_id,
            node_type=node_type,
            relation_path=relation_path,
            score=round(min(1.0, max(0.0, score)), 4),
            reason=reason,
            content=dict(node),
        )

    def get_prerequisite_chain(
        self,
        concept_id_or_name: str,
        depth: int = 2,
    ) -> list[RetrievalEvidence]:
        concept, _, _ = self._resolve_concept(concept_id_or_name)
        if concept is None or depth <= 0:
            return []
        root_id = str(concept["concept_id"])
        result: list[RetrievalEvidence] = []
        seen = {root_id}
        frontier: list[tuple[str, list[str]]] = [(root_id, [f"Concept:{root_id}"])]
        for level in range(1, depth + 1):
            next_frontier: list[tuple[str, list[str]]] = []
            for current_id, path in frontier:
                for edge in self._outgoing.get((current_id, "REQUIRES"), []):
                    target_id = str(edge["to"])
                    if target_id in seen or target_id not in self._nodes:
                        continue
                    seen.add(target_id)
                    relation_path = path + ["REQUIRES", f"Concept:{target_id}"]
                    result.append(
                        self._evidence(
                            self._nodes[target_id],
                            target_id,
                            "Concept",
                            relation_path,
                            0.82 - (level - 1) * 0.08,
                            f"prerequisite chain match at depth {level}",
                        )
                    )
                    next_frontier.append((target_id, relation_path))
            frontier = next_frontier
        return sorted(result, key=lambda item: (-item.score, item.node_id))

    def _concept_misconception_ids(self, concept_id: str) -> list[str]:
        return [
            str(edge["to"])
            for edge in self._outgoing.get((concept_id, "HAS_MISCONCEPTION"), [])
            if str(edge["to"]) in self._nodes
        ]

    def _match_misconception(
        self,
        misconception: str | None,
        candidate_ids: list[str],
    ) -> str | None:
        if not misconception:
            return None
        value = misconception.strip()
        if value in candidate_ids:
            return value
        markers = [marker for marker in _MISCONCEPTION_MARKERS if marker in value]
        best_id: str | None = None
        best_overlap = 0
        for node_id in candidate_ids:
            node = self._nodes[node_id]
            text = " ".join(
                str(node.get(field, ""))
                for field in ("description", "example_wrong_step", "diagnostic_signal")
            )
            if value and value in text:
                return node_id
            overlap = sum(marker in text for marker in markers)
            if overlap > best_overlap:
                best_id, best_overlap = node_id, overlap
        return best_id

    def get_remediating_hints(self, misconception_id: str) -> list[RetrievalEvidence]:
        if self._node_types.get(misconception_id) != "Misconception":
            return []
        result: list[RetrievalEvidence] = []
        for edge in self._incoming.get((misconception_id, "REMEDIATES"), []):
            hint_id = str(edge["from"])
            hint = self._nodes.get(hint_id)
            if hint is None:
                continue
            result.append(
                self._evidence(
                    hint,
                    hint_id,
                    "Hint",
                    [
                        f"Misconception:{misconception_id}",
                        "REMEDIATES",
                        f"Hint:{hint_id}",
                    ],
                    0.96,
                    "current misconception exact match; remediation relation match",
                )
            )
        return sorted(result, key=lambda item: (-item.score, item.node_id))

    @staticmethod
    def _target_hint_level(mastery_score: float) -> int:
        if mastery_score < 0.3:
            return 1
        if mastery_score < 0.7:
            return 2
        return 3

    @staticmethod
    def _request_mode(request_type: str | None) -> str:
        value = request_type or ""
        if value in ("generate_easier_exercise", "request_easier_exercise", "easier"):
            return "easier"
        if value in ("generate_harder_exercise", "request_harder_exercise", "harder"):
            return "harder"
        if value in ("generate_similar_exercise", "request_similar_exercise", "similar"):
            return "similar"
        return "mastery"

    def _mastery_strategy(self, concept: dict[str, Any], mastery_score: float) -> str:
        band_name = "low" if mastery_score < 0.3 else "medium" if mastery_score < 0.7 else "high"
        for threshold_id in concept.get("mastery_thresholds", []):
            threshold = self._nodes.get(str(threshold_id), {})
            for band in threshold.get("bands", []):
                if band.get("band") == band_name:
                    return str(band.get("strategy_text", ""))
        return ""

    def retrieve_evidence(
        self,
        concept_id_or_name: str,
        *,
        misconception: str | None = None,
        mastery_score: float = 0.5,
        request_type: str | None = None,
        prerequisite_depth: int = 2,
    ) -> PedagogicalEvidenceResult:
        concept, concept_score, concept_reason = self._resolve_concept(concept_id_or_name)
        if concept is None:
            return PedagogicalEvidenceResult()
        concept_id = str(concept["concept_id"])
        evidence: list[RetrievalEvidence] = [
            self._evidence(
                concept,
                concept_id,
                "Concept",
                [f"Concept:{concept_id}"],
                concept_score,
                concept_reason,
            )
        ]
        evidence.extend(self.get_prerequisite_chain(concept_id, depth=prerequisite_depth))

        misconception_ids = self._concept_misconception_ids(concept_id)
        matched_misconception_id = self._match_misconception(misconception, misconception_ids)
        for misconception_id in misconception_ids:
            node = self._nodes[misconception_id]
            matched = misconception_id == matched_misconception_id
            evidence.append(
                self._evidence(
                    node,
                    misconception_id,
                    "Misconception",
                    [
                        f"Concept:{concept_id}",
                        "HAS_MISCONCEPTION",
                        f"Misconception:{misconception_id}",
                    ],
                    0.95 if matched else 0.7,
                    (
                        "current misconception exact or signal match"
                        if matched
                        else "concept-related misconception"
                    ),
                )
            )
            for edge in self._outgoing.get((misconception_id, "DIAGNOSE_WITH"), []):
                question_id = str(edge["to"])
                question = self._nodes.get(question_id)
                if question is None:
                    continue
                evidence.append(
                    self._evidence(
                        question,
                        question_id,
                        "DiagnosticQuestion",
                        [
                            f"Concept:{concept_id}",
                            "HAS_MISCONCEPTION",
                            f"Misconception:{misconception_id}",
                            "DIAGNOSE_WITH",
                            f"DiagnosticQuestion:{question_id}",
                        ],
                        0.9 if matched else 0.67,
                        (
                            "diagnostic question targets current misconception"
                            if matched
                            else "diagnostic question for a concept-related misconception"
                        ),
                    )
                )

        target_hint_level = self._target_hint_level(mastery_score)
        for hint_id in concept.get("leveled_hints", []):
            hint_id = str(hint_id)
            hint = self._nodes.get(hint_id)
            if hint is None:
                continue
            level = int(hint.get("level", 1))
            remediates_match = bool(
                matched_misconception_id
                and any(
                    str(edge["to"]) == matched_misconception_id
                    for edge in self._outgoing.get((hint_id, "REMEDIATES"), [])
                )
            )
            score = 0.62
            reasons = ["concept hint relation"]
            if level == target_hint_level:
                score += 0.22
                reasons.append("mastery threshold match")
            if remediates_match:
                score += 0.14
                reasons.append("current misconception remediation match")
            evidence.append(
                self._evidence(
                    hint,
                    hint_id,
                    "Hint",
                    [f"Concept:{concept_id}", "HINT_FOR", f"Hint:{hint_id}"],
                    score,
                    "; ".join(reasons),
                )
            )

        practice_ids = [str(item) for item in concept.get("practice_problems", [])]
        practices = [self._nodes[item] for item in practice_ids if item in self._nodes]
        if practices:
            difficulties = [int(item.get("difficulty", 1)) for item in practices]
            mode = self._request_mode(request_type)
            if mode == "easier" or (mode == "mastery" and mastery_score < 0.3):
                target_difficulty = min(difficulties)
                difficulty_label = "easy"
            elif mode == "harder" or (mode == "mastery" and mastery_score >= 0.7):
                target_difficulty = max(difficulties)
                difficulty_label = "harder_variant"
            else:
                target_difficulty = min(difficulties, key=lambda value: abs(value - 2))
                difficulty_label = "similar"
            for practice in practices:
                practice_id = str(practice["problem_id"])
                difficulty = int(practice.get("difficulty", 1))
                matches_difficulty = difficulty == target_difficulty
                score = 0.58 + (0.24 if matches_difficulty else 0.0)
                reasons = ["concept practice relation"]
                if matches_difficulty:
                    reasons.append(f"request difficulty match: {difficulty_label}")
                    reasons.append("mastery threshold match")
                if mode != "mastery" and matches_difficulty:
                    score += 0.08
                evidence.append(
                    self._evidence(
                        practice,
                        practice_id,
                        "PracticeProblem",
                        [
                            f"Concept:{concept_id}",
                            "PRACTICE_FOR",
                            f"PracticeProblem:{practice_id}",
                        ],
                        score,
                        "; ".join(reasons),
                    )
                )

        evidence.sort(key=lambda item: (-item.score, item.node_type, item.node_id))
        return PedagogicalEvidenceResult(
            concept_id=concept_id,
            evidence=evidence,
            mastery_strategy=self._mastery_strategy(concept, mastery_score),
        )


@lru_cache
def get_retriever() -> JsonGraphRetriever:
    """进程级单例。"""
    return JsonGraphRetriever()


@lru_cache
def get_local_pedagogical_graph_retriever() -> LocalPedagogicalGraphRetriever:
    """Process-level singleton for local traversal and evidence scoring."""
    return LocalPedagogicalGraphRetriever()
