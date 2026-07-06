"""Graph RAG mock: concept graph backed by local JSON files.

第一版用 data/sample/*.json 模拟 Neo4j 概念图检索。
接口保持稳定，后续换 Neo4j 实现时上层不需要改动。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

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


@lru_cache
def get_retriever() -> JsonGraphRetriever:
    """进程级单例。"""
    return JsonGraphRetriever()
