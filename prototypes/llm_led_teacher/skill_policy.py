"""Canonical teaching Skill catalog and deterministic selection policy."""

from __future__ import annotations

from collections.abc import Mapping

from prototypes.llm_led_teacher.orchestration_models import TeachingContract


DEFAULT_SKILL_CATALOG: dict[str, str] = {
    "socratic_guidance": "一次推进一个关键步骤，用一个必要问题检查理解。",
    "error_diagnosis": "指出具体错误证据，再给最小可执行修正。",
    "concept_explanation": "先讲核心因果，再用贴近学生的例子解释。",
    "complete_solution": "按明确授权给出完整步骤、关键理由与最终结论。",
}


def available_skill_descriptors(
    catalog: Mapping[str, str],
) -> list[dict[str, str]]:
    """Return the exact names and descriptions exposed to the action model."""

    return [
        {"name": name, "description": catalog[name]}
        for name in sorted(catalog)
    ]


def select_skill_for_contract(contract: TeachingContract) -> str:
    """Choose a canonical Skill without expanding the teaching contract."""

    return {
        "concept_explanation": "concept_explanation",
        "one_step": "socratic_guidance",
        "guided_solution": "socratic_guidance",
        "full_solution": "complete_solution",
    }.get(contract.response_mode.value, "")

