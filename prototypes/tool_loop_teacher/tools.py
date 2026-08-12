"""Tool registry and adapters around existing teaching capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from app.agent.generic_math_reasoning import GenericMathReasoner
from app.agent.optional_verifier import LinearEquationVerifier, VerificationRequest
from app.agent.state import MathAnalysis, VerificationResult
from app.agent.student_model import StudentModel
from app.memory.schemas import EpisodeState
from app.retrieval.context_provider import (
    GraphContextProvider,
    GraphContextRequest,
    get_graph_context_provider,
)
from prototypes.tool_loop_teacher.models import ToolResult
from prototypes.tool_loop_teacher.skills import SkillCatalog


class NoArguments(BaseModel):
    pass


class LoadSkillArguments(BaseModel):
    name: str = Field(min_length=1)


class RetrieveContextArguments(BaseModel):
    concept_id: str = ""


class ReasonAboutProblemArguments(BaseModel):
    focus: str = ""


@dataclass
class ToolRuntimeContext:
    problem_context: str
    student_message: str
    analysis: MathAnalysis
    episode: EpisodeState
    previous_verified_step: str = ""
    graph_provider: GraphContextProvider | None = None
    reasoner: GenericMathReasoner | None = None
    skill_catalog: SkillCatalog | None = None


ToolHandler = Callable[[BaseModel, ToolRuntimeContext], dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: ToolHandler

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments_schema": self.arguments_model.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"duplicate tool: {definition.name}")
        self._tools[definition.name] = definition

    def descriptors(self) -> list[dict[str, Any]]:
        return [tool.descriptor() for tool in self._tools.values()]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolRuntimeContext,
    ) -> ToolResult:
        definition = self._tools.get(name)
        if definition is None:
            return ToolResult(
                tool_name=name,
                ok=False,
                error=f"unknown tool: {name}",
            )
        try:
            parsed = definition.arguments_model.model_validate(arguments)
            payload = definition.handler(parsed, context)
            return ToolResult(tool_name=name, payload=payload)
        except (ValidationError, ValueError, KeyError) as exc:
            return ToolResult(
                tool_name=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - tool failure becomes loop evidence
            return ToolResult(
                tool_name=name,
                ok=False,
                error=f"tool failed: {type(exc).__name__}: {exc}",
            )


def _verify_math_attempt(
    _arguments: BaseModel,
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    verifier = LinearEquationVerifier()
    request = VerificationRequest(
        problem_context=context.problem_context,
        student_attempt=context.student_message,
        previous_verified_step=context.previous_verified_step,
    )
    if not verifier.supports(request):
        result = VerificationResult(
            summary="当前题型不在确定性一元一次方程 verifier 的支持范围内。",
            verifier_name="none",
        )
    else:
        result = verifier.verify(request)
    return {"verification": result.model_dump(mode="json")}


def _inspect_student_context(
    _arguments: BaseModel,
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    snapshot = StudentModel().snapshot(
        context.episode,
        scope_id=context.problem_context,
    )
    return {"student_evidence": snapshot.model_dump(mode="json")}


def _retrieve_pedagogical_context(
    arguments: BaseModel,
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    parsed = RetrieveContextArguments.model_validate(arguments)
    provider = context.graph_provider or get_graph_context_provider()
    graph_context = provider.get_context(
        GraphContextRequest(
            problem_context=context.problem_context,
            concept_id=parsed.concept_id,
        )
    )
    return {"graph_context": graph_context.model_dump(mode="json")}


def _reason_about_problem(
    arguments: BaseModel,
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    parsed = ReasonAboutProblemArguments.model_validate(arguments)
    reasoner = context.reasoner or GenericMathReasoner()
    reasoning = reasoner.analyze(
        context.student_message,
        context.problem_context,
        context.analysis,
    )
    payload = reasoning.model_dump(mode="json")
    if parsed.focus:
        payload["requested_focus"] = parsed.focus
    return {"reasoning": payload}


def _load_skill(
    arguments: BaseModel,
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    parsed = LoadSkillArguments.model_validate(arguments)
    catalog = context.skill_catalog or SkillCatalog()
    return {"skill_name": parsed.name, "instructions": catalog.load(parsed.name)}


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="verify_math_attempt",
            description=(
                "用现有 Fraction math_checker 验证学生当前步骤。该结果优先于模型判断；"
                "运行时会在支持的题型上强制预调用一次。"
            ),
            arguments_model=NoArguments,
            handler=_verify_math_attempt,
        )
    )
    registry.register(
        ToolDefinition(
            name="inspect_student_context",
            description="读取当前题目范围内的学生证据，不把掌握度当作硬流程控制。",
            arguments_model=NoArguments,
            handler=_inspect_student_context,
        )
    )
    registry.register(
        ToolDefinition(
            name="retrieve_pedagogical_context",
            description="按需读取教学图谱证据；检索材料只能辅助教学决策。",
            arguments_model=RetrieveContextArguments,
            handler=_retrieve_pedagogical_context,
        )
    )
    registry.register(
        ToolDefinition(
            name="reason_about_problem",
            description="按需生成结构化数学推理和下一步候选，不负责判定学生正误。",
            arguments_model=ReasonAboutProblemArguments,
            handler=_reason_about_problem,
        )
    )
    registry.register(
        ToolDefinition(
            name="load_skill",
            description="按需加载一种教学方法的本地 Skill 指令。",
            arguments_model=LoadSkillArguments,
            handler=_load_skill,
        )
    )
    return registry
