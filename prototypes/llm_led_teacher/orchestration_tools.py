"""Guarded tools for the bounded multisubject teacher loop."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from prototypes.llm_led_teacher.orchestration_models import (
    RequestClassification,
    RequestScope,
    SchoolStage,
    Subject,
    ToolResult,
)
from prototypes.llm_led_teacher.rag_hook import (
    RagContext,
    retrieve_local_context,
    retrieve_web_context,
)


class LocalSearchArguments(BaseModel):
    query: str = Field(min_length=1, max_length=800)
    subject: Subject
    stage: SchoolStage
    top_k: int = Field(default=5, ge=1, le=10)
    sources: tuple[Literal["textbook", "exercise"], ...] = Field(
        default=("textbook", "exercise"),
        min_length=1,
    )


class WebSearchArguments(BaseModel):
    query: str = Field(min_length=1, max_length=800)
    subject: Subject
    stage: SchoolStage
    evidence_gap: str = Field(min_length=1, max_length=400)


@dataclass
class OrchestrationRuntime:
    classification: RequestClassification
    problem_context: str
    student_message: str
    local_search_succeeded: bool = False
    web_search_count: int = 0
    skill_catalog: dict[str, str] = field(default_factory=dict)
    local_search_signatures: set[str] = field(default_factory=set)


TimeoutRunner = Callable[[Callable[..., Any], dict[str, Any], float], Any]


def _bounded_call(
    handler: Callable[..., Any],
    kwargs: dict[str, Any],
    timeout_seconds: float,
) -> Any:
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="teacher-tool")
    future = pool.submit(handler, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _rag_data(context: RagContext) -> dict[str, Any]:
    return {
        "text": context.text,
        "refs": context.refs,
        "is_exact_match": context.is_exact_match,
        "source": context.source,
        "hits": context.hits,
        "online_triggered": context.online_triggered,
        "online_status": context.online_status,
        "online_reason": context.online_reason,
        "local_top_score": context.local_top_score,
        "diagnostics": context.diagnostics,
    }


class OrchestrationToolRegistry:
    def __init__(
        self,
        *,
        local_handler: Callable[..., Any] | None = None,
        web_handler: Callable[..., Any] | None = None,
        timeout_runner: TimeoutRunner = _bounded_call,
        timeout_seconds: float | None = None,
    ) -> None:
        self.local_handler = local_handler or self._search_local
        self.web_handler = web_handler or self._search_web
        self.timeout_runner = timeout_runner
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("TEACHER_TOOL_TIMEOUT_SECONDS", "45")
        )

    @staticmethod
    def _search_local(**kwargs: Any) -> RagContext:
        return retrieve_local_context(
            kwargs["query"],
            subject=kwargs["subject"],
            stage=kwargs["stage"],
            topk=kwargs["top_k"],
            sources=kwargs["sources"],
        )

    @staticmethod
    def _search_web(**kwargs: Any) -> RagContext:
        return retrieve_web_context(
            kwargs["query"],
            subject=kwargs["subject"],
            stage=kwargs["stage"],
            evidence_gap=kwargs["evidence_gap"],
        )

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        runtime: OrchestrationRuntime,
    ) -> ToolResult:
        started = perf_counter()
        if name in {"search_local_knowledge", "search_web"} and (
            runtime.classification.scope is RequestScope.OFF_TOPIC
        ):
            return ToolResult(name=name, ok=False, error="off-topic knowledge search forbidden")
        if name == "search_local_knowledge":
            return self._execute_local(arguments, runtime, started)
        if name == "search_web":
            return self._execute_web(arguments, runtime, started)
        if name == "load_skill":
            skill_name = str(arguments.get("name", ""))
            content = runtime.skill_catalog.get(skill_name)
            if not content:
                return ToolResult(name=name, ok=False, error="unknown skill")
            return ToolResult(name=name, ok=True, data={"name": skill_name, "content": content[:1200]})
        if name == "inspect_student_context":
            return ToolResult(
                name=name,
                ok=True,
                data={
                    "problem_context": runtime.problem_context[-1200:],
                    "student_message": runtime.student_message[-800:],
                },
            )
        return ToolResult(name=name, ok=False, error="unknown tool")

    def _execute_local(
        self,
        arguments: dict[str, Any],
        runtime: OrchestrationRuntime,
        started: float,
    ) -> ToolResult:
        try:
            parsed = LocalSearchArguments.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult(name="search_local_knowledge", ok=False, error=f"invalid arguments: {exc.errors()[0]['msg']}")
        if parsed.subject is not runtime.classification.subject:
            return ToolResult(
                name="search_local_knowledge",
                ok=False,
                error="local search subject does not match classification",
            )
        if (
            runtime.classification.stage is not SchoolStage.UNKNOWN
            and parsed.stage is not runtime.classification.stage
        ):
            return ToolResult(
                name="search_local_knowledge",
                ok=False,
                error="local search stage does not match classification",
            )
        signature = parsed.model_dump_json()
        if signature in runtime.local_search_signatures:
            return ToolResult(name="search_local_knowledge", ok=False, error="duplicate local search")
        if len(runtime.local_search_signatures) >= 2:
            return ToolResult(name="search_local_knowledge", ok=False, error="local search budget exhausted")
        runtime.local_search_signatures.add(signature)
        return self._run_handler(
            "search_local_knowledge",
            self.local_handler,
            parsed.model_dump(),
            runtime,
            started,
            local=True,
        )

    def _execute_web(
        self,
        arguments: dict[str, Any],
        runtime: OrchestrationRuntime,
        started: float,
    ) -> ToolResult:
        if runtime.web_search_count >= 1:
            return ToolResult(name="search_web", ok=False, error="web search budget exhausted")
        if not runtime.local_search_succeeded:
            return ToolResult(name="search_web", ok=False, error="web search requires successful local search")
        try:
            parsed = WebSearchArguments.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult(name="search_web", ok=False, error=f"invalid arguments: {exc.errors()[0]['msg']}")
        runtime.web_search_count += 1
        return self._run_handler(
            "search_web",
            self.web_handler,
            parsed.model_dump(),
            runtime,
            started,
        )

    def _run_handler(
        self,
        name: str,
        handler: Callable[..., Any],
        kwargs: dict[str, Any],
        runtime: OrchestrationRuntime,
        started: float,
        *,
        local: bool = False,
    ) -> ToolResult:
        try:
            value = self.timeout_runner(handler, kwargs, self.timeout_seconds)
        except (TimeoutError, FutureTimeoutError):
            return ToolResult(name=name, ok=False, error="tool timed out", elapsed_seconds=perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - tools must not break the UI process
            return ToolResult(name=name, ok=False, error=f"tool error: {type(exc).__name__}", elapsed_seconds=perf_counter() - started)
        data = _rag_data(value) if isinstance(value, RagContext) else (value or {})
        if not isinstance(data, dict):
            data = {"value": str(data)[:600]}
        if local:
            runtime.local_search_succeeded = True
        return ToolResult(name=name, ok=True, data=data, elapsed_seconds=perf_counter() - started)
