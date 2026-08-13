from prototypes.llm_led_teacher.orchestration_models import (
    TeachingResponseMode,
    ToolResult,
)
from prototypes.llm_led_teacher.orchestrator import MultisubjectAgentLoop


class _NoModelCalls:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("high-confidence route must not call the model")


class _FastRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name, arguments, runtime) -> ToolResult:
        self.calls.append((name, arguments))
        if name == "load_skill":
            return ToolResult(
                name=name,
                ok=True,
                data={"name": arguments["name"], "content": "按需教学指令"},
            )
        if name == "search_local_knowledge":
            runtime.local_search_succeeded = True
            return ToolResult(
                name=name,
                ok=True,
                data={
                    "text": "本地教材证据",
                    "source": "local",
                    "hits": [{"document": "本地教材证据"}],
                },
            )
        raise AssertionError(f"unexpected tool: {name}")


def test_fast_physics_route_skips_model_decisions_and_calls_local_tool() -> None:
    client = _NoModelCalls()
    registry = _FastRegistry()
    result = MultisubjectAgentLoop(client=client, registry=registry).run(
        "",
        "请根据初中教材说明影响物体惯性大小的因素。",
        "",
        "",
    )

    assert client.calls == 0
    assert [name for name, _ in registry.calls] == [
        "load_skill",
        "search_local_knowledge",
    ]
    assert result.classification.subject.value == "物理"
    assert result.classification.stage.value == "初中"
    assert result.loaded_skills == ["concept_explanation"]
    assert result.loaded_skill_guidance == {
        "concept_explanation": "按需教学指令"
    }
    assert result.rag_context.source == "local"
    assert result.termination_reason == "deterministic_fast_path"
    assert all(event.tool_name != "search_web" for event in result.trace)


def test_fast_full_solution_loads_only_complete_solution_skill() -> None:
    client = _NoModelCalls()
    registry = _FastRegistry()
    result = MultisubjectAgentLoop(client=client, registry=registry).run(
        "3x^2+2x-11=0",
        "请用求根公式完整解这道题并给出答案",
        "数学",
        "初中",
    )

    assert client.calls == 0
    assert registry.calls == [
        ("load_skill", {"name": "complete_solution"})
    ]
    assert result.loaded_skills == ["complete_solution"]
    assert result.teaching_contract.response_mode is TeachingResponseMode.FULL_SOLUTION
    assert result.teaching_contract.may_reveal_final_answer is True


def test_fast_off_topic_route_uses_no_model_or_tools() -> None:
    client = _NoModelCalls()
    registry = _FastRegistry()
    result = MultisubjectAgentLoop(client=client, registry=registry).run(
        "",
        "请比较几款主流手机产品并推荐一款最值得购买的。",
        "",
        "",
    )

    assert client.calls == 0
    assert registry.calls == []
    assert result.classification.scope.value == "off_topic"
    assert result.direct_response
    assert result.termination_reason == "off_topic_redirect"
