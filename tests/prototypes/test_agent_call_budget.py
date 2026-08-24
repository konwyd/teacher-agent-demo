import pytest

from prototypes.llm_led_teacher import logic
from prototypes.llm_led_teacher.logic import PrototypeEpisode, run_turn
from prototypes.llm_led_teacher.orchestration_models import (
    AgentDecision,
    AgentOrchestrationResult,
    BoundaryAudit,
    RequestClassification,
    TeachingContract,
    ToolResult,
)
from prototypes.llm_led_teacher.orchestrator import MultisubjectAgentLoop
from prototypes.llm_led_teacher.rag_hook import RagContext


class _ResultOrchestrator:
    def __init__(self, result: AgentOrchestrationResult) -> None:
        self.result = result

    def run(self, *args, **kwargs) -> AgentOrchestrationResult:
        return self.result


class _BudgetClient:
    def __init__(self, response: str, *, allow_audit: bool) -> None:
        self.response = response
        self.allow_audit = allow_audit
        self.chat_limits: list[int | None] = []
        self.audit_limits: list[int | None] = []
        self.usage_history: list[dict] = []
        self.last_chat_metadata: dict = {}

    def chat(self, system, user, temperature=0.0, max_tokens=None):
        self.chat_limits.append(max_tokens)
        return self.response

    def chat_json(
        self,
        system,
        user,
        schema,
        temperature=0.0,
        max_tokens=None,
    ):
        if not self.allow_audit:
            raise AssertionError("this contract must not invoke semantic audit")
        self.audit_limits.append(max_tokens)
        return BoundaryAudit(passed=True)


class _EndToEndBudgetClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.usage_history: list[dict[str, object]] = []
        self.last_chat_metadata: dict[str, object] = {}

    def _record(self, call_type: str) -> None:
        self.calls.append(call_type)
        self.last_chat_metadata = {
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        self.usage_history.append(dict(self.last_chat_metadata))

    def chat(self, system, user, temperature=0.0, max_tokens=None):
        self._record("teacher")
        return "这个概念描述了对象之间的确定关系。你想先看它的定义吗？"

    def chat_json(
        self,
        system,
        user,
        schema,
        temperature=0.0,
        max_tokens=None,
    ):
        if schema is RequestClassification:
            self._record("classifier")
            return RequestClassification(
                scope="k9",
                subject="数学",
                stage="初中",
                is_new_problem=True,
                answer_mode="guided",
                reason="模型分类器确认普通数学概念",
            )
        if schema is BoundaryAudit:
            self._record("semantic_audit")
            return BoundaryAudit(passed=True)
        if schema is AgentDecision:
            self._record("action_provider")
            raise AssertionError("ordinary K9 must not invoke ActionProvider")
        raise AssertionError(f"unexpected schema: {schema}")


class _DynamicEndToEndClient(_EndToEndBudgetClient):
    def chat(self, system, user, temperature=0.0, max_tokens=None):
        self._record("teacher")
        return "课程标准应以官网当前公布版本为准。你想先核对发布日期吗？"

    def chat_json(
        self,
        system,
        user,
        schema,
        temperature=0.0,
        max_tokens=None,
    ):
        if schema is AgentDecision:
            self._record("action_provider")
            return AgentDecision(
                kind="call_tool",
                reason="本地资料后核对官网当前版本",
                tool_name="search_web",
                arguments={
                    "query": "最新初中数学课程标准",
                    "evidence_gap": "需要官网当前版本",
                },
            )
        return super().chat_json(
            system,
            user,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class _DynamicRegistry:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, name, arguments, runtime):
        self.calls.append(name)
        if name == "load_skill":
            return ToolResult(
                name=name,
                ok=True,
                data={"name": arguments["name"], "content": "教学指引"},
            )
        if name == "search_local_knowledge":
            runtime.local_search_succeeded = True
            return ToolResult(
                name=name,
                ok=True,
                data={"source": "local", "text": "本地课程标准", "hits": [{}]},
            )
        if name == "search_web":
            return ToolResult(
                name=name,
                ok=True,
                data={
                    "source": "web",
                    "text": "官网课程标准",
                    "hits": [{}],
                    "online_triggered": True,
                    "online_status": "used",
                },
            )
        raise AssertionError(f"unexpected tool: {name}")


def _result(response_mode: str) -> AgentOrchestrationResult:
    off_topic = response_mode == "brief_redirect"
    full = response_mode == "full_solution"
    concept = response_mode == "concept_explanation"
    return AgentOrchestrationResult(
        classification=RequestClassification(
            scope="off_topic" if off_topic else "k9",
            subject=None if off_topic else "数学",
            stage="未知" if off_topic else "初中",
            is_new_problem=off_topic,
            answer_mode="brief_redirect" if off_topic else "complete" if full else "guided",
            reason="test call budget",
        ),
        teaching_contract=TeachingContract(
            response_mode=response_mode,
            requested_target="测试教学目标",
            allowed_content=["请求范围内的内容"],
            forbidden_content=(
                ["代入或套用当前题", "最终答案"] if concept else []
            ),
            may_apply_to_active_problem=full,
            may_reveal_final_answer=full,
            reason="test contract",
        ),
        rag_context=RagContext(source="none"),
        termination_reason="test",
        direct_response=(
            "这个请求不在当前中小学课程辅导范围内。你可以换一个课程问题。"
            if off_topic
            else ""
        ),
    )


def test_concept_turn_uses_one_generation_and_one_small_audit() -> None:
    client = _BudgetClient("求根公式中的 a、b、c 是方程的系数。", allow_audit=True)
    episode = PrototypeEpisode(
        problem_context="3x^2+2x-11=0",
        active_subject="数学",
        active_stage="初中",
    )

    turn = run_turn(
        episode,
        "求根公式是什么？",
        client=client,
        orchestrator=_ResultOrchestrator(_result("concept_explanation")),
    )

    assert client.chat_limits == [1024]
    assert client.audit_limits == [320]
    assert turn.boundary_audit == BoundaryAudit(passed=True)
    assert turn.semantic_audit_performed is True


def test_deterministic_concept_end_to_end_uses_one_llm_call() -> None:
    client = _EndToEndBudgetClient()

    turn = run_turn(
        PrototypeEpisode(),
        "勾股定理是什么？",
        client=client,
        enable_agent_loop=True,
    )

    assert client.calls == ["teacher"]
    assert turn.model_generation_metadata["llm_call_count"] == 1
    assert turn.router_type == "deterministic_fast_path"
    assert turn.semantic_audit_performed is False


def test_classifier_direct_concept_end_to_end_uses_two_llm_calls() -> None:
    client = _EndToEndBudgetClient()

    turn = run_turn(
        PrototypeEpisode(),
        "欧拉线是什么？",
        client=client,
        enable_agent_loop=True,
    )

    assert client.calls == ["classifier", "teacher"]
    assert turn.model_generation_metadata["llm_call_count"] == 2
    assert turn.router_type == "model_classifier_direct_path"
    assert turn.agent_iterations == 0
    assert turn.semantic_audit_performed is False


def test_dynamic_web_end_to_end_stays_within_four_llm_calls() -> None:
    client = _DynamicEndToEndClient()
    registry = _DynamicRegistry()
    orchestrator = MultisubjectAgentLoop(
        client=client,
        registry=registry,
        enable_fast_routing=True,
    )

    turn = run_turn(
        PrototypeEpisode(),
        "请联网查一下最新的初中数学课程标准",
        client=client,
        orchestrator=orchestrator,
    )

    assert client.calls == ["action_provider", "teacher", "semantic_audit"]
    assert len(client.calls) <= 4
    assert turn.model_generation_metadata["llm_call_count"] == 3
    assert turn.router_type == "dynamic_agent_path"
    assert turn.agent_termination_reason == "dynamic_web_completed"
    assert turn.agent_iterations == 1
    assert registry.calls == [
        "load_skill",
        "search_local_knowledge",
        "search_web",
    ]


@pytest.mark.parametrize(
    "router_type",
    ["deterministic_fast_path", "model_classifier_direct_path"],
)
def test_low_risk_direct_concept_uses_one_generation_and_hard_recheck(
    monkeypatch,
    router_type: str,
) -> None:
    result = _result("concept_explanation").model_copy(
        update={
            "classification": _result("concept_explanation")
            .classification.model_copy(update={"is_new_problem": True}),
            "termination_reason": router_type,
            "routing_guidance": {"router": router_type},
        }
    )
    client = _BudgetClient(
        "勾股定理说明直角三角形三边之间的数量关系。你能指出哪条边是斜边吗？",
        allow_audit=False,
    )
    hard_rechecks: list[tuple[str, str]] = []

    def record_hard_recheck(problem_context, contract, response):
        hard_rechecks.append((problem_context, response))
        return []

    monkeypatch.setattr(logic, "hard_boundary_recheck", record_hard_recheck)

    turn = run_turn(
        PrototypeEpisode(),
        "勾股定理是什么？",
        client=client,
        orchestrator=_ResultOrchestrator(result),
    )

    assert client.chat_limits == [1024]
    assert client.audit_limits == []
    assert len(hard_rechecks) == 1
    assert turn.boundary_audit is None
    assert turn.semantic_audit_performed is False
    assert turn.repair_performed is False
    assert turn.router_type == router_type
    assert turn.agent_iterations == 0


def test_full_solution_turn_uses_one_generation_and_skips_semantic_audit() -> None:
    client = _BudgetClient(
        "按求根公式计算可得两个根，完整过程到此结束。",
        allow_audit=False,
    )
    episode = PrototypeEpisode(
        problem_context="3x^2+2x-11=0",
        active_subject="数学",
        active_stage="初中",
    )

    turn = run_turn(
        episode,
        "请完整解答并给出答案",
        client=client,
        orchestrator=_ResultOrchestrator(_result("full_solution")),
    )

    assert client.chat_limits == [2048]
    assert client.audit_limits == []
    assert turn.boundary_audit is None
    assert turn.repair_performed is False


def test_length_limited_generation_is_rewritten_once_before_audit() -> None:
    class LengthAwareClient(_BudgetClient):
        def __init__(self) -> None:
            super().__init__("", allow_audit=True)
            self.responses = iter(
                [
                    ("这是一段没有写完的概念解释，因为", "length"),
                    ("这是完整的概念解释。你想先练习哪个部分？", "stop"),
                ]
            )

        def chat(self, system, user, temperature=0.0, max_tokens=None):
            self.chat_limits.append(max_tokens)
            response, finish_reason = next(self.responses)
            self.last_chat_metadata = {"finish_reason": finish_reason}
            self.usage_history.append(dict(self.last_chat_metadata))
            return response

    client = LengthAwareClient()
    episode = PrototypeEpisode(active_subject="语文", active_stage="小学")

    turn = run_turn(
        episode,
        "如何学习语文",
        client=client,
        orchestrator=_ResultOrchestrator(_result("concept_explanation")),
    )

    assert client.chat_limits == [1024, 2048]
    assert client.audit_limits == [320]
    assert turn.teacher_output.response == "这是完整的概念解释。你想先练习哪个部分？"
    assert turn.model_generation_metadata["finish_reason"] == "stop"


def test_off_topic_direct_response_uses_zero_model_calls() -> None:
    client = _BudgetClient("不应使用", allow_audit=False)

    turn = run_turn(
        PrototypeEpisode(),
        "请推荐一款手机",
        client=client,
        orchestrator=_ResultOrchestrator(_result("brief_redirect")),
    )

    assert client.chat_limits == []
    assert client.audit_limits == []
    assert "中小学课程辅导范围" in turn.teacher_output.response
    assert turn.boundary_audit is None
