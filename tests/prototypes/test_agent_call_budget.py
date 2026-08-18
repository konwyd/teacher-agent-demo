from prototypes.llm_led_teacher.logic import PrototypeEpisode, run_turn
from prototypes.llm_led_teacher.orchestration_models import (
    AgentOrchestrationResult,
    BoundaryAudit,
    RequestClassification,
    TeachingContract,
)
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
