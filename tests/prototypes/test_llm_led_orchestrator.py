from __future__ import annotations

import pytest

from prototypes.llm_led_teacher.orchestration_models import (
    AgentDecision,
    RequestClassification,
    TeachingContract,
    TeachingResponseMode,
    ToolResult,
)
from prototypes.llm_led_teacher.orchestrator import (
    MultisubjectAgentLoop,
    RequestClassifier,
)
from prototypes.llm_led_teacher.orchestration_tools import OrchestrationToolRegistry


class _Classifier:
    def __init__(self, classification: RequestClassification) -> None:
        self.classification = classification

    def classify(self, *args, **kwargs) -> RequestClassification:
        return self.classification


class _Decisions:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def decide(self, payload) -> AgentDecision:
        self.calls += 1
        return self.decisions.pop(0)


class _ScriptedActions:
    def __init__(self, actions: list[object]) -> None:
        self.actions = list(actions)
        self.payloads: list[dict[str, object]] = []

    @property
    def calls(self) -> int:
        return len(self.payloads)

    def decide(self, payload):
        self.payloads.append(dict(payload))
        result = self.actions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _Registry:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, name, arguments, runtime) -> ToolResult:
        self.calls.append((name, arguments))
        if name == "search_local_knowledge":
            runtime.local_search_succeeded = True
        return ToolResult(name=name, ok=True, data={"text": "证据", "hits": []})


def _k9(subject: str, *, new: bool) -> RequestClassification:
    return RequestClassification(
        scope="k9",
        subject=subject,
        stage="初中",
        is_new_problem=new,
        answer_mode="guided",
        reason="test route",
    )


def _guided_contract() -> TeachingContract:
    return TeachingContract(
        response_mode="guided_solution",
        requested_target="引导学生推进当前学习问题",
        allowed_content=["必要概念", "分步提示"],
        forbidden_content=["未经请求的最终答案"],
        may_apply_to_active_problem=True,
        may_reveal_final_answer=False,
        reason="student did not request a complete solution",
    )


def _finish(contract: TeachingContract | None = None) -> AgentDecision:
    return AgentDecision(
        kind="finish",
        reason="evidence is enough",
        plan="guided",
        contract=contract or _guided_contract(),
    )


def test_loop_reports_real_phase_before_each_agent_component_runs() -> None:
    timeline: list[str] = []

    class RecordingClassifier(_Classifier):
        def classify(self, *args, **kwargs) -> RequestClassification:
            timeline.append("classifier_call")
            return super().classify(*args, **kwargs)

    class RecordingDecisions(_Decisions):
        def decide(self, payload) -> AgentDecision:
            timeline.append("decision_call")
            return super().decide(payload)

    def record_progress(event: str | object) -> None:
        if isinstance(event, str):
            timeline.append(event)
        else:
            timeline.append(f"event:{event.event_type}")

    loop = MultisubjectAgentLoop(
        classifier=RecordingClassifier(_k9("数学", new=False)),
        action_provider=RecordingDecisions([_finish()]),
        registry=_Registry(),
    )

    loop.run(
        "3x+2=11",
        "给我一点提示",
        "数学",
        "初中",
        progress_callback=record_progress,
    )

    assert timeline[:5] == [
        "classification",
        "classifier_call",
        "event:classification",
        "planning",
        "decision_call",
    ]


def test_followup_inherits_math_subject_when_classifier_marks_same_problem() -> None:
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_Decisions([_finish()]),
        registry=_Registry(),
    )

    result = loop.run(
        problem_context="解方程 2x+3=7",
        student_message="为什么要减去3？",
        active_subject="数学",
        active_stage="初中",
    )

    assert result.classification.subject.value == "数学"
    assert result.classification.is_new_problem is False


def test_explicit_physics_problem_switches_subject() -> None:
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("物理", new=True)),
        action_provider=_Decisions([_finish()]),
        registry=_Registry(),
    )

    result = loop.run(
        problem_context="解方程 2x+3=7",
        student_message="物体为什么会做匀速直线运动？",
        active_subject="数学",
        active_stage="初中",
    )

    assert result.classification.subject.value == "物理"
    assert result.classification.is_new_problem is True


def test_off_topic_finishes_without_knowledge_tools() -> None:
    classification = RequestClassification(
        scope="off_topic",
        subject=None,
        stage="未知",
        is_new_problem=True,
        answer_mode="brief_redirect",
        reason="company product question",
    )
    registry = _Registry()
    provider = _Decisions([_finish()])
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(classification),
        action_provider=provider,
        registry=registry,
    )

    result = loop.run(
        problem_context="",
        student_message="网易企业邮箱是什么",
        active_subject="",
        active_stage="",
    )

    assert registry.calls == []
    assert provider.calls == 0
    assert result.termination_reason == "off_topic_redirect"
    assert result.routing_guidance["answer_mode"] == "brief_redirect"
    assert result.teaching_contract.response_mode is TeachingResponseMode.BRIEF_REDIRECT
    assert result.teaching_contract.may_apply_to_active_problem is False
    assert result.teaching_contract.may_reveal_final_answer is False


def test_formula_question_finishes_with_concept_only_contract() -> None:
    proposed = TeachingContract(
        response_mode="concept_explanation",
        requested_target="解释求根公式",
        allowed_content=["通用公式", "符号含义"],
        forbidden_content=["代入当前题", "计算当前题的根"],
        may_apply_to_active_problem=False,
        may_reveal_final_answer=False,
        reason="concept request",
    )
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_Decisions([_finish(proposed)]),
        registry=_Registry(),
    )

    result = loop.run(
        "3x^2+2x-11=0",
        "求根公式是什么？",
        "数学",
        "初中",
    )

    assert result.teaching_contract == proposed
    assert result.trace[-1].event_type == "teaching_contract"
    assert result.trace[-1].details == proposed.model_dump(mode="json")


def test_finish_normalizes_untrusted_full_solution_contract() -> None:
    proposed = TeachingContract(
        response_mode="full_solution",
        requested_target="解答当前题",
        allowed_content=["完整步骤", "最终答案"],
        may_apply_to_active_problem=True,
        may_reveal_final_answer=True,
        reason="model selected a full solution",
    )
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_Decisions([_finish(proposed)]),
        registry=_Registry(),
    )

    result = loop.run("3x+2=11", "这道题怎么解？", "数学", "初中")

    assert result.teaching_contract.response_mode is TeachingResponseMode.ONE_STEP
    assert result.teaching_contract.may_reveal_final_answer is False
    assert result.trace[-1].details == result.teaching_contract.model_dump(mode="json")


def test_invalid_action_is_retried_once_then_valid_action_is_used() -> None:
    provider = _ScriptedActions(
        [
            {"kind": "finish", "reason": "missing required contract"},
            _finish(),
        ]
    )
    registry = _Registry()
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=provider,
        registry=registry,
    )

    result = loop.run("3x+2=11", "给我一点提示", "数学", "初中")

    assert provider.calls == 2
    assert provider.payloads[1]["action_format_repair"] == (
        "上次 JSON 不符合 schema，只输出合法 action。"
    )
    assert registry.calls == []
    assert result.termination_reason == "model_finished"
    retry_events = [event for event in result.trace if event.event_type == "action_retry"]
    assert len(retry_events) == 1
    assert retry_events[0].details == {"error_type": "ValidationError"}


def test_two_invalid_actions_for_new_physics_problem_fall_back_to_local_only() -> None:
    provider = _ScriptedActions(
        [
            {"kind": "finish", "reason": "missing required contract"},
            {"kind": "call_tool", "reason": "missing tool name"},
        ]
    )
    registry = _Registry()
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("物理", new=True)),
        action_provider=provider,
        registry=registry,
    )

    result = loop.run(
        "",
        "物理新题：请根据初中教材说明影响物体惯性大小的因素。",
        "",
        "",
    )

    assert provider.calls == 2
    assert registry.calls == [
        (
            "search_local_knowledge",
            {
                "query": "物理新题：请根据初中教材说明影响物体惯性大小的因素。",
                "subject": "物理",
                "stage": "初中",
                "top_k": 5,
                "sources": ["textbook", "exercise"],
            },
        )
    ]
    assert [item.name for item in result.observations] == [
        "search_local_knowledge"
    ]
    assert result.rag_context.text == "证据"
    assert result.termination_reason == "provider_error"
    assert all(name != "search_web" for name, _ in registry.calls)
    fallback_events = [
        event for event in result.trace if event.event_type == "fallback_tool"
    ]
    assert len(fallback_events) == 1
    assert fallback_events[0].tool_name == "search_local_knowledge"
    assert fallback_events[0].ok is True


def test_action_provider_is_never_called_for_off_topic_even_if_script_is_invalid() -> None:
    classification = RequestClassification(
        scope="off_topic",
        subject=None,
        stage="未知",
        is_new_problem=True,
        answer_mode="brief_redirect",
        reason="not a school question",
    )
    provider = _ScriptedActions(
        [ValueError("invalid action one"), ValueError("invalid action two")]
    )
    registry = _Registry()
    result = MultisubjectAgentLoop(
        classifier=_Classifier(classification),
        action_provider=provider,
        registry=registry,
    ).run("", "请介绍最新款手机", "", "")

    assert provider.calls == 0
    assert registry.calls == []
    assert result.teaching_contract.response_mode is TeachingResponseMode.BRIEF_REDIRECT


def test_provider_fallback_local_tool_stays_within_one_step_and_tool_budget() -> None:
    provider = _ScriptedActions(
        [ValueError("invalid action one"), ValueError("invalid action two")]
    )
    registry = _Registry()
    result = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("物理", new=True)),
        action_provider=provider,
        registry=registry,
        max_iterations=1,
    ).run("", "惯性是什么？", "", "")

    assert provider.calls == 2
    assert len(registry.calls) == 1
    assert len(result.observations) == 1
    assert all(name != "search_web" for name, _ in registry.calls)


def test_provider_failure_after_adequate_local_evidence_does_not_search_again() -> None:
    provider = _ScriptedActions(
        [
            AgentDecision(
                kind="call_tool",
                tool_name="search_local_knowledge",
                arguments={
                    "query": "惯性",
                    "subject": "物理",
                    "stage": "初中",
                },
                reason="先查询本地教材",
            ),
            ValueError("invalid action one"),
            ValueError("invalid action two"),
        ]
    )
    registry = _Registry()
    result = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("物理", new=False)),
        action_provider=provider,
        registry=registry,
    ).run("惯性概念", "为什么只与质量有关？", "物理", "初中")

    assert provider.calls == 3
    assert [name for name, _ in registry.calls] == ["search_local_knowledge"]
    assert len(result.observations) == 1
    assert result.rag_context.text == "证据"
    assert result.termination_reason == "provider_error"


def test_two_invalid_actions_for_concept_followup_use_one_local_search() -> None:
    provider = _ScriptedActions(
        [ValueError("invalid action one"), ValueError("invalid action two")]
    )
    registry = _Registry()
    result = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=provider,
        registry=registry,
    ).run("3x^2+2x-11=0", "求根公式是什么？", "数学", "初中")

    assert provider.calls == 2
    assert registry.calls == [
        (
            "search_local_knowledge",
            {
                "query": "求根公式是什么？",
                "subject": "数学",
                "stage": "初中",
                "top_k": 5,
                "sources": ["textbook", "exercise"],
            },
        )
    ]
    assert len(result.observations) == 1


def test_provider_error_returns_safe_contract() -> None:
    class _BrokenProvider:
        def decide(self, payload):
            raise ValueError("malformed decision")

    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_BrokenProvider(),
        registry=_Registry(),
    )

    result = loop.run("3x+2=11", "这道题怎么解？", "数学", "初中")

    assert result.termination_reason == "provider_error"
    assert result.teaching_contract.response_mode is TeachingResponseMode.ONE_STEP
    assert result.teaching_contract.may_reveal_final_answer is False


def test_provider_error_concept_question_returns_non_applying_contract() -> None:
    class _BrokenProvider:
        def decide(self, payload):
            raise ValueError("malformed decision")

    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_BrokenProvider(),
        registry=_Registry(),
    )

    result = loop.run("3x^2+2x-11=0", "求根公式是什么？", "数学", "初中")

    assert result.teaching_contract.response_mode is TeachingResponseMode.CONCEPT_EXPLANATION
    assert result.teaching_contract.may_apply_to_active_problem is False
    assert result.teaching_contract.may_reveal_final_answer is False


def test_provider_error_non_solve_fallback_does_not_apply_to_active_problem() -> None:
    class _BrokenProvider:
        def decide(self, payload):
            raise ValueError("malformed decision")

    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_BrokenProvider(),
        registry=_Registry(),
    )

    result = loop.run("3x+2=11", "给我一点提示", "数学", "初中")

    assert result.teaching_contract.response_mode is TeachingResponseMode.GUIDED_SOLUTION
    assert result.teaching_contract.may_apply_to_active_problem is False
    assert result.teaching_contract.may_reveal_final_answer is False


def test_provider_error_keeps_explicit_full_request_with_why_cue() -> None:
    class _BrokenProvider:
        def decide(self, payload):
            raise ValueError("malformed decision")

    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=False)),
        action_provider=_BrokenProvider(),
        registry=_Registry(),
    )

    result = loop.run(
        "3x+2=11",
        "请完整解答这道题，并解释为什么这样做",
        "数学",
        "初中",
    )

    assert result.teaching_contract.response_mode is TeachingResponseMode.FULL_SOLUTION
    assert result.teaching_contract.may_apply_to_active_problem is True
    assert result.teaching_contract.may_reveal_final_answer is True


def test_loop_stops_after_six_decisions() -> None:
    repeated = AgentDecision(
        kind="call_tool",
        tool_name="search_local_knowledge",
        arguments={"query": "力", "subject": "物理", "stage": "初中"},
        reason="try local evidence",
    )
    provider = _Decisions([repeated.model_copy(deep=True) for _ in range(8)])
    loop = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("物理", new=True)),
        action_provider=provider,
        registry=_Registry(),
        max_iterations=6,
    )

    result = loop.run(
        problem_context="",
        student_message="力是什么",
        active_subject="",
        active_stage="",
    )

    decision_events = [event for event in result.trace if event.event_type == "decision"]
    assert len(decision_events) == 6
    assert provider.calls == 6
    assert result.termination_reason == "iteration_budget_exhausted"
    assert result.teaching_contract.response_mode is TeachingResponseMode.CONCEPT_EXPLANATION
    assert result.teaching_contract.may_apply_to_active_problem is False
    assert result.teaching_contract.may_reveal_final_answer is False


def test_classifier_repairs_once_then_falls_back_off_topic() -> None:
    class BrokenClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat_json(self, **kwargs):
            self.calls += 1
            raise ValueError("malformed JSON")

    client = BrokenClient()
    result = RequestClassifier(client).classify("hello", "", "", "")

    assert client.calls == 2
    assert result.scope.value == "off_topic"
    assert result.answer_mode.value == "brief_redirect"

NINE_SUBJECT_SKILL_CASES = [
    ("数学", "老师，2x + 3 = 9 怎么解？", "socratic_guidance"),
    ("语文", "拟人这种修辞手法是什么意思？", "concept_explanation"),
    ("英语", "一般过去时什么时候使用？", "socratic_guidance"),
    ("物理", "为什么惯性只与质量有关？", "concept_explanation"),
    ("化学", "怎么判断一个反应是不是置换反应？", "socratic_guidance"),
    ("生物", "光合作用需要哪些条件？", "socratic_guidance"),
    ("历史", "辛亥革命的主要影响是什么？", "concept_explanation"),
    ("地理", "怎么判断一个地方属于什么气候？", "socratic_guidance"),
    ("道法", "权利和义务是什么关系？", "concept_explanation"),
]


@pytest.mark.parametrize(
    ("subject", "message", "expected_skill"),
    NINE_SUBJECT_SKILL_CASES,
)
def test_unknown_skill_recovers_once_for_every_subject(
    subject: str,
    message: str,
    expected_skill: str,
) -> None:
    provider = _ScriptedActions(
        [
            AgentDecision(
                kind="call_tool",
                tool_name="load_skill",
                arguments={"name": f"{subject}教学技能"},
                reason="加载教学方法",
            ),
            _finish(),
        ]
    )
    result = MultisubjectAgentLoop(
        classifier=_Classifier(_k9(subject, new=True)),
        action_provider=provider,
        registry=OrchestrationToolRegistry(),
        enable_fast_routing=False,
    ).run("", message, "", "")

    assert provider.calls == 2
    assert result.loaded_skills == [expected_skill]
    assert result.termination_reason == "model_finished"
    fallback_events = [
        event
        for event in result.trace
        if event.details.get("fallback_reason") == "unknown_skill_fallback"
    ]
    assert len(fallback_events) == 1
    assert fallback_events[0].details["requested_skill"] == f"{subject}教学技能"
    assert fallback_events[0].details["selected_skill"] == expected_skill
    assert all(
        event.summary != "load_skill 未执行：unknown skill"
        for event in result.trace
    )
    assert {
        item["name"] for item in provider.payloads[0]["available_skills"]
    } == {
        "socratic_guidance",
        "error_diagnosis",
        "concept_explanation",
        "complete_solution",
    }


@pytest.mark.parametrize(
    ("message", "expected_skill"),
    [
        ("请完整解答 3x-5=10，并说明每一步。", "complete_solution"),
        ("请引导我解 4x+1=13，但不要告诉我最后答案。", "socratic_guidance"),
        ("我觉得 2x+3=9 两边减3以后是 2x=5，对吗？", "socratic_guidance"),
    ],
)
def test_unknown_skill_fallback_preserves_request_boundary(
    message: str,
    expected_skill: str,
) -> None:
    provider = _ScriptedActions(
        [
            AgentDecision(
                kind="call_tool",
                tool_name="load_skill",
                arguments={"name": "数学教学技能"},
                reason="加载数学教学方法",
            ),
            _finish(),
        ]
    )
    result = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=True)),
        action_provider=provider,
        registry=OrchestrationToolRegistry(),
        enable_fast_routing=False,
    ).run("", message, "", "")

    assert result.loaded_skills == [expected_skill]
    if "不要告诉我" in message:
        assert result.teaching_contract.may_reveal_final_answer is False
    if "2x=5" in message:
        assert "error_diagnosis" not in result.loaded_skills


def test_empty_skill_catalog_stops_after_first_unknown_request() -> None:
    provider = _ScriptedActions(
        [
            AgentDecision(
                kind="call_tool",
                tool_name="load_skill",
                arguments={"name": "数学教学技能"},
                reason="加载数学教学方法",
            )
        ]
    )
    result = MultisubjectAgentLoop(
        classifier=_Classifier(_k9("数学", new=True)),
        action_provider=provider,
        registry=OrchestrationToolRegistry(),
        skill_catalog={},
        enable_fast_routing=False,
    ).run("", "老师，2x+3=9怎么解？", "", "")

    assert provider.calls == 1
    assert result.loaded_skills == []
    assert result.termination_reason == "skill_catalog_unavailable"


def test_search_subject_and_stage_are_bound_to_validated_classification() -> None:
    provider = _ScriptedActions(
        [
            AgentDecision(
                kind="call_tool",
                tool_name="search_local_knowledge",
                arguments={
                    "query": "一元一次方程",
                    "subject": "数学学科",
                    "stage": "初中数学",
                    "top_k": 5,
                    "sources": ["textbook", "exercise"],
                },
                reason="查询本地教材",
            ),
            _finish(),
        ]
    )
    registry = _Registry()
    classification = RequestClassification.model_validate(
        {**_k9("数学", new=True).model_dump(mode="json"), "stage": "未知"}
    )

    MultisubjectAgentLoop(
        classifier=_Classifier(classification),
        action_provider=provider,
        registry=registry,
        enable_fast_routing=False,
    ).run("", "老师，2x+3=9怎么解？", "", "")

    assert registry.calls[0] == (
        "search_local_knowledge",
        {
            "query": "一元一次方程",
            "subject": "数学",
            "stage": "未知",
            "top_k": 5,
            "sources": ["textbook", "exercise"],
        },
    )
