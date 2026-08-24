import pytest

from prototypes.llm_led_teacher import fast_routing
from prototypes.llm_led_teacher.fast_routing import choose_fast_route
from prototypes.llm_led_teacher.orchestration_models import (
    SchoolStage,
    Subject,
    TeachingResponseMode,
    ToolResult,
)
from prototypes.llm_led_teacher.retrieval_policy import RetrievalMode
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
    assert result.retrieval_decision.mode is RetrievalMode.LOCAL
    assert "教材" in result.retrieval_decision.reason
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


def test_fast_math_attempt_preserves_active_problem_continuity() -> None:
    client = _NoModelCalls()
    registry = _FastRegistry()
    result = MultisubjectAgentLoop(client=client, registry=registry).run(
        "初中数学：解方程 2x+3=7，请先引导我。",
        "我把两边都减去3，得到2x=5，对吗？",
        "数学",
        "初中",
    )

    assert client.calls == 0
    assert result.classification.is_new_problem is False
    assert registry.calls == [("load_skill", {"name": "socratic_guidance"})]
    assert result.termination_reason == "deterministic_fast_path"

    assert result.retrieval_decision.mode is RetrievalMode.NONE


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


@pytest.mark.parametrize("message", ["你好", "老师好！", "早上好", "谢谢老师", "再见"])
def test_pure_social_messages_skip_model_and_tools(message: str) -> None:
    client = _NoModelCalls()
    registry = _FastRegistry()

    result = MultisubjectAgentLoop(client=client, registry=registry).run(
        "", message, "", ""
    )

    assert client.calls == 0
    assert registry.calls == []
    assert result.classification.scope.value == "social"
    assert result.termination_reason == "social_response"
    assert "课程" in result.direct_response
    assert "不在当前" not in result.direct_response


@pytest.mark.parametrize("subject", ["数学", "语文", "英语", "物理"])
def test_explicit_subject_routes_as_curriculum_without_stage(subject: str) -> None:
    client = _NoModelCalls()
    registry = _FastRegistry()

    result = MultisubjectAgentLoop(client=client, registry=registry).run(
        "", f"老师好，如何学习{subject}？", "", ""
    )

    assert client.calls == 0
    assert result.classification.scope.value == "k9"
    assert result.classification.subject.value == subject
    assert result.classification.stage.value == "未知"
    assert result.termination_reason == "deterministic_fast_path"


def test_greeting_prefix_does_not_hide_curriculum_content() -> None:
    route = choose_fast_route("你好，请问小学数学分数怎么计算？", "", "", "")

    assert route is not None
    assert route.classification.scope.value == "k9"
    assert route.classification.subject.value == "数学"
    assert route.classification.stage.value == "小学"


def test_ni_hao_substring_is_not_treated_as_a_greeting() -> None:
    route = choose_fast_route("你好像算错了", "", "", "")

    assert route is None or route.classification.scope.value != "social"


def test_subject_word_does_not_override_clear_consumer_intent() -> None:
    client = _NoModelCalls()
    registry = _FastRegistry()

    result = MultisubjectAgentLoop(client=client, registry=registry).run(
        "", "数学老师用什么手机最值得买？", "", ""
    )

    assert client.calls == 0
    assert registry.calls == []
    assert result.classification.scope.value == "off_topic"


@pytest.mark.parametrize(
    ("message", "expected_subject"),
    [
        ("勾股定理应该怎么理解？", Subject.MATH),
        ("文言文里的倒装句是什么？", Subject.CHINESE),
        ("现在完成时怎么构成？", Subject.ENGLISH),
        ("惯性大小和什么有关？", Subject.PHYSICS),
        ("元素周期表如何分区？", Subject.CHEMISTRY),
        ("光合作用需要哪些条件？", Subject.BIOLOGY),
        ("辛亥革命有什么影响？", Subject.HISTORY),
        ("等高线越密说明什么？", Subject.GEOGRAPHY),
        ("公民基本权利和义务有哪些？", Subject.POLITICS),
    ],
)
def test_safe_unique_cues_cover_all_nine_subjects(
    message: str,
    expected_subject: Subject,
) -> None:
    route = choose_fast_route(message, "", "", "")

    assert route is not None
    assert route.classification.subject is expected_subject


@pytest.mark.parametrize(
    "message",
    [
        "勾股定理怎么证明？",
        "平方根是什么意思？",
        "一次函数的图像有什么特点？",
        "一元二次方程怎么配方？",
        "全等三角形有哪些判定方法？",
        "圆周角定理是什么？",
    ],
)
def test_middle_math_topic_cues_infer_math_and_middle_stage(message: str) -> None:
    route = choose_fast_route(message, "", "", "")

    assert route is not None
    assert route.classification.subject is Subject.MATH
    assert route.classification.stage is SchoolStage.MIDDLE


def test_short_why_followup_inherits_active_problem_context() -> None:
    route = choose_fast_route(
        "那为什么呢？",
        "初中数学：解方程 2x+3=7。",
        "数学",
        "初中",
    )

    assert route is not None
    assert route.classification.subject is Subject.MATH
    assert route.classification.stage is SchoolStage.MIDDLE
    assert route.classification.is_new_problem is False


@pytest.mark.parametrize("message", ["力是什么？", "这个变化是什么意思？"])
def test_ambiguous_generic_words_do_not_force_a_subject(message: str) -> None:
    assert choose_fast_route(message, "", "", "") is None


@pytest.mark.parametrize("message", ["锐角是什么？", "圆是什么？", "什么是圆？"])
def test_safe_basic_geometry_phrases_route_to_math(message: str) -> None:
    route = choose_fast_route(message, "", "", "")

    assert route is not None
    assert route.classification.subject is Subject.MATH


@pytest.mark.parametrize(
    "message",
    ["请搜索真实数据来源", "请帮我统计一下这段文章的字数"],
)
def test_ambiguous_math_topic_substrings_do_not_force_math(message: str) -> None:
    assert choose_fast_route(message, "", "", "") is None


@pytest.mark.parametrize("message", ["实数是什么？", "统计图怎么看？", "什么是统计？"])
def test_contextual_math_topic_phrases_still_route_to_math(message: str) -> None:
    route = choose_fast_route(message, "", "", "")

    assert route is not None
    assert route.classification.subject is Subject.MATH


def test_local_evidence_policy_requires_explicit_evidence_intent() -> None:
    assert fast_routing.needs_local_evidence("请根据教材解释惯性") is True
    assert fast_routing.needs_local_evidence("课标里怎样定义一次函数？") is True
    assert fast_routing.needs_local_evidence("惯性是什么？") is False


def test_dynamic_tool_policy_is_independent_from_reasoning_complexity() -> None:
    assert fast_routing.needs_dynamic_tool_loop("请联网查最新中考政策") is True
    assert fast_routing.needs_dynamic_tool_loop("搜索网页核对这条新闻") is True
    assert fast_routing.needs_dynamic_tool_loop("证明勾股定理") is False


def test_routine_concept_plan_does_not_search_any_retrieval_source() -> None:
    client = _NoModelCalls()
    registry = _FastRegistry()

    result = MultisubjectAgentLoop(client=client, registry=registry).run(
        "",
        "勾股定理是什么？",
        "",
        "",
    )

    assert client.calls == 0
    assert registry.calls == [("load_skill", {"name": "concept_explanation"})]
    assert result.retrieval_decision.mode is RetrievalMode.NONE
    assert result.rag_context.source == "none"


def test_explicit_textbook_request_uses_local_retrieval() -> None:
    client = _NoModelCalls()
    registry = _FastRegistry()

    result = MultisubjectAgentLoop(client=client, registry=registry).run(
        "",
        "请根据教材解释勾股定理。",
        "",
        "",
    )

    assert client.calls == 0
    assert [name for name, _ in registry.calls] == [
        "load_skill",
        "search_local_knowledge",
    ]
    assert result.retrieval_decision.mode is RetrievalMode.LOCAL


def test_short_concept_followup_does_not_repeat_local_retrieval() -> None:
    client = _NoModelCalls()
    registry = _FastRegistry()

    result = MultisubjectAgentLoop(client=client, registry=registry).run(
        "勾股定理是什么？",
        "那为什么呢？",
        "数学",
        "初中",
    )

    assert client.calls == 0
    assert registry.calls == [("load_skill", {"name": "concept_explanation"})]
    assert result.classification.is_new_problem is False
    assert result.retrieval_decision.mode is RetrievalMode.NONE
