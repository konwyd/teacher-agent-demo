from prototypes.llm_led_teacher.orchestration_models import (
    AnswerMode,
    RequestClassification,
    RequestScope,
    SchoolStage,
    Subject,
)
from prototypes.llm_led_teacher.retrieval_policy import (
    RetrievalDecision,
    RetrievalMode,
    apply_retrieval_guards,
)


def chinese_classification() -> RequestClassification:
    return RequestClassification(
        scope=RequestScope.K9,
        subject=Subject.CHINESE,
        stage=SchoolStage.MIDDLE,
        is_new_problem=True,
        answer_mode=AnswerMode.GUIDED,
        reason="语文写作指导",
    )


def test_rewriting_is_forced_to_no_retrieval() -> None:
    proposed = RetrievalDecision(
        mode=RetrievalMode.LOCAL_THEN_WEB,
        reason="模型提出搜索",
        confidence=0.8,
        needs_current_information=False,
    )
    guarded = apply_retrieval_guards(
        proposed,
        message="把这句话润色成一个有画面感的作文开头",
        classification=chinese_classification(),
    )
    assert guarded.mode is RetrievalMode.NONE
    assert guarded.reason == "这是基于用户文本的创作指导，无需外部资料"


def test_explicit_current_information_may_use_web() -> None:
    proposed = RetrievalDecision(
        mode=RetrievalMode.LOCAL_THEN_WEB,
        reason="需要核对最新信息",
        confidence=0.9,
        needs_current_information=True,
    )
    guarded = apply_retrieval_guards(
        proposed,
        message="查一下今天教育部最新发布的通知",
        classification=chinese_classification(),
    )
    assert guarded.mode is RetrievalMode.LOCAL_THEN_WEB
