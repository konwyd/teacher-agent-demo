"""High-confidence routing that avoids unnecessary serial LLM decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent import math_checker as mc
from prototypes.llm_led_teacher.orchestration_models import (
    AnswerMode,
    RequestClassification,
    RequestScope,
    SchoolStage,
    Subject,
)
from prototypes.llm_led_teacher.teaching_policy import (
    has_explicit_full_solution_request,
    is_concept_request,
)


@dataclass(frozen=True)
class FastRoute:
    """A deterministic route returned only when all routing facts are clear."""

    classification: RequestClassification
    reason: str


_SUBJECT_LABELS = {subject.value: subject for subject in Subject}
_PURE_SOCIAL_MESSAGES = {
    "你好",
    "您好",
    "老师好",
    "早上好",
    "上午好",
    "下午好",
    "晚上好",
    "谢谢",
    "谢谢老师",
    "再见",
    "老师再见",
}
_MIDDLE_STAGE_MARKERS = (
    "初中",
    "中考",
    "七年级",
    "八年级",
    "九年级",
    "初一",
    "初二",
    "初三",
)
_PRIMARY_STAGE_MARKERS = (
    "小学",
    "小升初",
    "一年级",
    "二年级",
    "三年级",
    "四年级",
    "五年级",
    "六年级",
)
_CONTINUITY_MARKERS = (
    "这道题",
    "本题",
    "当前题",
    "上一题",
    "刚才",
    "继续",
    "下一步",
    "这一步",
    "上一步",
    "这里",
    "这个式子",
    "上次",
    "之前说的",
    "接着",
    "这个作文",
    "那个作文",
    "这篇作文",
    "那篇作文",
)
_NEW_MATH_PROBLEM_MARKERS = (
    "解方程",
    "求解",
    "请解",
    "新题",
    "另一题",
    "换一道",
    "再做一题",
)
_OFF_TOPIC_OBJECT_MARKERS = (
    "手机",
    "平板",
    "笔记本电脑",
    "耳机",
    "汽车",
    "产品",
    "企业邮箱",
    "股票",
    "基金",
    "游戏",
    "明星",
    "电影",
)
_OFF_TOPIC_INTENT_MARKERS = (
    "推荐",
    "购买",
    "买哪",
    "值得买",
    "比较几款",
    "哪一款",
    "用什么",
    "介绍一下",
    "是什么东西",
)
_QUALITY_QA_BENEFIT_MARKERS = (
    "怎么办",
    "为什么",
    "为啥",
    "怎么解释",
    "怎么理解",
    "怎么记",
    "有什么方法",
    "有什么技巧",
    "容易错",
    "总是错",
    "区别",
    "思路",
    "反馈",
    "批改",
    "作文",
    "阅读理解",
)
_UNIQUE_SUBJECT_CUES: tuple[tuple[Subject, tuple[str, ...]], ...] = (
    (Subject.CHINESE, ("作文", "阅读理解", "文言文", "古诗词")),
    (
        Subject.PHYSICS,
        (
            "惯性",
            "牛顿第一定律",
            "匀速直线运动",
            "浮力",
            "压强",
            "欧姆定律",
            "串联电路",
            "并联电路",
        ),
    ),
    (
        Subject.CHEMISTRY,
        ("化学方程式", "元素周期表", "酸碱盐", "质量守恒定律"),
    ),
    (Subject.BIOLOGY, ("细胞结构", "光合作用", "呼吸作用", "遗传和变异")),
    (Subject.HISTORY, ("历史事件", "朝代", "辛亥革命", "工业革命")),
    (Subject.GEOGRAPHY, ("经纬度", "等高线", "季风气候", "地形图")),
)
_MATH_EXPRESSION = re.compile(
    r"(?:[a-zA-Z]\s*[+\-*/^=]|\d\s*[+\-*/^=]\s*[a-zA-Z\d]|"
    r"(?:方程|函数|几何|求根公式|判别式))"
)


def choose_fast_route(
    message: str,
    problem_context: str,
    active_subject: str,
    active_stage: str,
) -> FastRoute | None:
    """Route only explicit cases; ambiguous input stays in the model-led loop."""

    clean = message.strip()
    if _is_pure_social_message(clean):
        classification = RequestClassification(
            scope=RequestScope.SOCIAL,
            subject=None,
            stage=SchoolStage.UNKNOWN,
            is_new_problem=False,
            answer_mode=AnswerMode.BRIEF_REDIRECT,
            reason="deterministic pure social interaction",
        )
        return FastRoute(classification, "明确的简短问候或礼貌用语")

    explicit_subject = _subject_from_message(clean)
    inferred_subject = explicit_subject or _subject_from_unique_cue(clean)
    if _is_high_confidence_off_topic(clean, inferred_subject):
        classification = RequestClassification(
            scope=RequestScope.OFF_TOPIC,
            subject=None,
            stage=SchoolStage.UNKNOWN,
            is_new_problem=True,
            answer_mode=AnswerMode.BRIEF_REDIRECT,
            reason="deterministic high-confidence off-topic route",
        )
        return FastRoute(classification, "明确的非课程产品或消费请求")

    continuity_subject = _enum_or_none(Subject, active_subject)
    continuity_stage = _enum_or_none(SchoolStage, active_stage)
    math_response_kind = mc.classify_student_math_response(clean)
    is_math_response_continuation = bool(
        problem_context.strip()
        and continuity_subject is Subject.MATH
        and math_response_kind != "unknown"
        and not any(marker in clean for marker in _NEW_MATH_PROBLEM_MARKERS)
    )
    has_continuity = bool(problem_context.strip()) and (
        any(marker in clean for marker in _CONTINUITY_MARKERS)
        or has_explicit_full_solution_request(clean)
        or is_concept_request(clean)
        or is_math_response_continuation
    )
    if (
        has_continuity
        and continuity_subject is not None
        and continuity_stage in {SchoolStage.PRIMARY, SchoolStage.MIDDLE}
        and (inferred_subject is None or inferred_subject is continuity_subject)
    ):
        classification = _k9_classification(
            clean,
            continuity_subject,
            continuity_stage,
            is_new_problem=False,
            reason="deterministic continuation of active problem",
        )
        return FastRoute(classification, "当前题追问且学科、学段已由会话确定")

    if inferred_subject is None and _MATH_EXPRESSION.search(clean):
        inferred_subject = Subject.MATH
    inferred_stage = _stage_from_message(clean)
    if inferred_stage is SchoolStage.UNKNOWN:
        active_stage_value = _enum_or_none(SchoolStage, active_stage)
        if active_stage_value in {SchoolStage.PRIMARY, SchoolStage.MIDDLE}:
            inferred_stage = active_stage_value
        elif inferred_subject is Subject.PHYSICS and any(
            cue in clean for cue in _UNIQUE_SUBJECT_CUES[0][1]
        ):
            inferred_stage = SchoolStage.MIDDLE

    if inferred_subject is not None:
        classification = _k9_classification(
            clean,
            inferred_subject,
            inferred_stage,
            is_new_problem=True,
            reason="deterministic explicit K9 subject route",
        )
        return FastRoute(classification, "消息提供了明确的学科和学段证据")
    return None


def _k9_classification(
    message: str,
    subject: Subject,
    stage: SchoolStage,
    *,
    is_new_problem: bool,
    reason: str,
) -> RequestClassification:
    use_quality_qa = any(
        marker in message for marker in _QUALITY_QA_BENEFIT_MARKERS
    )
    return RequestClassification(
        scope=RequestScope.K9,
        subject=subject,
        stage=stage,
        is_new_problem=is_new_problem,
        answer_mode=(
            AnswerMode.COMPLETE
            if has_explicit_full_solution_request(message)
            else AnswerMode.GUIDED
        ),
        use_quality_qa=use_quality_qa,
        quality_qa_reason=(
            "当前问题包含解释、方法或经验型诉求，可参考历史优质问答"
            if use_quality_qa
            else "当前问题走教材、题库或当前步骤链路即可"
        ),
        reason=reason,
    )


def _is_pure_social_message(message: str) -> bool:
    normalized = re.sub(r"[\s，。！？!?、~～]+", "", message)
    return normalized in _PURE_SOCIAL_MESSAGES


def _subject_from_message(message: str) -> Subject | None:
    matches = {
        subject
        for label, subject in _SUBJECT_LABELS.items()
        if label in message
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _subject_from_unique_cue(message: str) -> Subject | None:
    matches = {
        subject
        for subject, cues in _UNIQUE_SUBJECT_CUES
        if any(cue in message for cue in cues)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _stage_from_message(message: str) -> SchoolStage:
    has_primary = any(marker in message for marker in _PRIMARY_STAGE_MARKERS)
    has_middle = any(marker in message for marker in _MIDDLE_STAGE_MARKERS)
    if has_primary == has_middle:
        return SchoolStage.UNKNOWN
    return SchoolStage.PRIMARY if has_primary else SchoolStage.MIDDLE


def _is_high_confidence_off_topic(
    message: str,
    _inferred_subject: Subject | None,
) -> bool:
    has_object = any(marker in message for marker in _OFF_TOPIC_OBJECT_MARKERS)
    has_intent = any(marker in message for marker in _OFF_TOPIC_INTENT_MARKERS)
    return has_object and has_intent


def _enum_or_none(enum_type, value: str):
    try:
        return enum_type(value)
    except ValueError:
        return None
