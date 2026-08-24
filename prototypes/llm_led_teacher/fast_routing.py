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
    "那为什么",
    "为什么呢",
    "接下来呢",
    "这是什么意思",
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
    "怎么记",
    "有什么方法",
    "有什么技巧",
    "容易错",
    "总是错",
    "总是写错",
    "常见错误",
    "反馈",
    "批改",
    "作文",
    "阅读理解",
)
_MATH_TOPIC_CUES = (
    "勾股定理",
    "平方根",
    "算术平方根",
    "立方根",
    "有理数",
    "无理数",
    "实数",
    "绝对值",
    "整式",
    "单项式",
    "多项式",
    "因式分解",
    "分式",
    "二次根式",
    "一元一次方程",
    "二元一次方程组",
    "一元二次方程",
    "不等式组",
    "不等式",
    "平面直角坐标系",
    "一次函数",
    "正比例函数",
    "反比例函数",
    "二次函数",
    "锐角",
    "钝角",
    "全等三角形",
    "相似三角形",
    "等腰三角形",
    "直角三角形",
    "三角形",
    "平行四边形",
    "矩形",
    "菱形",
    "正方形",
    "圆周角",
    "圆心角",
    "圆是什么",
    "什么是圆",
    "圆的定义",
    "圆的面积",
    "圆的周长",
    "平均数",
    "中位数",
    "众数",
    "方差",
    "概率",
    "统计",
)
_MIDDLE_MATH_TOPIC_CUES = (
    "勾股定理",
    "平方根",
    "算术平方根",
    "立方根",
    "无理数",
    "实数",
    "整式",
    "单项式",
    "多项式",
    "因式分解",
    "分式",
    "二次根式",
    "一元一次方程",
    "二元一次方程组",
    "一元二次方程",
    "不等式组",
    "平面直角坐标系",
    "一次函数",
    "正比例函数",
    "反比例函数",
    "二次函数",
    "全等三角形",
    "相似三角形",
    "平行四边形",
    "菱形",
    "圆周角",
    "圆心角",
    "方差",
)
_SUBJECT_CUES: dict[Subject, tuple[str, ...]] = {
    Subject.MATH: _MATH_TOPIC_CUES,
    Subject.CHINESE: (
        "作文",
        "阅读理解",
        "文言文",
        "古诗词",
        "修辞手法",
        "病句修改",
    ),
    Subject.ENGLISH: (
        "现在完成时",
        "一般过去时",
        "过去进行时",
        "被动语态",
        "定语从句",
        "宾语从句",
        "英语音标",
    ),
    Subject.PHYSICS: (
        "惯性",
        "牛顿第一定律",
        "匀速直线运动",
        "浮力",
        "压强",
        "欧姆定律",
        "串联电路",
        "并联电路",
    ),
    Subject.CHEMISTRY: (
        "化学方程式",
        "元素周期表",
        "酸碱盐",
        "质量守恒定律",
        "化合价",
    ),
    Subject.BIOLOGY: (
        "细胞结构",
        "光合作用",
        "呼吸作用",
        "遗传和变异",
        "生态系统",
    ),
    Subject.HISTORY: (
        "历史事件",
        "朝代",
        "辛亥革命",
        "工业革命",
        "鸦片战争",
    ),
    Subject.GEOGRAPHY: (
        "经纬度",
        "等高线",
        "季风气候",
        "地形图",
        "世界气候类型",
    ),
    Subject.POLITICS: (
        "公民基本权利和义务",
        "依法治国",
        "社会主义核心价值观",
        "未成年人保护法",
        "权利与义务的关系",
    ),
}
_LOCAL_EVIDENCE_MARKERS = (
    "教材",
    "课本",
    "课标",
    "课程标准",
    "原文",
    "原句",
    "出处",
    "哪一页",
    "练习册",
    "题库",
    "教材例题",
    "原题",
)
_DYNAMIC_TOOL_MARKERS = (
    "联网",
    "上网",
    "搜索",
    "查资料",
    "查官网",
    "官网",
    "网页",
    "最新",
    "实时",
    "新闻",
    "近期",
    "来源",
    "出处",
    "引用",
    "根据网页",
    "网络资料",
)
_CONTEXTUAL_SUBJECT_CUE_PATTERNS: dict[
    tuple[Subject, str], re.Pattern[str]
] = {
    (Subject.MATH, "实数"): re.compile(
        r"(?:^|什么是|数学中的?|学习|认识)实数"
        r"(?:$|[？?。！!]|是什么|包括|范围|集|轴|运算|与)"
    ),
    (Subject.MATH, "统计"): re.compile(
        r"(?:^|什么是|数学中的?|概率与|数据的?)统计"
        r"(?:$|[？?。！!]|是什么|图|表|量|学|知识|方法|规律|意义)"
        r"|统计(?:图|表|量|学)"
    ),
}
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
        elif inferred_subject in {Subject.PHYSICS, Subject.CHEMISTRY} and any(
            _cue_matches(inferred_subject, cue, clean)
            for cue in _SUBJECT_CUES[inferred_subject]
        ):
            inferred_stage = SchoolStage.MIDDLE
        elif inferred_subject is Subject.MATH and any(
            _cue_matches(Subject.MATH, cue, clean)
            for cue in _MIDDLE_MATH_TOPIC_CUES
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
            "当前问题包含经验、技巧、常见错误或反馈诉求，可参考历史优质问答"
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
        for subject, cues in _SUBJECT_CUES.items()
        if any(_cue_matches(subject, cue, message) for cue in cues)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _cue_matches(subject: Subject, cue: str, message: str) -> bool:
    contextual_pattern = _CONTEXTUAL_SUBJECT_CUE_PATTERNS.get((subject, cue))
    if contextual_pattern is not None:
        return bool(contextual_pattern.search(message.strip()))
    return cue in message


def needs_local_evidence(message: str) -> bool:
    """Return whether the student explicitly requests local source evidence."""

    return any(marker in message for marker in _LOCAL_EVIDENCE_MARKERS)


def needs_dynamic_tool_loop(
    message: str,
    classification: RequestClassification | None = None,
) -> bool:
    """Return whether the request needs dynamic external-tool selection."""

    if classification is not None and classification.scope is not RequestScope.K9:
        return False
    return any(marker in message for marker in _DYNAMIC_TOOL_MARKERS)


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
