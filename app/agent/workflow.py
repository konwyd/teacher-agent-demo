"""LangGraph structured teaching workflow, v3 (grounded).

六节点状态机：
  diagnose_node -> retrieve_node -> teaching_policy_node
  -> teach_node -> memory_update_node -> log_node

核心原则：Teacher Agent 必须始终围绕学生的 original_problem 教学。
- original_problem 在第一轮抽取，后续轮次不覆盖
- 学生的步骤回答优先用 math_checker 精确判定，不让 LLM 自由判断
- 检索例题只能辅助（仅 show_worked_example 可引入新题），用完必须回到原题
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from app.agent import math_checker as mc
from app.agent.math_checker import StepValidity
from app.agent.state import (
    DiagnosisResult,
    RetrievalResult,
    StudentGap,
    StudentIntent,
    StudentMasteryUpdate,
    TeacherState,
    TeachingAction,
    TurnLog,
)
from app.llm.deepseek_client import DeepSeekClient, DeepSeekError, is_configured
from app.memory.conversation_state import ConversationState, get_conversation_store
from app.memory.student_state import MasteryRecord, get_store
from app.retrieval.graph_retriever import ConceptNode, get_retriever

DEFAULT_CONCEPT_ID = "c_linear"

# ---------------------------------------------------------------- 规则表

_KEYWORD_RULES: list[tuple[tuple[str, ...], str]] = [
    (("分数", "分母", "/"), "c_frac"),
    (("括号", "(", "（"), "c_bracket"),
    (("同类项",), "c_combine"),
    (("移项",), "c_move"),
    (("等式性质", "等式的性质"), "c_eq_prop"),
    (("方程", "x", "解"), "c_linear"),
]

_INTENT_RULES: list[tuple[tuple[str, ...], StudentIntent]] = [
    (("对不对", "对吗", "检查", "我算出", "我的答案", "算得对"), StudentIntent.ANSWER_CHECKING),
    (("复习",), StudentIntent.REVIEW),
    (("是什么", "什么是", "为什么", "什么意思", "区别"), StudentIntent.CONCEPT_QUESTION),
    (("不会", "怎么解", "怎么做", "求解", "帮我解", "解方程", "="), StudentIntent.PROBLEM_SOLVING),
]

_GAP_RULES: list[tuple[tuple[str, ...], StudentGap]] = [
    (("算错", "总是错", "老是错", "计算错"), StudentGap.CALCULATION_ERROR),
    (("步骤", "顺序", "先做什么", "第一步"), StudentGap.PROCEDURE_ERROR),
]

_CONFUSED_WORDS = ("听不懂", "不明白", "还是不懂", "没听懂", "不理解", "越来越糊涂")
_REQUEST_SOLUTION_WORDS = ("直接告诉我", "告诉我答案", "直接给答案", "直接给我答案", "答案是多少", "直接说答案")
_ANSWER_CHECK_WORDS = ("对不对", "对吗", "检查", "算得对", "做得对")
_EXAMPLE_REQUEST_WORDS = ("举个例子", "举例", "给个例子", "来个例子", "看个例子")
_TOTALLY_LOST_WORDS = ("完全不懂", "完全听不懂", "一点都不懂", "一点也不懂")
_HELP_WORDS = ("怎么", "不会", "为什么", "什么", "吗", "帮", "?", "？", "呢", "教")


def is_step_message(message: str) -> bool:
    """消息是否应按“步骤回答”处理（供 diagnose 与 evaluator 共用）。

    带求助词的消息（如“我不会解 2x+3=7”）是提问不是步骤回答；
    但“我算的2x=4对不对”属于请求批改，仍走 checker。
    """
    return mc.has_math_content(message) and (
        any(k in message for k in _ANSWER_CHECK_WORDS)
        or not any(k in message for k in _HELP_WORDS)
    )


def _classify_equation_structure(message: str, formatted_eq: str) -> str:
    """按方程结构定位知识点：括号 > 分数系数 > 两边含x（移项）> 一般 ax+b=c。"""
    if "(" in message or "（" in message:
        return "c_bracket"
    parsed = mc.parse_equation(formatted_eq)
    if parsed is None:
        return DEFAULT_CONCEPT_ID
    (a1, b1), (a2, b2) = parsed
    if any(f.denominator != 1 for f in (a1, b1, a2, b2)):
        return "c_frac"
    if a1 != 0 and a2 != 0:
        return "c_move"
    return "c_linear"


def _detect_concept(message: str) -> tuple[ConceptNode, float]:
    retriever = get_retriever()
    concept = retriever.get_concept(message)
    if concept is not None:
        return concept, 0.9
    equation = mc.extract_equation(message)
    if equation is not None:
        found = retriever.get_concept(_classify_equation_structure(message, equation))
        if found is not None:
            return found, 0.85
    for keywords, concept_id in _KEYWORD_RULES:
        if any(k in message for k in keywords):
            found = retriever.get_concept(concept_id)
            if found is not None:
                return found, 0.6
    fallback = retriever.get_concept(DEFAULT_CONCEPT_ID)
    assert fallback is not None
    return fallback, 0.3


def _rule_diagnosis(student_id: str, message: str) -> tuple[DiagnosisResult, ConceptNode]:
    concept, confidence = _detect_concept(message)
    intent = StudentIntent.UNKNOWN
    for keywords, value in _INTENT_RULES:
        if any(k in message for k in keywords):
            intent = value
            break
    gap = StudentGap.UNKNOWN
    for keywords, value in _GAP_RULES:
        if any(k in message for k in keywords):
            gap = value
            break
    if gap is StudentGap.UNKNOWN:
        store = get_store()
        for prereq in get_retriever().get_prerequisites(concept.id):
            rec = store.get_mastery(student_id, prereq.id)
            if rec.attempts > 0 and rec.mastery < 0.3:
                gap = StudentGap.PREREQUISITE_MISSING
                break
    if gap is StudentGap.UNKNOWN and intent is StudentIntent.CONCEPT_QUESTION:
        gap = StudentGap.CONCEPT_MISSING
    diagnosis = DiagnosisResult(
        intent=intent,
        detected_concept=concept.name,
        student_gap=gap,
        confidence=confidence,
        reason=f"规则匹配：命中知识点「{concept.name}」，意图={intent.value}，卡点={gap.value}",
    )
    return diagnosis, concept


def _llm_diagnosis(student_id: str, message: str) -> tuple[DiagnosisResult, ConceptNode]:
    concept_names = [c.name for c in [get_retriever().get_concept(cid) for cid in
                     ("c_eq_prop", "c_combine", "c_move", "c_bracket", "c_linear", "c_frac")] if c]
    system = (
        "你是初中数学教学诊断器。根据学生消息判断：\n"
        "1. intent：concept_question/problem_solving/answer_checking/review/unknown\n"
        f"2. detected_concept：只能从这些知识点中选一个：{'、'.join(concept_names)}\n"
        f"3. student_gap：{'/'.join(g.value for g in StudentGap)}\n"
        "4. confidence：0.0-1.0\n"
        "5. reason：简要说明依据"
    )
    diagnosis = DeepSeekClient().chat_json(system, f"学生消息：{message}", schema=DiagnosisResult)
    concept = get_retriever().get_concept(diagnosis.detected_concept)
    if concept is None:
        concept, _ = _detect_concept(message)
        diagnosis.detected_concept = concept.name
    return diagnosis, concept


# ---------------------------------------------------------------- diagnose


def _init_problem(conv: ConversationState, equation: str) -> None:
    conv.original_problem = equation
    conv.target_equation = equation
    conv.current_step = equation
    conv.lesson_goal = "solve_for_x"
    conv.solved = False
    conv.solved_steps = []
    nxt = mc.suggest_next_step(equation)
    conv.expected_next_step = f"{nxt[0]}，得到 {nxt[1]}" if nxt else ""


def diagnose_node(state: TeacherState) -> dict:
    store = get_conversation_store()
    conv = store.get(state.student_id)
    message = state.message
    extracted = mc.extract_equation(message)
    first_turn_for_problem = False

    # original_problem：首轮抽取；已存在则绝不覆盖（除非上一题已解完且带来新题）
    if not conv.original_problem:
        if extracted:
            _init_problem(conv, extracted)
            first_turn_for_problem = True
            store.save(conv)
    elif conv.solved and extracted and not mc.is_equivalent(extracted, conv.original_problem):
        conv = store.reset(state.student_id)
        _init_problem(conv, extracted)
        first_turn_for_problem = True
        store.save(conv)

    # 基础诊断（知识点 / 基础意图 / 卡点）
    if is_configured():
        try:
            diagnosis, concept = _llm_diagnosis(state.student_id, message)
        except DeepSeekError:
            diagnosis, concept = _rule_diagnosis(state.student_id, message)
            diagnosis.reason += "（LLM 诊断失败，已用规则 fallback）"
    else:
        diagnosis, concept = _rule_diagnosis(state.student_id, message)

    # 含方程的消息：概念以确定性结构规则为准（防止 LLM 把 3(x-2)=9 判成一元一次方程）
    if extracted is not None:
        concept, _conf = _detect_concept(message)
        diagnosis.detected_concept = concept.name

    # 围绕 original_problem 的后续轮次：细分意图 + math_checker 优先判定
    validity = ""
    student_eq = ""
    step_explanation = ""
    if conv.original_problem and not first_turn_for_problem:
        followup: StudentIntent | None = None
        if "为什么" in message:
            followup = StudentIntent.ASKS_WHY
        elif any(k in message for k in _CONFUSED_WORDS):
            followup = StudentIntent.CONFUSED
        elif any(k in message for k in _REQUEST_SOLUTION_WORDS):
            followup = StudentIntent.REQUESTS_SOLUTION
        elif is_step_message(message):
            check = mc.check_step(conv.target_equation, message, prev_step=conv.current_step)
            if check.validity in (StepValidity.CORRECT_STEP, StepValidity.CORRECT_FINAL_ANSWER,
                                  StepValidity.INCORRECT_STEP):
                validity = check.validity.value
                student_eq = check.student_equation
                step_explanation = check.explanation
                if any(k in message for k in _ANSWER_CHECK_WORDS):
                    followup = StudentIntent.ANSWER_CHECKING
                elif check.validity is StepValidity.CORRECT_FINAL_ANSWER:
                    followup = StudentIntent.PROVIDES_FINAL_ANSWER
                else:
                    followup = StudentIntent.PROVIDES_STEP
        if followup is not None:
            diagnosis.intent = followup
            diagnosis.reason += f"；后续轮次围绕原题 {conv.original_problem}（意图细分={followup.value}）"

    return {
        "trace": state.trace + ["diagnose_node"],
        "diagnosis": diagnosis,
        "detected_concept_id": concept.id,
        "detected_concept_name": concept.name,
        "original_problem": conv.original_problem,
        "target_equation": conv.target_equation,
        "lesson_goal": conv.lesson_goal,
        "current_step": conv.current_step,
        "expected_next_step": conv.expected_next_step,
        "solved_steps": list(conv.solved_steps),
        "pending_teacher_question": conv.pending_teacher_question,
        "last_student_answer": message,
        "student_step_validity": validity,
        "student_equation": student_eq,
        "step_explanation": step_explanation,
        "consecutive_wrong_steps": conv.consecutive_wrong_steps,
    }


# ---------------------------------------------------------------- retrieve


def retrieve_node(state: TeacherState) -> dict:
    retriever = get_retriever()
    cid = state.detected_concept_id or DEFAULT_CONCEPT_ID
    concept = retriever.get_concept(cid)
    retrieval = RetrievalResult(
        concept_id=cid,
        concept_name=concept.name if concept else "",
        prerequisites=[p.name for p in retriever.get_prerequisites(cid)],
        exercises=[e.stem for e in retriever.get_related_exercises(cid)[:2]],
        misconceptions=[m.description for m in retriever.get_misconceptions(cid)],
    )
    return {"trace": state.trace + ["retrieve_node"], "retrieval": retrieval}


# ---------------------------------------------------------------- policy


def _choose_action(
    diagnosis: DiagnosisResult,
    record: MasteryRecord,
    message: str = "",
    validity: str = "",
    consecutive_wrong: int = 0,
) -> TeachingAction:
    """确定性教学策略（可单测）。规则从上到下优先级递减。

    show_worked_example 只在三种情况使用：
    a) 学生明确要例子；b) 同一知识点连续两次答错；c) 学生说完全不懂。
    （“直接告诉我答案”视作索要完整示范，也走 worked example。）
    """
    if any(k in message for k in _EXAMPLE_REQUEST_WORDS):
        return TeachingAction.SHOW_WORKED_EXAMPLE  # (a) 明确要例子
    if diagnosis.intent in (StudentIntent.PROVIDES_STEP, StudentIntent.PROVIDES_FINAL_ANSWER):
        if validity == StepValidity.INCORRECT_STEP.value and consecutive_wrong >= 1:
            return TeachingAction.SHOW_WORKED_EXAMPLE  # (b) 连续第二次答错
        return TeachingAction.GRADE_AND_FEEDBACK  # checker 已判定，只做反馈
    if diagnosis.intent is StudentIntent.ASKS_WHY:
        return TeachingAction.EXPLAIN_CONCEPT  # 先解释这个为什么，不引新题
    if diagnosis.intent is StudentIntent.CONFUSED:
        if any(k in message for k in _TOTALLY_LOST_WORDS):
            return TeachingAction.SHOW_WORKED_EXAMPLE  # (c) 完全不懂
        return TeachingAction.GIVE_HINT_LEVEL_2
    if diagnosis.intent is StudentIntent.REQUESTS_SOLUTION:
        return TeachingAction.SHOW_WORKED_EXAMPLE
    # 基础意图
    if diagnosis.intent is StudentIntent.ANSWER_CHECKING:
        return TeachingAction.GRADE_AND_FEEDBACK
    if diagnosis.intent is StudentIntent.REVIEW:
        return TeachingAction.SCHEDULE_REVIEW
    if diagnosis.student_gap is StudentGap.PREREQUISITE_MISSING:
        return TeachingAction.EXPLAIN_CONCEPT
    if diagnosis.student_gap is StudentGap.PROCEDURE_ERROR:
        return TeachingAction.GIVE_HINT_LEVEL_2  # 不再默认丢例题
    if diagnosis.student_gap is StudentGap.CALCULATION_ERROR:
        return TeachingAction.GIVE_HINT_LEVEL_2
    if diagnosis.intent is StudentIntent.CONCEPT_QUESTION:
        return TeachingAction.EXPLAIN_CONCEPT
    if diagnosis.confidence < 0.4:
        return TeachingAction.ASK_DIAGNOSTIC_QUESTION
    if record.mastery >= 0.7:
        return TeachingAction.GIVE_PRACTICE_QUESTION
    if record.attempts >= 1:
        return TeachingAction.GIVE_HINT_LEVEL_2
    return TeachingAction.GIVE_HINT_LEVEL_1


def teaching_policy_node(state: TeacherState) -> dict:
    diagnosis = state.diagnosis or DiagnosisResult()
    record = get_store().get_mastery(state.student_id, state.detected_concept_id or DEFAULT_CONCEPT_ID)
    action = _choose_action(
        diagnosis,
        record,
        message=state.message,
        validity=state.student_step_validity,
        consecutive_wrong=state.consecutive_wrong_steps,
    )
    allowed = action in (TeachingAction.SHOW_WORKED_EXAMPLE, TeachingAction.GIVE_PRACTICE_QUESTION)
    return {
        "trace": state.trace + ["teaching_policy_node"],
        "action": action,
        "allowed_to_use_new_example": allowed,
    }


# ---------------------------------------------------------------- teach

_ACTION_INSTRUCTIONS: dict[TeachingAction, str] = {
    TeachingAction.ASK_DIAGNOSTIC_QUESTION: "提出一个诊断性问题，弄清学生具体卡在哪里，不讲解。",
    TeachingAction.EXPLAIN_CONCEPT: "解释学生问的这个点（或缺失的前置知识），配最简单的说明，并回到原题。",
    TeachingAction.GIVE_HINT_LEVEL_1: "给一级提示：只提示原题第一步的思考方向，不给任何计算。",
    TeachingAction.GIVE_HINT_LEVEL_2: "给二级提示：指出原题具体的下一步操作，但不给出最终答案。",
    TeachingAction.SHOW_WORKED_EXAMPLE: "演示一道类似例题的完整步骤，结尾必须回到学生的原题。",
    TeachingAction.GIVE_PRACTICE_QUESTION: "布置一道练习，说明考察点。",
    TeachingAction.GRADE_AND_FEEDBACK: "根据步骤检查结果反馈：先肯定对的部分，指出错在哪一步。",
    TeachingAction.SCHEDULE_REVIEW: "安排复习：总结要点，给出复习建议。",
}

_TEACH_SYSTEM_PROMPT = (
    "你是一名耐心的中国初中数学老师。你必须严格遵守以下约束：\n"
    "1. 你必须优先回答学生当前这句话\n"
    "2. 你必须始终围绕 original_problem（学生的原题）教学\n"
    "3. retrieved_examples 只能作为辅助例子，绝不能替代 original_problem\n"
    "4. 除非本轮 teaching_action 是 show_worked_example，否则不要引入任何新题目\n"
    "5. 如果学生给出了正确步骤，先明确肯定，再继续原题的下一步\n"
    "6. 如果学生问“为什么”，先解释这个为什么，不要跳到新题\n"
    "7. 每轮最多问一个问题：整条回复只允许出现一个问号，结尾只能给学生一个任务，"
    "严禁连续追问（如“你知道吗？该怎么做？能算出来吗？”）\n"
    "8. 不要在学生已经给出正确步骤时重新讲无关例子\n"
    "9. 如果使用了例子，必须先说明“这是类比”，并在结尾回到 original_problem\n"
    "10. 不要过早给出最终答案；循序渐进，一轮只讲一个小点。"
    "只有学生已经解出、或明确要求直接告诉答案时，才允许说出最终答案\n"
    "11. 回复中出现的任何方程都必须与 original_problem 同解；"
    "严禁无故输出 2x+3=7、3x=9 之类的示例方程（除非本轮是 show_worked_example）\n"
    "语气温和友好，回复控制在150字以内。"
)


def _build_teach_context(state: TeacherState) -> str:
    diagnosis = state.diagnosis or DiagnosisResult()
    retrieval = state.retrieval or RetrievalResult()
    record = get_store().get_mastery(state.student_id, state.detected_concept_id or DEFAULT_CONCEPT_ID)
    action = state.action or TeachingAction.GIVE_HINT_LEVEL_1
    return (
        f"学生当前这句话：{state.message}\n"
        f"original_problem（必须围绕它教学）：{state.original_problem or '（暂无，学生还没给出具体题目）'}\n"
        f"当前推进到：{state.current_step or '尚未开始'}\n"
        f"已完成的正确步骤：{'；'.join(state.solved_steps) or '无'}\n"
        f"预期下一步：{state.expected_next_step or '无'}\n"
        f"上一轮老师提出的问题：{state.pending_teacher_question or '无'}\n"
        f"步骤检查结果（math checker 权威判定，必须以此为准）："
        f"{state.student_step_validity or '本轮不是步骤回答'}"
        f"{('，' + state.step_explanation) if state.step_explanation else ''}\n"
        f"诊断：知识点={state.detected_concept_name}，意图={diagnosis.intent.value}，"
        f"卡点={diagnosis.student_gap.value}，掌握度={record.mastery:.2f}\n"
        f"retrieved_examples："
        f"{('；'.join(retrieval.exercises) or '无') if state.allowed_to_use_new_example else '（本轮禁止使用例题，不要引入任何新题目）'}\n"
        f"常见误解：{'；'.join(retrieval.misconceptions) or '无'}\n"
        f"本轮 teaching_action：{action.value} —— {_ACTION_INSTRUCTIONS[action]}\n"
        f"是否允许引入新例题：{'是' if state.allowed_to_use_new_example else '否'}"
    )


def _deepseek_response(state: TeacherState) -> str:
    return DeepSeekClient().chat(_TEACH_SYSTEM_PROMPT, _build_teach_context(state))


def _step_feedback_response(state: TeacherState) -> str:
    """math_checker 判定后的确定性反馈（有无 API key 都用它，避免 LLM 误判）。"""
    validity = state.student_step_validity
    original = state.original_problem
    if validity == StepValidity.CORRECT_STEP.value:
        question = mc.next_question(state.student_equation)
        return (
            f"对，这一步是正确的。{state.step_explanation}。\n"
            f"现在继续解原题 {original}：{question}"
        )
    if validity == StepValidity.CORRECT_FINAL_ANSWER.value:
        return (
            f"完全正确！{state.step_explanation}。\n"
            f"我们把原题 {original} 解完了。把 {state.student_equation} 代回原方程检验一下，两边相等吗？"
        )
    # incorrect_step
    back_to = state.current_step or original
    question = mc.next_question(back_to)
    return (
        f"先停一下：{state.step_explanation}。\n"
        f"我们回到 {back_to} 这一步。{question}"
    )


def _why_response(state: TeacherState) -> str:
    """解释“为什么移项变号”类问题：等式性质 + 抵消过程，严格围绕原题，不引入新题。"""
    original = state.original_problem
    base = state.current_step or original
    nxt = mc.suggest_next_step(base) if base else None
    if original and nxt is not None:
        desc, next_eq = nxt
        parsed = mc.parse_equation(base)
        cancel = ""
        if parsed is not None:
            (a1, b1), (a2, b2) = parsed
            if b1 != 0 and a2 == 0:
                op = "减" if b1 > 0 else "加"
                cancel = (
                    f"左边 {'+' if b1 > 0 else ''}{mc.fmt_frac(b1)} {op}去后相互抵消，"
                    f"右边 {mc.fmt_frac(b2)} 也同样{op}去 {mc.fmt_frac(abs(b1))}，"
                )
        return (
            f"好问题！“移过去变号”不是把它搬过去，而是等式性质：{desc}，等式仍然成立"
            f"（就像天平两边同时拿走同样的重量，天平仍然平衡）。\n"
            f"{cancel}所以得到 {next_eq}。\n"
            f"回到你的原题 {original}：{mc.next_question(next_eq)}"
        )
    concept = get_retriever().get_concept(state.detected_concept_id or DEFAULT_CONCEPT_ID)
    desc = concept.description if concept else ""
    back = f"回到你的原题 {original}：" if original else ""
    return f"好问题！{desc}\n{back}你能说说现在卡在哪一小步吗？"


def _mock_response(state: TeacherState) -> str:
    """未配置 API key 时的确定性教学回复（严格围绕 original_problem）。"""
    diagnosis = state.diagnosis or DiagnosisResult()
    retrieval = state.retrieval or RetrievalResult()
    action = state.action or TeachingAction.GIVE_HINT_LEVEL_1
    concept_name = state.detected_concept_name or "这个知识点"
    original = state.original_problem

    if diagnosis.intent is StudentIntent.ASKS_WHY:
        return _why_response(state)

    if action is TeachingAction.SHOW_WORKED_EXAMPLE:
        example = retrieval.exercises[0] if retrieval.exercises else ""
        back = f"现在回到你的原题 {original}，仿照这个思路，" if original else ""
        return (
            f"这是「{concept_name}」的题。我们先看一道类似的例题（这是类比，不是你的原题）：{example}\n"
            f"思路：先处理常数项，再把 x 的系数化为 1。\n"
            f"{back}你能写出第一步吗？"
        )

    if action is TeachingAction.GIVE_PRACTICE_QUESTION and retrieval.exercises:
        return f"你对「{concept_name}」掌握得不错。来做这道练习：{retrieval.exercises[0]}\n做完告诉我你的答案？"

    if original:
        # 围绕原题的分层提示（不引入检索例题）
        target = state.current_step or original
        nxt = mc.suggest_next_step(target)
        if action is TeachingAction.GIVE_HINT_LEVEL_2 and nxt:
            return (
                f"没关系，我们放慢一点。这是一道「{concept_name}」题，原题是 {original}，"
                f"现在做到 {target}。\n下一步：{nxt[0]}。你试着写出结果？"
            )
        question = mc.next_question(target)
        return (
            f"这是一道「{concept_name}」题。我们就用你的原题来做：{original}。\n"
            f"提示：想把含 x 的项留下来。{question}"
        )

    # 没有具体原题：概念层面的引导（不引入新题号）
    concept = get_retriever().get_concept(state.detected_concept_id or DEFAULT_CONCEPT_ID)
    parts = [f"我们来看「{concept_name}」。"]
    if concept:
        parts.append(concept.description)
    if retrieval.prerequisites:
        parts.append(f"先回忆前置知识：{'、'.join(retrieval.prerequisites)}。")
    if retrieval.misconceptions:
        parts.append(f"注意常见错误：{retrieval.misconceptions[0]}")
    parts.append("你手上的题目是什么？发给我，我们一步步来？")
    return "\n".join(parts)


def _keep_single_question(response: str) -> str:
    """若回复中有多个问句，只保留最后一个（其余问句整句删除）。"""
    parts = re.split(r"(?<=[。！？!?\n])", response)
    questions = [p for p in parts if ("？" in p or "?" in p)]
    if len(questions) <= 1:
        return response
    last = questions[-1]
    kept = [p for p in parts if ("？" not in p and "?" not in p) or p is last]
    return "".join(kept).strip()


def postprocess_teacher_response(response: str, state: TeacherState) -> str:
    """LLM 回复的合规后处理：

    1. 非 worked_example 时出现与原题不同解的新方程 => 降级为确定性模板
    2. 学生未解出却出现最终答案（除非明确索要答案）=> 降级为确定性模板
    3. 多个问句 => 只保留最后一个
    """
    target = state.target_equation
    if target:
        student_eqs = {e.replace(" ", "") for e in mc.find_equations(state.message)}
        alien = [
            e for e in mc.find_equations(response)
            if e.replace(" ", "") not in student_eqs and not mc.is_equivalent(e, target)
        ]
        if alien and not state.allowed_to_use_new_example:
            return _mock_response(state)
        intent = state.diagnosis.intent if state.diagnosis else None
        sol = mc.solve_str(target)
        if (
            sol is not None
            and state.student_step_validity != StepValidity.CORRECT_FINAL_ANSWER.value
            and intent is not StudentIntent.REQUESTS_SOLUTION
        ):
            flat = response.replace(" ", "")
            if f"x={mc.fmt_frac(sol)}" in flat or f"{mc.fmt_frac(sol)}=x" in flat:
                return _mock_response(state)
    return _keep_single_question(response)


def teach_node(state: TeacherState) -> dict:
    validity = state.student_step_validity
    retrieval = state.retrieval or RetrievalResult()
    if validity in (
        StepValidity.CORRECT_STEP.value,
        StepValidity.CORRECT_FINAL_ANSWER.value,
        StepValidity.INCORRECT_STEP.value,
    ):
        # checker 权威判定 => 确定性反馈，绝不让 LLM 重新判断对错
        if (
            validity == StepValidity.INCORRECT_STEP.value
            and state.action is TeachingAction.SHOW_WORKED_EXAMPLE
            and retrieval.exercises
            and state.original_problem
        ):
            # 连续两次答错：给类比例题，再回到原题（仍只保留一个问题）
            response = (
                f"先停一下：{state.step_explanation}。\n"
                f"我们看一道类似的例题（这是类比）：{retrieval.exercises[0]}\n"
                f"仿照它的第一步，回到你的原题 {state.original_problem} 再试一次？"
            )
        else:
            response = _step_feedback_response(state)
    elif is_configured():
        try:
            response = postprocess_teacher_response(_deepseek_response(state), state)
        except DeepSeekError as exc:
            response = f"[错误] {exc}\n请检查网络与 .env 配置后重试。"
    else:
        response = _mock_response(state)
    return {"trace": state.trace + ["teach_node"], "response": response}


# ---------------------------------------------------------------- memory


def memory_update_node(state: TeacherState) -> dict:
    store = get_store()
    cid = state.detected_concept_id or DEFAULT_CONCEPT_ID
    before = store.get_mastery(state.student_id, cid)
    validity = state.student_step_validity
    if validity in (StepValidity.CORRECT_STEP.value, StepValidity.CORRECT_FINAL_ANSWER.value):
        after = store.update_mastery(state.student_id, cid, correct=True, weight=0.2)
    elif validity == StepValidity.INCORRECT_STEP.value:
        after = store.update_mastery(state.student_id, cid, correct=False, weight=0.2)
    else:
        after = store.update_mastery(state.student_id, cid, correct=False, weight=0.1)
    update = StudentMasteryUpdate(
        concept_id=cid,
        mastery_before=before.mastery,
        mastery_after=after.mastery,
        attempts=after.attempts,
    )
    return {
        "trace": state.trace + ["memory_update_node"],
        "mastery_update": update,
        "updated_student_state": {cid: after.mastery},
    }


# ---------------------------------------------------------------- log


def _log_path() -> Path:
    return Path(os.getenv("TEACHER_LOG_DIR", "logs")) / "turn_logs.jsonl"


def _last_question(response: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[？?])", response) if "？" in s or "?" in s]
    return sentences[-1] if sentences else ""


def log_node(state: TeacherState) -> dict:
    log_id = uuid.uuid4().hex
    update = state.mastery_update

    used_new_example = (
        mc.uses_new_example(state.response, state.target_equation) if state.target_equation else False
    )
    returned_to_original = (
        mc.mentions_equation(state.response, state.original_problem) if state.original_problem else True
    )

    entry = TurnLog(
        log_id=log_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        student_id=state.student_id,
        user_message=state.message,
        diagnosis=state.diagnosis or DiagnosisResult(),
        retrieval=state.retrieval or RetrievalResult(),
        teaching_action=state.action.value if state.action else "",
        teacher_response=state.response,
        student_state_before={update.concept_id: update.mastery_before} if update else {},
        student_state_after={update.concept_id: update.mastery_after} if update else {},
        original_problem=state.original_problem,
        target_equation=state.target_equation,
        student_step_validity=state.student_step_validity,
        whether_response_used_new_example=used_new_example,
        whether_response_returned_to_original_problem=returned_to_original,
    )
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")

    # 回写跨轮对话状态
    store = get_conversation_store()
    conv = store.get(state.student_id)
    conv.last_student_answer = state.message
    conv.pending_teacher_question = _last_question(state.response)
    if state.student_step_validity == StepValidity.CORRECT_STEP.value and state.student_equation:
        conv.current_step = state.student_equation
        conv.solved_steps = conv.solved_steps + [state.student_equation]
        nxt = mc.suggest_next_step(state.student_equation)
        conv.expected_next_step = f"{nxt[0]}，得到 {nxt[1]}" if nxt else ""
    elif state.student_step_validity == StepValidity.CORRECT_FINAL_ANSWER.value:
        conv.current_step = state.student_equation
        conv.solved_steps = conv.solved_steps + [state.student_equation]
        conv.expected_next_step = ""
        conv.solved = True
    if state.student_step_validity == StepValidity.INCORRECT_STEP.value:
        conv.consecutive_wrong_steps += 1
    elif state.student_step_validity in (
        StepValidity.CORRECT_STEP.value, StepValidity.CORRECT_FINAL_ANSWER.value
    ):
        conv.consecutive_wrong_steps = 0
    store.save(conv)

    return {"trace": state.trace + ["log_node"], "log_id": log_id}


# ---------------------------------------------------------------- graph

_NODE_ORDER = [
    "diagnose_node",
    "retrieve_node",
    "teaching_policy_node",
    "teach_node",
    "memory_update_node",
    "log_node",
]

_NODE_FUNCS = {
    "diagnose_node": diagnose_node,
    "retrieve_node": retrieve_node,
    "teaching_policy_node": teaching_policy_node,
    "teach_node": teach_node,
    "memory_update_node": memory_update_node,
    "log_node": log_node,
}


def build_workflow():
    """Build and compile the teaching state machine."""
    graph = StateGraph(TeacherState)
    for name in _NODE_ORDER:
        graph.add_node(name, _NODE_FUNCS[name])
    graph.add_edge(START, _NODE_ORDER[0])
    for a, b in zip(_NODE_ORDER, _NODE_ORDER[1:]):
        graph.add_edge(a, b)
    graph.add_edge(_NODE_ORDER[-1], END)
    return graph.compile()


def run_turn(student_id: str, message: str) -> TeacherState:
    """Run one teaching turn and return the final state."""
    app = build_workflow()
    result = app.invoke(TeacherState(student_id=student_id, message=message))
    return TeacherState.model_validate(result)
