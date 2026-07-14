"""Compatibility adapter for the legacy linear-equation LangGraph workflow.

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
from app.agent.context_resolver import ContextResolver
from app.agent.math_checker import StepValidity
from app.agent.state import (
    DiagnosisResult,
    PedagogicalContext,
    RetrievalResult,
    StudentGap,
    StudentIntent,
    StudentMasteryUpdate,
    TeacherState,
    TeachingAction,
    TurnLog,
)
from app.agent.socratic_policy import choose_socratic_decision
from app.llm.deepseek_client import DeepSeekClient, DeepSeekError, is_configured
from app.memory.conversation_state import ConversationState, get_conversation_store
from app.memory.episode_state import get_episode_store
from app.memory.learning_history import append_learning_event
from app.memory.profile_updater import get_profile_store, summarize_profile, update_profile_from_event
from app.memory.schemas import EpisodeState, LearningEvent, now_iso
from app.memory.student_state import ConceptMastery, get_store
from app.retrieval.context_provider import (
    get_graph_context_provider,
    graph_context_enabled,
)
from app.retrieval.graph_retriever import (
    ConceptNode,
    RetrievalEvidence,
)

DEFAULT_CONCEPT_ID = "c_linear"

# These labels are compatibility metadata for the retained equation adapter,
# not a GraphRAG replacement or a core-harness knowledge package.  They let
# the adapter keep its stable state/API contract when graph context is off.
_FALLBACK_CONCEPT_NAMES = {
    "c_eq_prop": "等式性质",
    "c_combine": "合并同类项",
    "c_move": "移项",
    "c_bracket": "去括号",
    "c_linear": "一元一次方程",
    "c_frac": "分数系数方程",
}


def _use_pedagogical_rag() -> bool:
    """Feature flag for ablation: defaults on, false disables pedagogical graph use."""
    return graph_context_enabled() and os.getenv(
        "USE_PEDAGOGICAL_RAG", "true"
    ).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _fallback_concept(concept_id: str) -> ConceptNode:
    """Return minimal legacy metadata when no graph provider is configured."""

    return ConceptNode(
        id=concept_id,
        name=_FALLBACK_CONCEPT_NAMES.get(
            concept_id, _FALLBACK_CONCEPT_NAMES[DEFAULT_CONCEPT_ID]
        ),
    )

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
_PRACTICE_REQUEST_WORDS = ("再来一题", "再出一道", "再来一道", "再练一题", "出个题")
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
    # 分数系数只看 x 的系数：x=4/3 这类“答案是分数”的式子不属于分数系数方程
    if any(f.denominator != 1 for f in (a1, a2)):
        return "c_frac"
    if a1 != 0 and a2 != 0:
        return "c_move"
    return "c_linear"


def _detect_concept(message: str) -> tuple[ConceptNode, float]:
    provider = get_graph_context_provider()
    concept = provider.get_concept(message)
    if concept is not None:
        return concept, 0.9
    equation = mc.extract_equation(message)
    if equation is not None:
        structure = _classify_equation_structure(message, equation)
        found = provider.get_concept(structure) or _fallback_concept(structure)
        return found, 0.85
    for keywords, concept_id in _KEYWORD_RULES:
        if any(k in message for k in keywords):
            found = provider.get_concept(concept_id) or _fallback_concept(concept_id)
            return found, 0.6
    fallback = provider.get_concept(DEFAULT_CONCEPT_ID)
    if fallback is not None:
        return fallback, 0.3
    return _fallback_concept(DEFAULT_CONCEPT_ID), 0.3


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
        for prereq in get_graph_context_provider().get_prerequisites(concept.id):
            rec = store.get_concept_mastery(student_id, prereq.id)
            if rec.attempts > 0 and rec.mastery_score < 0.3:
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
    concept_names = [c.name for c in [get_graph_context_provider().get_concept(cid) for cid in
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
    concept = get_graph_context_provider().get_concept(diagnosis.detected_concept)
    if concept is None:
        concept, _ = _detect_concept(message)
        diagnosis.detected_concept = concept.name
    return diagnosis, concept


def _anchor_gap(
    diagnosis: DiagnosisResult, student_id: str, concept: ConceptNode, message: str
) -> None:
    """LLM 猜测的卡点必须有证据支撑，否则回退到规则判定。

    防止“我不会解 2x+3=7”这种普通求助被 LLM 过度诊断为 procedure_error /
    calculation_error / prerequisite_missing，导致教学动作过于激进（跳过一级提示）。
    """
    if diagnosis.student_gap not in (
        StudentGap.CALCULATION_ERROR,
        StudentGap.PROCEDURE_ERROR,
        StudentGap.PREREQUISITE_MISSING,
    ):
        return
    evidence_gap = StudentGap.UNKNOWN
    for keywords, value in _GAP_RULES:
        if any(k in message for k in keywords):
            evidence_gap = value
            break
    if evidence_gap is StudentGap.UNKNOWN:
        store = get_store()
        for prereq in get_graph_context_provider().get_prerequisites(concept.id):
            rec = store.get_concept_mastery(student_id, prereq.id)
            if rec.attempts > 0 and rec.mastery_score < 0.3:
                evidence_gap = StudentGap.PREREQUISITE_MISSING
                break
    if diagnosis.student_gap is not evidence_gap:
        if evidence_gap is StudentGap.UNKNOWN and diagnosis.intent is StudentIntent.CONCEPT_QUESTION:
            evidence_gap = StudentGap.CONCEPT_MISSING
        diagnosis.student_gap = evidence_gap
        diagnosis.reason += "（LLM 卡点判断缺乏证据，已回退规则判定）"


# ---------------------------------------------------------------- diagnose


def _conversation_key(state: TeacherState) -> str:
    return state.conversation_id or state.student_id


def _conversation_store_key(state: TeacherState) -> str:
    return f"{state.student_id}::{_conversation_key(state)}"


_PRACTICE_REQUEST_TYPES = {
    "generate_practice",
    "generate_similar_exercise",
    "generate_harder_exercise",
    "generate_easier_exercise",
}


_PRACTICE_ACTIONS = {
    TeachingAction.GENERATE_PRACTICE_QUESTION,
    TeachingAction.GENERATE_SIMILAR_EXERCISE,
    TeachingAction.GENERATE_HARDER_EXERCISE,
    TeachingAction.GENERATE_EASIER_EXERCISE,
    TeachingAction.SUMMARIZE_THEN_PRACTICE,
}


def _is_student_request(state: TeacherState) -> bool:
    return bool(state.context_resolution.student_request_type)


def context_resolver_node(state: TeacherState) -> dict:
    conversation_id = _conversation_key(state)
    episode = get_episode_store().get(state.student_id, conversation_id)
    profile = get_profile_store().get_profile(state.student_id)
    resolution = ContextResolver().resolve(
        state.message,
        episode_state=episode,
        recent_turns=episode.recent_turns,
        student_profile=profile,
    )
    pending_handled = (
        resolution.is_reply_to_previous_teacher
        and resolution.semantic_meaning == "verification_success"
    )
    return {
        "trace": state.trace + ["context_resolver_node"],
        "conversation_id": conversation_id,
        "episode_state": episode,
        "context_resolution": resolution,
        "student_request_type": resolution.student_request_type or "",
        "pending_reply_handled": pending_handled,
        "current_phase": episode.current_phase,
        "pending_question_type": episode.pending_question_type or "",
        "expected_student_response": episode.expected_student_response or "",
    }


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
    conv_key = _conversation_key(state)
    conv_store_key = _conversation_store_key(state)
    conv = store.get(conv_store_key)
    episode = state.episode_state or get_episode_store().get(state.student_id, conv_key)
    message = state.message
    extracted = mc.extract_equation(message)
    first_turn_for_problem = False

    if episode.original_problem and not conv.original_problem:
        conv.original_problem = episode.original_problem or ""
        conv.target_equation = episode.target_equation or ""
        conv.current_step = episode.current_solution_state or episode.target_equation or ""
        conv.solved_steps = list(episode.solved_steps)
        conv.solved = episode.is_problem_solved
        store.save(conv)

    if (
        state.context_resolution.semantic_meaning == "verification_success"
        and episode.original_problem
    ):
        concept, _confidence = _detect_concept(episode.original_problem)
        diagnosis = DiagnosisResult(
            intent=StudentIntent.ANSWER_CHECKING,
            detected_concept=concept.name,
            student_gap=StudentGap.UNKNOWN,
            confidence=0.95,
            reason="ContextResolver 判定学生在回答上一轮验算问题，跳过普通概念诊断。",
        )
        return {
            "trace": state.trace + ["diagnose_node"],
            "diagnosis": diagnosis,
            "detected_concept_id": concept.id,
            "detected_concept_name": concept.name,
            "original_problem": episode.original_problem or "",
            "target_equation": episode.target_equation or "",
            "lesson_goal": "solve_for_x",
            "current_step": episode.current_solution_state or "",
            "expected_next_step": "",
            "solved_steps": list(episode.solved_steps),
            "pending_teacher_question": episode.last_teacher_question or "",
            "last_student_answer": message,
            "student_step_validity": "verification_success",
            "student_equation": episode.current_solution_state or "",
            "step_explanation": "学生完成代回验算，确认左右两边相等",
            "response_kind": "verification_success",
            "plan_steps": [],
            "consecutive_wrong_steps": conv.consecutive_wrong_steps,
            "problem_solved": True,
            "pending_reply_handled": True,
            "current_phase": "verification",
            "pending_question_type": "verification_question",
            "expected_student_response": "verification_result",
        }

    if _is_student_request(state):
        source_problem = episode.original_problem or conv.original_problem or ""
        requested_concept, requested_confidence = _detect_concept(message)
        if requested_confidence >= 0.6:
            concept, confidence = requested_concept, requested_confidence
        else:
            concept, confidence = _detect_concept(source_problem or message)
        request_type = state.context_resolution.student_request_type or ""
        intent = StudentIntent.REVIEW if request_type in {
            "review_concept",
            "summarize_current_problem",
            "switch_topic",
            "end_current_episode",
        } else StudentIntent.PROBLEM_SOLVING
        diagnosis = DiagnosisResult(
            intent=intent,
            detected_concept=concept.name,
            student_gap=StudentGap.UNKNOWN,
            confidence=max(confidence, 0.8),
            reason=f"ContextResolver 判定这是教学请求：{request_type}，不按新数学题或原题延续处理。",
        )
        return {
            "trace": state.trace + ["diagnose_node"],
            "diagnosis": diagnosis,
            "detected_concept_id": concept.id,
            "detected_concept_name": concept.name,
            "original_problem": episode.original_problem or conv.original_problem,
            "target_equation": episode.target_equation or conv.target_equation,
            "lesson_goal": conv.lesson_goal or "solve_for_x",
            "current_step": episode.current_solution_state or conv.current_step,
            "expected_next_step": conv.expected_next_step,
            "solved_steps": list(episode.solved_steps or conv.solved_steps),
            "pending_teacher_question": episode.last_teacher_question or conv.pending_teacher_question,
            "last_student_answer": message,
            "consecutive_wrong_steps": conv.consecutive_wrong_steps,
            "problem_solved": episode.is_problem_solved or conv.solved,
            "current_phase": episode.current_phase,
            "pending_question_type": episode.pending_question_type or "",
            "expected_student_response": episode.expected_student_response or "",
        }

    # original_problem：首轮抽取；已存在则绝不覆盖（除非上一题已解完且带来新题）
    if not conv.original_problem:
        if extracted:
            _init_problem(conv, extracted)
            first_turn_for_problem = True
            store.save(conv)
            episode.original_problem = conv.original_problem
            episode.target_equation = conv.target_equation
            episode.current_phase = "solving"
            episode.current_solution_state = conv.current_step
            get_episode_store().save(episode)
    elif conv.solved and extracted and not mc.is_equivalent(extracted, conv.original_problem):
        conv = store.reset(conv_store_key)
        _init_problem(conv, extracted)
        first_turn_for_problem = True
        store.save(conv)
        episode = get_episode_store().reset(state.student_id, conv_key)
        episode.original_problem = conv.original_problem
        episode.target_equation = conv.target_equation
        episode.current_phase = "solving"
        episode.current_solution_state = conv.current_step
        get_episode_store().save(episode)

    # 基础诊断（知识点 / 基础意图 / 卡点）
    if is_configured():
        try:
            diagnosis, concept = _llm_diagnosis(state.student_id, message)
            _anchor_gap(diagnosis, state.student_id, concept, message)
        except DeepSeekError:
            diagnosis, concept = _rule_diagnosis(state.student_id, message)
            diagnosis.reason += "（LLM 诊断失败，已用规则 fallback）"
    else:
        diagnosis, concept = _rule_diagnosis(state.student_id, message)

    # 含方程的消息：概念以确定性结构规则为准（防止 LLM 把 3(x-2)=9 判成一元一次方程）
    if extracted is not None:
        concept, _conf = _detect_concept(message)
        diagnosis.detected_concept = concept.name

    # 围绕 original_problem 的后续轮次：按学生回答类型分派（why > 最终答案/多步计划/中间式/单步 > 普通诊断）。
    # 学生可以跳步、合并步骤、一次说完整思路——checker 整体判定，不强制一步一答。
    validity = ""
    student_eq = ""
    step_explanation = ""
    response_kind = ""
    plan_steps: list[str] = []
    if conv.original_problem and not first_turn_for_problem:
        followup: StudentIntent | None = None
        kind = mc.classify_student_math_response(message)
        step_gate = (
            any(k in message for k in _ANSWER_CHECK_WORDS)
            or not any(k in message for k in _HELP_WORDS)
        )
        if kind == "why_question":
            followup = StudentIntent.ASKS_WHY
        elif any(k in message for k in _CONFUSED_WORDS):
            followup = StudentIntent.CONFUSED
        elif any(k in message for k in _REQUEST_SOLUTION_WORDS):
            followup = StudentIntent.REQUESTS_SOLUTION
        elif kind in ("multi_step_plan", "single_step", "intermediate_equation", "final_answer") and step_gate:
            if kind == "multi_step_plan":
                plan = mc.check_multi_step_plan(
                    conv.current_step or conv.target_equation, message,
                    fallback_start=conv.target_equation,
                )
                if plan.valid:
                    validity = (StepValidity.CORRECT_FINAL_ANSWER if plan.reaches_final
                                else StepValidity.CORRECT_STEP).value
                    student_eq = plan.end_equation
                    plan_steps = list(plan.steps)
                    step_explanation = plan.explanation
                    response_kind = "multi_step_plan"
                elif plan.explanation != "不构成多步计划":
                    validity = StepValidity.INCORRECT_STEP.value
                    student_eq = plan.end_equation
                    step_explanation = plan.explanation
                    response_kind = "multi_step_plan"
            elif kind == "single_step":
                res = mc.check_single_operation(conv.current_step or conv.target_equation, message)
                if res.validity is not StepValidity.NOT_A_STEP:
                    validity = res.validity.value
                    student_eq = res.student_equation
                    step_explanation = res.explanation
                    response_kind = "single_step"
            elif kind == "final_answer":
                res = mc.check_final_answer(conv.target_equation, message)
                if res.validity is not StepValidity.NOT_A_STEP:
                    validity = res.validity.value
                    student_eq = res.student_equation
                    step_explanation = res.explanation
                    response_kind = "final_answer"
            else:  # intermediate_equation
                res = mc.check_step(conv.target_equation, message, prev_step=conv.current_step)
                if res.validity in (StepValidity.CORRECT_STEP, StepValidity.CORRECT_FINAL_ANSWER,
                                    StepValidity.INCORRECT_STEP):
                    validity = res.validity.value
                    student_eq = res.student_equation
                    step_explanation = res.explanation
                    response_kind = "intermediate_equation"
            if validity:
                if any(k in message for k in _ANSWER_CHECK_WORDS):
                    followup = StudentIntent.ANSWER_CHECKING
                elif validity == StepValidity.CORRECT_FINAL_ANSWER.value:
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
        "response_kind": response_kind,
        "plan_steps": plan_steps,
        "consecutive_wrong_steps": conv.consecutive_wrong_steps,
        "problem_solved": conv.solved,
    }


# ---------------------------------------------------------------- retrieve


def retrieve_node(state: TeacherState) -> dict:
    retriever = get_graph_context_provider()
    cid = state.detected_concept_id or DEFAULT_CONCEPT_ID
    concept = retriever.get_concept(cid)
    if state.pending_reply_handled:
        retrieval = RetrievalResult(concept_id=cid, concept_name=concept.name if concept else "")
        return {
            "trace": state.trace + ["retrieve_node"],
            "retrieval": retrieval,
            "pedagogical_context": PedagogicalContext(),
            "pedagogical_context_used": False,
            "retrieval_backend": retriever.name,
            "retrieval_evidence": [],
            "retrieved_concepts": [],
            "retrieved_misconceptions": [],
            "retrieved_hints": [],
            "retrieved_practice_problem_ids": [],
            "top_evidence_reasons": [],
        }
    record = get_store().get_concept_mastery(state.student_id, cid)
    retrieval = RetrievalResult(
        concept_id=cid,
        concept_name=concept.name if concept else "",
        prerequisites=[p.name for p in retriever.get_prerequisites(cid)],
        exercises=[e.stem for e in retriever.get_related_exercises(cid)[:2]],
        misconceptions=[m.description for m in retriever.get_misconceptions(cid)],
    )
    pedagogical_context = PedagogicalContext()
    pedagogical_context_used = False
    evidence: list[RetrievalEvidence] = []
    retrieval_backend = retriever.name
    retrieved_concepts: list[str] = []
    retrieved_misconceptions: list[str] = []
    retrieved_hints: list[str] = []
    retrieved_practice_problem_ids: list[str] = []
    top_evidence_reasons: list[str] = []
    if _use_pedagogical_rag() and retriever.available:
        graph_result = retriever.retrieve_pedagogical_evidence(
            cid,
            misconception=" ".join(
                filter(
                    None,
                    [
                        state.message,
                        state.step_explanation,
                        " ".join(state.known_misconceptions),
                    ],
                )
            ),
            mastery_score=record.mastery_score,
            request_type=state.context_resolution.student_request_type,
        )
        retrieval_backend = graph_result.retrieval_backend
        evidence = graph_result.evidence
        pedagogical_context_used = bool(graph_result.concept_id and evidence)

        concept_evidence = [item for item in evidence if item.node_type == "Concept"]
        misconception_evidence = [
            item for item in evidence if item.node_type == "Misconception"
        ]
        diagnostic_evidence = [
            item for item in evidence if item.node_type == "DiagnosticQuestion"
        ]
        hint_evidence = [item for item in evidence if item.node_type == "Hint"]
        practice_evidence = [
            item for item in evidence if item.node_type == "PracticeProblem"
        ]
        root_concept = next(
            (item for item in concept_evidence if item.node_id == graph_result.concept_id),
            None,
        )
        prerequisite_evidence = [
            item for item in concept_evidence if "REQUIRES" in item.relation_path
        ]
        hint_ladder = sorted(
            hint_evidence,
            key=lambda item: (int(item.content.get("level", 1)), -item.score),
        )
        retrieved_concepts = [item.node_id for item in concept_evidence]
        retrieved_misconceptions = [item.node_id for item in misconception_evidence]
        retrieved_hints = [item.node_id for item in hint_evidence]
        retrieved_practice_problem_ids = [item.node_id for item in practice_evidence]
        top_evidence_reasons = list(dict.fromkeys(item.reason for item in evidence[:5]))
        pedagogical_context = PedagogicalContext(
            concept_id=graph_result.concept_id,
            concept_name=str(root_concept.content.get("name", "")) if root_concept else "",
            prerequisites=[
                str(item.content.get("name") or item.node_id)
                for item in prerequisite_evidence
            ],
            common_misconceptions=[
                str(item.content.get("description", ""))
                for item in misconception_evidence
                if item.content.get("description")
            ],
            diagnostic_questions=[
                str(item.content.get("prompt", ""))
                for item in diagnostic_evidence
                if item.content.get("prompt")
            ],
            hint_levels=[
                str(item.content.get("text", ""))
                for item in hint_ladder
                if item.content.get("text")
            ],
            recommended_exercises=[
                str(item.content.get("stem", ""))
                for item in practice_evidence
                if item.content.get("stem")
            ],
            mastery_based_strategy=graph_result.mastery_strategy,
        )
    return {
        "trace": state.trace + ["retrieve_node"],
        "retrieval": retrieval,
        "pedagogical_context": pedagogical_context,
        "pedagogical_context_used": pedagogical_context_used,
        "retrieval_backend": retrieval_backend,
        "retrieval_evidence": evidence,
        "retrieved_concepts": retrieved_concepts,
        "retrieved_misconceptions": retrieved_misconceptions,
        "retrieved_hints": retrieved_hints,
        "retrieved_practice_problem_ids": retrieved_practice_problem_ids,
        "top_evidence_reasons": top_evidence_reasons,
    }


# ---------------------------------------------------------------- policy


def _choose_action(
    diagnosis: DiagnosisResult,
    record: ConceptMastery,
    message: str = "",
    validity: str = "",
    consecutive_wrong: int = 0,
) -> TeachingAction:
    """确定性教学策略（可单测）。规则从上到下优先级递减。

    Student Memory（掌握度）只影响教学深浅：解释详略、是否给类比、是否推荐
    复习、是否出变式题。学生的步骤/计划回答永远先由 math_checker 判定
    （PROVIDES_STEP/PROVIDES_FINAL_ANSWER 分支在所有 mastery 分支之前），
    memory 不得拒绝跳步、不得强制一步一答、不得覆盖 checker 的正确判定。

    show_worked_example 只在三种情况使用：
    a) 学生明确要例子；b) 同一知识点连续两次答错；c) 学生说完全不懂。
    （“直接告诉我答案”视作索要完整示范，也走 worked example。）
    """
    if any(k in message for k in _PRACTICE_REQUEST_WORDS):
        # 学生主动要题：掌握尚可给变式练习，否则先回到提示
        return (TeachingAction.GIVE_PRACTICE_QUESTION
                if record.mastery_score >= 0.5 else TeachingAction.GIVE_HINT_LEVEL_2)
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
    # 按掌握度分层（<0.3 多诊断少抽象；0.3-0.7 一级提示；>=0.7 独立做题）
    if record.mastery_score < 0.3:
        return TeachingAction.ASK_DIAGNOSTIC_QUESTION
    if record.mastery_score >= 0.7:
        return TeachingAction.GIVE_PRACTICE_QUESTION
    if record.attempts >= 1:
        return TeachingAction.GIVE_HINT_LEVEL_2
    return TeachingAction.GIVE_HINT_LEVEL_1


def _match_pedagogical_misconception(state: TeacherState) -> str:
    context = state.pedagogical_context
    candidates = context.common_misconceptions
    if not candidates:
        return ""
    evidence = " ".join(
        [
            state.message,
            state.step_explanation,
            state.student_equation,
            " ".join(state.known_misconceptions),
        ]
    )
    for item in candidates:
        if any(token and token in evidence for token in ("变号", "两边", "每一项", "分母", "符号")):
            if any(token in item for token in ("变号", "两边", "每一项", "分母", "符号")):
                return item
    if state.student_step_validity == StepValidity.INCORRECT_STEP.value:
        return candidates[0]
    if state.diagnosis and state.diagnosis.intent is StudentIntent.ASKS_WHY:
        return candidates[0]
    return ""


def _select_pedagogical_usage(
    state: TeacherState,
    record: ConceptMastery,
    action: TeachingAction,
) -> dict:
    """Pick the small part of Pedagogical RAG actually used this turn."""
    context = state.pedagogical_context
    if not state.pedagogical_context_used or not context.concept_id:
        return {
            "selected_hint_level": 0,
            "matched_misconception": "",
            "diagnostic_question_used": "",
            "recommended_exercise": "",
        }

    # Checker-judged student math always wins. Pedagogical RAG may label the
    # likely misconception, but it must not redirect correct steps into hints.
    checker_active = state.student_step_validity in (
        StepValidity.CORRECT_STEP.value,
        StepValidity.CORRECT_FINAL_ANSWER.value,
        StepValidity.INCORRECT_STEP.value,
    )
    matched = _match_pedagogical_misconception(state)
    if checker_active and action is TeachingAction.GRADE_AND_FEEDBACK:
        return {
            "selected_hint_level": 0,
            "matched_misconception": matched,
            "diagnostic_question_used": "",
            "recommended_exercise": "",
        }

    selected_hint_level = 0
    diagnostic_question = ""
    recommended_exercise = ""

    if record.mastery_score < 0.3:
        selected_hint_level = 1 if context.hint_levels else 0
        if action is TeachingAction.ASK_DIAGNOSTIC_QUESTION and context.diagnostic_questions:
            diagnostic_question = context.diagnostic_questions[0]
    elif record.mastery_score < 0.7:
        if action is TeachingAction.GIVE_HINT_LEVEL_2 and len(context.hint_levels) >= 2:
            selected_hint_level = 2
        elif action in (TeachingAction.GIVE_HINT_LEVEL_1, TeachingAction.EXPLAIN_CONCEPT):
            selected_hint_level = 1 if context.hint_levels else 0
    else:
        if action is TeachingAction.GIVE_PRACTICE_QUESTION and context.recommended_exercises:
            if not state.original_problem or state.problem_solved:
                recommended_exercise = context.recommended_exercises[0]

    return {
        "selected_hint_level": selected_hint_level,
        "matched_misconception": matched,
        "diagnostic_question_used": diagnostic_question,
        "recommended_exercise": recommended_exercise,
    }


def _practice_action_for_request(request_type: str) -> TeachingAction:
    if request_type == "generate_similar_exercise":
        return TeachingAction.GENERATE_SIMILAR_EXERCISE
    if request_type == "generate_harder_exercise":
        return TeachingAction.GENERATE_HARDER_EXERCISE
    if request_type == "generate_easier_exercise":
        return TeachingAction.GENERATE_EASIER_EXERCISE
    if request_type == "summarize_current_problem":
        return TeachingAction.SUMMARIZE_THEN_PRACTICE
    return TeachingAction.GENERATE_PRACTICE_QUESTION


def _extract_exercise_equation(text: str) -> str:
    return mc.extract_equation(text) or ""


def _practice_difficulty(record: ConceptMastery, request_type: str) -> str:
    if request_type == "generate_harder_exercise":
        return "hard"
    if request_type == "generate_easier_exercise":
        return "easy"
    if record.mastery_score < 0.3:
        return "easy"
    if record.mastery_score >= 0.7:
        return "hard"
    return "medium"


def _choose_practice_problem(state: TeacherState, record: ConceptMastery) -> dict:
    request_type = state.context_resolution.student_request_type or "generate_practice"
    difficulty = _practice_difficulty(record, request_type)
    source = state.detected_concept_id or DEFAULT_CONCEPT_ID
    original = state.original_problem or state.target_equation

    presets = {
        "easy": "x + 4 = 9",
        "medium": "3x + 4 = 13",
        "hard": "5x - 2 = 3x + 6",
    }
    problem = presets[difficulty]
    generated_id = f"generated_{source}_{difficulty}"
    reason = f"根据请求 {request_type} 和 mastery={record.mastery_score:.2f} 生成 {difficulty} 练习。"

    if state.pedagogical_context.recommended_exercises and difficulty != "easy":
        for idx, exercise in enumerate(state.pedagogical_context.recommended_exercises):
            candidate = _extract_exercise_equation(exercise)
            if candidate and (not original or not mc.is_equivalent(candidate, original)):
                problem = candidate
                generated_id = f"pedagogical_recommended_{idx + 1}"
                reason += " 优先参考 Pedagogical RAG 的 recommended_exercises。"
                break

    if original and mc.is_equivalent(problem, original):
        problem = "3x + 4 = 13" if difficulty != "hard" else "5x - 2 = 3x + 6"
        generated_id = f"generated_{source}_{difficulty}_fallback"
        reason += " 避免复用刚完成的原题。"

    return {
        "generated_practice_problem": problem,
        "generated_exercise_id": generated_id,
        "practice_difficulty": difficulty,
        "source_concept": source,
        "practice_generation_reason": reason,
    }


def teaching_policy_node(state: TeacherState) -> dict:
    diagnosis = state.diagnosis or DiagnosisResult()
    record = get_store().get_concept_mastery(state.student_id, state.detected_concept_id or DEFAULT_CONCEPT_ID)
    if state.pending_reply_handled:
        return {
            "trace": state.trace + ["teaching_policy_node"],
            "action": TeachingAction.SCHEDULE_REVIEW,
            "socratic_stage": "summary",
            "socratic_move": "confirm_verification",
            "socratic_reason": "ContextResolver 已确认学生在回答验算问题，直接总结原题。",
            "allowed_to_use_new_example": False,
            "selected_hint_level": 0,
            "matched_misconception": "",
            "diagnostic_question_used": "",
            "recommended_exercise": "",
        }
    request_type = state.context_resolution.student_request_type or ""
    if request_type:
        if request_type in _PRACTICE_REQUEST_TYPES or request_type == "summarize_current_problem":
            action = _practice_action_for_request(request_type)
            practice = _choose_practice_problem(state, record)
            nxt = mc.suggest_next_step(practice["generated_practice_problem"])
            expected = f"{nxt[0]}，得到 {nxt[1]}" if nxt else ""
            return {
                "trace": state.trace + ["teaching_policy_node"],
                "action": action,
                "socratic_stage": "rule_building",
                "socratic_move": "generate_practice",
                "socratic_reason": "学生请求老师生成练习题，使用画像和 Pedagogical RAG 选择难度。",
                "allowed_to_use_new_example": True,
                "original_problem": practice["generated_practice_problem"],
                "target_equation": practice["generated_practice_problem"],
                "current_step": practice["generated_practice_problem"],
                "expected_next_step": expected,
                "problem_solved": False,
                "current_phase": "practice_generation",
                "pending_question_type": "",
                "expected_student_response": "",
                "selected_hint_level": 0,
                "matched_misconception": "",
                "diagnostic_question_used": "",
                "recommended_exercise": practice["generated_practice_problem"],
                "student_request_type": request_type,
                "whether_new_episode_started": True,
                **practice,
            }
        if request_type == "review_concept":
            return {
                "trace": state.trace + ["teaching_policy_node"],
                "action": TeachingAction.SCHEDULE_REVIEW,
                "socratic_stage": "summary",
                "socratic_move": "review_known_concepts",
                "socratic_reason": "学生请求复习当前知识点。",
                "allowed_to_use_new_example": False,
                "selected_hint_level": 0,
                "matched_misconception": "",
                "diagnostic_question_used": "",
                "recommended_exercise": "",
                "student_request_type": request_type,
            }
    action = _choose_action(
        diagnosis,
        record,
        message=state.message,
        validity=state.student_step_validity,
        consecutive_wrong=state.consecutive_wrong_steps,
    )
    socratic = choose_socratic_decision(
        diagnosis=diagnosis,
        action=action,
        student_step_validity=state.student_step_validity,
        response_kind=state.response_kind,
        consecutive_wrong_steps=state.consecutive_wrong_steps,
    )
    pedagogical_usage = _select_pedagogical_usage(state, record, action)
    allowed = (
        action is TeachingAction.SHOW_WORKED_EXAMPLE
        or (
            action is TeachingAction.GIVE_PRACTICE_QUESTION
            and (not state.original_problem or state.problem_solved)
        )
    )
    return {
        "trace": state.trace + ["teaching_policy_node"],
        "action": action,
        "socratic_decision": socratic,
        "socratic_stage": socratic.stage,
        "socratic_move": socratic.move,
        "socratic_reason": socratic.reason,
        "allowed_to_use_new_example": allowed,
        **pedagogical_usage,
    }


# ---------------------------------------------------------------- teach

_ACTION_INSTRUCTIONS: dict[TeachingAction, str] = {
    TeachingAction.ASK_DIAGNOSTIC_QUESTION: "提出一个诊断性问题，弄清学生具体卡在哪里，不讲解。",
    TeachingAction.EXPLAIN_CONCEPT: "解释学生问的这个点（或缺失的前置知识），配最简单的说明，并回到原题。",
    TeachingAction.GIVE_HINT_LEVEL_1: "给一级提示：只提示原题第一步的思考方向，不给任何计算。",
    TeachingAction.GIVE_HINT_LEVEL_2: "给二级提示：指出原题具体的下一步操作，但不给出最终答案。",
    TeachingAction.SHOW_WORKED_EXAMPLE: "演示一道类似例题的完整步骤，结尾必须回到学生的原题。",
    TeachingAction.GIVE_PRACTICE_QUESTION: "布置一道练习，说明考察点。",
    TeachingAction.GENERATE_PRACTICE_QUESTION: "按学生请求生成一道新的练习题，并让学生先做第一步。",
    TeachingAction.GENERATE_SIMILAR_EXERCISE: "生成一道同类型但数字不同的练习题，并让学生先做第一步。",
    TeachingAction.GENERATE_HARDER_EXERCISE: "生成一道稍微进阶的练习题，并让学生独立尝试第一步。",
    TeachingAction.GENERATE_EASIER_EXERCISE: "生成一道更基础的练习题，并让学生先完成一个小步骤。",
    TeachingAction.SUMMARIZE_THEN_PRACTICE: "先简短总结当前题，再给一道新练习题。",
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
    "7. 每轮必须且只能问一个问题：整条回复恰好一个问号、放在结尾作为学生的下一个任务，"
    "严禁连续追问（如“你知道吗？该怎么做？能算出来吗？”）\n"
    "8. 不要在学生已经给出正确步骤时重新讲无关例子\n"
    "9. 如果使用了例子，必须先说明“这是类比”，并在结尾回到 original_problem\n"
    "10. 不要过早给出最终答案；循序渐进，一轮只讲一个小点。"
    "只有学生已经解出、或明确要求直接告诉答案时，才允许说出最终答案\n"
    "11. 回复中出现的任何方程都必须与 original_problem 同解；"
    "严禁无故输出 2x+3=7、3x=9 之类的示例方程（除非本轮是 show_worked_example）\n"
    "12. 学生不一定会严格按照你预期的一步一步回答：如果学生一次性说出了多个正确步骤，"
    "你应该认可完整思路，而不是强行拉回上一个小步骤；如果学生提前给出正确最终答案，"
    "你应该先认可，再用代回原式或简短解释确认；只有当学生步骤错误或缺失关键计算时，才继续追问\n"
    "13. pedagogical_context 是教学决策辅助：只使用与 student_message 和 teaching_action 相关的部分，"
    "不要照抄全部 RAG 内容；如果学生问为什么，优先使用 misconception/prerequisite/hint explanation\n"
    "14. recommended_exercise 只能在当前题完成后作为下一题推荐，不能污染 original_problem\n"
    "语气温和友好，回复控制在150字以内。"
)


def _build_teach_context(state: TeacherState) -> str:
    diagnosis = state.diagnosis or DiagnosisResult()
    retrieval = state.retrieval or RetrievalResult()
    pedagogical = state.pedagogical_context
    record = get_store().get_concept_mastery(state.student_id, state.detected_concept_id or DEFAULT_CONCEPT_ID)
    action = state.action or TeachingAction.GIVE_HINT_LEVEL_1
    if record.mastery_score < 0.3:
        mastery_note = "（掌握度较低：多问小的诊断问题，避免抽象术语，回复中必须使用“等式两边同时…”这样的基础语言）"
    elif record.mastery_score >= 0.7:
        mastery_note = "（掌握度较高：减少提示，让学生独立完成，可推荐变式题）"
    else:
        mastery_note = "（掌握度中等：给一级提示，让学生自己完成下一步）"
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
        f"卡点={diagnosis.student_gap.value}，掌握度={record.mastery_score:.2f}{mastery_note}\n"
        f"学生已知易错点：{'、'.join(record.common_misconceptions) or '无'}\n"
        f"retrieved_examples："
        f"{('；'.join(retrieval.exercises) or '无') if state.allowed_to_use_new_example else '（本轮禁止使用例题，不要引入任何新题目）'}\n"
        f"常见误解：{'；'.join(retrieval.misconceptions) or '无'}\n"
        f"pedagogical_context_used：{state.pedagogical_context_used}\n"
        f"pedagogical_prerequisites：{'；'.join(pedagogical.prerequisites[:3]) or '无'}\n"
        f"pedagogical_misconceptions：{'；'.join(pedagogical.common_misconceptions[:2]) or '无'}\n"
        f"pedagogical_hint_levels：{'；'.join(pedagogical.hint_levels[:2]) or '无'}\n"
        f"pedagogical_strategy：{pedagogical.mastery_based_strategy or '无'}\n"
        f"本轮实际选用：hint_level={state.selected_hint_level or '无'}；"
        f"diagnostic_question={state.diagnostic_question_used or '无'}；"
        f"matched_misconception={state.matched_misconception or '无'}；"
        f"recommended_exercise={state.recommended_exercise or '无'}\n"
        f"本轮 teaching_action：{action.value} —— {_ACTION_INSTRUCTIONS[action]}\n"
        f"本轮 Socratic 顾问建议：stage={state.socratic_stage or '无'}，"
        f"move={state.socratic_move or '无'}，reason={state.socratic_reason or '无'}\n"
        f"是否允许引入新例题：{'是' if state.allowed_to_use_new_example else '否'}"
    )


def _deepseek_response(state: TeacherState) -> str:
    return DeepSeekClient().chat(_TEACH_SYSTEM_PROMPT, _build_teach_context(state))


_CN_ORDINALS = ("一", "二", "三", "四", "五", "六", "七", "八", "九")


def _multi_step_plan_response(state: TeacherState) -> str:
    """认可学生的完整多步思路：先肯定，展开每一步，再按是否解完收尾。"""
    lines = [f"对，这个思路是正确的。对于 {state.original_problem}："]
    for i, step in enumerate(state.plan_steps):
        idx = _CN_ORDINALS[i] if i < len(_CN_ORDINALS) else str(i + 1)
        lines.append(f"第{idx}步，{step}。")
    if state.student_step_validity == StepValidity.CORRECT_FINAL_ANSWER.value:
        lines.append(
            f"所以这题已经解出来了。你愿意把 {state.student_equation} 代回原式检查一下吗？"
        )
    else:
        lines.append(f"到这里都对。{mc.next_question(state.student_equation)}")
    return "\n".join(lines)


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


def _verification_success_response(state: TeacherState) -> str:
    problem = state.original_problem
    answer = state.current_step or state.student_equation
    return (
        f"对，代回后两边一样，验算也成立。"
        f"这说明原题 {problem} 已经完成，答案 {answer} 是正确的。"
        f"我们可以进入下一题或复习易错点。"
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
    concept = get_graph_context_provider().get_concept(
        state.detected_concept_id or DEFAULT_CONCEPT_ID
    )
    desc = concept.description if concept else ""
    back = f"回到你的原题 {original}：" if original else ""
    return f"好问题！{desc}\n{back}你能说说现在卡在哪一小步吗？"


def _selected_hint_text(state: TeacherState) -> str:
    if state.selected_hint_level <= 0:
        return ""
    hints = state.pedagogical_context.hint_levels
    idx = state.selected_hint_level - 1
    if 0 <= idx < len(hints):
        return hints[idx]
    return ""


def _practice_generation_response(state: TeacherState) -> str:
    problem = state.generated_practice_problem or state.original_problem
    concept_name = state.detected_concept_name or "这个知识点"
    if state.practice_difficulty == "easy":
        level_note = "更基础的"
        prompt = "为了把 x 单独留下，应该先处理哪一项？"
    elif state.practice_difficulty == "hard":
        level_note = "稍微进阶的"
        prompt = "你先试着把含 x 的项移到同一边，可以吗？"
    else:
        level_note = "同类型但数字不同的"
        prompt = "你先试试第一步：应该先处理哪一项？"
    prefix = "可以。"
    if state.action is TeachingAction.SUMMARIZE_THEN_PRACTICE:
        prefix = f"小结一下：解{concept_name}时，通常先整理常数项，再把 x 的系数化为 1。"
    return (
        f"{prefix}我给你一题{level_note}练习：\n\n"
        f"{problem}\n\n"
        f"{prompt}"
    )


def _mock_response(state: TeacherState) -> str:
    """未配置 API key 时的确定性教学回复（严格围绕 original_problem）。"""
    diagnosis = state.diagnosis or DiagnosisResult()
    retrieval = state.retrieval or RetrievalResult()
    action = state.action or TeachingAction.GIVE_HINT_LEVEL_1
    concept_name = state.detected_concept_name or "这个知识点"
    original = state.original_problem

    if action in _PRACTICE_ACTIONS:
        return _practice_generation_response(state)

    if diagnosis.intent is StudentIntent.ASKS_WHY:
        return _why_response(state)

    if action is TeachingAction.ASK_DIAGNOSTIC_QUESTION and state.diagnostic_question_used:
        return (
            "先用“等式两边同时做同样变化”这个想法检查一下："
            f"{state.diagnostic_question_used}"
        )

    if action is TeachingAction.SHOW_WORKED_EXAMPLE:
        example = retrieval.exercises[0] if retrieval.exercises else ""
        back = f"现在回到你的原题 {original}，仿照这个思路，" if original else ""
        return (
            f"这是「{concept_name}」的题。我们先看一道类似的例题（这是类比，不是你的原题）：{example}\n"
            f"思路：先处理常数项，再把 x 的系数化为 1。\n"
            f"{back}你能写出第一步吗？"
        )

    if action is TeachingAction.GIVE_PRACTICE_QUESTION:
        exercise = state.recommended_exercise or (retrieval.exercises[0] if retrieval.exercises else "")
        if exercise and state.allowed_to_use_new_example:
            return f"你对「{concept_name}」掌握得不错。来做这道练习：{exercise}\n做完告诉我你的答案？"
        if original:
            return f"你对「{concept_name}」基础不错，这题先尽量自己推进一步。{mc.next_question(state.current_step or original)}"

    if original:
        # 围绕原题的分层提示（不引入检索例题）
        target = state.current_step or original
        nxt = mc.suggest_next_step(target)
        hint_text = _selected_hint_text(state)
        if action is TeachingAction.GIVE_HINT_LEVEL_2 and nxt:
            return (
                f"没关系，我们放慢一点。这是一道「{concept_name}」题，原题是 {original}，"
                f"现在做到 {target}。\n提示：{hint_text or nxt[0]}。你试着写出结果？"
            )
        question = mc.next_question(target)
        return (
            f"这是一道「{concept_name}」题。我们就用你的原题来做：{original}。\n"
            f"提示：{hint_text or '想把含 x 的项留下来'}。{question}"
        )

    # 没有具体原题：概念层面的引导（不引入新题号）
    concept = get_graph_context_provider().get_concept(
        state.detected_concept_id or DEFAULT_CONCEPT_ID
    )
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
    4. 没有问句 => 按教学动作补一个收尾小问题（每轮结尾必须给学生一个任务）
    5. 低掌握度学生的诊断式提问必须使用基础语言（等式两边同时…），
       否则降级为确定性模板
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
    if (
        state.action is TeachingAction.ASK_DIAGNOSTIC_QUESTION
        and state.original_problem
        and not any(m in response for m in BASIC_LANGUAGE_MARKERS)
    ):
        record = get_store().get_concept_mastery(
            state.student_id, state.detected_concept_id or DEFAULT_CONCEPT_ID
        )
        if record.mastery_score < 0.3:
            return _mock_response(state)  # 模板必然使用“两边同时…”的基础语言
    response = _keep_single_question(response)
    if response.count("？") + response.count("?") == 0:
        closing = _CLOSING_QUESTIONS.get(
            state.action or TeachingAction.GIVE_HINT_LEVEL_1,
            "你先试试这一步，写出来给我看看？",
        )
        response = response.rstrip("。！!.\n ") + "。" + closing
    return response


# 低掌握度学生要求的“基础语言”标记（等式两边同时操作类表述）
BASIC_LANGUAGE_MARKERS = ("两边同时", "等式两边", "两边都")

_CLOSING_QUESTIONS: dict[TeachingAction, str] = {
    TeachingAction.GIVE_PRACTICE_QUESTION: "做完把你的答案告诉我？",
    TeachingAction.GRADE_AND_FEEDBACK: "你觉得哪一步最没把握？",
    TeachingAction.SCHEDULE_REVIEW: "我们从哪个知识点开始复习好？",
    TeachingAction.ASK_DIAGNOSTIC_QUESTION: "你现在卡在哪一小步？",
}


def teach_node(state: TeacherState) -> dict:
    validity = state.student_step_validity
    retrieval = state.retrieval or RetrievalResult()
    if state.context_resolution.semantic_meaning == "verification_success":
        response = _verification_success_response(state)
    elif state.action in _PRACTICE_ACTIONS:
        response = _practice_generation_response(state)
    elif validity in (
        StepValidity.CORRECT_STEP.value,
        StepValidity.CORRECT_FINAL_ANSWER.value,
        StepValidity.INCORRECT_STEP.value,
    ):
        # checker 权威判定 => 确定性反馈，绝不让 LLM 重新判断对错
        if (
            state.response_kind == "multi_step_plan"
            and validity != StepValidity.INCORRECT_STEP.value
            and state.plan_steps
        ):
            # 学生一次说出完整思路：整体认可，不拆回单步追问
            response = _multi_step_plan_response(state)
        elif (
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


def _recommend_next_action(record: ConceptMastery) -> str:
    if record.mastery_score < 0.3:
        return "ask_diagnostic_question：先补基础，用“等式两边同时…”的语言做小步诊断"
    if record.mastery_score < 0.7:
        return "give_hint_level_1：给一级提示，让学生自己完成下一步"
    return "give_practice_question：掌握较好，推荐变式练习巩固"


def memory_update_node(state: TeacherState) -> dict:
    store = get_store()
    cid = state.detected_concept_id or DEFAULT_CONCEPT_ID
    before = store.get_concept_mastery(state.student_id, cid)
    validity = state.student_step_validity
    if validity in (
        StepValidity.CORRECT_STEP.value,
        StepValidity.CORRECT_FINAL_ANSWER.value,
        StepValidity.INCORRECT_STEP.value,
    ):
        step_result = validity
    elif state.context_resolution.semantic_meaning == "verification_success":
        step_result = "correct_step"
    elif state.context_resolution.student_request_type:
        step_result = "practice_requested"
    else:
        step_result = "help_requested"
    after = store.update_mastery_after_turn(
        state.student_id, cid, step_result, error_detail=state.student_equation
    )
    if validity == StepValidity.INCORRECT_STEP.value:
        misconception = mc.classify_misconception(
            state.current_step or state.target_equation, state.student_equation
        ) or f"同解性错误（{state.detected_concept_name or cid}）"
        after = store.record_misconception(state.student_id, cid, misconception)
    update = StudentMasteryUpdate(
        concept_id=cid,
        mastery_before=before.mastery_score,
        mastery_after=after.mastery_score,
        attempts=after.attempts,
    )
    return {
        "trace": state.trace + ["memory_update_node"],
        "mastery_update": update,
        "updated_student_state": {cid: after.mastery_score},
        "mastery_delta": round(after.mastery_score - before.mastery_score, 4),
        "known_misconceptions": list(after.common_misconceptions),
        "recommended_next_action": _recommend_next_action(after),
    }


# ---------------------------------------------------------------- log


def _log_path() -> Path:
    return Path(os.getenv("TEACHER_LOG_DIR", "logs")) / "turn_logs.jsonl"


def _last_question(response: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[？?])", response) if "？" in s or "?" in s]
    return sentences[-1] if sentences else ""


def _infer_skill_id(state: TeacherState, event_type: str) -> str | None:
    text = " ".join([state.message, state.response, state.step_explanation, state.detected_concept_name or ""])
    if "除" in text or "系数" in text:
        return "divide_by_coefficient"
    if "移项" in text or "变号" in text:
        return "move_terms"
    if "括号" in text:
        return "remove_parentheses"
    if event_type in ("correct_step", "solved_problem", "verification_success"):
        return "solve_linear_equation"
    return None


def _generate_learning_events(state: TeacherState) -> list[LearningEvent]:
    cid = state.detected_concept_id or DEFAULT_CONCEPT_ID
    base = {
        "student_id": state.student_id,
        "conversation_id": _conversation_key(state),
        "concept_id": cid,
        "problem": state.original_problem or None,
    }
    events: list[LearningEvent] = []
    validity = state.student_step_validity
    if validity == StepValidity.CORRECT_STEP.value:
        events.append(LearningEvent(
            **base,
            skill_id=_infer_skill_id(state, "correct_step"),
            event_type="correct_step",
            evidence_text=state.student_equation or state.message,
            confidence=0.95,
        ))
    elif validity == StepValidity.CORRECT_FINAL_ANSWER.value:
        events.append(LearningEvent(
            **base,
            skill_id=_infer_skill_id(state, "solved_problem"),
            event_type="solved_problem",
            evidence_text=state.student_equation or state.message,
            confidence=0.98,
        ))
    elif validity == StepValidity.INCORRECT_STEP.value:
        events.append(LearningEvent(
            **base,
            skill_id=_infer_skill_id(state, "wrong_step"),
            event_type="wrong_step",
            evidence_text=state.student_equation or state.message,
            confidence=0.9,
        ))
        misconception = mc.classify_misconception(
            state.current_step or state.target_equation, state.student_equation
        )
        if misconception:
            events.append(LearningEvent(
                **base,
                skill_id=_infer_skill_id(state, "misconception"),
                event_type="misconception",
                evidence_text=state.message,
                confidence=0.9,
                metadata={"misconception": misconception},
            ))
    elif state.context_resolution.semantic_meaning == "verification_success":
        events.append(LearningEvent(
            **base,
            skill_id="verify_solution",
            event_type="verification_success",
            evidence_text=state.message,
            confidence=0.95,
        ))
        events.append(LearningEvent(
            **base,
            skill_id="solve_linear_equation",
            event_type="completed_episode",
            evidence_text=state.response,
            confidence=0.95,
        ))
    elif state.diagnosis and state.diagnosis.intent is StudentIntent.ASKS_WHY:
        events.append(LearningEvent(
            **base,
            skill_id=None,
            event_type="asked_why",
            evidence_text=state.message,
            confidence=0.8,
        ))
    elif state.diagnosis and state.diagnosis.intent is StudentIntent.CONFUSED:
        events.append(LearningEvent(
            **base,
            skill_id=None,
            event_type="expressed_confusion",
            evidence_text=state.message,
            confidence=0.75,
        ))
    elif state.context_resolution.dialogue_act == "off_topic":
        events.append(LearningEvent(
            **base,
            skill_id=None,
            event_type="off_topic",
            evidence_text=state.message,
            confidence=0.7,
        ))
    elif state.context_resolution.student_request_type:
        events.append(LearningEvent(
            **base,
            skill_id=None,
            event_type="practice_requested",
            evidence_text=state.message,
            confidence=0.85,
            metadata={
                "student_request_type": state.context_resolution.student_request_type,
                "generated_problem": state.generated_practice_problem,
                "practice_difficulty": state.practice_difficulty,
            },
        ))
    return events


def _infer_pending_question(response: str, state: TeacherState) -> tuple[str, str, str]:
    if state.action in _PRACTICE_ACTIONS:
        return "practice_solving", "next_step_question", "remove_constant_term"
    if not _last_question(response):
        return "completed" if state.problem_solved else state.current_phase or "solving", "", ""
    if "代回" in response or "两边相等" in response:
        return "verification", "verification_question", "verification_result"
    if state.action is TeachingAction.ASK_DIAGNOSTIC_QUESTION:
        return "checking_step", "diagnostic_question", "diagnostic_answer"
    if "下一步" in response or "除以" in response or "写出结果" in response:
        return "solving", "next_step_question", "next_step_or_answer"
    if state.action is TeachingAction.GIVE_PRACTICE_QUESTION:
        return "solving", "practice_answer", "practice_solution"
    return state.current_phase or "solving", "summary_check", "student_ack"


def log_node(state: TeacherState) -> dict:
    log_id = uuid.uuid4().hex
    update = state.mastery_update
    events = _generate_learning_events(state)
    for event in events:
        append_learning_event(state.student_id, event)
        update_profile_from_event(event)
    profile_summary = summarize_profile(state.student_id)

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
        socratic_stage=state.socratic_stage,
        socratic_move=state.socratic_move,
        socratic_reason=state.socratic_reason,
        pedagogical_context_used=state.pedagogical_context_used,
        retrieval_backend=state.retrieval_backend,
        retrieval_evidence=state.retrieval_evidence,
        retrieved_concepts=state.retrieved_concepts,
        retrieved_misconceptions=state.retrieved_misconceptions,
        retrieved_hints=state.retrieved_hints,
        retrieved_practice_problem_ids=state.retrieved_practice_problem_ids,
        top_evidence_reasons=state.top_evidence_reasons,
        selected_hint_level=state.selected_hint_level,
        matched_misconception=state.matched_misconception,
        diagnostic_question_used=state.diagnostic_question_used,
        recommended_exercise=state.recommended_exercise,
        student_request_type=state.context_resolution.student_request_type or state.student_request_type,
        generated_practice_problem=state.generated_practice_problem,
        generated_exercise_id=state.generated_exercise_id,
        practice_generation_reason=state.practice_generation_reason,
        practice_difficulty=state.practice_difficulty,
        source_concept=state.source_concept,
        whether_new_episode_started=state.whether_new_episode_started,
        conversation_id=_conversation_key(state),
        current_phase=state.current_phase,
        pending_question_type=state.pending_question_type,
        expected_student_response=state.expected_student_response,
        context_resolution=state.context_resolution,
        learning_events_this_turn=events,
        updated_profile_summary=profile_summary,
    )
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")

    # 回写跨轮对话状态
    store = get_conversation_store()
    if state.generated_practice_problem:
        conv = store.reset(_conversation_store_key(state))
        _init_problem(conv, state.generated_practice_problem)
    else:
        conv = store.get(_conversation_store_key(state))
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

    # 回写 Layer 2 episode state
    episode = state.episode_state or get_episode_store().get(state.student_id, _conversation_key(state))
    episode.original_problem = state.original_problem or episode.original_problem
    episode.target_equation = state.target_equation or episode.target_equation
    episode.current_solution_state = state.student_equation or state.current_step or episode.current_solution_state
    if state.generated_practice_problem:
        episode.original_problem = state.generated_practice_problem
        episode.target_equation = state.generated_practice_problem
        episode.current_solution_state = state.generated_practice_problem
        episode.pending_practice_problem = state.generated_practice_problem
        episode.solved_steps = []
    if state.student_step_validity in (
        StepValidity.CORRECT_STEP.value,
        StepValidity.CORRECT_FINAL_ANSWER.value,
    ) and state.student_equation:
        if state.student_equation not in episode.solved_steps:
            episode.solved_steps.append(state.student_equation)
    phase, pending_type, expected = _infer_pending_question(state.response, state)
    if state.context_resolution.semantic_meaning == "verification_success":
        phase, pending_type, expected = "completed", "", ""
        episode.is_problem_solved = True
    elif state.student_step_validity == StepValidity.CORRECT_FINAL_ANSWER.value:
        episode.is_problem_solved = False
    else:
        episode.is_problem_solved = conv.solved and pending_type == ""
    episode.current_phase = phase
    episode.last_teacher_question = _last_question(state.response) or None
    episode.pending_question_type = pending_type or None
    episode.expected_student_response = expected or None
    episode.recent_turns = (episode.recent_turns + [{
        "timestamp": now_iso(),
        "student_message": state.message,
        "teacher_response": state.response,
        "dialogue_act": state.context_resolution.dialogue_act,
        "student_request_type": state.context_resolution.student_request_type,
        "generated_practice_problem": state.generated_practice_problem,
        "practice_difficulty": state.practice_difficulty,
        "teaching_action": state.action.value if state.action else "",
    }])[-8:]
    get_episode_store().save(episode)

    return {
        "trace": state.trace + ["log_node"],
        "log_id": log_id,
        "episode_state": episode,
        "current_phase": episode.current_phase,
        "pending_question_type": episode.pending_question_type or "",
        "expected_student_response": episode.expected_student_response or "",
        "learning_events_this_turn": events,
        "updated_profile_summary": profile_summary,
        "student_request_type": state.context_resolution.student_request_type or state.student_request_type,
        "generated_practice_problem": state.generated_practice_problem,
        "generated_exercise_id": state.generated_exercise_id,
        "practice_generation_reason": state.practice_generation_reason,
        "practice_difficulty": state.practice_difficulty,
        "source_concept": state.source_concept,
        "whether_new_episode_started": state.whether_new_episode_started,
    }


# ---------------------------------------------------------------- graph

_NODE_ORDER = [
    "context_resolver_node",
    "diagnose_node",
    "retrieve_node",
    "teaching_policy_node",
    "teach_node",
    "memory_update_node",
    "log_node",
]

_NODE_FUNCS = {
    "context_resolver_node": context_resolver_node,
    "diagnose_node": diagnose_node,
    "retrieve_node": retrieve_node,
    "teaching_policy_node": teaching_policy_node,
    "teach_node": teach_node,
    "memory_update_node": memory_update_node,
    "log_node": log_node,
}


_LEGACY_TURN_MARKERS = (
    "方程",
    "移项",
    "去括号",
    "同类项",
    "等式",
    "分数系数",
    "验算",
    "代入",
    "两边",
    "x",
    "X",
    "=",
)


def supports_legacy_equation_turn(
    student_id: str,
    message: str,
    conversation_id: str | None = None,
) -> bool:
    """Return whether a turn belongs to the retained equation compatibility path.

    All equation-specific recognition remains inside this adapter so the generic
    workflow never needs topic keywords, equation parsing, or a checker import.
    """

    if any(marker in message for marker in _LEGACY_TURN_MARKERS):
        return True
    episode = get_episode_store().find(student_id, conversation_id)
    return bool(episode and episode.target_equation)


def build_legacy_equation_workflow():
    """Build and compile the retained equation state machine."""
    graph = StateGraph(TeacherState)
    for name in _NODE_ORDER:
        graph.add_node(name, _NODE_FUNCS[name])
    graph.add_edge(START, _NODE_ORDER[0])
    for a, b in zip(_NODE_ORDER, _NODE_ORDER[1:]):
        graph.add_edge(a, b)
    graph.add_edge(_NODE_ORDER[-1], END)
    return graph.compile()


def run_legacy_equation_turn(
    student_id: str,
    message: str,
    conversation_id: str | None = None,
) -> TeacherState:
    """Run a supported linear-equation turn through the retained workflow."""

    app = build_legacy_equation_workflow()
    result = app.invoke(TeacherState(
        student_id=student_id,
        message=message,
        conversation_id=conversation_id,
    ))
    return TeacherState.model_validate(result)
