"""Deterministic Socratic advisor layer.

This module adapts KELE-style consultant/teacher separation into the current
math-grounded workflow. It only emits a secondary teaching-stage signal; it
does not grade student math and must not override math_checker.
"""

from __future__ import annotations

from app.agent.math_checker import StepValidity
from app.agent.state import DiagnosisResult, SocraticDecision, StudentIntent, TeachingAction


def choose_socratic_decision(
    *,
    diagnosis: DiagnosisResult,
    action: TeachingAction,
    student_step_validity: str = "",
    response_kind: str = "",
    consecutive_wrong_steps: int = 0,
) -> SocraticDecision:
    """Choose a KELE-inspired advisor stage/move for this turn.

    The returned decision is intentionally advisory. Existing workflow policy
    and math_checker remain authoritative for correctness and allowed actions.
    """
    if student_step_validity == StepValidity.CORRECT_FINAL_ANSWER.value:
        return SocraticDecision(
            stage="summary",
            move="ask_student_to_verify",
            reason="学生已经给出正确最终答案，进入总结/代回检验阶段",
        )
    if student_step_validity == StepValidity.CORRECT_STEP.value:
        return SocraticDecision(
            stage="rule_building",
            move="ask_next_step_reason",
            reason="学生步骤正确，要求其继续解释或推进下一步",
        )
    if student_step_validity == StepValidity.INCORRECT_STEP.value:
        if action is TeachingAction.SHOW_WORKED_EXAMPLE or consecutive_wrong_steps >= 1:
            return SocraticDecision(
                stage="rule_building",
                move="offer_counterexample_or_worked_example",
                reason="同一知识点连续出错，使用类比或反例帮助学生重建规则",
            )
        return SocraticDecision(
            stage="induction",
            move="ask_student_to_reconsider",
            reason="学生步骤错误，先让学生重新检查同解变形",
        )

    if diagnosis.intent is StudentIntent.ASKS_WHY:
        return SocraticDecision(
            stage="induction",
            move="ask_reason",
            reason="学生追问原因，适合围绕等式性质追问/解释为什么",
        )
    if action is TeachingAction.EXPLAIN_CONCEPT:
        return SocraticDecision(
            stage="concept_probe",
            move="check_concept_understanding",
            reason="需要解释或检查概念掌握情况",
        )
    if action in (TeachingAction.GIVE_HINT_LEVEL_1, TeachingAction.ASK_DIAGNOSTIC_QUESTION):
        return SocraticDecision(
            stage="concept_probe",
            move="generate_subquestion",
            reason="处于概念探查阶段，用一个小问题探查学生当前理解",
        )
    if action is TeachingAction.GIVE_HINT_LEVEL_2:
        return SocraticDecision(
            stage="induction",
            move="guide_key_point",
            reason="需要更具体地引导关键步骤，但仍保留学生思考空间",
        )
    if action is TeachingAction.SHOW_WORKED_EXAMPLE:
        return SocraticDecision(
            stage="rule_building",
            move="offer_worked_example",
            reason="满足例题触发条件，用类比例题帮助学生建构规则",
        )
    if action is TeachingAction.GIVE_PRACTICE_QUESTION:
        return SocraticDecision(
            stage="rule_building",
            move="ask_student_to_apply_rule",
            reason="学生掌握度较高，要求应用规则完成变式练习",
        )
    if action is TeachingAction.SCHEDULE_REVIEW:
        return SocraticDecision(
            stage="summary",
            move="review_known_concepts",
            reason="学生请求复习，整理已学概念并安排巩固",
        )
    return SocraticDecision(
        stage="concept_probe",
        move="generate_subquestion",
        reason="默认用诊断性小问题继续探查学生理解",
    )
