"""Throwaway LLM-led teaching loop with verification-only orchestration.

Prototype question: can DeepSeek lead a complete multi-turn junior-math lesson
without planner, strategy, RAG, or mastery control, while a thin Agent shell
preserves context and audits mathematical claims?
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from app.agent import math_checker as mc
from app.agent.optional_verifier import LinearEquationVerifier, VerificationRequest
from app.agent.state import VerificationResult, VerificationStatus
from app.llm.deepseek_client import DeepSeekClient


class TeachingBehaviorLabel(str, Enum):
    """Observed behavior only; never controls response generation."""

    FOCUS = "focus"
    SCAFFOLD = "scaffold"
    EXPLAIN = "explain"
    CORRECT = "correct"
    CHECK = "check"
    MODEL = "model"
    PRACTICE = "practice"
    CONSOLIDATE = "consolidate"
    OTHER = "other"


class StudentAssessment(str, Enum):
    NOT_AN_ATTEMPT = "not_an_attempt"
    APPEARS_CORRECT = "appears_correct"
    APPEARS_INCORRECT = "appears_incorrect"
    UNCERTAIN = "uncertain"


class DisclosureLevel(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    FINAL = "final"


class DisclosureReason(str, Enum):
    NONE = "none"
    STUDENT_REQUESTED = "student_requested"
    STUDENT_ALREADY_PROPOSED = "student_already_proposed"
    VERIFIED_CORRECTION = "verified_correction"
    TEACHER_JUDGMENT = "teacher_judgment"


class TaskShape(str, Enum):
    DIRECT_FACT = "direct_fact"
    CONCEPTUAL = "conceptual"
    MULTI_STEP_PROBLEM = "multi_step_problem"
    OTHER = "other"


class ResponseScope(str, Enum):
    SINGLE_MOVE = "single_move"
    PARTIAL_SOLUTION = "partial_solution"
    FULL_SOLUTION = "full_solution"


class LLMTeachingTurn(BaseModel):
    """One autonomous teaching decision and response from the LLM."""

    response: str = Field(min_length=1)
    action_label: TeachingBehaviorLabel = TeachingBehaviorLabel.OTHER
    student_assessment: StudentAssessment = StudentAssessment.UNCERTAIN
    expects_student_response: bool = True
    disclosure_level: DisclosureLevel = DisclosureLevel.NONE
    disclosure_reason: DisclosureReason = DisclosureReason.NONE
    task_shape: TaskShape = TaskShape.OTHER
    response_scope: ResponseScope = ResponseScope.SINGLE_MOVE
    evidence_turn_ids: list[int] = Field(default_factory=list)
    mathematical_claims: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ClaimAudit(BaseModel):
    claim: str
    verification: VerificationResult


class PrototypeTurnRecord(BaseModel):
    turn_id: int
    student_message: str
    teacher_output: LLMTeachingTurn
    student_verification: VerificationResult
    evidence_level: str
    claim_audits: list[ClaimAudit] = Field(default_factory=list)
    repair_reasons: list[str] = Field(default_factory=list)
    remaining_warnings: list[str] = Field(default_factory=list)


class PrototypeEpisode(BaseModel):
    problem_context: str = ""
    previous_verified_step: str = ""
    turns: list[PrototypeTurnRecord] = Field(default_factory=list)


class StructuredTeacherClient(Protocol):
    def chat_json(
        self,
        system: str,
        user: str,
        schema: type[LLMTeachingTurn],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMTeachingTurn: ...


_SYSTEM_PROMPT = """你是一名中文初中数学老师，由你自主完成整轮教学。

你要自己理解题目、判断学生当前需要、选择方法、控制讲解深度，并生成自然的教师回复。
不存在外部 Planner、策略卡、RAG 或掌握度规则来指挥你。

必须遵守：
1. 始终围绕 problem_context 和真实对话历史，不要丢失原题。
2. deterministic_verification 是工具证据，优先级高于你的判断；不得与它冲突。
3. 没有确定性验证时可以正常推理和教学，但对正确性的表述使用“看起来”“按当前计算”等合适措辞。
4. 不要因为缺少 verifier 就退回“我不假设题型”或让学生重新说明已经清楚的信息。
5. 学生问为什么时直接解释原因；学生跳步时按实际内容回应。
6. 每轮最多提出一个真正必要的问题，也可以不提问；不要为了格式机械追问或要求反复复述依据。
7. 对 multi_step_problem，学生只说“怎么解”“我不会”“帮我”时，默认一轮只推进一个有意义的步骤；具体教哪一步由你决定。
8. 只有学生明确要求“直接给答案”“完整步骤”“从头到尾讲解”，或历史回合显示持续卡住时，才可以一次给出 full_solution。
9. direct_fact 可以直接回答；conceptual 可以完整解释概念，不要为了拆步骤而故意回避结论。
10. 必须如实填写 task_shape 和 response_scope；不能把完整解法标成 single_move。
11. 由你判断是否披露局部结果或最终答案，但必须如实填写 disclosure_level、disclosure_reason 和证据回合。
12. 学生明确说不要答案时不得给最终答案。
13. mathematical_claims 只列出你在回复中明确断言成立的关键算式、数值或结论；不要列问题、假设或原题复述。
14. action_label 只是对本轮行为的事后描述，不会限制你的教学。
"""


_FIRM_CORRECTNESS_PHRASES = (
    "完全正确",
    "你说得对",
    "是正确的",
    "算对了",
    "结果正确",
)

_NO_ANSWER_MARKERS = (
    "不要告诉我答案",
    "别告诉我答案",
    "不要直接给答案",
    "先别给答案",
)

_FULL_SOLUTION_REQUEST_MARKERS = (
    "直接告诉我答案",
    "直接给答案",
    "只要答案",
    "完整答案",
    "完整步骤",
    "全部步骤",
    "完整解答",
    "从头到尾",
    "完整讲解",
)


def _verify(
    problem_context: str,
    attempt: str,
    previous_verified_step: str = "",
) -> VerificationResult:
    verifier = LinearEquationVerifier()
    request = VerificationRequest(
        problem_context=problem_context,
        student_attempt=attempt,
        previous_verified_step=previous_verified_step,
    )
    if not verifier.supports(request):
        return VerificationResult(
            status=VerificationStatus.NOT_APPLICABLE,
            summary="当前题型没有确定性 verifier；允许 LLM 继续教学，但不写入已验证掌握证据。",
            verifier_name="none",
        )
    return verifier.verify(request)


def _history_payload(episode: PrototypeEpisode) -> list[dict[str, object]]:
    return [
        {
            "turn_id": turn.turn_id,
            "student": turn.student_message,
            "teacher": turn.teacher_output.response,
            "evidence_level": turn.evidence_level,
        }
        for turn in episode.turns[-12:]
    ]


def _teacher_payload(
    episode: PrototypeEpisode,
    message: str,
    verification: VerificationResult,
) -> str:
    return json.dumps(
        {
            "problem_context": episode.problem_context,
            "current_turn_id": len(episode.turns) + 1,
            "current_student_message": message,
            "conversation_history": _history_payload(episode),
            "deterministic_verification": verification.model_dump(mode="json"),
            "previous_verified_step": episode.previous_verified_step,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _audit_claims(
    problem_context: str,
    claims: list[str],
    previous_verified_step: str,
) -> list[ClaimAudit]:
    return [
        ClaimAudit(
            claim=claim,
            verification=_verify(problem_context, claim, previous_verified_step),
        )
        for claim in claims
    ]


def _question_count(text: str) -> int:
    return text.count("？") + text.count("?")


def _explicitly_requests_full_solution(message: str) -> bool:
    return any(marker in message for marker in _FULL_SOLUTION_REQUEST_MARKERS)


def _has_prior_disclosure_evidence(
    episode: PrototypeEpisode,
    output: LLMTeachingTurn,
) -> bool:
    return any(1 <= turn_id <= len(episode.turns) for turn_id in output.evidence_turn_ids)


def _has_verified_final_claim(claim_audits: list[ClaimAudit]) -> bool:
    for audit in claim_audits:
        if audit.verification.status is not VerificationStatus.VERIFIED_CORRECT:
            continue
        equation_text = mc.extract_equation(audit.claim)
        if equation_text is None:
            continue
        equation = mc.parse_equation(equation_text)
        if equation is not None and mc.is_final_form(equation):
            return True
    return False


def _collect_warnings(
    episode: PrototypeEpisode,
    message: str,
    verification: VerificationResult,
    output: LLMTeachingTurn,
    claim_audits: list[ClaimAudit],
) -> list[str]:
    warnings: list[str] = []
    if verification.status is VerificationStatus.VERIFIED_CORRECT:
        if output.student_assessment is StudentAssessment.APPEARS_INCORRECT:
            warnings.append("LLM把 verifier 已确认正确的学生步骤判断成错误。")
    elif verification.status is VerificationStatus.VERIFIED_INCORRECT:
        if output.student_assessment is not StudentAssessment.APPEARS_INCORRECT:
            warnings.append("LLM没有按 verifier 证据纠正错误步骤。")
    elif any(phrase in output.response for phrase in _FIRM_CORRECTNESS_PHRASES):
        warnings.append("没有确定性正确证据，却使用了绝对正确性肯定。")

    for audit in claim_audits:
        if audit.verification.status is VerificationStatus.VERIFIED_INCORRECT:
            warnings.append(f"教师数学断言与 verifier 冲突：{audit.claim}")

    if _question_count(output.response) > 1:
        warnings.append("教师回复包含多个问题。")

    if (
        any(marker in message for marker in _NO_ANSWER_MARKERS)
        and output.disclosure_level is DisclosureLevel.FINAL
    ):
        warnings.append("学生明确不要答案，但回复声明披露了最终答案。")

    current_turn_id = len(episode.turns) + 1
    if any(turn_id < 1 or turn_id > current_turn_id for turn_id in output.evidence_turn_ids):
        warnings.append("披露理由引用了不存在的对话回合。")
    if (
        output.disclosure_level is DisclosureLevel.FINAL
        and output.disclosure_reason is DisclosureReason.TEACHER_JUDGMENT
        and not output.evidence_turn_ids
    ):
        warnings.append("教师自主披露最终答案，但没有提供对话证据。")
    full_solution_claimed = (
        output.response_scope is ResponseScope.FULL_SOLUTION
        or (
            not episode.turns
            and (
                output.disclosure_level is DisclosureLevel.FINAL
                or _has_verified_final_claim(claim_audits)
            )
        )
    )
    full_solution_allowed = (
        _explicitly_requests_full_solution(message)
        or (
            output.disclosure_reason is DisclosureReason.TEACHER_JUDGMENT
            and _has_prior_disclosure_evidence(episode, output)
        )
    )
    if (
        output.task_shape is TaskShape.MULTI_STEP_PROBLEM
        and full_solution_claimed
        and not full_solution_allowed
    ):
        warnings.append(
            "多步题的普通求助不是完整解答请求；本轮应只推进一个有意义的步骤。"
        )
    return warnings


def _evidence_level(
    verification: VerificationResult,
    output: LLMTeachingTurn,
) -> str:
    if verification.status in {
        VerificationStatus.VERIFIED_CORRECT,
        VerificationStatus.VERIFIED_INCORRECT,
    }:
        return "verified"
    if output.student_assessment in {
        StudentAssessment.APPEARS_CORRECT,
        StudentAssessment.APPEARS_INCORRECT,
    }:
        return "plausible"
    return "unverified"


def _chat_teaching_turn(
    client: StructuredTeacherClient,
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int,
) -> LLMTeachingTurn:
    try:
        return client.chat_json(
            system,
            user,
            schema=LLMTeachingTurn,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ValidationError as exc:
        schema_retry_system = (
            system
            + "\n你上一次输出的 JSON 字段没有通过 LLMTeachingTurn Schema 校验。"
            "请根据下面的具体错误重新生成完整 JSON，只修正结构和字段归属。"
            "特别注意：direct_fact 只能用于 task_shape；disclosure_reason 必须从其自身枚举中选择。"
            "只输出修正后的 JSON 对象，不要解释错误。"
        )
        schema_retry_payload = json.dumps(
            {
                "original_request": json.loads(user),
                "schema_validation_errors": exc.errors(include_url=False),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return client.chat_json(
            schema_retry_system,
            schema_retry_payload,
            schema=LLMTeachingTurn,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def _repair(
    client: StructuredTeacherClient,
    payload: str,
    draft: LLMTeachingTurn,
    warnings: list[str],
) -> LLMTeachingTurn:
    repair_system = (
        _SYSTEM_PROMPT
        + "\n你上一版回复与审计证据存在冲突。只修正列出的冲突，保持自然教学和原有有效内容；"
        "不得退回泛化诊断文案。"
    )
    repair_payload = json.dumps(
        {
            "turn_context": json.loads(payload),
            "previous_draft": draft.model_dump(mode="json"),
            "must_fix": warnings,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _chat_teaching_turn(
        client,
        repair_system,
        repair_payload,
        temperature=0.2,
        max_tokens=900,
    )


def run_turn(
    episode: PrototypeEpisode,
    message: str,
    *,
    client: StructuredTeacherClient | None = None,
) -> PrototypeTurnRecord:
    """Run one autonomous LLM teaching turn with thin verification audits."""

    clean_message = message.strip()
    if not clean_message:
        raise ValueError("student message must not be empty")
    is_first_turn = not episode.turns
    if not episode.problem_context:
        episode.problem_context = clean_message

    if is_first_turn:
        verification = VerificationResult(
            status=VerificationStatus.UNABLE_TO_VERIFY,
            summary="首轮消息作为稳定原题，不把题目本身当作学生步骤判定。",
            verifier_name="none",
        )
    else:
        verification = _verify(
            episode.problem_context,
            clean_message,
            episode.previous_verified_step,
        )

    active_client = client or DeepSeekClient()
    payload = _teacher_payload(episode, clean_message, verification)
    output = _chat_teaching_turn(
        active_client,
        _SYSTEM_PROMPT,
        payload,
        temperature=0.4,
        max_tokens=900,
    )
    claim_audits = _audit_claims(
        episode.problem_context,
        output.mathematical_claims,
        episode.previous_verified_step,
    )
    repair_reasons = _collect_warnings(
        episode,
        clean_message,
        verification,
        output,
        claim_audits,
    )
    if repair_reasons:
        output = _repair(active_client, payload, output, repair_reasons)
        claim_audits = _audit_claims(
            episode.problem_context,
            output.mathematical_claims,
            episode.previous_verified_step,
        )

    remaining_warnings = _collect_warnings(
        episode,
        clean_message,
        verification,
        output,
        claim_audits,
    )
    record = PrototypeTurnRecord(
        turn_id=len(episode.turns) + 1,
        student_message=clean_message,
        teacher_output=output,
        student_verification=verification,
        evidence_level=_evidence_level(verification, output),
        claim_audits=claim_audits,
        repair_reasons=repair_reasons,
        remaining_warnings=remaining_warnings,
    )
    episode.turns.append(record)

    if verification.status is VerificationStatus.VERIFIED_CORRECT:
        verified_equation = mc.extract_equation(clean_message)
        if verified_equation is None and verification.evidence:
            verified_equation = mc.extract_equation(verification.evidence[0])
        if verified_equation:
            episode.previous_verified_step = verified_equation
    return record
