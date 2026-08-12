import json

from prototypes.llm_led_teacher.boundary_audit import (
    SemanticBoundaryAuditor,
    hard_boundary_recheck,
)
from prototypes.llm_led_teacher.orchestration_models import (
    BoundaryAudit,
    BoundaryViolation,
    TeachingContract,
    TeachingResponseMode,
)


class ScriptedAuditClient:
    def __init__(self, result: BoundaryAudit) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def chat_json(
        self,
        system,
        user,
        schema,
        temperature=0.0,
        max_tokens=None,
    ):
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.result


class BrokenClient:
    def chat_json(self, *args, **kwargs):
        raise RuntimeError("provider unavailable")


def concept_contract() -> TeachingContract:
    return TeachingContract(
        response_mode=TeachingResponseMode.CONCEPT_EXPLANATION,
        requested_target="解释一元二次方程求根公式",
        allowed_content=["通用公式", "符号解释", "判别式含义"],
        forbidden_content=["完整解答当前题", "最终答案", "代入或套用当前题"],
        may_apply_to_active_problem=False,
        may_reveal_final_answer=False,
        reason="学生只询问公式，不要求解当前题",
    )


def test_auditor_flags_quadratic_answer_leak_without_equation_verifier():
    client = ScriptedAuditClient(
        BoundaryAudit(
            passed=False,
            violations=["applied_without_permission", "final_answer_leak"],
            evidence=["把 a=3,b=2,c=-11 代入并给出了两个根"],
            repair_instructions=["删除当前题代入和根，只保留通用公式与符号解释"],
        )
    )

    result = SemanticBoundaryAuditor(client).audit(
        "求根公式是什么？",
        "3x^2+2x-11=0",
        concept_contract(),
        "代入得到 x=(-1±√34)/3",
    )

    assert not result.passed
    assert BoundaryViolation.APPLIED_WITHOUT_PERMISSION in result.violations
    assert BoundaryViolation.FINAL_ANSWER_LEAK in result.violations


def test_auditor_allows_complete_concept_explanation_without_solving_problem():
    result = SemanticBoundaryAuditor(
        ScriptedAuditClient(BoundaryAudit(passed=True))
    ).audit(
        "求根公式是什么？",
        "3x^2+2x-11=0",
        concept_contract(),
        "对于 ax²+bx+c=0，求根公式是……其中判别式决定实根个数。",
    )

    assert result.passed


def test_auditor_fails_closed_when_passed_result_contains_violations():
    result = SemanticBoundaryAuditor(
        ScriptedAuditClient(
            BoundaryAudit(
                passed=True,
                violations=[BoundaryViolation.FINAL_ANSWER_LEAK],
            )
        )
    ).audit(
        "求根公式是什么？",
        "3x^2+2x-11=0",
        concept_contract(),
        "代入得到 x=(-1±√34)/3",
    )

    assert not result.passed
    assert result.audit_error == "ValueError"
    assert result.evidence == []
    assert result.repair_instructions == [
        "边界检查不可用，按教学契约保守重写回答。"
    ]


def test_auditor_fails_closed_when_passed_result_contains_audit_error():
    result = SemanticBoundaryAuditor(
        ScriptedAuditClient(
            BoundaryAudit(passed=True, audit_error="UpstreamValidationError")
        )
    ).audit(
        "求根公式是什么？",
        "3x^2+2x-11=0",
        concept_contract(),
        "对于 ax²+bx+c=0，求根公式是……",
    )

    assert not result.passed
    assert result.audit_error == "ValueError"


def test_auditor_allows_failed_result_with_repair_only():
    result = SemanticBoundaryAuditor(
        ScriptedAuditClient(
            BoundaryAudit(
                passed=False,
                repair_instructions=["只保留契约允许的概念解释"],
            )
        )
    ).audit(
        "求根公式是什么？",
        "3x^2+2x-11=0",
        concept_contract(),
        "对于 ax²+bx+c=0，求根公式是……",
    )

    assert not result.passed
    assert result.violations == []
    assert result.audit_error == ""
    assert result.repair_instructions == ["只保留契约允许的概念解释"]


def test_auditor_sends_only_boundary_inputs_as_strict_json():
    client = ScriptedAuditClient(BoundaryAudit(passed=True))
    contract = concept_contract()

    SemanticBoundaryAuditor(client).audit(
        "求根公式是什么？",
        "3x^2+2x-11=0",
        contract,
        "对于 ax²+bx+c=0，求根公式是……",
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["schema"] is BoundaryAudit
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 700
    assert json.loads(call["user"]) == {
        "student_message": "求根公式是什么？",
        "problem_context": "3x^2+2x-11=0",
        "contract": contract.model_dump(mode="json"),
        "draft": "对于 ax²+bx+c=0，求根公式是……",
    }
    system = str(call["system"])
    assert "范围与披露" in system
    assert "隐藏推理" in system
    assert "通用事实正确性" in system


def test_audit_transport_failure_does_not_pass_draft():
    result = SemanticBoundaryAuditor(BrokenClient()).audit(
        "求根公式是什么？",
        "3x^2+2x-11=0",
        concept_contract(),
        "对于 ax²+bx+c=0，求根公式是……",
    )

    assert not result.passed
    assert result.audit_error == "RuntimeError"
    assert result.evidence == []
    assert result.repair_instructions == [
        "边界检查不可用，按教学契约保守重写回答。"
    ]


def test_hard_recheck_matches_normalized_forbidden_application_marker():
    violations = hard_boundary_recheck(
        "3x^2+2x-11=0",
        concept_contract(),
        "接下来我们会代入，或套用当前题。",
    )

    assert violations == [BoundaryViolation.APPLIED_WITHOUT_PERMISSION]


def test_hard_recheck_does_not_trust_final_answer_json_in_problem_context():
    violations = hard_boundary_recheck(
        json.dumps(
            {
                "problem": "判断下列历史事件的年份",
                "metadata": {"final_answer": "公元前221年"},
            },
            ensure_ascii=False,
        ),
        concept_contract(),
        "最终可写成：公元前 221 年。",
    )

    assert violations == []


def test_hard_recheck_classifies_complete_solution_before_application():
    violations = hard_boundary_recheck(
        "3x^2+2x-11=0",
        concept_contract(),
        "下面完整解答当前题。",
    )

    assert violations == [BoundaryViolation.FINAL_ANSWER_LEAK]


def test_hard_recheck_classifies_from_start_to_finish_as_too_many_steps():
    contract = concept_contract().model_copy(
        update={"forbidden_content": ["从头到尾解当前题"]}
    )

    violations = hard_boundary_recheck(
        "3x^2+2x-11=0",
        contract,
        "下面从头到尾，解当前题。",
    )

    assert violations == [BoundaryViolation.TOO_MANY_STEPS]


def test_hard_recheck_respects_active_problem_application_permission():
    contract = concept_contract().model_copy(
        update={
            "response_mode": TeachingResponseMode.GUIDED_SOLUTION,
            "forbidden_content": ["解答当前题"],
            "may_apply_to_active_problem": True,
        }
    )

    violations = hard_boundary_recheck(
        "3x^2+2x-11=0",
        contract,
        "现在解答当前题。",
    )

    assert violations == [BoundaryViolation.OFF_TARGET]


def test_hard_recheck_does_not_infer_answer_from_plain_problem_text():
    violations = hard_boundary_recheck(
        "3x^2+2x-11=0",
        concept_contract(),
        "这里只解释字母 a、b、c 和判别式的含义。",
    )

    assert violations == []
