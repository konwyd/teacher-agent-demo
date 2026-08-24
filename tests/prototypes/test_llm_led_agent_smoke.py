from types import SimpleNamespace

from scripts import run_llm_led_agent_smoke as smoke


def test_smoke_requires_explicit_live_flag_and_bounded_count(capsys) -> None:
    assert smoke.main([]) == 2
    assert "--live" in capsys.readouterr().out

    assert smoke.main(["--live", "--max-scenarios", "13"]) == 2
    assert "1 到 12" in capsys.readouterr().out


def test_smoke_refuses_non_loopback_model_endpoint(capsys) -> None:
    assert smoke.main(
        [
            "--live",
            "--max-scenarios",
            "1",
            "--base-url",
            "https://api.deepseek.com/v1",
        ]
    ) == 2

    assert "只允许" in capsys.readouterr().out


def test_smoke_matrix_covers_handoff_categories() -> None:
    assert len(smoke.SCENARIOS) == 12
    assert {scenario.category for scenario in smoke.SCENARIOS} == {
        "简单数学引导",
        "数学概念解释",
        "学生错误步骤",
        "物理",
        "化学",
        "语文",
        "英语",
        "历史地理生物代表样本",
        "追问续答",
        "明确索要完整答案",
        "课外问题",
        "无效输入",
    }
    assert sum(len(scenario.messages) for scenario in smoke.SCENARIOS) == 14


def test_turn_record_keeps_public_trace_and_usage_only() -> None:
    classification = SimpleNamespace(
        model_dump=lambda **_kwargs: {
            "scope": "k9",
            "subject": "数学",
            "stage": "初中",
            "is_new_problem": True,
        }
    )
    contract = SimpleNamespace(
        model_dump=lambda **_kwargs: {
            "response_mode": "one_step",
            "may_reveal_final_answer": False,
        }
    )
    turn = SimpleNamespace(
        turn_id=1,
        student_message="初中数学：2x+3=7 怎么解？",
        teacher_output=SimpleNamespace(response="先把等式两边都减去3。"),
        classification=classification,
        loaded_skills=["socratic_scaffolding"],
        agent_trace=[
            SimpleNamespace(
                step=2,
                event_type="tool_result",
                tool_name="load_skill",
                ok=True,
                elapsed_seconds=0.001,
                summary="load_skill 完成，命中 0 条",
            )
        ],
        agent_termination_reason="deterministic_fast_path",
        router_type="deterministic_fast_path",
        agent_iterations=0,
        semantic_audit_performed=False,
        teaching_contract=contract,
        student_verification=SimpleNamespace(status=SimpleNamespace(value="unable_to_verify")),
        boundary_audit=SimpleNamespace(passed=True, violations=[]),
        post_repair_boundary_audit=None,
        boundary_fallback_reason="",
        repair_performed=False,
        remaining_warnings=[],
        rag_source="local",
        rag_hits=[{}],
        stage_timings_seconds={"planning": 0.1},
        backend_total_seconds=1.2,
        model_generation_metadata={
            "llm_call_count": 2,
            "total_tokens": 500,
            "prompt_tokens": 400,
            "completion_tokens": 100,
            "cache_hit_rate": 0.25,
        },
    )

    record = smoke._turn_record(turn, elapsed_seconds=1.25)

    assert record["agent_termination_reason"] == "deterministic_fast_path"
    assert record["loaded_skills"] == ["socratic_scaffolding"]
    assert record["tool_calls"][0]["tool_name"] == "load_skill"
    assert record["safe_output_ready_seconds"] == 1.25
    assert record["llm_call_count"] == 2
    assert record["router_type"] == "deterministic_fast_path"
    assert record["agent_iterations"] == 0
    assert record["semantic_audit_performed"] is False
    assert record["boundary_violations"] == []
    assert "reasoning_content" not in record


def test_scenario_checks_allow_same_tool_in_separate_turns() -> None:
    base_record = {
        "classification": {
            "scope": "k9",
            "subject": "物理",
            "is_new_problem": True,
        },
        "teaching_contract": {
            "response_mode": "concept_explanation",
            "may_reveal_final_answer": False,
        },
        "agent_termination_reason": "deterministic_fast_path",
        "tool_calls": [
            {
                "event_type": "tool_result",
                "tool_name": "load_skill",
                "summary": "load_skill 完成，命中 0 条",
            }
        ],
    }
    follow_up = {
        **base_record,
        "classification": {
            **base_record["classification"],
            "is_new_problem": False,
        },
    }

    checks = smoke._scenario_checks(smoke.SCENARIOS[8], [base_record, follow_up])

    assert all(checks.values())


def test_scenario_checks_reject_placeholder_safety_fallback() -> None:
    record = {
        "classification": {
            "scope": "k9",
            "subject": "物理",
            "is_new_problem": True,
        },
        "teaching_contract": {
            "response_mode": "concept_explanation",
            "may_reveal_final_answer": False,
        },
        "teacher_response": (
            "我会只说明这个概念本身和相关符号的含义，不展开当前题的计算。"
        ),
        "agent_termination_reason": "deterministic_fast_path",
        "tool_calls": [],
    }

    checks = smoke._scenario_checks(smoke.SCENARIOS[3], [record])

    assert checks["no_placeholder_fallback"] is False
