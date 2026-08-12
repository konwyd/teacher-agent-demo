# Teaching Contract Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 8502 enforce a subject-agnostic teaching disclosure contract, render textbook-style mathematics, and persistently show real Agent/Tool activity instead of the legacy fixed-stage summary.

**Architecture:** Extend the existing orchestration contracts so every successful Agent finish produces a `TeachingContract`. Remove the independent Observer LLM call from the 8502 turn path, generate against that contract, run one local-model semantic `BoundaryAudit`, repair violations once, then normalize math Markdown before persistence and display. Keep narrow deterministic math verification as optional evidence, not the general disclosure gate.

**Tech Stack:** Python 3.10, Pydantic v2, Streamlit/MathJax Markdown, existing `DeepSeekClient`, pytest, existing Chroma/RAG Tool Registry.

## Global Constraints

- 8502 remains the only user-facing entrypoint.
- Supported subjects remain 数学、语文、英语、物理、化学、生物、历史、地理、道法.
- All semantic planning and auditing calls use the existing local `DeepSeekClient`; add no provider or paid API.
- Only `full_solution` may reveal a final answer, and only after an explicit user request for a complete solution.
- `concept_explanation` must not apply a concept to the active problem unless `may_apply_to_active_problem=true`.
- Off-topic requests call neither local RAG nor web search.
- Existing deterministic math verification remains available where supported but is no longer the generic disclosure auditor.
- Preserve login, user isolation, conversation IDs, pending-message recovery, token totals, KV/prompt-cache metrics, and the current context-meter API.
- Preserve concurrent uncommitted files `prototypes/llm_led_teacher/context_usage.py`, `tests/prototypes/test_context_usage.py`, `docs/superpowers/plans/2026-08-12-stable-current-context-meter.md`, and their edits in `streamlit_app.py` / `test_llm_led_streaming.py`.
- Do not stage, reset, rewrite, or commit `data/china_textbook_k9_vectordb/**`.
- Use TDD for every production behavior change: observe RED, make the smallest implementation, then observe GREEN.
- Stage explicit files only; do not use `git add -A` in this mixed worktree.

---

## File Structure

- Modify `prototypes/llm_led_teacher/orchestration_models.py`: teaching-contract and boundary-audit Pydantic contracts; finish decisions require a contract.
- Modify `prototypes/llm_led_teacher/orchestrator.py`: contract-aware Agent prompt, deterministic contract normalization, and finish trace.
- Create `prototypes/llm_led_teacher/teaching_policy.py`: shared explicit-solution/no-answer intent predicates and deterministic contract normalization, avoiding a `logic.py`/orchestrator import cycle.
- Create `prototypes/llm_led_teacher/boundary_audit.py`: local-model semantic boundary auditor and safe fallback result.
- Create `prototypes/llm_led_teacher/math_markdown.py`: high-confidence LaTeX delimiter normalization.
- Modify `prototypes/llm_led_teacher/streaming.py`: stream complete display-math blocks atomically.
- Modify `prototypes/llm_led_teacher/logic.py`: remove independent Observer call, generate from the contract, audit/repair once, normalize formulas, and persist new fields.
- Modify `prototypes/llm_led_teacher/live_progress.py`: Agent-first and guardrail progress events.
- Modify `prototypes/llm_led_teacher/streamlit_app.py`: persistent contract, Agent trace, boundary audit, and optional verifier rendering while retaining context-meter edits.
- Modify `tests/prototypes/test_llm_led_orchestration_models.py`.
- Modify `tests/prototypes/test_llm_led_orchestrator.py`.
- Create `tests/prototypes/test_teaching_policy.py`.
- Create `tests/prototypes/test_boundary_audit.py`.
- Create `tests/prototypes/test_math_markdown.py`.
- Modify `tests/prototypes/test_llm_led_turn_contract.py`.
- Modify `tests/prototypes/test_llm_led_streaming.py` without deleting the concurrent context-meter assertions.
- Modify `tests/prototypes/test_llm_led_auth_ui.py` and `tests/test_conversation_archive.py` for old-record compatibility.
- Modify `README.md` and `prototypes/llm_led_teacher/README.md` after behavior is verified.

---

### Task 1: Teaching Contract and Boundary Audit Contracts

**Files:**
- Modify: `prototypes/llm_led_teacher/orchestration_models.py`
- Modify: `prototypes/llm_led_teacher/logic.py` model defaults only
- Modify: `tests/prototypes/test_llm_led_orchestration_models.py`
- Modify: `tests/test_conversation_archive.py`

**Interfaces:**
- Produces: `TeachingResponseMode(str, Enum)` with `concept_explanation`, `one_step`, `guided_solution`, `full_solution`, `brief_redirect`.
- Produces: `TeachingContract(response_mode, requested_target, allowed_content, forbidden_content, may_apply_to_active_problem, may_reveal_final_answer, reason)`.
- Produces: `BoundaryViolation(str, Enum)` and `BoundaryAudit(passed, violations, evidence, repair_instructions, audit_error)`.
- Changes: `AgentDecision.contract: TeachingContract | None`; `finish` requires a contract.
- Changes: `AgentOrchestrationResult.teaching_contract: TeachingContract`.
- Extends: `PrototypeTurnRecord.teaching_contract`, `boundary_audit`, `formula_normalized`, all with legacy-safe defaults.

- [ ] **Step 1: Write failing contract-validation tests**

```python
def test_concept_contract_cannot_reveal_or_apply_to_active_problem():
    with pytest.raises(ValidationError):
        TeachingContract(
            response_mode="concept_explanation",
            requested_target="解释求根公式",
            allowed_content=["给出通用公式"],
            forbidden_content=[],
            may_apply_to_active_problem=True,
            may_reveal_final_answer=False,
            reason="concept only",
        )


def test_only_full_solution_may_reveal_final_answer():
    with pytest.raises(ValidationError):
        TeachingContract(
            response_mode="guided_solution",
            requested_target="引导解题",
            may_reveal_final_answer=True,
            reason="invalid disclosure",
        )


def test_finish_decision_requires_teaching_contract():
    with pytest.raises(ValidationError):
        AgentDecision(kind="finish", reason="done")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python3 -m pytest tests/prototypes/test_llm_led_orchestration_models.py -q`

Expected: FAIL because the new types and `contract` field do not exist.

- [ ] **Step 3: Implement the Pydantic contracts and invariants**

```python
class TeachingResponseMode(str, Enum):
    CONCEPT_EXPLANATION = "concept_explanation"
    ONE_STEP = "one_step"
    GUIDED_SOLUTION = "guided_solution"
    FULL_SOLUTION = "full_solution"
    BRIEF_REDIRECT = "brief_redirect"


class TeachingContract(BaseModel):
    response_mode: TeachingResponseMode
    requested_target: str = Field(min_length=1, max_length=300)
    allowed_content: list[str] = Field(default_factory=list, max_length=5)
    forbidden_content: list[str] = Field(default_factory=list, max_length=5)
    may_apply_to_active_problem: bool = False
    may_reveal_final_answer: bool = False
    reason: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def validate_disclosure(self):
        if self.may_reveal_final_answer and self.response_mode is not TeachingResponseMode.FULL_SOLUTION:
            raise ValueError("only full_solution may reveal final answer")
        if self.response_mode is TeachingResponseMode.CONCEPT_EXPLANATION and self.may_apply_to_active_problem:
            raise ValueError("concept explanation cannot apply to active problem")
        return self
```

- [ ] **Step 4: Add legacy episode/turn tests**

Validate an old serialized turn without the three new fields and assert:

```python
assert turn.teaching_contract is None
assert turn.boundary_audit is None
assert turn.formula_normalized is False
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_llm_led_orchestration_models.py tests/test_conversation_archive.py -q`

Expected: all pass.

- [ ] **Step 6: Commit Task 1 files only**

```bash
git add prototypes/llm_led_teacher/orchestration_models.py prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_orchestration_models.py tests/test_conversation_archive.py
git commit -m "feat: add teaching disclosure contracts"
```

---

### Task 2: Textbook Math Markdown and Formula-Safe Streaming

**Files:**
- Create: `prototypes/llm_led_teacher/math_markdown.py`
- Modify: `prototypes/llm_led_teacher/streaming.py`
- Create: `tests/prototypes/test_math_markdown.py`
- Modify: `tests/prototypes/test_llm_led_streaming.py`

**Interfaces:**
- Produces: `normalize_math_markdown(text: str) -> tuple[str, bool]`.
- Produces: `iter_markdown_display_chunks(text: str, text_chunk_size: int = 1, delay_seconds: float = 0.02) -> Iterator[str]`.
- Preserves: `iter_display_chunks()` for non-Markdown compatibility tests.

- [ ] **Step 1: Write failing normalization tests**

```python
def test_normalizes_model_bracket_latex_to_display_math():
    raw = "公式：\n\n[ x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a} ]\n\n其中 a 不为 0。"
    normalized, changed = normalize_math_markdown(raw)
    assert normalized == "公式：\n\n$$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$\n\n其中 a 不为 0。"
    assert changed is True


def test_does_not_change_ordinary_brackets_or_code_fences():
    raw = "取值范围是 [0, 1]。\n```python\nprint(r'\\frac{1}{2}')\n```"
    assert normalize_math_markdown(raw) == (raw, False)


def test_normalizes_standard_latex_delimiters():
    normalized, _ = normalize_math_markdown(r"行内 \\(a^2+b^2\\)，展示 \\[x=\\frac12\\]")
    assert normalized == r"行内 $a^2+b^2$，展示 $$x=\\frac12$$"
```

- [ ] **Step 2: Write a failing atomic-formula streaming test**

```python
def test_display_math_is_emitted_as_one_closed_chunk():
    chunks = list(iter_markdown_display_chunks("说明\n\n$$x=\\frac{1}{2}$$\n\n结束", delay_seconds=0))
    assert "$$x=\\frac{1}{2}$$" in chunks
    assert all(chunk.count("$$") in (0, 2) for chunk in chunks)
    assert "".join(chunks) == "说明\n\n$$x=\\frac{1}{2}$$\n\n结束"
```

- [ ] **Step 3: Run the new tests and verify RED**

Run: `python3 -m pytest tests/prototypes/test_math_markdown.py tests/prototypes/test_llm_led_streaming.py -q`

Expected: import failures for the missing functions.

- [ ] **Step 4: Implement high-confidence normalization outside code fences**

Split the input on fenced-code blocks. On non-code segments:

```python
segment = re.sub(r"\\\\\[(.*?)\\\\\]", lambda m: f"$${m.group(1).strip()}$$", segment, flags=re.S)
segment = re.sub(r"\\\\\((.*?)\\\\\)", lambda m: f"${m.group(1).strip()}$", segment, flags=re.S)
segment = _BRACKET_LATEX_BLOCK.sub(lambda m: f"$${m.group(1).strip()}$$", segment)
```

`_BRACKET_LATEX_BLOCK` must require a standalone paragraph and at least one of `\\frac`, `\\sqrt`, `\\pm`, `\\times`, `\\neq`, `\\le`, `\\ge`, `^`, or `_` before converting.

- [ ] **Step 5: Implement formula-safe display chunks**

Tokenize complete `$$...$$` blocks. Yield each display-math token atomically; feed only intervening text through the existing character chunker.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_math_markdown.py tests/prototypes/test_llm_led_streaming.py -q`

Expected: all pass, including existing exact reconstruction tests and concurrent context-meter source assertions.

- [ ] **Step 7: Commit Task 2 files only**

```bash
git add prototypes/llm_led_teacher/math_markdown.py prototypes/llm_led_teacher/streaming.py tests/prototypes/test_math_markdown.py tests/prototypes/test_llm_led_streaming.py
git commit -m "fix: render textbook math safely"
```

---

### Task 3: Contract-Aware Agent Finish and Deterministic Policy Normalization

**Files:**
- Modify: `prototypes/llm_led_teacher/orchestrator.py`
- Modify: `prototypes/llm_led_teacher/orchestration_models.py`
- Create: `prototypes/llm_led_teacher/teaching_policy.py`
- Modify: `tests/prototypes/test_llm_led_orchestrator.py`
- Create: `tests/prototypes/test_teaching_policy.py`

**Interfaces:**
- Produces: `has_explicit_full_solution_request(message)`, `has_no_answer_request(message)`, and `normalize_teaching_contract(contract, student_message, classification) -> TeachingContract` from `teaching_policy.py`.
- Changes: every `MultisubjectAgentLoop.run()` result contains `teaching_contract`.
- Emits: a persistent `AgentTraceEvent(event_type="teaching_contract", summary=..., details=<public contract>)` on finish.

- [ ] **Step 1: Write failing policy tests**

```python
def test_formula_question_finishes_with_concept_only_contract():
    proposed = TeachingContract(
        response_mode="concept_explanation",
        requested_target="解释求根公式",
        allowed_content=["通用公式", "符号含义"],
        forbidden_content=["代入当前题", "计算当前题的根"],
        may_apply_to_active_problem=False,
        may_reveal_final_answer=False,
        reason="concept request",
    )
    result = loop_with_finish(proposed).run(
        "3x^2+2x-11=0",
        "求根公式是什么？",
        "数学",
        "初中",
    )
    assert result.teaching_contract == proposed
    assert result.trace[-1].event_type == "teaching_contract"


def test_no_answer_phrase_overrides_model_full_solution_contract():
    normalized = normalize_teaching_contract(
        full_solution_contract(may_reveal_final_answer=True),
        "讲讲思路，但不要告诉我答案",
        math_classification(),
    )
    assert normalized.response_mode is TeachingResponseMode.GUIDED_SOLUTION
    assert normalized.may_reveal_final_answer is False


def test_full_solution_requires_explicit_complete_solution_marker():
    normalized = normalize_teaching_contract(
        full_solution_contract(may_reveal_final_answer=True),
        "这道题怎么解？",
        math_classification(),
    )
    assert normalized.response_mode is TeachingResponseMode.ONE_STEP
    assert normalized.may_reveal_final_answer is False
```

- [ ] **Step 2: Run orchestrator tests and verify RED**

Run: `python3 -m pytest tests/prototypes/test_llm_led_orchestrator.py -q`

Expected: failures because finish decisions/results lack contracts and normalization.

- [ ] **Step 3: Update the Agent action prompt and finish schema**

The prompt must distinguish “complete explanation of the requested concept” from “full solution of active problem”, list the five response modes, require `contract` on finish, and explicitly say that `may_reveal_final_answer=true` is illegal without an explicit complete-solution request.

- [ ] **Step 4: Implement deterministic normalization in the shared policy module**

Move the existing `_FULL_SOLUTION_REQUEST_MARKERS` and `_NO_ANSWER_MARKERS` values out of `logic.py` into `teaching_policy.py`; both `logic.py` and `orchestrator.py` import the predicates from there. For an illegal `full_solution`, choose `one_step` when the message is a generic solve request and `guided_solution` otherwise. For off-topic classifications, force `brief_redirect` with both booleans false. This module must not import `logic.py` or `orchestrator.py`.

- [ ] **Step 5: Persist the contract trace**

On a finish decision, normalize the contract, append a public trace event, and return it as `AgentOrchestrationResult.teaching_contract`. The trace details may contain the contract JSON but never the Agent prompt or hidden reasoning.

- [ ] **Step 6: Run orchestrator/tool tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_teaching_policy.py tests/prototypes/test_llm_led_orchestrator.py tests/prototypes/test_llm_led_orchestration_tools.py -q`

Expected: all pass; local-before-web and call budgets remain enforced.

- [ ] **Step 7: Commit Task 3 files only**

```bash
git add prototypes/llm_led_teacher/teaching_policy.py prototypes/llm_led_teacher/orchestrator.py prototypes/llm_led_teacher/orchestration_models.py tests/prototypes/test_teaching_policy.py tests/prototypes/test_llm_led_orchestrator.py
git commit -m "feat: make agent submit teaching contract"
```

---

### Task 4: Subject-Agnostic Semantic Boundary Auditor

**Files:**
- Create: `prototypes/llm_led_teacher/boundary_audit.py`
- Create: `tests/prototypes/test_boundary_audit.py`

**Interfaces:**
- Produces: `SemanticBoundaryAuditor(client).audit(student_message, problem_context, contract, draft) -> BoundaryAudit`.
- Produces: `safe_boundary_fallback(contract: TeachingContract, error: Exception) -> BoundaryAudit`.
- Produces: `hard_boundary_recheck(problem_context, contract, response) -> list[BoundaryViolation]` for high-confidence post-repair leaks.
- Consumes: `DeepSeekClient.chat_json` and the contracts from Task 1.

- [ ] **Step 1: Write failing semantic-auditor tests with a scripted JSON client**

```python
def test_auditor_flags_quadratic_answer_leak_without_equation_verifier():
    client = ScriptedAuditClient(BoundaryAudit(
        passed=False,
        violations=["applied_without_permission", "final_answer_leak"],
        evidence=["把 a=3,b=2,c=-11 代入并给出了两个根"],
        repair_instructions=["删除当前题代入和根，只保留通用公式与符号解释"],
    ))
    result = SemanticBoundaryAuditor(client).audit(
        "求根公式是什么？",
        "3x^2+2x-11=0",
        concept_contract(),
        "代入得到 x=(-1±√34)/3",
    )
    assert not result.passed
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
```

- [ ] **Step 2: Write a failing fail-closed test**

```python
def test_audit_transport_failure_does_not_pass_draft():
    result = SemanticBoundaryAuditor(BrokenClient()).audit(...)
    assert not result.passed
    assert result.audit_error == "RuntimeError"
    assert result.repair_instructions == ["边界检查不可用，按教学契约保守重写回答。"]
```

Also add a post-repair hard-check test whose concept contract forbids applying the active problem and whose response repeats a forbidden disclosure marker; assert the corresponding violation is returned.

- [ ] **Step 3: Run tests and verify RED**

Run: `python3 -m pytest tests/prototypes/test_boundary_audit.py -q`

Expected: import failure because `boundary_audit.py` does not exist.

- [ ] **Step 4: Implement the strict JSON audit prompt**

The system prompt must say it judges scope/disclosure only, not hidden reasoning or universal factual correctness. Serialize exactly `student_message`, `problem_context`, `contract`, and `draft`. Use `temperature=0.0`, `max_tokens=700`, and `BoundaryAudit` as the schema.

- [ ] **Step 5: Implement fail-closed behavior**

Catch provider, validation, and timeout exceptions. Return `passed=False`, no fabricated evidence, `audit_error=type(exc).__name__`, and the single conservative repair instruction shown in Step 2.

- [ ] **Step 6: Implement the narrow post-repair hard check**

Normalize whitespace and punctuation, then check only high-confidence evidence: literal/normalized forbidden-content markers emitted by the contract and final-answer strings explicitly present in the active problem metadata when available. Do not pretend this is a universal mathematical verifier. A hit blocks the repaired draft and records the exact public marker that matched.

- [ ] **Step 7: Run boundary-audit tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_boundary_audit.py -q`

Expected: all pass.

- [ ] **Step 8: Commit Task 4 files only**

```bash
git add prototypes/llm_led_teacher/boundary_audit.py tests/prototypes/test_boundary_audit.py
git commit -m "feat: add semantic teaching boundary audit"
```

---

### Task 5: Replace Observer-Centric Turn Flow with Contract Generation

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py`

**Interfaces:**
- `run_turn()` keeps its public parameters and `PrototypeTurnRecord` return type.
- Removes the independent `_observe_turn()` call from all 8502 Agent-enabled turns.
- Consumes: `AgentOrchestrationResult.teaching_contract`, `SemanticBoundaryAuditor`, and `normalize_math_markdown`.
- Persists: contract, boundary audit, normalized final response, Agent trace, and aggregate usage across planning/audit/generation/repair.

- [ ] **Step 1: Replace observer-centric tests with failing contract-flow tests**

```python
def test_formula_explanation_does_not_solve_active_quadratic():
    orchestrator = FakeOrchestrator(concept_formula_result())
    client = ScriptedTurnClient(
        drafts=[
            "通用公式是 [ x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a} ]。代入本题得到 x=...",
            "通用公式是 [ x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a} ]。其中 a、b、c 是系数。",
        ],
        audits=[BoundaryAudit(
            passed=False,
            violations=["applied_without_permission", "final_answer_leak"],
            repair_instructions=["删除本题代入和答案"],
        )],
    )
    episode = PrototypeEpisode(
        problem_context="3x^2+2x-11=0",
        active_subject="数学",
        active_stage="初中",
    )
    turn = run_turn(episode, "求根公式是什么？", client=client, orchestrator=orchestrator)
    assert "代入本题" not in turn.teacher_output.response
    assert "$$x=\\frac" in turn.teacher_output.response
    assert turn.repair_performed
    assert turn.boundary_audit.violations
```

- [ ] **Step 2: Add a failing no-observer-call and usage-aggregation test**

Use a fake client whose `chat_json` records schemas. Assert `TurnGuidance` is never requested, `BoundaryAudit` is requested exactly once, and `model_generation_metadata.llm_call_count` includes orchestration, generation, audit, and repair records from the shared client.

- [ ] **Step 3: Run turn-contract tests and verify RED**

Run: `python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q`

Expected: failures because `run_turn()` still calls Observer, does not invoke the semantic auditor, and does not normalize formulas.

- [ ] **Step 4: Derive legacy `TurnGuidance` from the contract without an LLM call**

Map `one_step` to `PacingMode.SINGLE_MOVE`; map the other modes to `COMPLETE_RESPONSE`. Preserve `observer_raw_guidance` and `observer_error` fields for old UI/schema compatibility by storing the derived guidance and `observer_error="observer_replaced_by_teaching_contract"`.

- [ ] **Step 5: Generate, audit, repair once, hard-recheck, then normalize**

Pass `teaching_contract` in `_teacher_payload`. After the draft, call `SemanticBoundaryAuditor.audit`. If it fails, call `_repair` with contract plus `repair_instructions`, then run `hard_boundary_recheck` on the repaired draft. If that high-confidence check still finds a forbidden disclosure, replace the output with a contract-safe concise fallback instead of showing the leaking draft. Run existing supported claim verification as optional evidence, but do not use lack of verifier support to pass or fail disclosure. Normalize the accepted response using `normalize_math_markdown` before creating `TeacherResponse`/record data.

- [ ] **Step 6: Fail closed when the boundary audit is unavailable**

On `audit_error`, perform one conservative contract-driven repair. If repair generation fails, let the existing pending-message exception path preserve the student input. Never display the unaudited draft.

- [ ] **Step 7: Preserve episode, pending-message, and usage behavior**

Keep active problem/subject/stage updates from orchestration, `previous_verified_step` updates for supported verified math, and `summarize_usage_metadata` over the entire turn offset. Do not rename token/cache metadata keys.

- [ ] **Step 8: Run focused contract/auth tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py tests/prototypes/test_llm_led_auth_ui.py tests/test_conversation_archive.py -q`

Expected: all pass, including old-record and pending-message tests.

- [ ] **Step 9: Commit Task 5 files only**

```bash
git add prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_turn_contract.py
git commit -m "feat: enforce teaching contract in 8502 turns"
```

---

### Task 6: Persist Real Agent/Tool Trace in the 8502 UI

**Files:**
- Modify: `prototypes/llm_led_teacher/live_progress.py`
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Modify: `tests/prototypes/test_llm_led_streaming.py`
- Modify: `tests/prototypes/test_llm_led_auth_ui.py`

**Interfaces:**
- Live status displays `Agent Loop` events and `Answer Guardrails` events.
- `render_turn_details()` renders `turn.teaching_contract`, every `turn.agent_trace` event, `turn.boundary_audit`, and optional verifier evidence.
- Current and historical answers both render the persisted normalized Markdown.
- Consumes: `iter_markdown_display_chunks` for current response streaming.

- [ ] **Step 1: Strengthen the UI source contract before editing**

Add assertions without deleting the concurrent context-meter assertions:

```python
assert "build_context_usage_display(" in source
assert "teaching_contract" in source
assert "agent_trace" in source
assert "boundary_audit" in source
assert "iter_markdown_display_chunks" in source
assert "观察器原始输出" not in source
assert "processing_steps = [" not in source
assert source.count("render_context_usage(episode)") == 1
assert source.count("render_turn_token_usage(turn)") == 2
```

- [ ] **Step 2: Add a failing persistent-trace rendering test**

Extract a pure helper `agent_trace_rows(turn) -> list[dict[str, str]]` or test the Streamlit source contract. Assert a trace containing `load_skill`, `search_local_knowledge`, and `teaching_contract` produces three visible rows with public summary, success state, and elapsed time; assert private `details` keys are not rendered wholesale.

- [ ] **Step 3: Run UI tests and verify RED**

Run: `python3 -m pytest tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py tests/prototypes/test_context_usage.py -q`

Expected: failures for missing contract/trace/audit UI and old Observer summary still present.

- [ ] **Step 4: Replace the fixed processing summary with real persisted trace**

Render a table with columns `步骤`, `类型`, `Tool`, `状态`, `耗时`, `摘要`. Render only `AgentTraceEvent.summary`, `tool_name`, `ok`, and `elapsed_seconds`; never dump full prompts or arbitrary `details`.

- [ ] **Step 5: Render contract and boundary audit separately**

Show requested target, response mode, answer/application permissions, allowed/forbidden content, audit status, violation names, evidence summaries, and repair instructions. Rename verifier section to `可选确定性验证`, and show `not_applicable` as neutral information rather than a failed global audit.

- [ ] **Step 6: Make live progress Agent-first**

Replace legacy `observing/retrieving/generating/auditing/repairing` presentation text with:

- Agent classification/planning/tool events from `update_agent_event`;
- `generating`, `boundary_check`, `repairing`, `formatting`, `complete` guardrail events.

Stage timings remain available, but labels must correspond to the real components.

- [ ] **Step 7: Use formula-safe rendering for current and history answers**

Use `iter_markdown_display_chunks(turn.teacher_output.response)` for the current turn. Continue `st.markdown(turn.teacher_output.response)` for history; both consume the same normalized persisted response.

- [ ] **Step 8: Run UI/auth/context tests and verify GREEN**

Run: `python3 -m pytest tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py tests/prototypes/test_context_usage.py tests/test_auth.py tests/test_conversation_archive.py -q`

Expected: all pass; context-meter labels, login/history, pending message, token totals, and cache hit/miss displays remain intact.

- [ ] **Step 9: Commit Task 6 files explicitly**

Because `streamlit_app.py` and `test_llm_led_streaming.py` already contain concurrent user changes, inspect the combined diff before staging and include those files only after confirming the context-meter hunks are preserved.

```bash
git add prototypes/llm_led_teacher/live_progress.py prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py
git commit -m "feat: show real agent trace and teaching boundary"
```

---

### Task 7: Full Regression, Documentation, and Real 8502 Acceptance

**Files:**
- Modify: `README.md`
- Modify: `prototypes/llm_led_teacher/README.md`
- Modify design spec only if implementation required a factual correction.

**Interfaces:**
- Documents Agent contract flow, semantic boundary audit, optional verifier, textbook math, and real trace.
- Leaves a healthy 8080/8002/8502 stack running.

- [ ] **Step 1: Run all prototype and context-meter tests**

Run: `python3 -m pytest tests/prototypes -q`

Expected: all pass.

- [ ] **Step 2: Run the complete suite**

Run: `python3 -m pytest`

Expected: zero failures.

- [ ] **Step 3: Run the offline teaching evaluation**

Run: `python3 scripts/run_eval.py`

Expected: 38 passed, 0 failed.

- [ ] **Step 4: Update documentation**

Document that `complete_response` is replaced by a disclosure-aware teaching contract in 8502, formula output is normalized to MathJax-compatible Markdown, the semantic audit checks scope rather than universal correctness, and deterministic verifiers are optional subject tools.

- [ ] **Step 5: Check diff and secret hygiene**

Run: `git diff --check -- README.md prototypes/llm_led_teacher tests/prototypes docs/superpowers`

Run: `git -c filter.lfs.process= -c filter.lfs.required=false status -sb --untracked-files=all`

Expected: no Chroma file staged; context-meter files remain present; no `.env`, SQLite account/conversation DB, log, cache, or secret is staged.

- [ ] **Step 6: Restart/reload 8502 and verify service health**

Run the existing launcher if services are not healthy, then check:

```bash
curl -fsS --noproxy '*' http://127.0.0.1:8080/health
curl -fsS --noproxy '*' http://127.0.0.1:8002/api/v2/heartbeat
curl -fsS --noproxy '*' http://127.0.0.1:8502/_stcore/health
```

- [ ] **Step 7: Real acceptance — concept boundary**

Create/restore an episode whose active problem is `3x^2+2x-11=0`, then ask `求根公式是什么？` using the local model. Expected:

- contract is `concept_explanation`;
- `may_apply_to_active_problem=false` and `may_reveal_final_answer=false`;
- answer contains a rendered general quadratic formula and symbol/discriminant explanation;
- answer does not contain `a=3`, `b=2`, `c=-11`, `136`, or `(-1±√34)/3`;
- trace shows actual Agent actions and contract submission.

- [ ] **Step 8: Real acceptance — explicit full solution**

Ask `请用求根公式完整解这道题并给出答案`. Expected contract is `full_solution`, final disclosure is allowed, formula is textbook-rendered, and boundary audit passes or records a justified repair.

- [ ] **Step 9: Real acceptance — other routing**

Run one physics local-RAG question and one off-topic product question. Confirm physics trace includes local search with `Physics/物理` filters and no unnecessary web call; off-topic uses zero knowledge tools.

- [ ] **Step 10: Final verification and documentation commit**

```bash
git add README.md prototypes/llm_led_teacher/README.md
git commit -m "docs: explain teaching contract agent flow"
```

Re-run `python3 -m pytest` after the final documentation/source tree is fixed and inspect every staged file before any push.
