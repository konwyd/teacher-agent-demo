# Single-Move Turn Contract Design

Date: 2026-08-10

## Objective

Strengthen two existing teaching invariants after generation in the LLM-led
Teacher Agent:

1. a teacher turn never contains multiple questions, and repair leaves one
   genuine question at the end; and
2. when effective pacing is `single_move`, the teacher advances the active
   supported problem by at most one meaningful mathematical move.

The observer and generator prompts remain useful soft guidance. Deterministic
post-generation auditing is the enforcement boundary. The change is scoped to
`prototypes/llm_led_teacher` and does not alter RAG, memory, the production
workflow, or the mathematical authority of `app/agent/math_checker.py`.

## Motivation

Current behavior audits multiple questions only after the response is already
accepted. It records an advisory warning but does not repair the response.

The existing pacing audit also focuses on explicit final-answer forms. A
`single_move` response can therefore derive an intermediate equation and then
describe the next operation in prose without writing the final answer. This is
still pedagogically a multi-move response, even though the current equation
audit sees only one new equation state.

## Chosen approach

Use a deterministic hybrid audit:

- question count is a hard repair reason;
- verified equation-state transitions are the primary pacing signal; and
- narrow sequential-operation prose is a secondary pacing signal after the
  response has already introduced one new equation state.

This is preferred over equation-only auditing because equation-only auditing
misses verbal leakage of the next move. It is preferred over an additional LLM
judge because pacing must remain deterministic, offline-testable, and free of
extra latency and cost.

## Question contract

The existing punctuation-based question counter remains the first enforcement
mechanism. A draft with more than one `?` or `？` produces a repair reason during
the pre-acceptance audit, not only a remaining advisory warning. Zero-question
drafts retain their current behavior in this narrowly scoped patch.

The repair instruction must tell the generator to:

- keep only one genuine question;
- place it at the end of the response;
- turn earlier interrogative material into statements or remove it; and
- preserve all other verified teaching constraints.

The repaired response is audited again. If it still contains multiple
questions, the remaining warning is recorded exactly as today; the turn is not
placed in an unbounded repair loop.

This patch does not introduce a natural-language question classifier. The
punctuation rule is deliberately transparent and consistent with the existing
contract.

## Single-move contract

### Applicability

The hard pacing audit runs whenever the normalized, effective guidance is
`PacingMode.SINGLE_MOVE`, including later turns in the same episode. It is not
limited to a newly established problem.

`PacingMode.COMPLETE_RESPONSE` remains available for explicit requests for a
complete solution and for turns whose purpose is conceptual explanation rather
than one-step progression.

The deterministic mathematical part is initially limited to verifier-supported
linear-equation problems. Unsupported topics continue to receive prompt-level
pacing guidance and the universal one-question audit, but no speculative
mathematical step counting is added.

### Equation states

The audit extracts equation claims through the existing claim-extraction and
verification path. A candidate state counts only when it:

- contains the active problem's unknown;
- is verified as equivalent to the active problem by the deterministic checker;
- is not merely the original equation repeated;
- is not a duplicate of another normalized state in the same response; and
- is not a variable-free equality or tautology such as `15=15`.

One distinct new verified state is allowed. Two or more distinct new verified
states constitute more than one move and trigger repair.

Final-answer protection remains independently active. The new audit complements
that protection rather than replacing it.

### Verbal next-move leakage

After one new verified equation state has already appeared, the response must
not also instruct or announce a subsequent algebraic operation. The secondary
detector is intentionally narrow: it requires both a sequencing cue and a
supported equation-operation cue in text occurring after the first new state.

Sequencing cues include forms such as:

- `接下来`, `下一步`, `然后`, `再`, `继续`, `第二步`;

and operation cues include forms such as:

- both sides adding or subtracting a quantity;
- both sides multiplying or dividing by a quantity;
- moving a term, combining like terms, expanding, or factoring when those
  operations are supported by the current checker scope.

An explanation of why the already-performed move is valid does not count as a
second move. A bare sequencing word without an operation cue also does not
count. This paired requirement limits false positives.

For example, after deriving `5x=15`:

- `因为等式两边同时减去 2，等式仍然成立。` is an explanation of the current
  move and is allowed;
- `接下来两边再除以 5。` announces a second move and triggers repair.

### Repair behavior

When the single-move audit fails, the repair instruction contains the detected
reason and requires the generator to:

- keep only the first meaningful transformation;
- remove later equation states and later-operation instructions;
- avoid disclosing the final answer unless the effective pacing allows it; and
- finish with exactly one question asking the student to perform or interpret
  the next move.

Repair remains one-shot. Existing deterministic claim verification, final-form
protection, and no-answer-request protection remain authoritative.

## Data and control flow

The LLM-led turn remains:

1. observe the student message;
2. normalize the effective pacing;
3. establish or retain the active problem;
4. verify the student's submitted work when applicable;
5. retrieve bounded optional RAG context;
6. generate a teacher draft;
7. extract and verify supported mathematical claims;
8. audit question count, answer disclosure, and effective pacing;
9. perform at most one repair call when a hard audit fails;
10. record the accepted response and any remaining warnings.

RAG and memory remain soft context. Neither can override pacing or mathematical
verification.

## Test strategy

Implementation follows test-driven development. Tests are added or changed
before production code.

Required focused cases:

1. a draft with two questions triggers one repair call and the accepted response
   has one ending question;
2. `single_move` with original, intermediate, and final equation states triggers
   repair;
3. `single_move` with one intermediate state followed by a verbal next operation
   triggers repair;
4. one intermediate state followed by an explanation of that same move does not
   trigger repair;
5. the audit applies on later turns, not only on a newly established problem;
6. duplicate equations, original-equation repetition, and variable-free
   tautologies do not inflate the move count;
7. an explicit complete-solution request remains allowed under
   `complete_response`;
8. unsupported mathematical topics retain existing behavior except that the
   universal one-question contract is enforced;
9. existing flexible-step, retrieval, memory, termination, and final-answer
   regressions remain green.

Required repository gates:

```text
python -m pytest
python scripts/run_eval.py
```

Real DeepSeek validation is optional and reported separately because it incurs
external API usage. Offline fake-client tests must reproduce both observed
failure modes without external services.

## Non-goals

- Reworking the observer or replacing LLM-led teaching with a scripted flow.
- Adding another LLM call solely to judge pacing.
- Changing RAG example-exposure or retrieved-example claim auditing.
- Expanding deterministic algebra support beyond the current checker.
- Refactoring unrelated production-agent, memory, retrieval, API, or UI code.
- Repeating repair until a response passes.

## Rollback and integration

The implementation remains on `integration/teacher-agent-consolidation` and is
committed separately from this design. No branch or worktree is deleted, no
force push is used, and no unrelated local state is modified.

If focused or full regression tests reveal unresolved false positives, the code
change remains on the integration branch and is not merged back to the previous
canonical branch.
