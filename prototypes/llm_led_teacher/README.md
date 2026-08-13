# LLM-led Teacher Prototype

This prototype asks whether a bounded DeepSeek Agent can lead a multi-turn K9
lesson while local, testable contracts preserve the active problem and control
what the student-facing answer may disclose. It deliberately bypasses the legacy
TeachingPlanner, mastery control, and Pedagogical Graph RAG.

The 8502 path runs `run_turn(..., enable_agent_loop=True)`. Its Agent replaces
the experimental Observer as the source of teaching intent: after classifying
scope, subject, stage, and problem continuity, the bounded loop may load a Skill
or call guarded local/web knowledge tools, then must submit a `TeachingContract`.
The contract records one of these response modes plus an explicit target,
allowed/forbidden content, active-problem application permission, and final-answer
permission:

- `concept_explanation`: explain the general concept without applying it to the
  active problem or revealing that problem's answer;
- `one_step` / `guided_solution`: apply only the bounded help the contract allows;
- `full_solution`: apply to the active problem and reveal the final answer only
  after an explicit complete-solution request from the student's original text;
- `brief_redirect`: answer an off-topic request briefly with zero knowledge-tool
  calls and invite a return to K9 learning.

Local policy normalizes the model-proposed permissions from the original student
message, so the Agent cannot grant itself a wider disclosure scope. The legacy
`single_move` / `complete_response` `TurnGuidance` remains only as a compatibility
projection derived from the trusted contract; it is not the authority for the
8502 Agent path. The old Observer remains available only when the Agent loop is
explicitly disabled.

After teacher generation, the configured local model performs a semantic
boundary audit against the contract. It checks target, application permission,
answer disclosure, and step scope; a failed or unavailable audit causes a
conservative repair and high-confidence literal recheck before display. This is
**not a universal factual verifier** and an audit pass must not be presented as
proof that every math, physics, or other subject claim is correct.

Deterministic verification is an optional subject tool. The current adapter
exposes the existing Fraction-based linear-equation `math_checker`; unsupported
problems record `not_applicable` or `unable_to_verify` and continue through the
same teaching-contract boundary. Missing verification never becomes fabricated
verified evidence and does not block a response.

Before persistence and display, unambiguous model LaTeX delimiters and textbook
formula blocks are normalized to MathJax-compatible `$...$` / `$$...$$`
Markdown without modifying fenced code. The normalized teacher text is reused in
later history. Successful turns are stored in the per-user SQLite conversation
archive together with the classification, TeachingContract, boundary audit,
optional verifier result, formula-normalization marker, and the real public
Agent/Tool trace. The UI re-renders those public events and does not expose
prompts, private event details, or chain-of-thought.

Run the local 8080/8002/8502 stack from the repository or worktree root:

```bash
python3 scripts/run_llm_led_teacher.py
```

No real model or paid API is called by automated tests. Starting the app and
sending a message uses the model endpoint selected by the launcher/configuration;
the standard launcher pins Teacher subprocesses to the local 8080 endpoint.
