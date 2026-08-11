# Canonical Documentation Refresh Design

Date: 2026-08-11

## Objective

Make the default branch documentation describe the actual canonical Teacher
Agent: the autonomous LLM-led teaching loop with deterministic mathematical
auditing and optional bounded semantic RAG.

The refresh must prevent readers and coding agents from mistaking the retained
structured workflow under `app/agent` for the primary runtime. It must also
preserve the historical project specification instead of rewriting history.

## Confirmed problem

The current root `README.md` and `CODEX_HANDOFF.md` predate the LLM-led
prototype and the RAG consolidation. They still present the following as the
current product:

- a deterministic teaching-policy workflow;
- mastery-driven flow control;
- mock Graph RAG;
- a one-variable linear-equation MVP boundary; and
- `app/ui/streamlit_app.py` as the main UI.

The accurate description currently exists only in
`prototypes/llm_led_teacher/README.md`. `CLAUDE.md` is the original project
specification and also describes the superseded state-machine direction.

## Chosen approach

Use a three-document update:

1. Rewrite the root `README.md` as the concise canonical entry point.
2. Rewrite `CODEX_HANDOFF.md` as the detailed current engineering handoff.
3. Preserve the body of `CLAUDE.md`, but add a prominent historical-status
   banner pointing readers to the current authority documents.

This is preferred over updating only the root README because coding agents are
explicitly instructed to read `CODEX_HANDOFF.md` and `CLAUDE.md`. It is
preferred over turning the root README into a link-only index because the
default branch should be understandable without navigating into a prototype
subdirectory.

## Source of truth hierarchy

The refreshed documents establish this order:

1. `AGENTS.md` — mandatory coding and teaching invariants.
2. `CODEX_HANDOFF.md` — current implementation and operational handoff.
3. root `README.md` — public overview and quick start.
4. `prototypes/llm_led_teacher/README.md` — focused runtime and RAG details.
5. `CLAUDE.md` — retained historical original specification, not current
   architecture authority.

No document may imply that branch names alone establish architecture truth.
The current runtime paths and tested behavior are authoritative.

## Root README design

The root README will be concise and user-facing. It will contain:

- project positioning: Chinese junior-high mathematics Teacher Agent research
  demo focused on teacher quality;
- current architecture: DeepSeek autonomously chooses teaching moves and
  wording, while a thin shell preserves the active problem, verifies supported
  mathematics, supplies optional RAG evidence, and repairs hard violations;
- an explicit statement that `TeachingPlanner`, `StrategyRegistry`, and
  mastery-based hard flow control are outside the canonical loop;
- a scope distinction: the teacher can discuss general junior-high mathematics,
  while deterministic verification is strongest for supported one-variable
  linear equations;
- the canonical Streamlit command;
- optional RAG installation and configuration at a useful summary level;
- offline verification commands;
- a short repository map distinguishing canonical runtime, shared safety
  modules, evaluation assets, and legacy/research baseline;
- links to `AGENTS.md`, `CODEX_HANDOFF.md`, and the focused prototype README.

It will not claim that missing external services passed real validation, and it
will not expose API keys, workstation paths, or machine-specific configuration.

## CODEX_HANDOFF design

The handoff will be rewritten as the current technical snapshot. It will
include:

- canonical branch/worktree context using the audited commit lineage;
- current product goal and trust boundaries;
- canonical turn flow from observation through optional retrieval, generation,
  deterministic audit, one-shot repair, and persistence;
- module ownership for the LLM-led runtime, shared checker, retrieval provider,
  DeepSeek client, UI, tests, and evaluation;
- the status of the retained `app/agent` workflow as a compatibility and
  research baseline, not the main loop;
- RAG architecture, configuration, failure behavior, and example-exposure
  rules;
- current known limitations, including single-move and multi-question issues
  that are designed but not yet implemented;
- startup and verification commands;
- external-service verification distinctions;
- safe Git and development rules consistent with `AGENTS.md`;
- a short historical branch-consolidation summary with important SHAs.

Historical bug tables, stale test counts, old roadmap items, and claims that
mock Graph RAG is the current retrieval architecture will be removed rather
than carried forward as current status.

## CLAUDE.md treatment

The original body will remain unchanged. A banner at the top will state:

- this file records the original project specification;
- its state-machine, mastery, and Graph RAG architecture is historical;
- the canonical runtime is now the LLM-led prototype; and
- current work must follow `AGENTS.md` and `CODEX_HANDOFF.md` when they conflict
  with this historical document.

This preserves provenance while preventing accidental reintroduction of the
superseded architecture.

## Accuracy rules

- Do not describe semantic RAG as mandatory; it is opt-in and non-blocking.
- Do not describe RAG or memory as workflow authorities.
- Do not imply the deterministic checker covers all junior-high mathematics.
- Do not describe the legacy workflow as deleted.
- Do not claim real Chroma, embedding, or DeepSeek validation without current
  evidence.
- Keep all commands valid for Windows PowerShell, with optional POSIX notes only
  when useful.
- Keep the required offline gates exactly as specified in `AGENTS.md`:
  `python -m pytest` and `python scripts/run_eval.py`.

## Verification

Because this change is documentation-only, verification focuses on consistency:

1. search the three documents for stale canonical claims;
2. verify every referenced path exists;
3. verify documented commands match current entry points and dependency groups;
4. inspect Markdown structure and links;
5. run `git diff --check`;
6. confirm no runtime source, tests, data, or environment files changed;
7. run `python -m pytest`; and
8. run `python scripts/run_eval.py`.

Although the implementation is prose-only, the repository-wide gates in
`AGENTS.md` remain mandatory. The handoff will report their actual results and
will not claim a fresh pass unless both commands complete successfully.

## Non-goals

- Modifying runtime behavior, prompts, RAG, memory, UI, or tests.
- Renaming or moving `prototypes/llm_led_teacher`.
- Deleting the legacy workflow or historical branches.
- Rewriting the original `CLAUDE.md` specification body.
- Implementing the pending single-move and multi-question repair design.
- Publishing or pushing commits without separate authorization.

## Commit structure

The design document is committed separately. After user review, the three
documentation changes will be implemented and committed together as one
coherent documentation refresh.
