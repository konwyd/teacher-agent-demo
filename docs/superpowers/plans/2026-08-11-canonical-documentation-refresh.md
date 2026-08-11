# Canonical Documentation Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stale default-branch documentation with an accurate description of the canonical LLM-led Teacher Agent while preserving the original historical specification.

**Architecture:** The root README becomes the concise public entry point, `CODEX_HANDOFF.md` becomes the detailed current engineering snapshot, and `CLAUDE.md` keeps its original body behind a clear historical-status banner. Runtime code, tests, data, and configuration remain unchanged.

**Tech Stack:** Markdown, Git, PowerShell validation commands, existing Python/pytest evaluation tooling.

## Global Constraints

- The canonical runtime is `prototypes/llm_led_teacher`; `app/agent` remains a legacy/research baseline.
- DeepSeek autonomously chooses teaching moves and wording; the shell preserves context, verifies supported mathematics, supplies optional RAG evidence, and performs one-shot repair.
- RAG and memory are soft constraints and cannot override `app/agent/math_checker.py`.
- General junior-high mathematics is the conversational scope; deterministic verification is strongest for supported one-variable linear equations.
- Do not claim fresh real-service validation without current evidence.
- Do not change runtime source, tests, data, environment files, branches, or worktrees.
- Do not publish or push commits without separate authorization.
- Run `python -m pytest` and `python scripts/run_eval.py` after the edits.

---

### Task 1: Replace the root README with the canonical public overview

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: current runtime entry point, dependency groups, RAG environment names, and repository invariants.
- Produces: the default-branch landing page and quick-start instructions consumed by users and contributors.

- [x] **Step 1: Replace the stale positioning and architecture description**

Use these exact top-level sections in this order:

```markdown
# Teacher Agent — LLM 主导的中文初中数学教师智能体

## 当前架构
## 教学与安全边界
## 快速开始
## 可选 RAG
## 验证
## 仓库结构
## 当前状态与已知限制
## 开发文档
```

State explicitly under `当前架构`:

- the runtime LLM is DeepSeek;
- the LLM chooses the teaching move, explanation depth, and wording;
- the deterministic shell owns active-problem continuity, supported math verification, bounded optional retrieval, response audit, and at most one repair call;
- `TeachingPlanner`, `StrategyRegistry`, and mastery-driven hard flow control are not part of the canonical loop;
- `app/agent` is retained as a legacy/research baseline.

- [x] **Step 2: Correct the scope language**

Replace the one-variable-equation MVP claim with the explicit distinction:

```text
The Teacher Agent may teach general junior-high mathematics through DeepSeek.
Deterministic mathematical verification is currently strongest for supported
one-variable linear equations; unsupported topics continue without pretending
that the checker verified them.
```

Render the final copy naturally in Chinese without expanding the checker claim.

- [x] **Step 3: Document the canonical quick start**

Use PowerShell commands that point to the actual UI:

```powershell
conda create -n teacher-agent python=3.11 -y
conda activate teacher-agent
pip install -e ".[dev,ui]"
Copy-Item .env.example .env
streamlit run prototypes/llm_led_teacher/streamlit_app.py
```

Explain that `DEEPSEEK_API_KEY` is required for real conversations and must
never be committed.

- [x] **Step 4: Document optional RAG without making it mandatory**

Include the install command and two connection modes:

```powershell
pip install -e ".[dev,ui,rag]"
$env:TEACHER_RAG_ENABLED = "true"
$env:CHROMA_SERVER = "localhost:8002"
```

Also mention `TEACHER_RAG_VECTORDB_PATH` as the alternative local persistent
path, and link to `prototypes/llm_led_teacher/README.md` for the complete list.
State that disabled, empty, or failed retrieval does not block teaching.

- [x] **Step 5: Add verification, repository map, and current limitations**

Use the exact offline gates:

```powershell
python -m pytest
python scripts/run_eval.py
```

The repository map must distinguish:

- `prototypes/llm_led_teacher/`: canonical autonomous loop and UI;
- `app/llm/`: sole DeepSeek boundary;
- `app/retrieval/`: reusable optional semantic retrieval;
- `app/agent/math_checker.py`: deterministic mathematical authority;
- `app/agent/`, `app/api/`, `app/ui/`, `app/memory/`: retained structured-workflow baseline and shared research modules;
- `app/evaluation/`, `data/eval/`, `scripts/run_eval.py`, `tests/`: regression gates.

List the pending multi-question and `single_move` enforcement work as known
limitations and link to
`docs/superpowers/specs/2026-08-10-single-move-turn-contract-design.md`.

- [x] **Step 6: Run focused README checks**

Run:

```powershell
rg -n "当前 baseline|MVP 范围：一元一次方程|streamlit run app/ui|下一步才是.*Graph RAG" README.md
rg -n "prototypes/llm_led_teacher/streamlit_app.py|TeachingPlanner|StrategyRegistry|TEACHER_RAG_ENABLED|python -m pytest|python scripts/run_eval.py" README.md
```

Expected: the first command returns no matches; the second confirms all current
architecture and operational markers.

### Task 2: Rewrite CODEX_HANDOFF as the current engineering snapshot

**Files:**
- Modify: `CODEX_HANDOFF.md`

**Interfaces:**
- Consumes: `AGENTS.md`, the RAG consolidation design, current runtime modules, and audited branch SHAs.
- Produces: the detailed handoff coding agents must read before changing the repository.

- [x] **Step 1: Replace the legacy handoff structure**

Use these exact top-level sections:

```markdown
# CODEX_HANDOFF.md — 当前项目交接

## 1. 当前结论
## 2. Canonical Teacher Agent 架构
## 3. 回合数据流与信任边界
## 4. 关键模块
## 5. RAG 架构与配置
## 6. Legacy baseline 的位置
## 7. 教学行为不变式
## 8. 运行与验证
## 9. 当前验证状态
## 10. 已知问题与下一步
## 11. Git consolidation 记录
## 12. 接手规则
```

- [x] **Step 2: Describe the canonical loop and module ownership**

Document this flow exactly in meaning:

```text
observe message and soft pacing
  -> establish or retain active problem
  -> verify supported student mathematics
  -> retrieve or reuse bounded optional evidence
  -> let DeepSeek generate the teaching response
  -> extract and verify supported teacher claims
  -> repair once only when a hard audit fails
  -> persist the audited turn and safe retrieval metadata
```

Map the implementation to:

- `prototypes/llm_led_teacher/logic.py`;
- `prototypes/llm_led_teacher/rag_adapter.py`;
- `prototypes/llm_led_teacher/streamlit_app.py`;
- `app/agent/math_checker.py` and `app/agent/optional_verifier.py`;
- `app/retrieval/semantic_provider.py` and
  `app/retrieval/chroma_semantic_provider.py`;
- `app/llm/deepseek_client.py`.

- [x] **Step 3: Record RAG behavior and external dependencies accurately**

State that RAG is opt-in, non-blocking, and separate from the active problem.
Document statuses `disabled`, `success`, `empty`, and `error`; reference versus
example content; example-exposure gates; cache/refresh behavior; Chroma HTTP or
persistent-path modes; and bge-m3-compatible embeddings.

List configuration names from the focused prototype README without inserting
machine-specific values. Distinguish offline fake-provider coverage from real
Chroma, embedding-model, and DeepSeek checks. Do not preserve stale claims that
only mock Graph RAG exists.

- [x] **Step 4: Preserve the legacy baseline without presenting it as canonical**

Explain that `app/agent/workflow.py`, `app/api/main.py`,
`app/ui/streamlit_app.py`, and mastery persistence remain for compatibility,
regression research, and architecture comparisons. State that they must not be
merged back into the LLM-led loop as hard planning or mastery control.

- [x] **Step 5: Record known issues and Git lineage**

List these current unresolved runtime issues without claiming they are fixed:

- multiple question marks are currently advisory after acceptance;
- `single_move` can still leak a second operation in prose;
- retrieved worked examples need separate audit scoping from the original
  problem;
- optional external services may be unavailable and must be reported as such.

Reference the existing single-move design and documentation-refresh design.

Record these audited SHAs:

- canonical source before consolidation:
  `eee8059cbb52e7abd995583c666ec6b01fb62f53`;
- server RAG snapshot:
  `20af21b76a8d965f37958889e7588e45ae7f20fd`;
- historical shared-remote RAG base:
  `1e535c131a3edf3d010a78abbe10bce701c4829c`;
- completed RAG integration state:
  `9a30b0c16d1e1eda167f9f1f811a6f1b70614256`.

State that RAG was manually adapted rather than merging either old Agent loop.

- [x] **Step 6: Add operational and safety instructions**

Document the canonical Streamlit command, optional RAG setup, both required
offline gates, optional real evaluation command, and a reminder that real LLM
checks incur cost. Reiterate no `.env`, keys, vector database artifacts,
runtime logs, or `data/runtime` contents may be committed.

- [x] **Step 7: Run focused handoff checks**

Run:

```powershell
rg -n "六节点 structured teaching workflow.*✅|当前只保留 mock graph|MVP 范围：一元一次方程|streamlit run app/ui" CODEX_HANDOFF.md
rg -n "LLM-led|prototypes/llm_led_teacher|9a30b0c16d1e1eda167f9f1f811a6f1b70614256|20af21b76a8d965f37958889e7588e45ae7f20fd|single_move" CODEX_HANDOFF.md
```

Expected: the first command returns no matches; the second confirms current
architecture, lineage, and known-issue coverage.

### Task 3: Mark CLAUDE.md historical and validate the complete refresh

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-08-11-canonical-documentation-refresh-design.md`
- Create: `docs/superpowers/plans/2026-08-11-canonical-documentation-refresh.md`

**Interfaces:**
- Consumes: the refreshed README and handoff.
- Produces: unambiguous authority ordering and a verified documentation-only commit.

- [x] **Step 1: Add the historical-status banner without changing the original body**

Insert this meaning immediately after `# CLAUDE.md`:

```markdown
> **历史规范说明（2026-08-11）**
>
> 本文件保留项目最初的 state-machine / mastery / Graph RAG 设想，用于历史追溯，
> 不再代表当前 canonical runtime。当前主实现是
> `prototypes/llm_led_teacher` 下的 LLM-led Teacher Agent。
> 若本文与 `AGENTS.md` 或 `CODEX_HANDOFF.md` 冲突，以后两者为准。
```

Do not edit the remaining original specification.

- [x] **Step 2: Verify every documented path exists**

Run:

```powershell
@(
  'AGENTS.md',
  'CODEX_HANDOFF.md',
  'CLAUDE.md',
  'prototypes/llm_led_teacher/README.md',
  'prototypes/llm_led_teacher/logic.py',
  'prototypes/llm_led_teacher/rag_adapter.py',
  'prototypes/llm_led_teacher/streamlit_app.py',
  'app/agent/math_checker.py',
  'app/agent/optional_verifier.py',
  'app/retrieval/semantic_provider.py',
  'app/retrieval/chroma_semantic_provider.py',
  'app/llm/deepseek_client.py',
  'scripts/run_eval.py'
) | ForEach-Object { if (-not (Test-Path -LiteralPath $_)) { throw "Missing documented path: $_" } }
```

Expected: exit code 0 with no missing-path error.

- [x] **Step 3: Verify the diff is documentation-only and clean**

Run:

```powershell
git diff --check
git status --short
git diff --name-only 9a30b0c16d1e1eda167f9f1f811a6f1b70614256..HEAD
git diff --name-only
```

Expected: only `README.md`, `CODEX_HANDOFF.md`, `CLAUDE.md`, the two design
documents, and this plan appear across the complete documentation series; the
working-tree diff contains only the three target documents plus the spec
correction and plan before commit.

- [x] **Step 4: Run repository-wide offline gates**

Run:

```powershell
python -m pytest
python scripts/run_eval.py
```

Expected: both commands exit 0. Record actual totals from command output in the
final handoff rather than retaining stale hard-coded test counts.

- [x] **Step 5: Review the final prose diff**

Run:

```powershell
git diff -- README.md CODEX_HANDOFF.md CLAUDE.md docs/superpowers/specs/2026-08-11-canonical-documentation-refresh-design.md docs/superpowers/plans/2026-08-11-canonical-documentation-refresh.md
$markers = @('T' + 'BD', 'T' + 'ODO', 'PLACE' + 'HOLDER', 'FIX' + 'ME')
Select-String -Path README.md,CODEX_HANDOFF.md,CLAUDE.md,docs/superpowers/specs/2026-08-11-canonical-documentation-refresh-design.md,docs/superpowers/plans/2026-08-11-canonical-documentation-refresh.md -Pattern $markers
```

Expected: no placeholders, no machine-specific paths, no secrets, no claims
that external services passed, and no runtime source changes.

- [x] **Step 6: Commit the coherent documentation refresh**

Run:

```powershell
git add README.md CODEX_HANDOFF.md CLAUDE.md docs/superpowers/specs/2026-08-11-canonical-documentation-refresh-design.md docs/superpowers/plans/2026-08-11-canonical-documentation-refresh.md
git commit -m "docs: align default branch with llm-led runtime"
```

Do not push. Report the new commit SHA and the branch's ahead/behind state.
