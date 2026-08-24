# AGENTS.md — Coding Agent Operating Protocol

本仓库是中文初中数学 Teacher Agent research demo。

8502 是当前唯一主运行时。
`prototypes/llm_led_teacher/**` 中的正式 runtime 代码不能因为位于 `prototypes/` 下而被视为一次性实验代码。

8501/8000 为冻结的 legacy baseline：
- 不再进行新功能开发
- 仅接受 regression fix 和 critical fix
- 8502 新功能不需要维持 legacy feature compatibility

Runtime LLM 为 DeepSeek API / local inference depending on environment。

你是 coding agent，不是 runtime teacher model。


## 1. Required Context Before Work

开始任何开发任务前，必须先阅读：

- `AGENTS.md`
- `OWNERSHIP.md`
- `PROGRESS.md`
- `DECISIONS.md`
- 当前 subsystem 相关文档

其中：

- `AGENTS.md` 定义你应该如何工作
- `OWNERSHIP.md` 是 subsystem ownership、primary paths、protected interfaces、shared interfaces 和 mandatory co-review 的唯一 source of truth
- subsystem 文档描述当前功能线的设计与实验状态

不要在 `AGENTS.md` 中自行重新解释或覆盖 `OWNERSHIP.md`。


## 2. Active Session Scope

本仓库由多个开发者和多个 Codex sessions 并行维护。

不同 Codex conversations 可能来自同一个 Codex 账号。

因此你绝不能根据以下信息推断当前开发者身份或 ownership：

- Codex account
- conversation history
- git author
- workspace 名称
- 文件所在目录

开始修改代码前，必须能够明确：

Active subsystem:
[SUBSYSTEM]

Acting owner:
[OWNER]

Task:
[SHORT DESCRIPTION]

如果当前任务已经明确提供这些信息，可以直接使用。

如果无法明确判断 active subsystem 或 acting owner，在修改代码前必须询问开发者。

当前 session 默认只能：

- 修改 active subsystem 的 primary paths
- 修改当前任务明确批准的 shared interfaces
- 修改明确获得批准的跨 subsystem 文件

如果发现其他 subsystem 存在问题：

不要顺手修复。

先报告：

- affected subsystem
- affected files
- dependency/blocker
- 为什么当前任务会受到影响

未经批准不要扩展 task scope。


## 3. Git and Worktree Preflight

本仓库存在多人、多 branch、多 worktree 并行开发。

绝不能假设：

- 当前 branch 是 default branch
- `main` 是 default branch
- 当前 checkout 是主 checkout
- 当前 workspace 是唯一 checkout
- 当前 dirty files 属于你的任务
- repository root 与当前工作目录相同
- push 当前 branch 就等于更新 GitHub default branch

开始修改前至少确认：

- repository root
- current branch
- current HEAD
- upstream / tracking branch
- remote
- current worktree
- other linked worktrees
- dirty / untracked files

至少执行或等价检查：

git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
git worktree list
git remote -v

如任务涉及 commit / push / merge，还必须额外确认：

- 当前 branch 的 upstream
- 实际 push target
- remote default branch（如果相关）

不要根据 branch 名字猜测这些信息。


## 4. Multi-Agent Safety Rules

你不是本仓库唯一正在工作的 agent。

永远不要：

- 删除你没有创建的 dirty changes
- reset 其他开发者的修改
- restore unrelated files
- 清理看似“无关”的 untracked files
- 修改另一个 worktree
- 因测试失败而自动修改另一个 subsystem
- 因当前任务需要而自动扩大 ownership
- 自动 merge 实验 branch
- 自动 cherry-pick 其他开发者的 commit

以下 destructive Git 操作未经明确批准不得执行：

- `git reset --hard`
- `git clean -fd`
- 对无关文件执行 `git restore`
- 覆盖其他开发者未提交修改
- force push

并行独立开发任务优先使用独立 git worktree。

一个 worktree 应对应：

- 明确 branch
- 明确 task
- 明确 active subsystem

不要通过在共享 checkout 中频繁 `git switch` / `git checkout` 来模拟并行开发。


## 5. Ownership and Shared Interfaces

Ownership 统一由 `OWNERSHIP.md` 定义。

不要：

- 在本文件重新维护 ownership map
- 仅根据目录位置推断 ownership
- 因创建新文件而自动决定它属于哪个 subsystem
- 因文件移动而自动改变 semantic ownership

修改 shared interface 前：

1. 查阅 `OWNERSHIP.md`
2. 确认 single owner
3. 确认 mandatory co-review subsystems
4. 确认当前任务是否已获得修改许可

Shared contract change 必须按照 `OWNERSHIP.md` 的 protocol 分析至少：

- producer
- consumers
- schema compatibility
- persistence compatibility
- evaluation impact
- migration/default behavior
- rollback behavior

如果无法确定是否属于 contract change，按 shared-interface change 处理并先报告。


## 6. General Engineering Rules

### 6.1 Incremental Changes

不要重构整个项目。

现有 architecture 和 subsystem boundaries 是刻意设计的。

优先：

- 最小修改
- 局部实现
- 保持现有 public contracts
- 使用已有 infrastructure

除非当前任务明确要求，不要进行 unrelated cleanup 或 architecture rewrite。


### 6.2 Evaluation Is a Regression Gate

不要删除、绕过或弱化 evaluation。

禁止为了让测试通过而：

- 删除 case
- 放宽 assertion
- 修改 expected behavior
- 修改 golden/holdout label
- 改 metric 定义

如果 evaluation expectation 确实需要变化，必须先说明：

- 原行为
- 新行为
- 教学或架构理由
- 受影响 subsystem

Memory-specific evaluation 的 ownership 和保护规则以 `OWNERSHIP.md` 为准。


### 6.3 Math Checker Is Authoritative

学生步骤、多步计划和最终答案的数学正确性，在适用范围内必须由：

`app/agent/math_checker.py`

进行 deterministic verification。

LLM、Memory、RAG、Teaching Policy 都不得推翻 checker 结论。


### 6.4 Memory Is a Soft Teaching Signal

Memory 可以影响：

- explanation depth
- analogy
- review
- hint level
- practice / variation selection

Memory 不得：

- 决定数学正确性
- 阻止学生跳步
- 强制一步一答
- 覆盖当前学生消息
- 覆盖 TeachingContract
- 阻断正常教学流程

Memory failure 应保持 fail-open。


### 6.5 Memory V2 Remains Shadow

Memory V2 admission/controller 当前不得：

- authorize baseline write
- block baseline write
- modify authoritative baseline write semantics

除非有新的明确 architecture decision，否则保持 shadow。


### 6.6 Student Memory and Agent Experience Memory Must Stay Separate

Quality-QA / Agent Experience Memory 不等于 Student Memory。

不得：

- 将 Quality-QA 自动写入学生 profile/topic memory
- 将 agent experience 当成某个学生的个人事实
- 混淆 curated knowledge、agent experience 与 student memory 的 provenance


### 6.7 RAG Is Teaching Evidence

RAG 可以提供：

- prerequisite knowledge
- misconceptions
- diagnostic information
- teaching strategies
- hints
- worked-example candidates
- variation suggestions

RAG 不得：

- 决定学生答案正确性
- 覆盖 math_checker
- 覆盖 TeachingContract
- 自动替换 active/original problem

Retrieved examples 只能在明确允许例题或练习题的 teaching action 中使用。


### 6.8 LLM Gateway

Production LLM 调用必须经过：

`app/llm/deepseek_client.py`

结构化输出使用：

- JSON mode
- Pydantic validation

不要在其他 subsystem 中自行创建新的 OpenAI / Anthropic / compatible client。


### 6.9 Secrets and Runtime Artifacts

永远不要提交、打印或硬编码：

- `.env`
- API keys
- credentials
- private reasoning

不要将以下 runtime artifacts 当作源码提交：

- `logs/`
- `data/runtime/`
- local databases
- temporary model/runtime artifacts


## 7. Teaching Behavior Invariants

修改 workflow、prompt、Teaching Policy、response generation 或相关模板前，必须确保：

- `original_problem` 首轮确定后跨轮保持，不被无意覆盖
- 普通讲解持续围绕当前 active problem
- 非 worked-example / practice action 不得无条件引入新题污染当前问题
- 学生未解出且未索要最终答案时，不应提前泄露最终答案
- 学生可以：
  - 单步作答
  - 多步作答
  - 跳步
  - 直接给答案
  - 询问原因
- 这些输入都必须被正常识别和处理

已有更具体的教学 invariant、golden case 或 evaluator 规则优先保持。


## 8. Testing Protocol

实现过程中优先运行与当前修改最相关的 targeted tests。

任务完成前必须运行：

python -m pytest

python scripts/run_eval.py

两者默认应保持 offline / mock mode。

如果当前 subsystem 有额外 benchmark 或 regression command，也必须运行相关检查。

例如修改：

- math_checker → 更新并运行对应 checker/flexible-step tests
- shared schema → 运行所有相关 consumer tests
- Memory admission → 运行对应 Memory admission evaluation/robustness
- teaching behavior → 更新或新增相应 golden/evaluator coverage

如果测试失败：

优先定位真实原因。

不要为了让测试变绿而：

- 删除测试
- 放宽断言
- 修改正确预期
- 绕过失败路径

如果失败来自当前 task scope 之外：

停止跨域修改，并报告 blocker。


## 9. Commit / Push / Merge Protocol

不要把“代码修改完成”和“应该 commit/push/merge”视为同一件事。

除非开发者明确要求，否则不要擅自：

- merge
- rebase
- cherry-pick
- push 到其他 branch
- 修改 remote default branch

在 commit / push 前必须报告：

- Current branch
- Current HEAD
- Upstream
- Intended commit scope
- Intended push target
- Dirty files
- Unrelated changes
- Current worktree

Push 前必须明确：

本次 push 的目标 branch 是什么。

绝不能因为：

`git push`

成功，就宣称 GitHub default branch 已更新。

如果开发者要求更新 default branch，应先确认：

- remote default branch
- 当前 branch 与 default branch 的关系
- 需要 push、merge 还是其他操作

然后再执行。


## 10. Completion Protocol

任务结束前：

1. 确认实际修改文件。
2. 确认没有修改不属于当前 scope 的文件。
3. 运行 required tests/evaluation。
4. 检查 shared-interface impact。
5. 检查 git/worktree 状态。
6. 明确说明剩余 blocker 或未完成项。

最终汇报至少包含：

- Active subsystem
- What changed
- Files changed
- Tests/evaluation run
- Test results
- Shared interfaces affected
- Cross-subsystem impact
- Remaining blockers
- Current branch / worktree status

不要把未验证的修改描述为完成。

If the task materially changes project state:

- update only the affected subsystem section in `PROGRESS.md`
- record current phase, completed work, blockers and next step
- do not rewrite unrelated subsystem progress
- do not duplicate architecture decisions from `DECISIONS.md`

## 11. Final Checklist

开始修改前：

- [ ] 已阅读 `OWNERSHIP.md`
- [ ] 已确认 active subsystem
- [ ] 已确认 acting owner
- [ ] 已确认 task scope
- [ ] 已检查 branch / HEAD / worktree / dirty files
- [ ] 已确认不存在 ownership 冲突

实现过程中：

- [ ] 没有修改无关 subsystem
- [ ] 没有覆盖其他开发者修改
- [ ] shared interface 修改已按规则处理
- [ ] 没有进行 unrelated refactor
- [ ] 没有弱化 evaluation

结束前：

- [ ] targeted tests 已运行
- [ ] `python -m pytest` 已运行
- [ ] `python scripts/run_eval.py` 已运行
- [ ] subsystem-specific evaluation 已按需运行
- [ ] 无 secrets/runtime artifacts 被提交
- [ ] 已报告 shared-interface / cross-subsystem impact
- [ ] 已报告 branch / worktree 状态
- [ ] commit / push / merge target 未被假设