# 当前工作交接

## 最新增量：结构化学生记忆（2026-08-17）

- 已按 `user_id -> student_profile_id -> global / subject / domain / topic` 接入 SQLite 结构化记忆；8502 UI 只绑定身份，不再抢先选择最近会话。
- Agent Loop 在现有 Fast Router / RequestClassifier 之后解析主题：唯一高置信候选恢复上下文并以软参考注入；多个候选先返回一个确定性澄清问题；无候选只注入 global 档案。
- 澄清候选保存在 Episode，下一轮支持主题名或“第一个/第二个/第三个”，通过 `memory_clarification_reply` 继续，不重复调用分类模型。
- 回答完成后投递 SQLite outbox，由单后台 worker 确定性提取年级、称呼、教学偏好、学科主题、进度和困难；失败最多重试三次，进程恢复时重置遗留 processing 任务。
- 旧 `cross_session_memory` 仅保留反序列化兼容，不再进入 Teacher payload；旧 Conversation Archive 只作为结构化表回填前的候选来源。
- 关键实现：`app/memory/learner_memory.py`、`prototypes/llm_led_teacher/orchestrator.py`、`prototypes/llm_led_teacher/logic.py`、`prototypes/llm_led_teacher/streamlit_app.py`。
- 行为测试：`tests/behavior/test_structured_student_memory.py`，覆盖异步写回、用户隔离、唯一/多候选/无候选、旧 Archive fallback、零 LLM 澄清和下一轮确定性选题。
- 本次最终验证：`python3 -m pytest -q --tb=short` 为 681/681 通过；`python3 scripts/run_eval.py` 为 38/38 通过；`git diff --check` 通过。
- 当前运行参数已核实为 launcher `--parallel 4 --unified-kv`，llama-server `--parallel 4 --kv-unified --cont-batching`。`TEACHER_KV_PERSIST=0` 只关闭 SSD slot 快照，不关闭 unified KV。
- 权威链路说明见 `docs/8502_Teacher_Agent_Loop_全链路说明.md`。当前工作树还有其他未提交改动与运行时数据，后续提交时必须只选择本功能相关文件。

---


## 0. 第二轮优化完成状态（2026-08-14，优先于下方第一轮快照）

- 隔离分支已提交 `5b5288f fix: stabilize live agent loop boundaries`，完整测试 604/604、离线评测 38/38 通过。
- 12 类真实模型矩阵 12/12 通过；受最终修复影响的聚焦复测分别为 4/4 和 3/3 通过。
- 同一矩阵的已测对比：调用 44 -> 35，token 76,985 -> 57,249，耗时 474.671s -> 355.519s，修复轮次 10 -> 6。后续个案继续下降，但没有重跑整套聚合，因此不记录未经测量的新总数。
- 已修复：数学步骤误判新题、整段 fenced JSON 解析、概念边界误审、格式性多问题触发修复、VERIFIED_INCORRECT 反馈含糊，以及占位回复假绿。新增 `scripts/run_llm_led_agent_smoke.py`。
- 两组并发工作已分别提交为 `6f0d88f`（本地/API 后端切换）和 `bf66a41`（安全实时思考展示）；第二轮 Agent Loop 已无损合入主分支 `b342313`，8502 已加载新代码。
- 主分支合并后验证为 613/613 pytest、38/38 离线评测和 4/4 真实模型聚焦烟测；8080、8002、8502 均健康，部署参数保持不变。


> 更新时间：2026-08-14。本文是本轮 Skill 编排接口修复的权威交接入口。根目录原有的 `CODEX_HANDOFF.md` 和 `HANDOFF.md` 记录较早架构，仅供历史背景参考；若状态冲突，以本文、`progress.md` 和实际 Git/进程检查为准。

## 1. 第一轮历史结论（已被第 0 节取代）

- Agent Loop 的核心接口修复已经完成，并以等价补丁合入主开发分支。
- 完整自动化测试为 **587/587 通过**，离线教学评测为 **38/38 通过**。
- 8080 本地模型、8002 Chroma 和 8502 Streamlit 当前均由主工作树启动。
- 尚未完成的是 8502 常驻环境中的 12 组跨学科真实模型烟测，以及把本轮三份文档同步回主分支。
- 不要再次合并或 cherry-pick 隔离分支中的功能提交；主分支已经包含相同补丁，只是 commit hash 不同。

## 2. 分支与目录

### 正在运行的主工作树

- 目录：`/home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo`
- 分支：`codex/llm-led-teaching-agent`
- 最新功能提交：`11bca60 fix: guard verified skill boundaries`
- 8502 进程从此目录启动。

### 本轮隔离工作树

- 目录：`/home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo/.worktrees/skill-orchestration-interface-fix`
- 分支：`codex/skill-orchestration-interface-fix`
- 对应功能提交：`25d478e fix: guard verified skill boundaries`
- `25d478e` 与主分支 `11bca60` 的 stable patch-id 均为 `858c54e68ec299130f9cf1d68cf44548b4adb9e4`，确认是同一补丁。
- `progress.md`、`DECISIONS.md` 和本文目前只在此隔离分支。

## 3. 本轮解决了什么

目标是修复“Agent 找不到合适 Skill、反复调用、检索范围漂移、简单问题消耗过高”的编排接口问题，同时保留现有登录、上下文长度、缓存命中率、公式渲染和教学边界功能。

已落地行为：

1. Skill 名称和说明统一由 `skill_policy.py` 的规范目录提供。
2. 模型生成未知 Skill 时，只允许一次确定性安全回退。
3. 成功和失败的 Skill/Tool 调用均按签名去重；重复加载已成功 Skill 会短路循环。
4. 教材检索的 `subject` 和 `stage` 强制绑定已验证分类结果，Agent 不能自行改写检索边界。
5. 未获得可靠数学验证证据前，不向 Agent 暴露 `error_diagnosis`。
6. Agent 继续按需选择 Skill/Tool，不恢复固定六阶段流程。

关键实现文件：

- `prototypes/llm_led_teacher/skill_policy.py`
- `prototypes/llm_led_teacher/orchestrator.py`
- `prototypes/llm_led_teacher/orchestration_tools.py`
- `prototypes/llm_led_teacher/logic.py`
- `tests/prototypes/test_skill_policy.py`
- `tests/prototypes/test_llm_led_orchestrator.py`
- `tests/prototypes/test_llm_led_orchestration_tools.py`

## 4. 功能提交映射

隔离分支和主分支内容一致，但由于提交方式不同，hash 不同：

| 作用 | 隔离分支 | 主分支 |
|---|---|---|
| 集中 Skill policy | `41ed3f7` | `1ec3862` |
| Skill 加载幂等 | `4c050d1` | `ce1bfd6` |
| policy 文件格式修正 | `3efa1c2` | `bac1592` |
| 决策绑定规范 Skill | `622a1bf` | `5da0db0` |
| 检索绑定分类边界 | `3aecf25` | `5d8e8db` |
| 验证边界与重复短路 | `25d478e` | `11bca60` |

## 5. 验证证据

在隔离分支最新功能代码上已执行：

```bash
python3 -m pytest -q
# 587 passed in 45.57s

python3 scripts/run_eval.py
# 38 cases, 38 passed, 0 failed（mock LLM）

git diff --check
# passed
```

真实本地模型的一次简单数学题编排测试：

- 编排模型调用从旧基线约 9 次降为 5 次。
- 总 token 从旧基线约 8,453 降为约 5,788，下降约 31.5%。
- 该数字来自单次样本，只能说明方向，不能代替 12 组烟测。

## 6. 当前服务快照

截至本文更新时：

| 服务 | 端口 | 状态 | 关键参数 |
|---|---:|---|---|
| DeepSeek-V4 llama-server | 8080 | 运行中 | ctx 524288，parallel 1，batch/ubatch 128/128 |
| Chroma 教材库 | 8002 | 运行中 | 主工作树 `data/china_textbook_k9_vectordb` |
| LLM-led Teacher | 8502 | 运行中 | 主工作树 `scripts/run_warmed_streamlit.py` |

标准启动入口：

```bash
cd /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo
python3 scripts/run_llm_led_teacher.py \
  --model-script /home/dockerContainerSave/data/wyd-workspace/run_deepseek_v4_8x4080.sh \
  --ctx-size 524288 \
  --parallel 1 \
  --batch-size 128 \
  --ubatch-size 128 \
  --app-port 8502 \
  --rag-port 8002 \
  --db-path /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo/data/china_textbook_k9_vectordb
```

启动器会为本地地址设置 `NO_PROXY/no_proxy`。若绕过启动器直接跑诊断脚本，也必须保证 `127.0.0.1,localhost` 不经过 HTTP 代理，否则访问 8080/8002 可能出现误导性的 502。

## 7. 已知问题与风险

1. 隔离冷启动进程中，RAG embedding 初始化曾超过 3 秒并触发超时；需要在已经预热的 8502 常驻环境复核。
2. 一次真实模型运行曾选择两个不同的有效 Skill，后续未稳定复现；应观察真实调用轨迹，不要仅看最终回答。
3. 简单数学题仍可能触发 5 次编排模型调用，快速路由还有优化空间。
4. 自动评测使用 mock LLM；本地真实模型具有随机性，必须完成跨学科烟测后才能下最终性能结论。
5. `error_diagnosis` 当前被全局隐藏。这是刻意的保守边界，不是遗漏；只有接入可靠验证证据后才应重新开放。

## 8. 下一位接手者的执行顺序

1. 先阅读 `progress.md` 与 `DECISIONS.md`，不要从两份 2026-07 历史交接文档推断当前架构。
2. 确认 8502 页面可访问，并检查页面调用轨迹确实出现按需 Skill/Tool，而非固定阶段文案。
3. 在同一登录会话中测试至少 12 组：简单数学引导、数学概念解释、学生错误步骤、物理、化学、语文、英语、历史/地理/生物、追问续答、明确索要完整答案、课外问题和无效输入。
4. 每组记录：分类结果、加载的 Skill、Tool 调用、循环终止原因、首字延迟、总耗时、总 token、缓存命中率、是否遵守教学边界。
5. 若失败，优先判断是 RAG 冷启动、Skill 重复、检索边界、教学契约还是模型生成问题；一次只修一个根因。
6. 修改后重新执行完整 `pytest`、`scripts/run_eval.py` 和相关真实烟测。
7. 仅将文档提交同步到主分支；不要重复同步表格中的六个功能提交。

## 9. 文档同步方式

在主工作树中先查看隔离分支相对主分支新增的文档提交：

```bash
git -C /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo \
  log --oneline codex/llm-led-teaching-agent..codex/skill-orchestration-interface-fix
```

然后只 cherry-pick `docs:` 提交。同步前先检查主工作树是否干净；不要用 destructive reset，也不要覆盖用户的未提交修改。

## 10. 不要破坏的约束

- 不修改当前 KV/context/parallel/batch 参数来掩盖编排问题。
- 不破坏登录、会话历史、上下文用量和缓存命中率 UI。
- 不允许 Agent 自行扩大教学契约或提前给出题目最终答案。
- 不把语义边界审计描述成通用事实或数学正确性证明。
- 不让联网搜索成为本地教材检索失败的默认兜底。
- 不提交 `.env`、运行日志、SQLite/Chroma 运行时变化或模型缓存。
