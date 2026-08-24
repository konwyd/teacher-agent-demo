# 项目进度
ARCHIVED DOCUMENT

This document describes an earlier project state and is not authoritative.
Do not use it as current runtime or architecture guidance.

Current sources of truth:
- AGENTS.md
- OWNERSHIP.md
- PROGRESS.md
- DECISIONS.md
## 第二轮完成快照（2026-08-14，优先于下方第一轮状态）

- 功能提交：隔离分支 `5b5288f`，主分支等价合入提交 `b342313`
- 验证：隔离分支 604/604 pytest；合并主分支 613/613 pytest、38/38 离线评测、12/12 完整真实模型矩阵及合并后 4/4 聚焦烟测
- 性能：调用 44 -> 35，token 76,985 -> 57,249，耗时 474.671s -> 355.519s，修复 10 -> 6
- 已完成：真实烟测 runner、续题识别、严格 fenced JSON、概念审计、单问题收敛、错误反馈一致性、占位回复门禁
- 部署：`b342313` 已由 8502 加载，8080/8002/8502 健康检查均为 HTTP 200


## 第一轮历史状态（已被上方完成快照取代）
- 当前分支: `codex/skill-orchestration-interface-fix`
- 最新功能 commit: 隔离分支 `25d478e`，主分支 `11bca60`（补丁内容一致）
- 测试状态: 587/587 通过
- 离线教学评测: 38/38 通过
- Lint: 项目未配置独立命令；`git diff --check` 通过
- 集成状态: 功能修复已合并到 `codex/llm-led-teaching-agent`；8502 正从主工作树运行

## 已完成
- [x] 将教师 Skill 名称和说明集中到规范目录
- [x] 将 Agent 决策绑定到规范 Skill，未知名称只允许一次确定性回退
- [x] 对成功和失败的 Skill/Tool 调用做硬去重，重复加载时立即短路
- [x] 将教材检索的学科和学段绑定到已验证分类结果
- [x] 未取得数学验证证据前隐藏 `error_diagnosis`，避免越权诊断
- [x] 覆盖九学科、别名、重复调用、检索过滤和边界保护测试
- [x] 完成全量自动化测试和离线教学评测

## 进行中
- [ ] 在 8502 的常驻进程中执行跨学科真人烟测
- [ ] 将本轮进度、决策和交接文档同步回主分支

## 已知问题
- 隔离冷启动进程中，RAG embedding 初始化可能超过 3 秒并触发超时；需在 8502 常驻环境复核
- 一次真实模型测试曾选择两个有效 Skill；后续未稳定复现，需要纳入跨学科烟测观察
- 简单数学题仍可能走 5 次编排模型调用；已由约 8,453 tokens 降至约 5,788 tokens，但快速路由仍有优化空间
- 12 组真实模型跨学科验收尚未完成，当前 38 组评测使用 mock LLM

## 下一步
1. 在 8502 常驻环境运行 12 组跨学科真实模型烟测，记录 Tool/Skill 轨迹、耗时和 token
2. 若复现 RAG 超时或重复 Skill，再开聚焦修复轮
3. 根据烟测数据评估简单数学题快速路由，减少不产生教学价值的模型调用
4. 将本轮文档提交同步回主分支
