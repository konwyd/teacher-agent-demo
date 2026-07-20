# LLM 回合观察器实验设计

## 目标

在不恢复 Planner、策略族或强制知识点流程的前提下，用一次额外 LLM 调用区分：

- 当前消息是否开启了一道新题；
- 当前回合应当只推进一个有意义步骤，还是可以完整解释。

该观察器扩展 `2026-07-17-llm-led-nonblocking-verification-design.md`，不得破坏其
“观察失败不阻断主教学回答”的原则。

## 最小输出协议

观察器只返回三个字段：

- `starts_new_problem: bool`
- `pacing: single_move | complete_response`
- `reason: str`，仅用于研究界面诊断

观察器不判断数学正确性，不生成教学动作，不填写披露原因，也不输出改写后的题目。
当 `starts_new_problem=true` 时，Agent直接使用学生原始消息作为新的 active problem，
避免 LLM 改写或污染题目。

## 每轮数据流

1. Agent把当前 active problem、最近对话和学生消息交给观察器。
2. 观察器判断换题与节奏。
3. 若换题，Agent更新 `problem_context` 并清空 `previous_verified_step`。
4. 新题陈述不作为学生步骤判定；后续回复继续走现有 verifier。
5. Agent把观察结果作为提示交给主 LLM，主 LLM仍只返回自然语言。
6. 回答中的可解析方程继续由 `math_checker.find_equations` 和 verifier 审计。

## 非阻断降级

观察器的 JSON、枚举、网络或配置错误必须被捕获并记录。降级时：

- 如果当前没有 active problem，当前消息成为 active problem；
- 明确索要完整答案时使用 `complete_response`；
- 当前消息含可解析方程且像求解请求时使用 `single_move`；
- 其它情况使用 `complete_response`；
- 无论如何继续调用主 LLM并展示其回答。

## 关键场景

- 概率题之后输入“老师，3x-5=10怎么解？”：观察器判定新题和 `single_move`，
  active problem 切换到该方程，最终形式 `x=5` 可被 checker 识别并修正。
- “为什么两边要同时加5？”：沿用当前方程，使用 `complete_response`。
- “3的相反数是什么？”：新概念题，使用 `complete_response`；没有 verifier 仍正常输出。
- 观察器失败：主回答继续，详情区显示降级原因。

## 成本与风险

- 正常每轮两次 API 调用；硬冲突修正时最多三次。
- 观察器可能误判换题，因此 UI 必须展示 active problem 和观察理由，便于真人评测。
- 本阶段只在独立原型中试验，不迁移到主 Teacher Agent。
