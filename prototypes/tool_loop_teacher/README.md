# Tool-loop Teacher Prototype

这是一个与现有运行路径完全隔离的技术原型，用来验证：固定教学 pipeline
能否改成“模型按需选择 Skill 和 Tool，再由确定性安全层收口”的编排方式。

`app/`、API 路由、现有 workflow、配置和正式测试均未修改，也不会自动导入本目录。

## 架构

```text
原题 + 学生消息
  │
  ├─ 强制 math_checker verifier（不可绕过）
  ├─ SafetyGate 计算允许的教学动作
  │
  └─ bounded Agent Loop
       ├─ load_skill（按需）
       ├─ inspect_student_context（按需）
       ├─ retrieve_pedagogical_context（按需）
       ├─ reason_about_problem（按需）
       └─ finish: 提出 TeachingActionPlan
              │
              ├─ TeachingPlanner 再验证计划
              ├─ TeachingBrief 限定可披露内容
              └─ ResponseGuard / 安全模板生成回复
```

当前 `DeepSeekClient` 没有暴露原生 `tool_calls` 接口，因此原型使用其
`chat_json()` 实现等价的结构化动作协议：每轮返回 `call_tool` 或 `finish`。
所有真实 LLM 调用仍经过 `app/llm/deepseek_client.py`。

本地 DeepSeek-V4 会先生成 `reasoning_content`，动作调用默认预留 1200 个输出
token，避免思考占满预算后没有最终 JSON；可用 `--decision-max-tokens` 调整。

## 已接入的 Tools

- `verify_math_attempt`：复用 `LinearEquationVerifier`，底层必须经过现有
  Fraction `math_checker`；支持的题型会在 loop 前强制执行。
- `inspect_student_context`：读取当前题目范围内的学生证据。
- `retrieve_pedagogical_context`：读取现有只读 Graph Context Provider。
- `reason_about_problem`：调用现有通用数学 reasoner，提供候选下一步。
- `load_skill`：加载本目录 `skills/` 中的教学方法说明。

## 运行离线测试

从仓库根目录执行：

```bash
python -m pytest prototypes/tool_loop_teacher/tests -q
```

测试使用脚本化 action provider，不调用真实 API。

## 运行真实 DeepSeek 演示

配置 `DEEPSEEK_API_KEY` 后，从仓库根目录执行：

```bash
python -m prototypes.tool_loop_teacher.demo \
  --problem "解方程 2x+3=7" \
  --message "我不会，下一步怎么做？" \
  --show-trace
```

如果机器设置了 HTTP/HTTPS 代理，直连本地 llama.cpp 时还应设置
`NO_PROXY=127.0.0.1,localhost`（及小写 `no_proxy`），否则 OpenAI/httpx 客户端
可能把 `127.0.0.1:8080` 请求送进代理并得到 502。

该命令会产生真实 API 调用和费用。正常情况下每个工具决策占一次调用，最终回复
再占一次调用；`reason_about_problem` 被选择时还可能产生一次内部结构化推理调用。

### 本地真人体验页面

本地 llama.cpp 已在 `127.0.0.1:8080` 提供 `deepseek-v4-flash` 时，可启动：

```bash
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
streamlit run prototypes/tool_loop_teacher/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8503
```

页面强制使用本地模型地址，不会从 `.env` 回退到付费 endpoint。左侧可修改原题、
循环预算，并查看每轮实际加载的 Skill、调用的 Tool、验证结果和完整编排轨迹。

## 当前刻意未做

- 不接 FastAPI 或 Streamlit，不改变产品入口。
- 不写正式 episode store，不影响现有学生数据。
- 不实现新题自动切换、子 Agent、并行委派或 Skill 自动学习。
- 不让模型自由决定是否运行数学 checker。
- 不将 RAG、memory 或 Skill 提升为可以覆盖确定性规则的硬控制。

这些边界使原型只回答编排可行性问题；验证有效后，再决定是否增加实验性产品入口。
