# LLM-led Teacher 安全流式显示设计

日期：2026-07-21

## 目标

改善当前网页提交问题后长时间没有可见反馈、随后一次性显示全部教师回答的体验，同时保持现有 LLM-led Teacher Prototype 的教学节奏、确定性审计和 RAG 策略不变。

成功标准：

- 用户提交消息后立即看到当前阶段，而不是页面静默等待。
- 页面依次反馈观察、检索、生成和审计阶段。
- 未经后置审计的教师初稿绝不显示给学生。
- 审计通过或修正完成后，学生可见的最终回答逐块显示。
- 历史消息在 Streamlit rerun 后仍一次性正常渲染，不重复播放动画。
- API、网络或 RAG 失败继续沿用现有降级和错误处理。

## 方案选择

采用“状态进度 + 审计后流式显示”，不采用原始模型 token 直出。

原因：`run_turn()` 必须先提取教师初稿中的方程并执行确定性审计；初稿可能包含不应出现的最终答案，甚至需要第三次 LLM 调用修正。若直接把模型 token 发给浏览器，审计发现问题时违规内容已经不可撤回地展示。

## 设计

### 安全边界

`run_turn()` 继续同步完成完整回合，并保持其返回值、episode 写入时机和审计逻辑不变。只有 `PrototypeTurnRecord.teacher_output.response` 中最终确认的文本可以进入学生可见区域。

不修改：

- observer 的请求和规范化；
- RAG 的 local/online 策略及阈值；
- 教师 system prompt；
- 方程提取、verifier、repair 条件；
- active problem 和历史记录语义。

### 阶段进度

为 `run_turn()` 增加可选的进度回调。核心逻辑只报告稳定的阶段标识，不依赖 Streamlit：

1. `observing`：观察器判断与 guidance 规范化；
2. `retrieving`：本地/联网 RAG；
3. `generating`：教师初稿；
4. `auditing`：方程提取和确定性审计；
5. `repairing`：仅在触发修正时出现；
6. `complete`：最终记录已安全写入 episode。

Streamlit 将这些标识映射为中文状态。回调异常不得影响教学回合。

### 最终回答流式显示

`run_turn()` 返回后，在当前 assistant chat message 中使用一个纯函数把最终文本切分成适合显示的短块，再通过 Streamlit placeholder/`write_stream` 逐块渲染。

切分必须满足：

- 拼接后与最终文本逐字符完全一致；
- 不改变 Markdown、LaTeX、标点和换行；
- 空文本不产生块；
- 测试不依赖真实时间休眠或真实 LLM。

这里的“流式”是安全展示流，不缩短后端首个最终 token 的生成时间；阶段进度负责让等待过程可见。历史回合直接使用 `st.markdown()`，只对本次新生成的最终回答播放一次。

### 错误处理

- 观察器失败：继续使用现有 fallback，并推进状态。
- RAG 失败：继续使用现有 `retrieval_error` 降级。
- 教师调用失败：状态标记失败并显示现有 `DeepSeekError`。
- 进度 UI 自身失败：不得改变 `run_turn()` 的业务结果。

## TDD 与验证

先增加失败测试，至少覆盖：

1. 回调按正常路径报告阶段，且 `complete` 最后出现。
2. 触发 repair 时报告 `repairing`。
3. 回调抛异常不阻断回答。
4. 最终文本分块后可无损重组，包含中文、Markdown 和 LaTeX。
5. 流式展示使用的是审计后的修正版，不是初稿。

验证命令：

```bash
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
python3 -m pytest -q
TEACHER_FORCE_MOCK=1 python3 scripts/run_eval.py
```

不调用真实付费 API。

## 非目标

- 不将未经审计的 DeepSeek token 直接展示。
- 不改变模型、temperature、base URL 或 RAG 检索策略。
- 不重构 LLM-led Teacher Prototype。
- 不修改旧 Teacher Agent 的 8501 页面。
