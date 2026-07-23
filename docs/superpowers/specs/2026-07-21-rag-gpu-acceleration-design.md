# RAG GPU 加速设计

日期：2026-07-21

## 目标

在不改变原项目检索策略、召回结果语义和联网阈值的前提下，将 bge-m3 的计算从 CPU 迁移到单张 NVIDIA GPU，降低实时网页请求延迟和 CPU 负载。

成功标准：

- 本地教材与题库仍使用原有 BM25 + bge-m3 + RRF。
- 仍使用 0.65 本地有效命中阈值、0.44 最终过滤阈值和原 Tavily 联网逻辑。
- `TEACHER_RAG_DEVICE=cuda:0` 时，本地和联网 bge-m3 均使用 GPU 0。
- CUDA 不可用或设备初始化失败时自动回退 CPU，教师回答不被阻断。
- 启动阶段完成模型预热，首位用户不承担模型冷启动。
- GPU 热启动后的本地 RAG 基准明显优于当前 11–18 秒，并记录实测结果。

## 设计

### 统一设备配置

新增一个由环境变量读取的统一设备值：

```text
TEACHER_RAG_DEVICE=cuda:0
```

原项目 `llm-test/rag_retriever.py` 的 `TextbookRAG` 和 `ExerciseRAG` 在构造 Chroma `SentenceTransformerEmbeddingFunction` 时显式传入该设备。

`llm-test/online_rag.py` 的共享 bge-m3 重排序模型读取同一设备值。禁止本地检索和联网重排序各自硬编码不同设备。

### 自动回退

设备解析遵循：

1. 环境变量未设置时，CUDA 可用则默认 `cuda:0`，否则 `cpu`。
2. 显式设置 `cpu` 时始终使用 CPU。
3. 设置 CUDA 设备但 CUDA 不可用或模型初始化失败时，记录不含敏感信息的警告并回退 CPU。
4. 回退只改变计算设备，不改变检索、排序或过滤策略。

### 启动预热

Streamlit 在同一服务进程首次启动时执行一次带缓存的预热；launcher 只注入设备与线程环境：

- 初始化原项目本地教材和题库检索器；
- 使用轻量数学 query 完成一次 embedding；
- 初始化联网重排序模型但不调用 Tavily，不产生联网费用；
- 预热失败时输出清晰告警，仍允许 Streamlit 启动。

### CPU 约束

GPU 模式下限制 PyTorch/Tokenizer 的 CPU 线程数，避免单个 Streamlit 请求再次占满数十个核心。线程数通过环境变量配置并设置保守默认值，不影响 BM25/RRF 算法。

### 可观测性

启动日志显示：

- 选择的 RAG device；
- 是否发生 CPU 回退；
- 预热耗时；
- 不输出任何 API key。

## 测试

先写失败测试，覆盖：

1. `cuda:0` 配置传入本地两个 Chroma embedding function。
2. 联网重排序使用同一设备。
3. CUDA 不可用时回退 CPU。
4. 预热不调用 Tavily。
5. 预热失败不阻止服务启动。
6. 原 RAG 策略相关测试保持通过。

验证：

```bash
python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q
python3 -m pytest -q
TEACHER_FORCE_MOCK=1 python3 scripts/run_eval.py
```

另执行同一问题的 GPU 冷/热本地 RAG 基准，报告耗时与 CPU 占用变化。

## 非目标

- 不修改 BM25、RRF、top-k、相似度阈值或 Tavily 策略。
- 不做多 GPU 调度。
- 不并行化 observer、RAG 和 teacher。
- 不改 DeepSeek 模型或参数。
- 不公开 Chroma 端口。
