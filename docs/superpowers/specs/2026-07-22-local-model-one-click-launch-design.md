# 本地 DeepSeek-V4 一键启动整合设计

日期：2026-07-22

## 目标

让 `python3 scripts/run_llm_led_teacher.py` 在一次命令中启动或复用本地
DeepSeek-V4 llama.cpp 服务、Chroma Server 和 LLM-led Teacher 网页，并把
observer、教师生成及修正请求全部指向本地模型。

本地模型不可用时整套服务必须停止，不得回退到 DeepSeek 付费 API。

## 范围

- 扩展现有 `scripts/run_llm_led_teacher.py`；
- 复用 `/home/dockerContainerSave/data/wyd-workspace/run_deepseek_v4_8x4080.sh`；
- 为 `app/llm/deepseek_client.py` 增加环境变量驱动的请求超时；
- 增加纯离线启动器和客户端配置测试；
- 更新根 README 的一键启动与配置说明。

不修改教学策略、RAG 策略、模型权重、llama.cpp 源码或网页业务流程。

## 默认运行配置

| 配置 | 默认值 |
|---|---|
| 模型地址 | `127.0.0.1:8080` |
| 模型别名 | `deepseek-v4-flash` |
| 总上下文 | `131072` |
| 并发槽 | `1` |
| batch / ubatch | `128 / 128` |
| K cache | `q8_0` |
| V cache | `q4_0` |
| 模型健康检查等待 | `900` 秒 |
| 教师 LLM 请求超时 | `300` 秒 |
| RAG 设备 | `cuda:7` |
| Chroma | `127.0.0.1:8002` |
| Streamlit | `0.0.0.0:8502` |

这些值可以通过启动参数或环境变量覆盖，但默认行为必须可以直接运行。

## 架构

Python 启动器继续负责子进程生命周期。模型命令不在 Python 中复制，而是调用
现有 shell 脚本，并通过环境变量传入 `PORT`、`CTX_SIZE`、`PARALLEL`、
`LLAMA_ARG_CACHE_TYPE_K` 和 `LLAMA_ARG_CACHE_TYPE_V`。

启动顺序：

1. 检查模型端口；
2. 已存在服务则验证 `/health` 和 `/v1/models` 中的模型别名后复用；
3. 没有服务则启动模型脚本并轮询健康状态；
4. 模型健康后启动或复用 Chroma；
5. 构造仅供 Streamlit 子进程使用的本地模型环境；
6. 预热 RAG 并启动 8502 页面。

## 本地模型环境

启动器显式覆盖 Streamlit 子进程中的以下变量：

- `DEEPSEEK_API_KEY=local`；
- `DEEPSEEK_BASE_URL=http://127.0.0.1:8080/v1`；
- `DEEPSEEK_MODEL=deepseek-v4-flash`；
- `DEEPSEEK_TIMEOUT_SECONDS=300`；
- `TEACHER_RAG_DEVICE=cuda:7`，但尊重用户显式设置的值。

显式覆盖前三项可以防止仓库 `.env` 中旧的付费 API 配置优先进入当前网页。

## 健康检查与错误处理

- HTTP 检查使用 Python 标准库，不新增运行依赖；
- `/health` 必须返回 2xx；
- `/v1/models` 必须返回包含 `deepseek-v4-flash` 的模型列表；
- 端口已占用但检查失败时立即退出，不接管或杀死未知进程；
- 新启动模型在 900 秒内未健康、提前退出或返回错误模型名时，整套启动失败；
- Chroma 或 Streamlit 启动失败时，清理由本次启动器创建的所有子进程；
- 复用的外部模型或 Chroma 不属于启动器，退出时不得终止；
- 任何失败都不得恢复或使用原 DeepSeek API。

## 超时

`DeepSeekSettings` 新增 `deepseek_timeout_seconds: float = 300.0`。客户端构造器
未显式传入 `timeout` 时使用该配置；测试仍可传入更短的显式值。错误信息保留
model 和 base URL，但不记录 API key。

## 测试设计

所有测试使用 fake process、fake HTTP response 或 mock OpenAI client，不启动模型、
不占用 GPU、不访问网络、不调用付费 API。

必须覆盖：

1. 本地模型环境正确覆盖旧 API 配置；
2. 健康且别名正确的 8080 服务被复用；
3. 端口占用但不健康时 fail-closed；
4. 新模型健康后才继续启动 Chroma/Streamlit；
5. 模型超时或提前退出时清理已拥有进程并返回非零；
6. 外部复用服务不会被清理；
7. 环境变量超时传入 OpenAI 客户端；
8. 显式构造器超时仍能覆盖环境默认值。

## 验收

- 目标测试先 RED 后 GREEN；
- `python -m pytest tests/prototypes/test_llm_led_launcher.py -q` 通过；
- DeepSeek 客户端目标测试通过；
- 完整 `python -m pytest` 和 `python scripts/run_eval.py` 通过；
- `--help` 展示本地模型端口、别名、健康等待和请求超时配置；
- 不运行真实付费 API。
