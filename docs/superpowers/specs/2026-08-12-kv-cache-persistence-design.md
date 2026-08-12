# KV Cache Persistence Design

## Goal

为本地 `llama-server` 推理增加按“学生账号 + 对话”隔离的 KV cache 自动恢复与保存。模型或页面重启后，教师正式回复可以尝试复用此前落盘状态；任何 KV 故障都不得导致聊天记录丢失或阻断正常回答。

## Scope

- 只接入 `prototypes/llm_led_teacher` 的完整教学回合。
- 保留项目已有的模型地址、端口、数据库地址和其他启动参数；KV 功能复用这些已有值，只新增必要的 KV 配置，不迁移或替换现有地址。
- 聊天历史仍由 `data/runtime/conversations.sqlite3` 保存，并且始终是权威数据源。
- KV 文件只作为可删除、可重建的性能缓存。
- Observer、分类、RAG、审计等内部阶段可以使用被分配的 slot，但只有最终面向学生的教师状态会落盘。
- 不改动教学规则、数学 verifier、RAG 决策或账号模型。
- 本版本不提供 Git 自动回退界面。

## Chosen Approach

采用“版本化 KV 文件 + 确定性 slot 映射 + slot 级锁 + 自动降级”。相比直接覆盖固定 `.bin`，版本化保存不会在一次中断写入后立即破坏最后一个可用版本；相比只依赖 llama-server 内存 cache，它能够跨模型进程重启恢复。

每个缓存身份使用：

```text
cache_key = SHA-256(user_id + NUL + conversation_id)
slot_id   = cache_key modulo configured_parallel
filename  = kv_<cache_key前32位>_<time_ns>.bin
```

文件名不直接暴露用户名、学生 ID 或对话 ID。每个身份保留最近两个成功版本，恢复时从新到旧尝试。

## Components

### `app/llm/kv_cache.py`

独立负责本地 llama-server 的 KV 生命周期：

- 从环境变量读取启用状态、server URL、保存目录、slot 数和 HTTP 超时。
- 校验仅连接本地/明确配置的 OpenAI-compatible llama-server。
- 为 `(user_id, conversation_id)` 计算稳定 cache key 与 slot。
- 使用进程内线程锁和 `fcntl.flock` 文件锁保护同一 slot 的完整教学回合。
- 调用 `POST /slots/{id_slot}?action=restore|save`。
- 恢复失败时尝试上一版本；全部失败时返回降级状态。
- 保存成功后清理该身份更旧的版本，只保留两个。
- 暴露结构化状态，包括 slot、恢复结果、保存结果、文件名、token/byte 数和安全错误摘要。

### `app/llm/deepseek_client.py`

继续作为唯一模型出口。新增临时 slot 上下文，使 `chat()` 和 `chat_json()` 在一个教学回合内自动向 OpenAI 请求加入：

```python
extra_body={"id_slot": slot_id, "cache_prompt": True}
```

上下文退出后恢复原状态，避免影响其他调用。未启用 KV 时请求格式保持不变。

### `prototypes/llm_led_teacher/logic.py`

`run_turn()` 接收可选缓存身份。启用时执行：

1. 获取该身份对应 slot 的独占 lease。
2. Observer、分类和其他前置模型调用固定到这个 slot，但不保存。
3. 在教师初稿生成前恢复该对话最新 KV 文件，覆盖前置内部调用留下的临时 slot 状态。
4. 生成教师初稿并执行确定性审计；如需 repair，则 repair 结果成为最终 slot 状态。
5. 最终回复成功后保存一个新版本 KV 文件。
6. 将 KV 状态合并进 `model_generation_metadata`。
7. 释放 slot lease。

若回合异常，不保存半成品 KV；原有异常处理保持不变。

### Streamlit UI

调用 `run_turn()` 时传入已登录用户的 `user_id` 和当前 `conversation_id`。诊断详情中显示：

- KV 是否启用；
- 使用的 slot；
- 是否成功恢复；
- 是否成功保存；
- 当前 `.bin` 文件名与本地目录；
- 降级原因（如有）。

UI 不读取或渲染 `.bin` 的二进制内容，因为它不是可读文本，也不能显示成对话。

### Launcher and model script

`run_deepseek_v4_8x4080.sh` 在启动前创建 KV 目录，并向 llama-server 加入：

```bash
--slot-save-path "$KV_SLOT_SAVE_PATH"
```

一键启动器把同一绝对目录、模型地址和 `--parallel` 数量传给 Streamlit：

```text
TEACHER_KV_PERSIST=1
TEACHER_KV_CACHE_DIR=<absolute path>
TEACHER_KV_SERVER_URL=http://127.0.0.1:8080
TEACHER_LLAMA_PARALLEL=<parallel>
```

这里的 server URL 从启动器当前已有的 `--model-host` 与 `--model-port` 派生；默认值继续是项目原有的 `127.0.0.1:8080`，用户传入其他已有地址时原样复用，不新增固定端口或改写现有连接地址。

设置 `TEACHER_KV_PERSIST=0` 时，Agent 完全跳过 slot API 和请求附加字段，聊天继续正常工作。

## Data Flow

```text
登录并选择对话
  -> user_id + conversation_id
  -> 计算 slot 并加锁
  -> 内部调用固定使用该 slot（不落盘）
  -> 恢复该对话最新可用 .bin
  -> 最终教师生成/必要的 repair
  -> llama-server 保存新 .bin
  -> 保存完整聊天到 conversations.sqlite3
  -> UI 显示 KV 状态
```

对话切换不需要手工保存或恢复：下一次发送消息时，系统按新的 `(user_id, conversation_id)` 自动恢复对应文件。同一个用户的不同对话不会共享 cache key；不同用户也不会共享。

## Concurrency and `--parallel`

`--parallel 1` 使用 slot 0，多个会话按 slot 锁串行执行。`--parallel 4` 使用 slot 0 到 3，cache key 稳定映射到其中一个 slot；落在不同 slot 的回合可以并行，碰撞到同一 slot 的回合串行，并在教师生成前恢复自己的文件，因此不会把另一个学生的 KV 当成自己的状态。

锁覆盖整个教学回合，是因为 Observer 等内部调用也会改变 llama-server 的 slot。只锁 `restore -> save` 而放任其他请求自动选 slot，会存在保存错状态的竞态。

## Failure Handling

- 没有缓存文件：标记 `miss`，直接正常生成并在结束后创建第一份文件。
- restore 失败：尝试上一版本；仍失败则标记 `fallback`，清空/重算当前 prompt，不中断回答。
- save 失败：保留旧版本，返回本轮回答并在 UI 显示警告。
- 文件损坏或与 GGUF、llama.cpp、上下文/KV 配置不兼容：restore 会失败，系统从 SQLite 历史重算并生成新版本。
- server 没有以 `--slot-save-path` 启动：视为 KV 不可用，不影响教学。
- KV 目录不可访问：禁用本轮 KV，不修改聊天数据库。

错误信息不得包含 API key、密码或完整 prompt。

## Cache-hit Limitation

恢复成功表示 llama-server 已把保存的 slot 状态载入内存，不等于下一次 prompt 的全部 token 都命中。llama-server 按“新 prompt 与已缓存 token 的最长严格相同前缀”复用。

当前教师 prompt 每轮重新构造 JSON，`current_turn_id` 和历史结构会变化，所以第一版可能主要命中稳定 system prompt。该版本先验证保存、重启恢复和隔离正确性；重排消息格式以提高前缀命中率属于后续独立优化，避免本次同时改变教学提示词行为。

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `TEACHER_KV_PERSIST` | `0`（普通启动）/ 一键启动器设为 `1` | 是否启用 KV 持久化 |
| `TEACHER_KV_CACHE_DIR` | `data/runtime/kv-states` | Agent 与 llama-server 共享的本地目录 |
| `TEACHER_KV_SERVER_URL` | 从 `DEEPSEEK_BASE_URL` 去掉 `/v1` 推导 | slot API 根地址 |
| `TEACHER_LLAMA_PARALLEL` | `1` | 可用 slot 数量 |
| `TEACHER_KV_HTTP_TIMEOUT_SECONDS` | `30` | save/restore HTTP 超时 |
| `TEACHER_KV_KEEP_VERSIONS` | `2` | 每个对话保留版本数 |

## Tests

- cache key 不泄露身份且相同身份稳定映射到相同 slot。
- 不同对话独立选择/扫描版本文件。
- restore 从新到旧回退，save 成功后才清理旧版本。
- HTTP 请求路径、JSON body 和禁用模式正确。
- `DeepSeekClient` 仅在 slot 上下文中发送 `id_slot` 与 `cache_prompt`。
- `run_turn()` 保证内部调用之后、教师生成之前 restore，最终回复之后 save；异常时不 save。
- launcher 正确传递目录、URL 和 parallel，模型脚本包含 `--slot-save-path`。
- UI 把登录用户和当前对话作为缓存身份传入，并显示状态。
- 完整运行 `python -m pytest` 和 `python scripts/run_eval.py`，确认教学行为无回归。

## Acceptance Criteria

1. 第一次对话完成后，KV 目录出现该对话的 `.bin` 文件。
2. 重启 llama-server 和页面后，继续同一对话会先 restore 再生成，并在 UI 显示恢复成功。
3. 切换学生或对话时使用不同缓存身份，不会串用上下文。
4. `--parallel 1` 与 `--parallel 4` 都能正确运行。
5. 删除全部 `.bin`、关闭开关或 restore 失败时，聊天历史和正常生成仍可用。
6. 现有 pytest 与离线 evaluation 全部通过。
