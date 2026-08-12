# KV Cache Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic per-user, per-conversation llama-server KV slot restore and versioned save to the authenticated LLM-led Teacher prototype.

**Architecture:** `app.llm.kv_cache` owns identity hashing, deterministic slot assignment, version discovery, locking, and llama-server slot HTTP calls. `DeepSeekClient` receives slot routing through a temporary `ContextVar`; `run_turn` holds one lease, restores immediately before teacher generation, and saves after the final draft or repair. SQLite history remains authoritative and KV errors degrade to normal prompt recomputation.

**Tech Stack:** Python 3.10+, stdlib urllib/hashlib/threading/fcntl, OpenAI-compatible client, Streamlit, pytest, Bash.

## Global Constraints

- Preserve all existing model addresses, ports, database addresses, and unrelated launch parameters.
- Keep all inference calls routed through `app/llm/deepseek_client.py`.
- Persist only final teacher state, never an internal Observer/classification state.
- Treat KV files as disposable acceleration data; SQLite remains authoritative.
- KV failures must not block replies or history saves.
- Support parallel 1 and parallel 4 with deterministic mapping and slot locks.
- Run `python -m pytest` and `python scripts/run_eval.py` before completion.

---

### Task 1: Slot-aware DeepSeek requests

**Files:**
- Modify: `app/llm/deepseek_client.py`
- Modify: `tests/test_deepseek_integration.py`

**Interfaces:**
- Produces `llama_slot(slot_id: int | None)`.
- `chat` and `chat_json` add `extra_body={"id_slot": N, "cache_prompt": True}` only within that context.

- [ ] **Step 1: Write failing request-shape tests**

```python
with llama_slot(3):
    client.chat("system", "user")
client.chat("system", "next")
assert requests[0]["extra_body"] == {"id_slot": 3, "cache_prompt": True}
assert "extra_body" not in requests[1]
```

Add the equivalent JSON-mode assertion and a negative-slot validation test.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_deepseek_integration.py -k llama_slot -v`

Expected: import failure because `llama_slot` is absent.

- [ ] **Step 3: Implement minimal ContextVar routing**

Add a `ContextVar`, a context manager that always resets its token, and conditional request-dictionary updates in both methods. Leave ordinary request bodies unchanged.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_deepseek_integration.py -k "llama_slot or records_usage" -v`

- [ ] **Step 5: Commit**

```bash
git add app/llm/deepseek_client.py tests/test_deepseek_integration.py
git commit -m "feat: route local LLM calls to a slot"
```

### Task 2: Versioned KV manager

**Files:**
- Create: `app/llm/kv_cache.py`
- Create: `tests/test_kv_cache.py`

**Interfaces:**
- Produces `KvCacheIdentity(user_id, conversation_id)`.
- Produces `KvCacheSettings.from_env()`.
- Produces `KvCacheManager.lease(identity)` and lease methods `restore`, `save`, `status`.
- Produces `KvTurnStatus.to_metadata()`.

- [ ] **Step 1: Write failing identity/settings tests**

Assert SHA-256 identity stability and secrecy, valid slots for parallel 1 and 4, disabled default, invalid-value rejection, and server URL derivation that only removes the existing `/v1` suffix.

```python
identity = KvCacheIdentity("usr_secret", "conv_secret")
assert manager.slot_for(identity) in range(4)
assert "usr_secret" not in manager.cache_prefix(identity)
```

- [ ] **Step 2: Verify identity tests RED**

Run: `python -m pytest tests/test_kv_cache.py -k "identity or settings or slot" -v`

- [ ] **Step 3: Implement identity, settings, and locking**

Hash `user_id + NUL + conversation_id`. Reject blanks, parallel below 1, non-positive timeout, and retention below 1. Hold a process `threading.Lock` plus `<cache_dir>/.slot-N.lock` using `fcntl.flock` for the full lease.

- [ ] **Step 4: Write failing lifecycle tests**

Assert these exact calls and JSON bodies:

```text
POST {existing_url}/slots/2?action=restore {"filename":"kv_hash_version.bin"}
POST {existing_url}/slots/2?action=erase   {}
POST {existing_url}/slots/2?action=save    {"filename":"kv_hash_new.bin"}
```

Cover newest-to-oldest restore fallback, erase on miss/failure, no deletion before successful save, metadata extraction, and retention of two versions.

- [ ] **Step 5: Verify lifecycle tests RED**

Run: `python -m pytest tests/test_kv_cache.py -k "restore or save or erase or version" -v`

- [ ] **Step 6: Implement safe lifecycle calls**

Use proxy-free stdlib HTTP, basename-only filenames, bounded safe errors, and `time.time_ns()` version names. Catch network/JSON/filesystem/server failures into status. Prune old files only after a successful save.

- [ ] **Step 7: Verify GREEN and commit**

Run: `python -m pytest tests/test_kv_cache.py -v`

```bash
git add app/llm/kv_cache.py tests/test_kv_cache.py
git commit -m "feat: manage versioned llama KV states"
```

### Task 3: Teaching lifecycle integration

**Files:**
- Modify: `prototypes/llm_led_teacher/logic.py`
- Modify: `tests/prototypes/test_llm_led_turn_contract.py`

**Interfaces:**
- Extends `run_turn` with optional `kv_identity` and `kv_manager`.
- Adds `kv_persistence_*` keys to model-generation metadata.

- [ ] **Step 1: Write failing order tests**

With fake lease/client events, assert:

```text
lease-enter -> internal calls -> restore -> teacher draft -> optional repair -> save -> lease-exit
```

Assert no identity means no manager activity, generation exceptions do not save, and restore/save failure still returns a reply.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/prototypes/test_llm_led_turn_contract.py -k kv_persistence -v`

- [ ] **Step 3: Implement wrapper and hooks**

Wrap the existing body with lease acquisition and `llama_slot`. Call restore immediately before `_chat_teaching_turn`; call save after the optional repair. Merge status metadata after usage aggregation so it is retained.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/prototypes/test_llm_led_turn_contract.py -v`

```bash
git add prototypes/llm_led_teacher/logic.py tests/prototypes/test_llm_led_turn_contract.py
git commit -m "feat: persist final teacher KV state"
```

### Task 4: Additive launcher configuration

**Files:**
- Modify: `../run_deepseek_v4_8x4080.sh`
- Modify: `scripts/run_llm_led_teacher.py`
- Modify: `.env.example`
- Modify: `tests/prototypes/test_llm_led_launcher.py`

**Interfaces:**
- Keeps existing host, port, parallel, DB, model, context, batch, and GPU defaults.
- Adds `KV_SLOT_SAVE_PATH` to model environment.
- Adds `TEACHER_KV_PERSIST`, `TEACHER_KV_CACHE_DIR`, `TEACHER_KV_SERVER_URL`, and `TEACHER_LLAMA_PARALLEL` to service environment.

- [ ] **Step 1: Write failing preservation tests**

```python
env = build_service_env(base, model_host="10.0.0.8", model_port=9099,
                        kv_cache_dir=tmp_path, parallel=4, ...)
assert env["DEEPSEEK_BASE_URL"] == "http://10.0.0.8:9099/v1"
assert env["TEACHER_KV_SERVER_URL"] == "http://10.0.0.8:9099"
assert env["TEACHER_LLAMA_PARALLEL"] == "4"
```

Assert the model script retains `${PORT:-8080}` and adds directory creation plus `--slot-save-path`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/prototypes/test_llm_led_launcher.py -k "kv or service_environment or model_environment" -v`

- [ ] **Step 3: Implement additive settings**

Create `data/runtime/kv-states`, pass its absolute path with a trailing separator to llama-server, and derive the KV URL from the current model host/port. Document new variables without changing existing values.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/prototypes/test_llm_led_launcher.py -v`

```bash
git add scripts/run_llm_led_teacher.py .env.example tests/prototypes/test_llm_led_launcher.py ../run_deepseek_v4_8x4080.sh
git commit -m "feat: enable llama slot state directory"
```

### Task 5: Authenticated UI wiring and diagnostics

**Files:**
- Modify: `prototypes/llm_led_teacher/streamlit_app.py`
- Modify: `tests/prototypes/test_llm_led_auth_ui.py`
- Modify: `tests/prototypes/test_llm_led_streaming.py`

**Interfaces:**
- Constructs identity from internal `authenticated_user["user_id"]` and `prototype_conversation_id`.
- Displays slot, restore/save outcome, basename/directory, and safe fallback warning.

- [ ] **Step 1: Write failing UI tests**

Assert the one `run_turn` site receives both internal IDs, not username/title. Assert rendered/source details include KV persistence, slot, restored/saved state, and safe fallback.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/prototypes/test_llm_led_auth_ui.py tests/prototypes/test_llm_led_streaming.py -k kv -v`

- [ ] **Step 3: Wire and render**

Extend the existing token/KV details. Display filename and directory but never parse the binary file or expose prompt/error internals.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/prototypes/test_llm_led_auth_ui.py tests/prototypes/test_llm_led_streaming.py -v`

```bash
git add prototypes/llm_led_teacher/streamlit_app.py tests/prototypes/test_llm_led_auth_ui.py tests/prototypes/test_llm_led_streaming.py
git commit -m "feat: show conversation KV persistence status"
```

### Task 6: Full verification

**Files:** Modify only if verification reveals an implementation defect.

- [ ] **Step 1: Run all tests**

Run: `python -m pytest`

- [ ] **Step 2: Run required offline evaluation**

Run: `python scripts/run_eval.py`

- [ ] **Step 3: Validate script and configuration**

Run: `bash -n ../run_deepseek_v4_8x4080.sh`

Run: `rg -n -- "--slot-save-path|TEACHER_KV_|KV_SLOT_SAVE_PATH" ../run_deepseek_v4_8x4080.sh scripts/run_llm_led_teacher.py .env.example`

- [ ] **Step 4: Check data safety and address preservation**

Run: `git status --short` and `git diff --check`. Confirm no `.env`, database, logs, keys, or `.bin` files are tracked and existing host/port defaults are unchanged.

- [ ] **Step 5: Report direct operation**

Keep the existing command:

```bash
cd /home/dockerContainerSave/data/wyd-workspace/teacher-agent-demo
python scripts/run_llm_led_teacher.py
```

After the first successful reply, `.bin` files should appear under `data/runtime/kv-states/`. Continuing the same conversation after restart should show restore success.
