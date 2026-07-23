# Local DeepSeek-V4 One-Click Launch Implementation Plan

**Goal:** Make the existing launcher manage a fail-closed local llama.cpp model,
Chroma, and the 8502 Teacher page while using a configurable local request timeout.

**Architecture:** Keep model command details in the existing 8-GPU shell script.
Add small testable launcher functions for environment construction, health checks,
reuse, waiting, and cleanup. Point only the Streamlit child process at the local
OpenAI-compatible endpoint.

**Tech Stack:** Python standard library, subprocess, urllib, pytest, OpenAI client.

## Global Constraints

- Never call a real paid API in tests.
- Never start the real GGUF model in tests.
- Fail closed when the local model is unhealthy.
- Reused external processes must not be terminated.
- Preserve all unrelated uncommitted work.

---

### Task 1: Launcher contract tests

**Files:**
- Create: `tests/prototypes/test_llm_led_launcher.py`
- Modify: `scripts/run_llm_led_teacher.py`

- [ ] Write tests for local API environment override, model process environment,
  healthy reuse, occupied-unhealthy failure, successful readiness wait, timeout
  cleanup, and external-process ownership.
- [ ] Run `python -m pytest tests/prototypes/test_llm_led_launcher.py -q`.
  Expected: RED because the model launcher interfaces do not exist.
- [ ] Add `ModelStartupError`, `build_model_env`,
  `model_server_ready`, and `ensure_model_server`.
- [ ] Add CLI defaults for port 8080, alias `deepseek-v4-flash`, 524288 context,
  900-second startup wait, and 300-second request timeout.
- [ ] Integrate model readiness before Chroma and Streamlit.
- [ ] Re-run the target test. Expected: GREEN.

### Task 2: Client timeout contract

**Files:**
- Modify: `tests/test_deepseek_integration.py`
- Modify: `app/llm/deepseek_client.py`

- [ ] Add tests proving `DEEPSEEK_TIMEOUT_SECONDS` reaches the OpenAI client and
  an explicit constructor timeout overrides settings.
- [ ] Run the two tests. Expected: RED because settings lack this field.
- [ ] Add `deepseek_timeout_seconds` and make constructor timeout optional.
- [ ] Re-run the tests. Expected: GREEN.

### Task 3: Documentation and verification

**Files:**
- Modify: `README.md`

- [ ] Document that the one-click command starts the local model first, list
  health/timeout overrides, and remove wording that says the API is still default.
- [ ] Run `python -m pytest tests/prototypes/test_llm_led_launcher.py -q`.
- [ ] Run the DeepSeek integration target tests.
- [ ] Run `python -m pytest`.
- [ ] Run `python scripts/run_eval.py`.
- [ ] Run `python3 scripts/run_llm_led_teacher.py --help`.
- [ ] Run `git diff --check` and report modified files/status.
