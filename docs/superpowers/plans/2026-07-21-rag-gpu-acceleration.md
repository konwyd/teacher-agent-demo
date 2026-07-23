# RAG GPU Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all bge-m3 inference used by the original RAG strategy to one configurable GPU with safe CPU fallback and startup warmup.

**Architecture:** Add a small device-selection module shared by local and online RAG, pass its result into both SentenceTransformer constructions, and invoke a non-network warmup from the one-command launcher before Streamlit starts. Retrieval ranking, thresholds, Tavily triggering, and prompt context remain unchanged.

**Tech Stack:** Python 3.10, PyTorch 2.4, sentence-transformers, ChromaDB, pytest, Streamlit.

## Global Constraints

- Default device is `cuda:0` when CUDA is available, otherwise `cpu`.
- `TEACHER_RAG_DEVICE` can explicitly select a device.
- CUDA initialization failure falls back to CPU without blocking teaching.
- Preserve BM25, bge-m3, RRF, top-k, 0.65 online threshold, 0.44 final filtering, and Tavily behavior.
- Warmup must not call Tavily or a paid LLM API.
- Do not print API keys.
- Do not commit, push, merge, or switch branches.

---

### Task 1: Shared device selection

**Files:**
- Create: `llm-test/rag_device.py`
- Test: `tests/prototypes/test_rag_gpu.py`

**Interfaces:**
- Produces: `resolve_rag_device(requested: str | None = None) -> str`
- Produces: `configure_cpu_threads() -> int`

- [ ] Write failing tests for explicit CPU, default CUDA, unavailable-CUDA fallback, and thread configuration.
- [ ] Run `python3 -m pytest tests/prototypes/test_rag_gpu.py -q` and confirm RED.
- [ ] Implement environment-driven device resolution without loading a model.
- [ ] Run the target tests and confirm GREEN.

### Task 2: Local and online bge-m3 use one device

**Files:**
- Modify: `llm-test/rag_retriever.py`
- Modify: `llm-test/online_rag.py`
- Test: `tests/prototypes/test_rag_gpu.py`

**Interfaces:**
- Consumes: `resolve_rag_device()` from Task 1.

- [ ] Add failing tests that capture the `device` passed to Chroma SentenceTransformer embedding functions and online SentenceTransformer.
- [ ] Confirm tests fail because the current code uses the CPU default/hard-coded CPU.
- [ ] Pass the shared device into TextbookRAG, ExerciseRAG, and OnlineRAG model initialization.
- [ ] Add guarded model-initialization fallback to CPU and verify tests pass.

### Task 3: Non-network startup warmup

**Files:**
- Create: `scripts/warmup_rag.py`
- Modify: `scripts/run_llm_led_teacher.py`
- Test: `tests/prototypes/test_rag_gpu.py`

**Interfaces:**
- Produces: `warmup_local_rag(query: str = "一元一次方程") -> dict[str, object]`.
- Streamlit executes the cached warmup in its own process; the launcher supplies the same device/thread environment.

- [ ] Write failing tests proving warmup initializes local retrievers, never calls Tavily, reports elapsed/device, and failures do not block launcher startup.
- [ ] Confirm RED.
- [ ] Implement cached same-process warmup and invoke it during Streamlit startup.
- [ ] Configure conservative CPU thread environment defaults in the launcher.
- [ ] Confirm GREEN.

### Task 4: Verification and benchmark

**Files:**
- Modify only if tests expose a defect in the preceding task files.

- [ ] Run `python3 -m pytest tests/prototypes/test_rag_gpu.py -q`.
- [ ] Run `python3 -m pytest tests/prototypes/test_llm_led_turn_contract.py -q`.
- [ ] Run `python3 -m pytest -q`.
- [ ] Run `TEACHER_FORCE_MOCK=1 python3 scripts/run_eval.py`.
- [ ] Restart the one-command service with `TEACHER_RAG_DEVICE=cuda:0`.
- [ ] Benchmark identical local-RAG questions twice to report cold and warm latency, GPU memory, CPU usage, and selected device.
- [ ] Confirm ports 8002 and 8502 return HTTP 200.
