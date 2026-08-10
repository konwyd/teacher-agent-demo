# LLM-led Teacher Agent

This is the primary autonomous teaching loop. DeepSeek chooses the teaching
move, explanation depth, and wording. The surrounding Agent shell keeps the
active problem stable, preserves recent conversation, verifies supported
mathematics, supplies optional bounded RAG evidence, and repairs responses only
when deterministic audits find a conflict.

## Turn flow

```text
observe active problem and soft pacing
  -> verify the student's supported mathematics
  -> retrieve or reuse optional RAG evidence
  -> let DeepSeek teach autonomously
  -> audit supported teacher claims
  -> repair only when required
  -> save the audited turn and source metadata
```

TeachingPlanner, StrategyRegistry, and mastery-based flow control remain outside
this loop. RAG is also not a planner: it cannot change the active problem,
override the verifier, determine correctness, or force the student through a
script.

## RAG boundaries

- RAG is disabled by default and failure never blocks teaching.
- New problems retrieve once; ordinary follow-ups reuse the active-problem
  cache.
- Queries preserve equations, LaTeX, fractions, roots, and mathematical
  symbols.
- Normal turns query reference collections only.
- Exercise/example retrieval is allowed only when the student explicitly asks
  for an example or practice problem, makes two consecutive verifier-confirmed
  wrong attempts, or explicitly says they completely do not understand.
- The teacher prompt receives bounded text plus source IDs and similarity.
  Streamlit displays only source metadata, not raw retrieved text.

## Installation

The standard offline tests do not need RAG packages. Install the UI and optional
RAG runtime explicitly:

```powershell
pip install -e ".[dev,ui,rag]"
Copy-Item .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env` or the launching shell. To enable semantic
retrieval, set `TEACHER_RAG_ENABLED=true` and choose one Chroma connection:

- `CHROMA_SERVER=localhost:8002` for a Chroma HTTP service; or
- clear `CHROMA_SERVER` and set `TEACHER_RAG_VECTORDB_PATH` for a local
  persistent database.

`TEACHER_RAG_EMBED_MODEL` accepts a sentence-transformers model ID or a local
model directory. Use `TEACHER_RAG_DEVICE=cpu` for predictable deployment, or a
validated `cuda:N` device when available. The default metadata filters match
the audited databases: `TEACHER_RAG_SUBJECT_FILTER=数学` for reference
collections and `TEACHER_RAG_EXERCISE_SUBJECT_FILTER=Math` for the exercise
collection.

## Run

```powershell
streamlit run prototypes/llm_led_teacher/streamlit_app.py
```

The UI shows the active problem, audited answer, verifier result, retrieval
status, source IDs, similarity, example permission, and stage timing. It does
not expose chain of thought, API keys, raw exceptions, or local paths.

## Verify

```powershell
python -m pytest
python scripts/run_eval.py
```

Both commands are offline by default. Starting the UI and sending a message
uses the configured DeepSeek endpoint and may incur cost.
