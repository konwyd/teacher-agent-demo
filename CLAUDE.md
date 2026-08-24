# CLAUDE.md
ARCHIVED DOCUMENT

This document describes an earlier project state and is not authoritative.
Do not use it as current runtime or architecture guidance.

Current sources of truth:
- AGENTS.md
- OWNERSHIP.md
- PROGRESS.md
- DECISIONS.md
## Project

This repository is a research demo for a Chinese K9/K12 Teacher Agent.

Claude Code is only the coding assistant for this repository.  
The runtime LLM used by the product is DeepSeek API, not Claude.

## Goal

Build a small research demo of a Teacher Agent that can guide students step by step.

The agent should support:

1. Diagnose student misunderstanding.
2. Retrieve curriculum knowledge from Graph RAG.
3. Give progressive hints instead of directly dumping answers.
4. Generate or select practice questions.
5. Grade student responses.
6. Update student mastery state.
7. Produce logs for research evaluation.

## MVP Scope

Focus on Chinese junior high school math first.

Do not build a general K12 product in the first version.

## Tech Stack

- Python
- FastAPI
- LangGraph
- DeepSeek API
- Neo4j
- Qdrant or pgvector
- PostgreSQL or SQLite
- Streamlit or simple web UI
- pytest

## Architecture Rules

Separate these layers clearly:

- `app/api`: FastAPI routes
- `app/agent`: LangGraph teaching workflow
- `app/llm`: DeepSeek client wrapper
- `app/retrieval`: vector retrieval and graph retrieval
- `app/memory`: student state and mastery tracking
- `app/evaluation`: benchmark and scoring scripts
- `app/data_pipeline`: dataset parsing and ingestion
- `configs`: model, database, and retrieval configs
- `tests`: unit and integration tests

## DeepSeek Rules

Use DeepSeek through OpenAI-compatible API.

Do not hard-code API keys.

All LLM calls must go through `app/llm/deepseek_client.py`.

For routing, grading, and state updates, prefer JSON output mode.

Always validate JSON with Pydantic models.

## Agent Rules

The Teacher Agent is a state machine, not a free-form chatbot.

Each turn should follow:

1. understand_input
2. diagnose_gap
3. retrieve_evidence
4. choose_teaching_action
5. generate_response
6. update_student_state

Allowed teaching actions:

- ask_diagnostic_question
- explain_concept
- give_hint_level_1
- give_hint_level_2
- show_worked_example
- give_practice_question
- grade_and_feedback
- schedule_review

## RAG Rules

Do not put all data into one vector index.

Use graph retrieval for:

- prerequisite relations
- concept dependencies
- chapter/section structure
- learning path planning

Use vector retrieval for:

- textbook explanations
- worked examples
- similar exercises
- misconception descriptions

## Testing Rules

Before considering a feature complete, run:

```bash
pytest
