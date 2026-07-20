# LLM-led Teacher Prototype

This throwaway prototype answers one question: can DeepSeek lead a complete
multi-turn junior-math lesson while the Agent framework only preserves context
and verifies the mathematical claims it can deterministically support?

It deliberately bypasses TeachingPlanner, strategy capabilities, Pedagogical
RAG, and mastery-based control. State is kept only in Streamlit session memory.
For multi-step problems, the LLM still chooses the teaching move, but one normal
turn advances only one meaningful step unless the student explicitly requests a
full solution or prior turns provide evidence of sustained difficulty.

The primary LLM call returns student-facing plain text, not research metadata.
The Agent extracts parseable equations with the existing `math_checker` and
audits those equations. Unsupported prose claims remain `unverified` and are
still shown to the student; missing verification never blocks the response.

An experimental non-blocking LLM observer runs before teaching to detect a new
active problem and choose `single_move` or `complete_response` pacing. Normal
turns therefore use two API calls; a hard-audit repair may use a third. Observer
schema or transport failures fall back locally and never block the plain-text
teacher response.

Run from the worktree root:

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
streamlit run prototypes/llm_led_teacher/streamlit_app.py
```

No real API is called by tests. Starting the app and sending a message uses the
configured DeepSeek API and may incur cost.
