# LLM-led Teacher Prototype

This throwaway prototype answers one question: can DeepSeek lead a complete
multi-turn junior-math lesson while the Agent framework only preserves context,
verifies supported mathematical claims, audits disclosure, and records labels?

It deliberately bypasses TeachingPlanner, strategy capabilities, Pedagogical
RAG, and mastery-based control. State is kept only in Streamlit session memory.

Run from the worktree root:

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
streamlit run prototypes/llm_led_teacher/streamlit_app.py
```

No real API is called by tests. Starting the app and sending a message uses the
configured DeepSeek API and may incur cost.
