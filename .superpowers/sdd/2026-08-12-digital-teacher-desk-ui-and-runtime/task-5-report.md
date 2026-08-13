# Task 5 Report: One-Way Thinking Panel State Machine

## Status

Complete. The public thinking panel now stays expanded while generation is running, collapses once on successful completion, stays expanded on failure, and preserves a user's later expansion choice across reruns. Historical state is isolated by both conversation ID and turn ID. The saved answer renders in the successful generation run without a forced `st.rerun()`.

## TDD evidence

RED:

```text
python3 -m pytest tests/prototypes/test_thinking_state.py -q
ERROR tests/prototypes/test_thinking_state.py
ModuleNotFoundError: No module named 'prototypes.llm_led_teacher.thinking_state'
```

GREEN:

```text
python3 -m pytest tests/prototypes/test_thinking_state.py tests/prototypes/test_llm_led_streaming.py -q
20 passed in 0.39s

python3 -m pytest tests/prototypes/test_thinking_state.py tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py::test_new_public_thinking_completes_collapsed_without_forced_rerun -q
21 passed in 4.53s
```

The existing AppTest `test_new_public_thinking_stays_expanded_for_one_rerun_then_collapses` encoded the bug rejected by this specification. With the new implementation it failed in the first full run (`449 passed, 1 failed`) because no historical expander was intentionally manufactured by a success-path rerun. With root-agent authorization, that one obsolete test was migrated to assert the new behavior: the live status completes collapsed in the same run, and its keyed historical expander remains collapsed on the next ordinary rerun.

## Regression evidence

```text
python3 -m pytest
450 passed in 46.59s

python3 scripts/run_eval.py
评测模式：mock LLM（离线）
总 case 数：38  通过：38  失败：0
```

## Implementation notes

- Added immutable `ThinkingPanelState` transitions. Completion is idempotent, so later user choices are not overwritten by reruns.
- `thinking_key(conversation_id, turn_id)` prevents state from leaking between conversations or turns.
- `LiveThinkingPanel.append` writes public summaries and updates only the elapsed label; it never changes expansion.
- `LiveThinkingPanel.complete` updates to `备课完成 · {seconds}秒`, `state="complete"`, `expanded=False` once.
- Removed `_JUST_GENERATED_TURN_KEY`, its `pop`, and the unconditional success-path `st.rerun()`.
- Public summaries remain whitelisted; no private chain-of-thought field is rendered.

## Commit

`fix: make thinking panel transitions stable`

## Concerns

- The offline evaluation updates the tracked local Chroma SQLite file as a runtime side effect. That unrelated binary change is intentionally excluded from this commit.
- `tests/prototypes/test_llm_led_auth_ui.py` is an additional file beyond the original five-file implementation brief; root explicitly authorized migrating the one obsolete AppTest because its old assertion directly required the rerun flicker being fixed. No other test in that file was changed.

## Repair round 1/5

Review found that keyed `st.expander` parameters used by the implementation require Streamlit 1.55+, while the UI extra still allowed 1.35. No project lockfile or deployment dependency manifest exists; `llm-test/requirements.txt` belongs to a separate model-test utility and does not declare Streamlit. The authoritative UI extra is now pinned to `streamlit>=1.55`, with a source contract test preventing regression.

RED:

```text
python3 -m pytest tests/prototypes/test_llm_led_streaming.py::test_ui_extra_requires_streamlit_with_keyed_expander_support tests/prototypes/test_llm_led_auth_ui.py::test_new_public_thinking_completes_collapsed_without_forced_rerun -q
1 failed, 1 passed
FAILED test_ui_extra_requires_streamlit_with_keyed_expander_support
```

GREEN:

```text
python3 -m pytest tests/prototypes/test_thinking_state.py tests/prototypes/test_llm_led_streaming.py tests/prototypes/test_llm_led_auth_ui.py -q
30 passed in 15.10s

python3 -m pytest
451 passed in 51.39s

python3 scripts/run_eval.py
评测模式：mock LLM（离线）
总 case 数：38  通过：38  失败：0
```

The AppTest now proves the audited teacher answer renders in the original successful run before any subsequent `app.run()`. Streamlit AppTest exposes no click or `set_value` operation for expanders, so the test drives the real keyed widget session value (`public_thinking:{conversation_id}:turn:1:expanded`) and reruns the application through the same `render_public_thinking` synchronization path. It verifies the historical expander remains open across consecutive reruns and that both its public summary and the teacher answer remain rendered.

Repair concern: offline evaluation again touched the tracked local Chroma SQLite runtime artifact; it remains deliberately excluded from the repair commit.
