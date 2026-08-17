# Unified KV Dynamic Context MVP Design

**Date:** 2026-08-17

## Goal

Allow up to four concurrent generations to share one 524,288-token KV pool while preserving a logical 524,288-token ceiling for any single conversation when the other conversations are small or idle.

This MVP deliberately relies on llama.cpp's native unified KV, continuous batching, RAM prompt cache, and idle-slot eviction. It does not introduce a custom application-level Slot Broker.

## Current Problem

The current deployment starts llama-server with `--ctx-size 524288 --parallel 1`. Every conversation hashes to slot 0, so all turns run serially. The application persists per-conversation slot snapshots to SSD, but the model-facing history is limited to the latest 12 turns and the context meter reports the final internal model call rather than the conversation working set.

Changing only to fixed `--parallel 4` without unified KV would divide the total context into roughly 131K per slot. That would increase concurrency but prevent any conversation from reaching 512K.

## Approved Scope

The MVP will:

1. Start llama-server with four slots and unified KV.
2. Enable continuous batching, a 64 GiB RAM prompt cache, and idle-slot caching.
3. Preserve the 524,288-token logical context limit for each sequence while all active sequences share the same physical 524,288-token pool.
4. Send the complete stored conversation session to the teacher request instead of retaining only the latest 12 turns.
5. Report session/peak context occupancy instead of the last small internal audit request.
6. Limit SSD KV snapshots to 100 GiB with high-water cleanup.
7. Preserve SQLite conversation history regardless of KV eviction or restore failure.

The MVP will not add a custom Slot Broker, request priorities, preemption, multiple model pools, automatic semantic summarization, or a distributed cache.

## Runtime Configuration

The target llama-server configuration is:

```text
--ctx-size 524288
--parallel 4
--kv-unified
--cont-batching
--cache-ram 65536
--cache-idle-slots
```

With unified KV enabled, llama.cpp gives each sequence a logical `n_ctx_seq` equal to the total context while the sequences compete for the shared physical KV buffer. Up to four requests may decode concurrently, but their combined live KV demand must fit the physical pool.

If all active requests require more KV than the pool can provide, requests may wait or fail at the llama-server boundary. The MVP will validate this behavior under controlled load instead of implementing a separate admission controller.

## Storage Tiers

### GPU KV

Contains actively used sequence state. It is fastest and disappears whenever llama-server stops.

### RAM Prompt Cache

llama.cpp receives a 64 GiB cache budget. When unified KV needs room, idle slots can be serialized into the RAM prompt cache and cleared from the GPU buffer. RAM cache entries disappear when llama-server or the machine restarts.

### SSD KV Snapshots

Existing per-conversation slot snapshots remain under `data/runtime/kv-states/`, which is already located on the external 15 TB SSD. The cache has:

- a 100 GiB hard budget;
- cleanup beginning at 80 GiB;
- cleanup continuing until usage is at or below 70 GiB;
- one stable snapshot per conversation in steady state;
- temporary retention of the previous snapshot only while a replacement is being validated;
- oldest inactive and incompatible snapshots removed first.

Snapshot compatibility must include the model identity, llama.cpp/cache format, total context, parallel count, unified-KV mode, and relevant KV datatype settings. An incompatible snapshot is a cache miss, not a conversation failure.

### SQLite Conversation Archive

`data/runtime/conversations.sqlite3` remains the durable source of truth. KV loss affects latency only. If GPU, RAM, and SSD caches all miss, the application reconstructs the model request from the stored conversation.

## Conversation Context Semantics

The conversation archive keeps the complete episode. The teacher-facing history builder will no longer slice `episode.turns[-12:]`; it will include the complete session until the request approaches the configured safety boundary.

The first MVP will not silently summarize or discard history. It will reserve room for system instructions, the current user message, tool/RAG content, and output. A controlled context-full response is preferable to hidden truncation. The exact safety boundary will be chosen from load-test evidence; an initial operational target of about 500K tokens is acceptable but is not a hard-coded requirement of the design.

## Context Meter

The composer meter must describe the active conversation, not the final internal LLM call. Per turn, the application will retain the maximum actual context occupancy observed across classification, generation, audit, and repair calls. Because the teacher generation contains the complete conversation history, this peak represents the session working set more accurately than the current last-call value.

The UI will show at least:

```text
Current conversation context: X / 524.3K
Concurrent generations: N / 4
```

It must not label cumulative API token billing as context occupancy and must not sum each turn's prompt tokens, because repeated history would be double-counted.

## Failure Handling

- RAM cache miss: try the SSD conversation snapshot.
- SSD restore failure or incompatibility: erase the target slot and rebuild from SQLite text.
- SSD save failure: return the teacher response and retain the last known-good snapshot.
- SSD budget exceeded: remove incompatible files, superseded versions, then least-recently-used inactive snapshots.
- Unified KV capacity pressure: let llama-server queue or reject according to its native behavior; record the result during the MVP load test.
- CUDA OOM or server exit: the launcher stops the owned stack and reports failure; no automatic fallback to a paid external API.

## Verification

### Unit and integration tests

- Unified-KV flags and RAM-cache limits propagate from the launcher to llama-server.
- Runtime metadata reports a 524,288-token per-conversation ceiling with four unified slots.
- A conversation with more than 12 turns keeps its earliest turn in the model payload.
- Turn usage aggregation chooses the maximum context occupancy rather than the last internal call.
- SSD cleanup honors 100/80/70 GiB thresholds and never deletes active or last-known-good state during replacement.
- Cache restore failures fall back to text without losing the response.

### Live validation

Run staged tests with one, two, then four concurrent requests. Measure:

- startup and steady-state GPU memory;
- prompt and generation throughput;
- first-token and total latency;
- actual per-slot and total KV occupancy;
- idle-slot RAM cache save/restore;
- SSD save/restore latency;
- behavior when combined live contexts approach 524,288 tokens;
- CUDA OOM, fragmentation, and process stability.

The deployment proceeds only if four-way concurrency remains stable on the current 8×RTX 4080 host. If it does not, the runtime falls back to two unified slots while keeping the rest of the design unchanged.

## Rollout

1. Add configuration plumbing and tests without changing the running service.
2. Fix complete-session context construction and meter semantics.
3. Add bounded SSD cleanup.
4. Run the full offline test and evaluation suite.
5. Restart with two unified slots and run live validation.
6. Increase to four unified slots and repeat the same validation.
7. Keep `parallel=4` only after the four-way test passes; otherwise retain `parallel=2` and record the measured constraint.

