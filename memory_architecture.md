# Memory Architecture

This file documents how Micky remembers things — across a single utterance, across one session, across multiple sessions, and across the lifetime of the user's machine. Read this before changing any retrieval, prompt-injection, or memory-store code.

---

## Mental model

There is no single "memory" in this app. There are **four independent layers**, each with a different lifetime, retrieval strategy, and write trigger. They all converge on the same place at runtime: bytes appended to the system prompt or the request `contents` before each call to Gemini.

| Layer | Lifetime | Storage | Retrieval | Write trigger |
|---|---|---|---|---|
| 1. Conversation history | This task | RAM (Swift array) | Verbatim, every turn | Every user/model turn |
| 2. Long-term memory store | Forever (FIFO at 300) | `~/.micky_memory.json` | Keyword overlap scoring, top 5 | After every `[TASK_DONE]` |
| 3. Personal context | Until the user edits it | `micky_context.md` (and KB sources it compiles from) | Whole file dumped, mtime-cached | Manual edit or `kb_compile.py` |
| 4. Knowledge graph (optional) | Until the user rebuilds it | `~/micky_kb/*.sqlite` | Gemini-driven slice, per-transcript | `MICKY_USE_GRAPH=1` + manual rebuild |

---

## Layer 1 — Conversation history (short-term)

**Where:** Swift-side `CompanionManager` keeps an array of `{role, text}` turns. Sent in every `/chat` request body as the Vertex `contents` array.

**Read:** Always. The full history is sent on every turn — Gemini relies on it for follow-ups like "do that again", "the other one", "scroll back".

**Write:** Eagerly, at the end of each loop iteration. Even an interrupt mid-loop preserves what already happened.

**Eviction:** `compactConversationHistoryIfNeeded()` trims older turns when the request body would otherwise blow past Vertex's input limits. The "new task" shortcut (`ctrl+option`) clears the history for a fresh task.

**Important invariant:** the history sent to the proxy is the *only* short-term context. The proxy does not store anything between requests.

---

## Layer 2 — Long-term memory store (procedural / personal / tool)

**Where:** [`clicky/leanring-buddy/MickyMemoryStore.swift`](clicky/leanring-buddy/MickyMemoryStore.swift) → JSON file at `~/.micky_memory.json`.

**Schema:**
```swift
struct MickyMemory {
    let id: UUID
    let category: .personal | .procedural | .tool
    let keywords: [String]   // lowercase, min 3 chars
    let content: String      // one natural-language sentence
    let createdAt: Date
    var accessCount: Int     // bumped on retrieval
}
```

**Three categories:**
- `personal` — facts about the user (preferences, recurring workflows, name)
- `procedural` — what action sequences worked or failed for a given app or task type
- `tool` — which action tags (`APPLESCRIPT`, `AXCLICK`, etc.) succeed for which app states

**Read:** Before each task, `relevantMemoriesSystemBlock(for: queryText)` runs `retrieveRelevant(for:)`, which:
1. Tokenizes the query into `Set<String>` of words ≥ 3 chars
2. For each memory, counts keyword overlaps with the query (bidirectional substring match)
3. Scores `matchCount * 10 + accessCount`, returns top 5
4. Wraps the results in `── RELEVANT PAST EXPERIENCE ──` and appends to the system prompt

**Write:** After every `[TASK_DONE]`, the task transcript + outcome summary is stored as `procedural` memory. ⚠️ **Known bug**: write happens unconditionally on `[TASK_DONE]`, even when the model lied about completion. See [agent_contract.md](agent_contract.md) and the BEHAVIOR RULES section in [proxy.py](proxy.py) for the verify-before-done rule that mitigates this from the model side. A proper Swift-side fix should also gate the write on user non-correction in the next turn.

**Eviction:** FIFO at 300 entries. `memories.removeFirst(memories.count - 300)`.

**Persistence:** `JSONEncoder` with `.iso8601` dates, atomic write to `~/.micky_memory.json` on every `store()`.

---

## Layer 3 — Personal context (the markdown dump)

**Where:** `micky_context.md` in the repo root. Compiled from KB sources by `kb_compile.py` (not in this repo — lives in the user's `~/micky_kb/` workspace).

**Read:** [`build_personal_context_section`](proxy.py) in the proxy, on every `/chat`. The whole file (~20 KB) is wrapped in `── PERSONAL CONTEXT ──` and appended to the system prompt augmentation.

**Caching:** `_personal_context_md_cache` keyed by `os.stat(MICKY_CONTEXT_PATH).st_mtime_ns`. Re-read only when the file changes. Same pattern for the apps catalog and folders wiki — see `_apps_section_cache`, `_folders_section_cache`.

**Write:** Manual edit, or by running `kb_compile.py` to regenerate from upstream KB sources.

**Why a flat MD dump:** at ~20 KB the cost of always-on personal context is negligible compared to the value of zero retrieval bugs. The byte-stable bytes also make Vertex AI's implicit prompt cache hit hard once warmed (verified empirically: ~74% cache rate after ~5 turns; with the mtime cache stabilizing the bytes, expected to materialize sooner).

---

## Layer 4 — Knowledge graph (optional, off by default)

**Where:** `~/micky_kb/graph_query.py` + a SQLite DB. Lazy-imported by [`_load_graph_query_fn`](proxy.py) only when `MICKY_USE_GRAPH=1`.

**Read:** When the env var is set AND the current turn has a transcript, `query_for_transcript()` makes a Flash Lite call to retrieve only the slice of nodes relevant to this transcript (~1–3 KB instead of the full 20 KB MD).

**Trade-off:** smaller per-turn context cost, but adds a second LLM round-trip and is non-deterministic. Currently off because the MD path is good enough and prompt caching makes the size cost trivial.

---

## What is NOT in this stack

For the avoidance of doubt:

- **No vector embeddings.** Anywhere. No `text-embedding-*` calls, no cosine similarity, no FAISS, no sqlite-vec.
- **No vector database.** No Pinecone, Weaviate, Qdrant, Chroma, etc.
- **No retrieval framework.** No LangChain, LlamaIndex, Haystack. The agentic loop is hand-rolled in [`CompanionManager.swift:706`](clicky/leanring-buddy/CompanionManager.swift) (~`agentic loop`); retrieval is hand-rolled in `MickyMemoryStore.retrieveRelevant`.
- **No fancy similarity scoring.** The keyword scorer is bidirectional substring match. The fuzzy app/folder matchers in [proxy.py](proxy.py) (`fuzzy_match_folder_basename`, `fuzzy_match_app_name`) use normalized substring matching with tiered scoring.

This is deliberate. At single-user / single-machine scale with ~300 memory entries and ~10 KB of context, none of those tools earn their keep — they add operational surface for marginal retrieval-quality wins.

---

## Where it all goes at request time

The full system prompt sent to Gemini per `/chat` is assembled in [`handle_chat`](proxy.py) as:

```
[Swift-baked agent contract]              ← stable, ~43 KB in current binary
+ [── BEHAVIOR RULES ──]                  ← static const in proxy, ~2 KB
+ [── INSTALLED APPS ──]                  ← mtime-cached
+ [── KNOWN FOLDERS ON THIS MAC ──]       ← mtime-cached
+ [── PERSONAL CONTEXT ──]                ← mtime-cached (or graph slice)
```

Plus, in `contents`:
- Conversation history (Layer 1, verbatim)
- The current user turn
- Optional `── PRE-RESOLVED ENTITIES ──` block injected by the resolver
- Optional screenshots (vision-stripped on text-only turns)
- Optional `── RELEVANT PAST EXPERIENCE ──` block from Layer 2 (Swift side)

Every byte of this assembly is observable in `gemini_debug.log` (rotating, 20 MB cap). Per-turn token + timing breakdowns land in `metrics.jsonl`.

---

## When to upgrade what

| Trigger | Upgrade to |
|---|---|
| `~/.micky_memory.json` consistently >2k entries | Embedding-based scoring inside `MickyMemoryStore` (Gemini `text-embedding-004` + numpy cosine, no DB) |
| `metrics.jsonl` shows keyword recall is missing relevant past experience even with the inverted index | Same as above — embeddings |
| Multi-machine sync becomes a requirement | SQLite + sqlite-vec, replicated; still no vector DB |
| Multi-step retrieval (memory → wiki → action) becomes common | LlamaIndex query engines or LangGraph for the retrieval graph; still no need for the rest of LangChain |

None of these are needed today. Document them here when they are.

---

## Pointers

| Concern | File:symbol |
|---|---|
| Add/retrieve a memory | [`clicky/leanring-buddy/MickyMemoryStore.swift`](clicky/leanring-buddy/MickyMemoryStore.swift) → `store`, `retrieveRelevant`, `relevantMemoriesSystemBlock` |
| Inject memory into Gemini's prompt | Swift caller of `relevantMemoriesSystemBlock` (in `CompanionManager`) |
| Conversation history shape | `CompanionManager` agentic loop (`maxIterations = 8`) |
| Personal-context build + cache | [proxy.py](proxy.py) → `_load_personal_context_md_cached`, `build_kb_augmentation` |
| Graph path | `~/micky_kb/graph_query.py` (out-of-repo); proxy hook in [`_load_graph_query_fn`](proxy.py) |
| Per-turn metrics emission | [proxy.py](proxy.py) → `record_turn_metrics`; live file at `metrics.jsonl` |
