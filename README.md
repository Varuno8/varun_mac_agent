# Micky / Clicky — macOS Voice Agent

A push-to-talk voice agent that lives in the macOS menu bar. Hold `ctrl+option`, speak a command, release — Micky executes it: opens apps, types text, clicks UI elements, sends messages, all hands-free.

**Clicky** is the Swift menu bar app — eyes and hands.  
**Micky** is the Python proxy — the brain (Vertex AI Gemini + pydantic-ai orchestrator).

Two execution tracks run in parallel per turn:

- **Track A — Vision/UI**: Clicky captures screenshots; Gemini drives UI via bracket tags; Swift executes clicks/keystrokes/AppleScript.
- **Track B — API-first**: A pydantic-ai orchestrator pre-pass routes common tasks (Slack, Drive, Gmail, Calendar, GitHub, Notion, etc.) directly through an MCP bridge — no screenshots, no OCR. Results are injected as structured hints before the main Gemini call. Track A is the fallback when Track B has no coverage.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CLICKY  (Swift, macOS)                              ║
║                           EYES & HANDS                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  User holds ctrl+option                                                    ║
║        │                                                                   ║
║        ▼                                                                   ║
║  GlobalPushToTalkShortcutMonitor  (listen-only CGEvent tap)                ║
║        │ start/stop signal                                                 ║
║        ▼                                                                   ║
║  BuddyDictationManager  (AVAudioEngine → PCM16)                           ║
║        │ audio buffers                                                     ║
║        ▼                                                                   ║
║  Transcription Provider  ──────────────────────────────────────────────    ║
║    Primary : AssemblyAI WebSocket (u3-rt-pro, real-time streaming)         ║
║    Fallback: POST /transcribe → proxy → Gemini STT                         ║
║    Fallback: OpenAI Whisper upload                                         ║
║    Fallback: Apple Speech (on-device)                                      ║
║        │ final transcript                                                  ║
║        ▼                                                                   ║
║  ScreenCaptureKit  (JPEG per display, max 1280px, cursor-screen first)     ║
║        │ screenshot(s)                                                     ║
║        ▼                                                                   ║
║  CompanionManager  ◄──────────────────────────────────────────────────┐   ║
║    Agentic loop · max 8 iterations                                    │   ║
║    Injects: history + memory block + screenshots                       │   ║
║        │                                                              │   ║
║        │  POST /chat  (SSE, localhost:8081)                           │   ║
║        ▼                                                              │   ║
╠════════════════════════════════╦═════════════════════════════════════╝   ║
║                                ║                                         ║
║        PROXY  (Python)         ║  ← HTTP SSE response stream             ║
║           THE BRAIN            ║                                         ║
╠════════════════════════════════╣                                         ║
║                                                                          ║
║  FastAPI /chat handler                                                   ║
║        │                                                                 ║
║        ▼                                                                 ║
║  Orchestrator pre-pass  (pydantic-ai Agent — Flash Lite)                 ║
║    • Track B: routes API tasks through Corsair MCP bridge                ║
║      (Slack, Drive, Gmail, Calendar, GitHub, Notion, Linear…)            ║
║    • Searches ~/.clicky_wiki/ for relevant prior notes                   ║
║    • Appends durable facts to core memory                                ║
║    • Returns OrchestratorHints → injected as ORCHESTRATOR HINTS block    ║
║        │                                                                 ║
║        ▼                                                                 ║
║  Resolver pre-pass  (Flash Lite — cheap entity fuzzy-match)              ║
║    • Fuzzy-matches app names + folder paths from transcript               ║
║    • Injects PRE-RESOLVED ENTITIES block into prompt                     ║
║        │                                                                 ║
║        ▼                                                                 ║
║  System prompt assembly  (see Memory section)                            ║
║    [base contract] + [KB: apps·folders·context] + [core memory]          ║
║    + [task slot]  ← ordered for Vertex prefix-cache stability            ║
║        │                                                                 ║
║        ▼                                                                 ║
║  Vertex AI  ─► Gemini (streaming SSE)                                    ║
║        │ SSE chunks                                                      ║
║        ▼                                                                 ║
║  Streaming action interceptor                                            ║
║    • [APPLESCRIPT:…] → run via osascript server-side (Clicky is          ║
║      sandboxed and cannot touch ~/Downloads, ~/Desktop etc.)             ║
║    • [TASK_DONE] after unverified UI action → rewritten to [SCREENSHOT]  ║
║    • All other tags forwarded verbatim to Swift                          ║
║        │                                                                 ║
║        ▼                                                                 ║
║  SSE stream → Swift                                                      ║
║        │                                                                 ║
║        │ (background, after [TASK_DONE])                                 ║
║        ▼                                                                 ║
║  Wiki summarizer  (Flash Lite classifier → ~/.clicky_wiki/)              ║
║  BigQuery telemetry  (turns · task_events · errors tables)               ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                        CLICKY  (continued)                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  AgenticTagParser  ──►  ToolGuard  ──►  AgenticActionExecutor           ║
║                                                │                        ║
║               ┌──────────────┬────────────────┼──────────────┐         ║
║               ▼              ▼                ▼              ▼         ║
║         NSAppleScript  AccessibilityClicker  CGEvent    OCRClicker     ║
║         (open / shell)  (AX tree + AXPress)  (mouse/kbd) (Vision fw)  ║
║               │              │                │              │         ║
║               └──────────────┴────────────────┴──────────────┘         ║
║                                       │                                ║
║                                  macOS Apps / UI                       ║
║                                                                        ║
║  TTS Client  ──►  AVAudioPlayer  ──►  Speaker                         ║
║  OverlayWindow  (blue cursor · bezier fly-to · all monitors)          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Repository layout

```
varun_agent/
│
├── proxy.py                   FastAPI server (port 8081). Three routes:
│                                POST /chat       → Vertex AI Gemini (SSE)
│                                POST /tts        → Vertex AI Gemini TTS → WAV
│                                POST /transcribe → Vertex AI Gemini STT → JSON
│
├── agent_orchestrator.py      pydantic-ai orchestrator (Track B pre-pass).
│                              Runs before every /chat call. Tools: search_wiki,
│                              use_corsair, list_corsair_operations, get_corsair_schema,
│                              core_memory_append, update_wiki_knowledge, list_wiki_topics,
│                              read_core_memory. Returns OrchestratorHints injected
│                              into the Gemini system prompt.
│                              Enable: MICKY_ORCHESTRATOR=1
│
├── agent_memory.py            Tier-2 (Core) + Tier-4 (Wiki) memory.
│                              Owns ~/.clicky_core_memory.json and ~/.clicky_wiki/
│
├── agent_schemas.py           Pydantic schemas: MacAction, TaskDone, TaskAbandon,
│                              CorsairCall/Result, CoreMemoryFact, WikiSearchHit.
│                              Validates Gemini output + round-trips to bracket tags.
│
├── agent_runtime.py           CorsairMCPBridge (real MCP stdio client when
│                              MICKY_MCP_BRIDGE is set), MockMCPServer (stub otherwise),
│                              use_corsair(), bracket-tag encode/decode bridge.
│
├── agent_wiki_summarizer.py   Post-task Flash Lite classifier. Decides if a task
│                              was wiki-worthy and appends to ~/.clicky_wiki/.
│
├── agent_contract.md          The system prompt contract (Micky's identity, action
│                              tag vocabulary, rules, failure modes).
│
├── memory_architecture.md     Documentation of all memory layers.
│
├── micky_context.md           Personal context dump (auto-generated by kb_compile.py,
│                              do not edit by hand).
│
├── task_state.json            Persistent task slot — survives conversation compaction.
│
├── metrics.jsonl              Per-turn structured log: tokens, latency, cache hits.
├── gemini_debug.log           Full Gemini wire log (rotating, 20 MB cap × 3 backups).
├── proxy.log                  stdout/stderr of the proxy process.
│
├── run_micky.sh               Full start: kill old → start proxy → run Clicky binary.
├── run_proxy.sh               Proxy only (when Clicky is already running).
│
├── pyproject.toml             Python deps (uv/pip). Requires Python >=3.13.
├── .env                       API keys + model names. Never commit.
│
├── clicky-mcp-bridge/         Node.js stdio MCP server (Track B executor).
│   ├── corsair.ts             Canonical Corsair instance: 10 plugins, sqlite DB,
│   │                          auto-managed KEK at ./.clicky_kek.
│   ├── src/mcp_server.ts      Entry point — spawned by Python as a subprocess.
│   ├── src/generate_wiki.ts   Reflects all 194 Corsair operations into
│   │                          ~/.clicky_wiki/tools/<service>.md (debug docs).
│   └── src/list_tools.ts      Debug helper: dump every exposed API path.
│
└── clicky/
    └── leanring-buddy/        Swift Xcode project (typo in name is legacy, do not rename)
        ├── CompanionManager.swift          Central state machine + agentic loop
        ├── AgenticActionExecutor.swift     753-line bracket-tag dispatcher
        ├── AgenticCoordinateMapper.swift   Screenshot pixel → screen point math
        ├── MickyMemoryStore.swift          Tier-5 episodic (~/.micky_memory.json)
        ├── CompanionScreenCaptureUtility.swift  ScreenCaptureKit multi-monitor capture
        ├── OverlayWindow.swift             Blue cursor + response bubble overlay
        ├── BuddyDictationManager.swift     AVAudioEngine push-to-talk pipeline
        └── worker/src/index.ts             Cloudflare Worker (production API proxy)
```

---

## How to run

### Prerequisites

- macOS 14.2+ (ScreenCaptureKit)
- Python 3.13+
- GCP project with Vertex AI API enabled + a service account
- A built `Clicky.app` (from Xcode — one-time build)

### 1. Set up `.env`

```bash
# Required
GOOGLE_PROJECT_ID="your-gcp-project-id"
GOOGLE_CLIENT_EMAIL="your-sa@your-project.iam.gserviceaccount.com"
GOOGLE_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n..."
GOOGLE_PRIVATE_KEY_ID="..."

# Models — use Flash for lower cost, upgrade if needed
GEMINI_MODEL="gemini-2.5-flash-preview-05-20"
GEMINI_TTS_MODEL="gemini-2.5-flash-preview-tts"
GEMINI_STT_MODEL="gemini-3-flash-preview"

# Voice transcription
ASSEMBLYAI_API_KEY="..."

# TTS voice
ELEVENLABS_API_KEY="..."
ELEVENLABS_VOICE_ID="..."

# Optional: BigQuery telemetry (leave unset to disable)
BQ_DATASET="micky_agent"
# BQ_DISABLED=1
```

> **Cost warning:** `gemini-3.1-pro-preview` is a frontier model and burns credits fast. `gemini-2.5-flash-preview-05-20` handles all tasks at ~10× lower cost per turn.

### 2. Install Python dependencies

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Build Clicky (requires Xcode, one-time)

```
open clicky/leanring-buddy.xcodeproj
# Cmd+R to build and run
```

> **Do NOT run `xcodebuild` from the terminal** — it invalidates TCC permissions (Screen Recording, Accessibility). You will have to re-grant them in System Settings.

The built app lands at:
```
~/Library/Developer/Xcode/DerivedData/leanring-buddy-*/Build/Products/Debug/Clicky.app
```

### 4. (Optional) Set up the MCP bridge for API-first execution

Without this step, the orchestrator pre-pass is still available but calls fall back to `MockMCPServer` (returns empty results). Track A vision/UI path is unaffected either way.

```bash
# Prerequisites: Node 22+
cd clicky-mcp-bridge
npm install
cp .env.example .env
# Fill in OAuth client IDs/secrets and API keys for the integrations you want.

# First boot: creates corsair.db + auto-generates KEK at ./.clicky_kek (back it up)
npm run generate-wiki   # verifies everything loads; writes ~/.clicky_wiki/tools/

# OAuth setup for each Google/GitHub/Slack/Linear/Notion integration you want:
npx corsair setup gmail
npx corsair setup googledrive
npx corsair setup googlecalendar
npx corsair setup slack
# ... repeat for github, linear, notion

# Verify: list every API path the agent can call
npm run list-tools
# Expected: slack.api.channels.list, googledrive.api.files.list, gmail.api.messages.send, …
```

### 5. Run

```bash
# Track A only (default)
./run_micky.sh

# Track A + Track B (API-first orchestrator + MCP bridge)
MICKY_ORCHESTRATOR=1 \
MICKY_MCP_BRIDGE=/Users/varuntyagi/Downloads/varun_agent/clicky-mcp-bridge \
./run_micky.sh

# Proxy only (Clicky already running)
./run_proxy.sh
```

`run_micky.sh` auto-picks the newest binary between `/Applications/Clicky.app` (CI/CD builds) and DerivedData (local Xcode builds), waits for the proxy to be ready on port 8081, then runs the binary.

### Verify

- Clicky icon appears in the menu bar
- Hold `ctrl+option` — waveform animates in the overlay
- Say *"what's on my screen"*
- Screenshot is captured, Gemini replies, blue cursor points at referenced elements

---

## Per-turn pipeline

Every voice interaction flows through these stages in order:

```
1.  KEY DOWN   ctrl+option held
2.  RECORD     AVAudioEngine → PCM16 → AssemblyAI WebSocket → partial transcripts
3.  KEY UP     Final transcript emitted
4.  CAPTURE    ScreenCaptureKit → JPEG per display (sorted: cursor screen first)
5.  ORCHESTRATE Pydantic-ai pre-pass (Track B, Flash Lite, gated by MICKY_ORCHESTRATOR=1):
               a. list_corsair_operations → use_corsair via MCP bridge (API calls)
               b. search_wiki → inject relevant prior notes
               c. core_memory_append if new durable fact found
               Returns OrchestratorHints → injected as ORCHESTRATOR HINTS block
               Skips screenshot follow-up turns (orchestrator output is stale there)
5.5 RESOLVE    Proxy resolver pre-pass (Flash Lite):
               fuzzy-match app names + folder paths → PRE-RESOLVED ENTITIES block
6.  ASSEMBLE   System prompt built (see Memory section for ordering)
7.  GEMINI     Vertex AI streaming SSE
               │ chunk arrives
               ├─ [APPLESCRIPT:…] → run server-side via osascript, strip from stream
               ├─ [TASK_DONE] after unverified click → rewrite to [SCREENSHOT]
               └─ everything else → forward to Swift
8.  PARSE      AgenticTagParser → ToolGuard → AgenticActionExecutor
               dispatches each tag to: NSAppleScript / AX tree / CGEvent / OCR
9.  TTS        Spoken text → ElevenLabs or Gemini TTS → WAV → AVAudioPlayer
10. POINT      [POINT:x,y:label] → blue cursor bezier fly-to on overlay
11. DONE       [TASK_DONE] → loop terminates, task slot cleared
               (background) → Wiki summarizer classifies task for wiki write
               (background) → BigQuery telemetry row inserted
```

If the model needs to verify state after an action (e.g. confirm a click landed), it emits `[SCREENSHOT]` and the loop continues — up to 8 iterations.

---

## Memory system

Micky has five independent memory layers. They converge as text blocks injected into the Gemini system prompt or request contents before each call. Order matters: stable layers come first so Vertex AI's prefix cache hits.

```
System prompt (assembled in this order for prefix-cache stability)
═══════════════════════════════════════════════════════════════════
 ① Agent contract        ~43 KB  baked into Swift binary
 ② Behavior rules        ~2 KB   static Python constant in proxy.py
 ③ Installed apps        varies  mtime-cached from disk
 ④ Known folders         varies  mtime-cached from disk
 ⑤ Personal context      ~20 KB  micky_context.md, mtime-cached
 ⑥ Core memory           <300 tok ~/.clicky_core_memory.json ← changes rarely
 ⑦ Task slot             small   task_state.json ← changes every turn (TAIL)

Request contents (mutable, sent on every call)
═══════════════════════════════════════════════
 ⑧ Conversation history  Swift array, compacted when too large
 ⑨ Current user turn     transcript + screenshots
 ⑩ PRE-RESOLVED ENTITIES resolver output, this turn only
 ⑪ RELEVANT PAST EXPERIENCE top-5 episodic memories (keyword-scored)
 ⑫ CURRENT TASK SLOT     echoed from task_state.json for Gemini to read
```

### The five tiers

| # | Name | File | Lifetime | How it's read | How it's written |
|---|---|---|---|---|---|
| T1 | **Working** | Swift RAM array | This task | Verbatim every turn | Every user/model turn |
| T2 | **Core** | `~/.clicky_core_memory.json` | Forever | Always in system prompt (⑥) | `core_memory_append` tool call |
| T3 | **Personal context** | `micky_context.md` | Until re-compiled | Whole file dumped (⑤) | `kb_compile.py` (manual) |
| T4 | **LLM Wiki** | `~/.clicky_wiki/**/*.md` | Forever | On-demand: `search_wiki(query)` | Post-task Flash Lite classifier |
| T5 | **Episodic** | `~/.micky_memory.json` | Forever, FIFO@300 | Keyword-scored top-5 (⑪) | After every `[TASK_DONE]` |

---

### T2 — Core memory (Letta-style)

Stable identity facts that Micky should never have to re-discover. Hard cap: **300 tokens**. Oldest non-pinned fact is evicted when full. Agent writes new facts mid-conversation via the `core_memory_append` tool.

Seed facts (written on first run):
```
★ user is Varun (garimatyagi268@gmail.com)        [pinned]
★ prefer APIs over UI/OCR when both are possible  [pinned]
• user prefers VS Code
• user is actively building TryOwn with Next.js 16
• Diagnxt is in prototype phase — no real data yet
```

Because core memory sits **after** the static KB layers in the system prompt, a `core_memory_append` write doesn't invalidate the prefix cache for the expensive preceding layers.

---

### T3 — Personal context

`micky_context.md` is compiled from sources under `~/micky_kb/` by `kb_compile.py`. It is read-only at runtime. The proxy caches it by `mtime_ns` — the bytes only change when you explicitly re-run the compiler, so the Vertex prefix cache hits every turn after the first. At ~20 KB the empirical cache hit rate reaches ~74% by turn 5.

---

### T4 — LLM Wiki (Karpathy-style)

The key unlock for unbounded knowledge growth **without** context bloat. Topic-organized flat markdown files under `~/.clicky_wiki/`:

```
~/.clicky_wiki/
├── projects/         ← project notes (e.g. clicky.md, tryown.md, diagnxt.md)
├── preferences/      ← coding style, communication preferences
├── debugging/        ← workarounds, AppleScript quirks, macOS permission notes
├── knowledge/        ← agent-written free-form notes (update_wiki_knowledge tool)
├── people/           ← facts about people (e.g. garima.md)
└── tools/            ← AUTO-GENERATED by `npm run generate-wiki` in clicky-mcp-bridge
    ├── slack.md      │  One file per Corsair plugin. Each file lists every exposed
    ├── googledrive.md│  API path + TypeScript-style input/output schema.
    ├── gmail.md      │  194 operations across 10 services. Regenerate after npm upgrades.
    └── …             │  NOT loaded into every prompt — searched only on demand.
```

**Reading:** Only via the `search_wiki(query)` tool — grep over filenames (weight 3) + content (weight 1), returns top-3 matching files (capped at 4000 chars each). The agent calls it when it decides it needs deeper context. Zero tokens paid otherwise.

**Writing:** After every `[TASK_DONE]`, a background Flash Lite call classifies `wiki_worthy: bool`. If true, it also extracts `{wiki_path, content}` and calls `wiki_append`. The write is best-effort and never blocks the chat response. A path-traversal guard (`_sanitize_wiki_path`) rejects any path that escapes `~/.clicky_wiki/`.

---

### T5 — Episodic store (existing, slated for deprecation)

`~/.micky_memory.json` — up to 300 entries, FIFO eviction. Retrieval: bidirectional keyword substring match scored by `(match_count × 10 + access_count)`, top-5 injected per turn. Will be deprecated once the LLM Wiki covers the same ground.

---

## Context window optimization

### 1. Prefix-cache-stable ordering

Static layers (contract → rules → apps → folders → context MD → core memory) are assembled **before** mutable layers (task slot). The task slot mutates every turn; everything before it stays byte-identical across most turns. Vertex AI's implicit prefix cache hits on the stable prefix.

The proxy logs `kb_prefix_stable=True/False` and a SHA256 prefix of the augmentation block on every turn. If `prefix_stable=True` but `main_cached_tokens` is low, something is unexpectedly mutating the stable region.

### 2. mtime-based file caching

`micky_context.md`, the apps catalog, and the folders wiki are re-read from disk only when their `mtime_ns` changes. Between compiler runs the bytes are identical → same prefix cache key.

### 3. Conversation history compaction

`CompanionManager.compactConversationHistoryIfNeeded()` trims older turns when the request body would exceed Vertex's input limit. The task slot (`task_state.json`) carries the active goal forward even after older history is dropped, so Gemini always knows what it was working on.

### 4. Vision-conditional screenshots

Screenshots are attached only on turns that need vision: the first turn of a task referencing the screen, and immediately after a `[SCREENSHOT]` tag. Text-only follow-up turns omit images — saves 10–50 KB per turn.

### 5. Resolver pre-pass on Flash Lite

Entity resolution (app name fuzzy-match, folder path fuzzy-match) runs on the cheap Flash Lite model before the main call. The main model never wastes tokens re-searching for something the resolver already resolved.

### 6. On-demand wiki retrieval

The LLM Wiki is never auto-loaded. Gemini calls `search_wiki(query)` only when it decides the turn needs deeper context. Turns that don't need it pay zero tokens for the wiki.

### 7. RESOURCE_EXHAUSTED retry with backoff

When the user presses `ctrl+option` immediately after a task, the previous Gemini HTTP stream may still be draining. Vertex sees two concurrent requests and returns 429. The proxy now detects `RESOURCE_EXHAUSTED`, waits 4 seconds for the old stream to finish, and retries once automatically — the user never needs to restart the proxy.

---

## Action tag vocabulary

| Tag | Effect |
|---|---|
| `[APPLESCRIPT:source]` | Run AppleScript / `do shell script` — intercepted server-side by proxy (Clicky sandbox) |
| `[AXCLICK:label]` | Click by accessibility label — **prefer over pixel click whenever any label exists** |
| `[CLICK:x,y]` / `[CLICK:x,y:screenN]` | Pixel click (screenshot pixel space, 0,0 = top-left). Only for unlabeled elements |
| `[DBLCLICK:x,y]` / `[RCLICK:x,y]` | Double / right click — same coordinate rules as `[CLICK]` |
| `[TYPE:text]` | Type into focused field. Escape `]` as `\]` |
| `[HOTKEY:cmd+s]` | Send a key chord |
| `[SCROLL:down:3:x,y]` | Scroll N clicks in direction at coordinate |
| `[WAIT:500]` | Sleep N ms (use after app launch, before clicking) |
| `[SCREENSHOT]` | Re-capture and re-evaluate on next turn |
| `[CONFIRM:message]` | Stop and require user confirmation before destructive actions |
| `[POINT:x,y:label]` | Move blue cursor overlay to coordinate (visual only) |
| `[PLAN:step1\|step2]` | Register a multi-step plan |
| `[SUBTASK_DONE:step]` | Mark plan step complete |
| `[TASK_DONE]` | Task complete — stops agentic loop, clears task slot |
| `[TASK_UPDATE:{…}]` | Write/update persistent task slot JSON |
| `[TASK_ABANDON:reason]` | Drop task, clear slot, no success logged |

### Coordinate math

`[CLICK:x,y]` is in **screenshot pixel space** (top-left origin). `AgenticCoordinateMapper` converts to screen points:

```
scaleX = displayWidthPoints  / screenshotWidthPixels
scaleY = displayHeightPoints / screenshotHeightPixels

screenX = displayFrame.minX + pixelX × scaleX
screenY = displayFrame.minY + displayHeightPoints − (pixelY × scaleY)
          ↑ flip Y axis: AppKit origin is bottom-left, screenshot origin is top-left
```

`screenshotWidthPixels` and `screenshotHeightPixels` are taken from `cgImage.width/height` (actual captured dimensions), not from `configuration.width/height` (requested). ScreenCaptureKit may round to GPU alignment boundaries, causing a systematic offset if the wrong values are used.

**Rule:** if `[POINT:x,y]` and `[CLICK:x,y]` target the same element, both coordinates must be identical.

---

## Server-side interceptors

### APPLESCRIPT interception

Clicky has the macOS app sandbox enabled (`com.apple.security.app-sandbox = true`). AppleScript that touches `~/Downloads`, `~/Desktop`, or any path outside the container fails with `error -54 / Operation not permitted`. The proxy runs as a plain Terminal process with no sandbox, so it intercepts every `[APPLESCRIPT:…]` tag, runs it via `osascript`, and strips the tag from the SSE stream. Clicky never sees it.

### TASK_DONE safety rewrite

If Gemini emits `[TASK_DONE]` in the same response as a UI-mutating action (`[CLICK]`, `[AXCLICK]`, `[DBLCLICK]`, `[RCLICK]`, `[TYPE]`, or `[APPLESCRIPT]`) without a `[SCREENSHOT]` in between, the proxy rewrites `[TASK_DONE]` → `[SCREENSHOT]`. This forces the agentic loop to verify that the action actually landed before declaring success.

The `rewritten_premature_task_done_count` metric tracks how often this fires — if it's consistently non-zero, the system-prompt rule-following is slipping.

---

## Dual-track architecture

Track A (vision/UI, always on) and Track B (API-first, gated) run as a pre-pass + main-call pipeline.

### Track B — Orchestrator pre-pass

`agent_orchestrator.py` is a pydantic-ai `Agent` with strict typed output (`OrchestratorHints`). It runs before every `/chat` call when `MICKY_ORCHESTRATOR=1` is set, and is skipped on screenshot follow-up turns (where orchestrator output would be stale).

Tools available to the orchestrator:

| Tool | Purpose |
| --- | --- |
| `use_corsair(integration, action, params)` | Execute one Corsair API call via MCP bridge |
| `list_corsair_operations(plugin?)` | Discover API paths before calling `use_corsair` |
| `get_corsair_schema(path)` | TypeScript-style schema for one path |
| `search_wiki(query)` | Search `~/.clicky_wiki/` for prior notes |
| `core_memory_append(fact)` | Persist a durable fact to T2 core memory |
| `update_wiki_knowledge(filename, content)` | Append a longer note to the wiki |
| `list_wiki_topics()` | List all wiki files |
| `read_core_memory()` | Read current T2 core memory verbatim |

The orchestrator's `render_for_prompt()` emits an `── ORCHESTRATOR HINTS ──` block injected into the Gemini system prompt. When the block is empty (vision-only task), it's skipped entirely to preserve prefix-cache stability.

### Track B — MCP Bridge

`clicky-mcp-bridge/` is a Node.js stdio MCP server spawned by `agent_runtime.CorsairMCPBridge`. It fronts 10 Corsair plugins (Slack, Drive, Gmail, Calendar, GitHub, Linear, Notion, Tavily, Discord, Stripe) over stdin/stdout. The Python side holds a background asyncio event loop that routes calls through `mcp.client.stdio`.

MCP tools exposed:

| Tool | Use |
|---|---|
| `list_operations` | All callable API paths |
| `get_schema` | TS-shaped schema for one path |
| `run_script` | Execute one Corsair operation |
| `corsair_setup` | Drive OAuth from inside a conversation |
| `request_permission` | Approval flow for guarded actions |

When `MICKY_MCP_BRIDGE` is unset, `agent_runtime._get_mcp()` returns a `MockMCPServer` that returns empty results — orchestrator fails open and the main call proceeds on Track A alone.

### Track A — Schema observability

`decode_bracket_tag(tag)` in `agent_runtime.py` parses each emitted tag through the `MacAction` Pydantic schema and records `schema_ok / schema_invalid / schema_unmodeled` in `metrics.jsonl`. No behavior changes to Swift — this is instrumentation only.

---

## BigQuery telemetry

Three append-only tables in `{PROJECT_ID}.micky_agent`:

| Table | One row per | Key columns |
| --- | --- | --- |
| `turns` | `/chat` request | transcript, response_text, response_action_tags, token counts, latency, finish_reason |
| `task_events` | `[TASK_UPDATE]` / `[TASK_DONE]` / `[TASK_ABANDON]` tag | goal, steps_remaining, step_done, outcome |
| `errors` | any component failure | component, error_type, message, context |

Writes are fire-and-forget on a 2-thread `BQ_EXECUTOR` pool — a BQ outage never blocks a `/chat` response. The local `metrics.jsonl` is the durable fallback.

Disable with `BQ_DISABLED=1` in `.env`.

---

## Debugging

| What to look at | Where |
| --- | --- |
| Live proxy output | `tail -f proxy.log` |
| Full Gemini request + response | `tail -f gemini_debug.log` |
| Per-turn token counts + latency | `tail -f metrics.jsonl \| jq .` |
| Active task state | `cat task_state.json` |
| Core memory | `cat ~/.clicky_core_memory.json` |
| LLM Wiki contents | `ls ~/.clicky_wiki/**` |
| Corsair API docs (Track B) | `cat ~/.clicky_wiki/tools/slack.md` (one file per service) |
| Episodic memories | `cat ~/.micky_memory.json \| jq .` |
| Track B orchestrator hints | Look for `── ORCHESTRATOR HINTS ──` in `gemini_debug.log` |
| MCP bridge operations | `cd clicky-mcp-bridge && npm run list-tools` |

Key `metrics.jsonl` fields:

| Field | Target |
| --- | --- |
| `main_cached_tokens` | Should be >60% of `main_prompt_tokens` after warm-up |
| `main_ttft_ms` | Time-to-first-token, target <600 ms |
| `kb_prefix_stable` | Should be `true` on all but the first turn of a session |
| `rewritten_premature_task_done_count` | Should be 0; non-zero means model is skipping verification |
| `intercepted_applescripts` | Count of APPLESCRIPT tags the proxy ran server-side |
| `orchestrator_ran` | `true` when the Track B pre-pass executed this turn |
| `orchestrator_latency_ms` | Pre-pass wall time (target <800 ms on Flash Lite) |
| `orchestrator_corsair_calls` | Number of Corsair MCP calls made by the orchestrator |
| `orchestrator_wiki_hits` | Number of wiki snippets injected into the Gemini prompt |
| `orchestrator_error` | Non-null when the pre-pass failed (always fails open) |

---

## Key design decisions

**Gemini is the brain — proxy is plumbing.** The proxy runs the orchestrator and resolver pre-passes, assembles the system prompt, and forwards everything to Gemini. It never silently executes intent or makes decisions on behalf of the model. Exception: the APPLESCRIPT interception and TASK_DONE rewrite are mechanical enforcement of existing rules the model is already supposed to follow — they are guardrails, not policy.

**APIs before vision.** When the user's request maps to a service with an API (Slack, Drive, Gmail, Calendar, GitHub, Notion, Linear, Tavily, Discord, Stripe), the Track B orchestrator calls it directly — ~1 s, fractions of a cent, zero screenshots. Track A vision/OCR is the fallback for legacy desktop apps and websites with no API. Core memory fact: `"prefer APIs over UI/OCR when both are possible"` is pinned in every system prompt.

**AXCLICK over CLICK.** AX tree clicks are resolution-independent and survive display scaling changes. Pixel clicks are only for canvas elements and unlabeled icons. The contract explicitly instructs the model to prefer `[AXCLICK]` whenever any label is visible.

**Inline bracket tags, not JSON tool calls.** Gemini streams prose and bracket tags interleaved. Swift parses tags as they arrive in the SSE stream, so TTS starts playing the spoken portion while actions are still queuing. This delivers sub-600 ms time-to-first-token UX while keeping the wire format simple.

**No vector database.** At single-user scale with ~300 memory entries, keyword scoring (bidirectional substring match + access count) is faster, more transparent, and easier to debug than embeddings. The wiki's grep-based search follows the same principle. The upgrade path to embeddings is documented in `memory_architecture.md` — it's a swap inside `MickyMemoryStore`, not a new infrastructure dependency.

**Server account credentials, never in the app.** Clicky holds no API keys. The proxy on localhost holds them, loaded from `.env` at startup via a GCP service account. The Cloudflare Worker mirrors this for production.

**APPLESCRIPT runs in the proxy, not in Clicky.** Clicky is sandboxed. The proxy is not. Moving AppleScript execution to the proxy unlocks access to `~/Downloads`, `~/Desktop`, and anything else the user's Terminal can reach.
