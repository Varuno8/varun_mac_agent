"""
Local proxy server — replaces the Cloudflare Worker.
Runs on http://localhost:8080 and proxies the three routes Clicky uses:

  POST /chat        → Vertex AI streaming vision (SSE)
  POST /tts         → Vertex AI Gemini TTS → WAV bytes
  POST /transcribe  → Vertex AI Gemini STT → { transcript }

Auth: service-account credentials from .env (same format as Agent_Swarm).
"""

import base64
import json
import re
import struct
import subprocess
import wave
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from google.genai import types
from google import genai
from google.oauth2 import service_account


# ── Load .env ─────────────────────────────────────────────────────────────────

def load_env(path: str = ".env") -> dict:
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        val = val.replace("\\n", "\n").replace("\\r", "\r")
        env[key.strip()] = val
    return env


def clean_pem_key(raw: str) -> str:
    """Strip non-base64 chars from PEM body lines (handles corrupted \\y \\K sequences)."""
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        if line.startswith("-----"):
            cleaned.append(line)
        else:
            cleaned.append(re.sub(r"[^A-Za-z0-9+/=]", "", line))
    return "\n".join(cleaned)


ENV = load_env(str(Path(__file__).parent / ".env"))

PROJECT_ID    = ENV["GOOGLE_PROJECT_ID"]
MODEL_TEXT    = ENV.get("GEMINI_MODEL",     "gemini-3.1-flash-lite-preview")
MODEL_TTS     = ENV.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
MODEL_STT     = ENV.get("GEMINI_STT_MODEL", "gemini-3-flash-preview")
# The intent resolver runs as a fast pre-pass on every /chat. Flash Lite is
# cheap, low-latency, and plenty for fuzzy entity resolution against the
# apps/folders catalogs already in the system prompt.
MODEL_RESOLVER = ENV.get("GEMINI_RESOLVER_MODEL", "gemini-3.1-flash-lite-preview")
TTS_VOICE     = ENV.get("GEMINI_TTS_VOICE", "Kore")

# Verbose Gemini wire log — the request/response logger appends here.
# Defined up front so the startup banner can reference its location.
GEMINI_DEBUG_LOG_PATH = Path(__file__).parent / "gemini_debug.log"


# ── Build Vertex AI client ────────────────────────────────────────────────────

def build_credentials():
    sa_info = {
        "type": "service_account",
        "client_email": ENV["GOOGLE_CLIENT_EMAIL"],
        "private_key": clean_pem_key(ENV["GOOGLE_PRIVATE_KEY"]),
        "private_key_id": ENV.get("GOOGLE_PRIVATE_KEY_ID", ""),
        "project_id": PROJECT_ID,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    return service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


CREDENTIALS = build_credentials()

GEMINI_CLIENT = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location="global",
    credentials=CREDENTIALS,
)

print(f"✅ Vertex AI client ready  project={PROJECT_ID}")
print(f"   text model : {MODEL_TEXT}")
print(f"   TTS  model : {MODEL_TTS}  voice={TTS_VOICE}")
print(f"   STT  model : {MODEL_STT}")
print(f"   resolver   : {MODEL_RESOLVER}")
print(f"📒 Full Gemini wire log: {GEMINI_DEBUG_LOG_PATH}")
print(f"   (tail -f this file to see every request/response in detail)")


# ── WAV helper ────────────────────────────────────────────────────────────────

def build_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ── Image stripping on initial turns ──────────────────────────────────────────
#
# Clicky takes a screenshot on every push-to-talk and sends it with the
# request. For tasks like "open X", "delete X", "send X to Y", Gemini
# doesn't need vision — the screenshot is wasted bandwidth/cost/latency.
#
# Rule: drop image parts from the request UNLESS one of:
#   1. This is a screenshot-driven follow-up (Clicky's prompt starts with
#      "[fresh screenshot attached" — emitted after a [SCREENSHOT] action).
#   2. The transcript clearly references the screen (click/type/look/etc.).
#
# If Gemini does need vision and we stripped, it can emit [SCREENSHOT] and
# the next turn will carry one.

VISION_TRIGGER_KEYWORDS = (
    "screen", "click", "press the", "press that", "press this",
    "type ",  # trailing space — catches "type my", "type the", "type into"
    "into the field", "into that field", "into this field",
    "into the box", "into that box", "into the input",
    "look at", "show me", "see this", "see that",
    "describe", "read this", "read that",
    "what's there", "what's on", "what does", "where is",
    "the button", "the icon", "the link", "the field", "the input",
    "this button", "that button", "this icon", "that icon",
    "this link", "that link", "this field", "that field",
    "this window", "that window", "this page", "that page",
    "scroll", "highlighted", "selected",
)
FOLLOWUP_PREFIX = "[fresh screenshot attached"


def initial_request_likely_needs_vision(transcript: str) -> bool:
    """Heuristic: does this transcript clearly reference visible screen state?"""
    if not transcript:
        return False
    lowered = transcript.lower()
    return any(keyword in lowered for keyword in VISION_TRIGGER_KEYWORDS)


def is_screenshot_followup_turn(last_user_text: str) -> bool:
    """Clicky prefixes follow-up turns triggered by [SCREENSHOT] with a marker."""
    return bool(last_user_text and last_user_text.startswith(FOLLOWUP_PREFIX))


def strip_images_from_contents(contents: list) -> tuple[list, int]:
    """
    Return (new_contents, stripped_count). Image parts (inline_data) are
    removed; text parts are kept. Content objects with no parts left are
    still kept (with role intact) — the SDK is fine with empty parts.
    """
    stripped_count = 0
    new_contents = []
    for content_item in contents:
        original_parts = getattr(content_item, "parts", None) or []
        kept_parts = []
        for part in original_parts:
            if hasattr(part, "inline_data") and part.inline_data:
                stripped_count += 1
            else:
                kept_parts.append(part)
        new_contents.append(types.Content(
            role=getattr(content_item, "role", "user"),
            parts=kept_parts,
        ))
    return new_contents, stripped_count


# ── Verbose Gemini wire log ───────────────────────────────────────────────────
#
# Full request/response bodies are too large for the main proxy.log
# (system prompt is ~37KB per request). They go into a separate file the
# user can `tail -f` when debugging. Image data is summarized, not dumped.
# (GEMINI_DEBUG_LOG_PATH is defined up top alongside other path constants.)


def log_gemini_request(
    call_label: str,
    model: str,
    system_prompt: str,
    contents: list,
    extra_meta: dict | None = None,
) -> None:
    """
    Append a fully-detailed entry to gemini_debug.log describing exactly
    what we're about to send to Gemini. Image parts are summarized as
    `[image: <mime>, ~<KB>]` rather than dumped.
    """
    try:
        with GEMINI_DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write("\n" + "=" * 88 + "\n")
            log_file.write(f"📤 → GEMINI  [{call_label}]  {datetime.now().isoformat(timespec='seconds')}\n")
            log_file.write(f"model: {model}\n")
            if extra_meta:
                for key, value in extra_meta.items():
                    log_file.write(f"{key}: {value}\n")
            if system_prompt:
                log_file.write(f"\n--- SYSTEM_INSTRUCTION ({len(system_prompt)} bytes) ---\n")
                log_file.write(system_prompt)
                log_file.write("\n")
            log_file.write(f"\n--- CONTENTS ({len(contents)} turn(s)) ---\n")
            for turn_index, turn in enumerate(contents):
                role = getattr(turn, "role", "?")
                turn_parts = getattr(turn, "parts", []) or []
                log_file.write(f"\n[turn {turn_index}, role={role}]\n")
                for part in turn_parts:
                    if hasattr(part, "text") and part.text:
                        log_file.write(f"  TEXT ({len(part.text)} bytes):\n")
                        for line in part.text.splitlines():
                            log_file.write(f"    {line}\n")
                    elif hasattr(part, "inline_data") and part.inline_data:
                        mime = part.inline_data.mime_type or "?"
                        size_kb = len(part.inline_data.data) // 1024
                        log_file.write(f"  IMAGE: mime={mime}, ~{size_kb}KB\n")
                    else:
                        log_file.write(f"  PART (unknown type)\n")
            log_file.write("=" * 88 + "\n")
    except Exception as debug_log_error:
        print(f"⚠️ Debug log write failed: {debug_log_error}")


def log_gemini_response(call_label: str, response_text: str, extra_meta: dict | None = None) -> None:
    """Append the full Gemini response text to gemini_debug.log."""
    try:
        with GEMINI_DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n📥 ← GEMINI  [{call_label}]  {datetime.now().isoformat(timespec='seconds')}\n")
            if extra_meta:
                for key, value in extra_meta.items():
                    log_file.write(f"{key}: {value}\n")
            log_file.write(f"--- RESPONSE ({len(response_text)} bytes) ---\n")
            for line in response_text.splitlines():
                log_file.write(f"  {line}\n")
            log_file.write("=" * 88 + "\n")
    except Exception as debug_log_error:
        print(f"⚠️ Debug log write failed: {debug_log_error}")


# ── KB injection (proxy-side, since the running binary doesn't do it) ────────
#
# The April-22 Clicky binary sends only its base system prompt — it doesn't
# have the apps catalog / folders wiki / personal context wiring that the
# Swift source has now. So the proxy injects all three sections itself,
# reading the same disk locations the (newer) Swift code would.

APPLICATION_SCAN_DIRECTORIES = [
    "/Applications",
    "/System/Applications",
    "/System/Applications/Utilities",
    f"{Path.home()}/Applications",
]
LAPTOP_WIKI_INDEX_PATH = Path.home() / "laptop_wiki" / "index.md"
MICKY_CONTEXT_PATH = Path(__file__).parent / "micky_context.md"

# Set MICKY_USE_GRAPH=1 to route the `── PERSONAL CONTEXT ──` slot through
# ~/micky_kb/graph_query.py (Gemini-driven retrieval over a SQLite graph)
# instead of dumping the full compiled micky_context.md every turn.
USE_GRAPH = os.environ.get("MICKY_USE_GRAPH", "").strip().lower() in (
    "1", "true", "yes", "on",
)
MICKY_KB_DIR = Path.home() / "micky_kb"


def _load_graph_query_fn():
    """Lazy import — pulls in graph_query.py only when the toggle is on, so a
    broken graph build never breaks proxy startup."""
    if not USE_GRAPH:
        return None
    try:
        if str(MICKY_KB_DIR) not in sys.path:
            sys.path.insert(0, str(MICKY_KB_DIR))
        from graph_query import query_for_transcript  # noqa: WPS433
        return query_for_transcript
    except Exception as graph_import_err:
        print(f"⚠️  MICKY_USE_GRAPH=1 but graph_query import failed: {graph_import_err}")
        return None


def scan_installed_app_names() -> list[str]:
    """
    Mirror of `InstalledApplicationCatalog.scanInstalledApps` from the
    Swift source. Lists every `.app` bundle name in the standard scan
    directories, sorted case-insensitively. The bundle name (without the
    `.app` suffix) is what `open -a` accepts.
    """
    discovered: set[str] = set()
    for directory in APPLICATION_SCAN_DIRECTORIES:
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for entry in entries:
            if entry.endswith(".app"):
                discovered.add(entry[:-len(".app")])
    return sorted(discovered, key=str.lower)


def build_apps_section() -> str:
    """
    Build the `── INSTALLED APPS ──` section the resolver looks for.
    """
    apps_list = ", ".join(scan_installed_app_names())
    return (
        "\n\n── INSTALLED APPS ──\n"
        "these are the apps actually installed on this machine. when you "
        "write `open -a 'NAME'`, prefer one of these exact names. you may "
        "write a casual short form (the executor fuzzy-matches it back to a "
        "real bundle name), but using a name from this list is faster and "
        "more reliable. if the user asks for an app that isn't on this list, "
        "tell them it's not installed instead of guessing.\n\n"
        f"{apps_list}"
    )


def build_folders_section() -> str:
    """
    Build the `── KNOWN FOLDERS ON THIS MAC ──` section by parsing the
    wiki index. Same shape as `laptopFoldersSection()` in the Swift
    source, so the existing resolver code works against either one.
    """
    if not LAPTOP_WIKI_INDEX_PATH.exists():
        return ""
    try:
        raw_index = LAPTOP_WIKI_INDEX_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""

    folder_entries: list[tuple[str, str]] = []
    for line in raw_index.splitlines():
        # Index rows look like: | [/abs/path](folders/slug.md) | Purpose. |
        if not line.startswith("| ["):
            continue
        try:
            path_open = line.index("[")
            path_close = line.index("](", path_open)
            absolute_path = line[path_open + 1:path_close]
            purpose_separator = line.index(" | ", path_close)
            purpose = line[purpose_separator + 3:].rstrip()
            if purpose.endswith(" |"):
                purpose = purpose[:-2]
            folder_entries.append((absolute_path, purpose.strip()))
        except ValueError:
            continue

    if not folder_entries:
        return ""

    formatted_entries = "\n".join(
        f"  {path} — {purpose}" for path, purpose in folder_entries
    )
    return (
        "\n\n── KNOWN FOLDERS ON THIS MAC ──\n"
        "the user's notable folders, with one-line summaries (sourced from "
        "~/laptop_wiki/index.md, regenerated when the user adds or moves "
        "files). when the user mentions a folder by partial or casual name, "
        "match it against this list FIRST to find the real absolute path. "
        "only fall back to `find` if no entry below plausibly matches. "
        "always pass absolute paths to shell commands — never `~`.\n\n"
        f"{formatted_entries}"
    )


def _extract_transcript_and_history(
    body: dict,
    fallback_prompt: str,
) -> tuple[str, list[dict]]:
    """
    Flatten whatever shape the client sent into (transcript, history_messages).

    Supports both:
      A. Vertex `contents`: [{role, parts:[{text}, {inlineData}, ...]}, ...]
      B. Legacy:           {userPrompt, history:[{userPlaceholder, assistantResponse}, ...]}

    `transcript` is the last user-text in this turn (text parts only — image
    parts ignored). `history_messages` is every prior turn flattened to
    {"role": "user"|"model", "text": ...}, oldest → newest, current turn
    excluded.
    """
    raw_contents = body.get("contents", []) or []

    def text_of_parts(parts: list) -> str:
        chunks = []
        for part in parts or []:
            text = part.get("text") if isinstance(part, dict) else None
            if text and text.strip():
                chunks.append(text.strip())
        return "\n".join(chunks).strip()

    if raw_contents:
        history: list[dict] = []
        for turn in raw_contents[:-1]:
            text = text_of_parts(turn.get("parts", []))
            if not text:
                continue
            role = turn.get("role", "user")
            if role not in ("user", "model"):
                role = "user"
            history.append({"role": role, "text": text})
        last_text = text_of_parts(raw_contents[-1].get("parts", []))
        return (last_text or fallback_prompt or "").strip(), history

    # Legacy shape
    history = []
    for turn in body.get("history", []) or []:
        user_placeholder = (turn.get("userPlaceholder") or "").strip()
        assistant_response = (turn.get("assistantResponse") or "").strip()
        if user_placeholder:
            history.append({"role": "user", "text": user_placeholder})
        if assistant_response:
            history.append({"role": "model", "text": assistant_response})
    return (fallback_prompt or "").strip(), history


def build_personal_context_section(
    transcript: str = "",
    history_messages: list[dict] | None = None,
) -> str:
    """
    Build the `── PERSONAL CONTEXT ──` slot. Two paths:

    1. Graph path (MICKY_USE_GRAPH=1, transcript present, graph built):
       call graph_query.query_for_transcript() so the slot contains only
       the slice of nodes relevant to this turn (~1-3 KB instead of 20+).
    2. MD path (default / fallback): drop the full compiled
       micky_context.md verbatim, like before.
    """
    if USE_GRAPH and transcript:
        graph_query_fn = _load_graph_query_fn()
        if graph_query_fn is not None:
            try:
                graph_md = graph_query_fn(
                    transcript=transcript,
                    history_messages=history_messages,
                    gemini_client=GEMINI_CLIENT,
                    resolver_model=MODEL_RESOLVER,
                    log_request=log_gemini_request,
                    log_response=log_gemini_response,
                )
            except Exception as graph_err:
                print(f"⚠️  graph_query failed → falling back to MD: {graph_err}")
                graph_md = ""
            if graph_md:
                md_size = (MICKY_CONTEXT_PATH.stat().st_size
                           if MICKY_CONTEXT_PATH.exists() else 0)
                print(
                    f"🕸️  Graph context: {len(graph_md.encode('utf-8'))}B "
                    f"(MD path would be {md_size}B)"
                )
                return f"\n\n── PERSONAL CONTEXT ──\n{graph_md}\n"

    if not MICKY_CONTEXT_PATH.exists():
        return ""
    try:
        context_body = MICKY_CONTEXT_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    return f"\n\n── PERSONAL CONTEXT ──\n{context_body}\n"


def build_kb_augmentation(
    transcript: str = "",
    history_messages: list[dict] | None = None,
) -> str:
    """
    Concatenated KB sections to append to whatever system prompt the
    binary sends. Re-reads files on every call so KB edits and `kb_compile.py`
    runs are picked up live.

    `transcript` and `history_messages` are only consulted by the graph path
    (build_personal_context_section). The apps + folders sections are still
    full-catalog dumps — those stay deterministic and small enough that
    selective retrieval doesn't pay off yet.
    """
    return (
        build_apps_section()
        + build_folders_section()
        + build_personal_context_section(
            transcript=transcript,
            history_messages=history_messages,
        )
    )


# ── Intent resolver (LLM pre-pass) ────────────────────────────────────────────

# Section markers as the Swift app emits them in the system prompt.
SECTION_MARKER_INSTALLED_APPS    = "── INSTALLED APPS ──"
SECTION_MARKER_KNOWN_FOLDERS     = "── KNOWN FOLDERS ON THIS MAC ──"
SECTION_MARKER_PERSONAL_CONTEXT  = "── PERSONAL CONTEXT ──"
SECTION_HEADER_PATTERN = re.compile(r"\n── [A-Z ]+ ──")
H2_PATTERN             = re.compile(r"^##\s+([^\n]+)$", re.MULTILINE)


def extract_prompt_section(system_instruction: str, marker: str) -> str:
    """
    Slice out the body of a `── <MARKER> ──` section from the assembled
    system prompt. The body runs until the next `── … ──` header or end
    of string.
    """
    section_start = system_instruction.find(marker)
    if section_start == -1:
        return ""
    body_start = section_start + len(marker)
    next_header = SECTION_HEADER_PATTERN.search(system_instruction, body_start)
    body_end = next_header.start() if next_header else len(system_instruction)
    return system_instruction[body_start:body_end].strip()


def extract_h2_subsection(body: str, heading: str) -> str:
    """
    Pull the body of a `## <heading>` section out of a markdown block,
    stopping at the next `## ` heading. Empty string if not found.
    """
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(body)
    if not match:
        return ""
    section_start = match.end()
    next_heading = re.search(r"^##\s+", body[section_start:], re.MULTILINE)
    section_end = section_start + next_heading.start() if next_heading else len(body)
    return body[section_start:section_end].strip()


def resolve_intent_with_llm(transcript: str, system_instruction: str) -> dict:
    """
    Pre-pass before the main agentic call. Asks Gemini Flash Lite to
    fuzzy-resolve any folder, app, or contact references in the user's
    transcript to canonical names from the apps/folders catalogs already
    in the system prompt.

    Why a separate call: the main model has the same data but sometimes
    distrusts it (runs `find`, hallucinates paths, etc.). A focused
    resolver with no other job, low temperature, and JSON-only output is
    far more reliable for this single sub-task — and at ~$0.0001 per
    call it's free in practice.

    Returns a JSON-shaped dict. Empty dict if anything goes wrong, so
    the main call can proceed without it.
    """
    apps_body = extract_prompt_section(system_instruction, SECTION_MARKER_INSTALLED_APPS)
    folders_body = extract_prompt_section(system_instruction, SECTION_MARKER_KNOWN_FOLDERS)
    personal_context = extract_prompt_section(system_instruction, SECTION_MARKER_PERSONAL_CONTEXT)
    preferences_body = extract_h2_subsection(personal_context, "Preferences") if personal_context else ""
    print(f"🔎 Resolver inputs: system={len(system_instruction)}B  "
          f"apps={len(apps_body)}B  folders={len(folders_body)}B  "
          f"context={len(personal_context)}B  prefs={len(preferences_body)}B")
    if not apps_body and not folders_body:
        # Diagnostic: extraction failed. Dump what's there so we can see the
        # actual marker format the binary is using.
        print(f"⚠️ Resolver: empty apps and folders sections")
        for keyword in ("INSTALLED", "KNOWN FOLDERS", "PERSONAL CONTEXT"):
            idx = system_instruction.find(keyword)
            if idx == -1:
                print(f"   '{keyword}' not found in system prompt at all")
            else:
                # Print the 60 chars around each occurrence with codepoints for the
                # leading/trailing non-ASCII marker chars so we can see what they are.
                window_start = max(0, idx - 30)
                window_end = min(len(system_instruction), idx + 50)
                window = system_instruction[window_start:window_end]
                codepoints = [f"U+{ord(c):04X}" for c in window if not (32 <= ord(c) < 127) and c != '\n']
                print(f"   '{keyword}' at offset {idx}: {window!r}")
                if codepoints:
                    print(f"     non-ASCII codepoints near it: {codepoints[:10]}")
        return {}

    preferences_block = (
        f"User preferences (apply these when the user names a category instead of a specific app):\n{preferences_body[:1500]}\n\n"
        if preferences_body else ""
    )

    resolver_prompt = (
        "You are a deterministic entity resolver for a macOS voice agent.\n\n"
        f"User said (possibly with transcription errors or noise): \"{transcript}\"\n\n"
        f"{preferences_block}"
        f"Apps installed on this machine (canonical names — pick exactly):\n{apps_body[:2000]}\n\n"
        f"Known folders (absolute path — purpose):\n{folders_body[:6000]}\n\n"
        "Match generously: case-insensitive, ignoring spaces, underscores, "
        "hyphens, and punctuation. Examples of valid matches:\n"
        "  'anti gravity' / 'anti-gravity' / 'AG'    → app 'Antigravity'\n"
        "  'new media' / 'newmedia' / 'new-media'   → folder ending '/new_media'\n"
        "  'kin care' / 'KinCare'                    → folder ending '/kincare'\n"
        "  'try own backend'                          → folder ending '/tryown_backend'\n\n"
        "Category words: ONLY the exact words 'editor', 'IDE', 'browser', "
        "'terminal', 'shell', 'messaging', 'text' (as a verb meaning 'send a "
        "text'), 'message' are categories. When the user uses one of these "
        "literal words and no specific app, resolve to the default for that "
        "category from the Preferences section above.\n\n"
        "Anything else that resembles a real app name — 'VS Code', 'vscode', "
        "'code insiders', 'safari', 'chrome', 'whatsapp', 'antigravity', etc. — "
        "is a SPECIFIC NAME. Honor it exactly even if a default exists. "
        "'open kincare in vs code' must resolve to 'Visual Studio Code - "
        "Insiders', NOT to the default editor.\n\n"
        "Path rule: only emit a `folder_abs_path` value that appears VERBATIM "
        "in the Known folders list above. If the user's spoken folder name "
        "doesn't match any listed path under reasonable fuzzy rules, set "
        "`folder_abs_path` to null and explain in `notes` — never construct a "
        "new path by substituting characters.\n\n"
        "Return JSON only — no prose, no markdown fences. Schema:\n"
        "{\n"
        '  "verb": "open" | "send" | "search" | "delete" | "rename" | '
        '"move" | "copy" | "create" | "click" | "type" | "other" | null,\n'
        '  "app_canonical_name": <exact name from apps list above, or null>,\n'
        '  "folder_abs_path": <absolute path from folders list above, or null>,\n'
        '  "file_query": <words user spoke for a specific file (not a '
        'folder), or null>,\n'
        '  "destination_folder_abs_path": <for move/copy: where to put the '
        'item. absolute path from folders list, or null>,\n'
        '  "new_name": <for rename/create: the new basename, or null>,\n'
        '  "contact_query": <name to look up in Contacts.app, or null>,\n'
        '  "message_text": <the actual message body the user wants to send, or null>,\n'
        '  "search_query": <for verb=search: the query to look up, or null>,\n'
        '  "raw_phrases": {\n'
        '    "app": <verbatim words user used for the app, or null>,\n'
        '    "folder": <verbatim words user used for the folder, or null>,\n'
        '    "file": <verbatim words user used for a specific file, or null>,\n'
        '    "contact": <verbatim words user used for the person, or null>\n'
        "  },\n"
        '  "notes": <one short sentence describing matches, for debugging>\n'
        "}\n\n"
        'Verb hints:\n'
        ' - "open the X video / X file / X document" → verb="open", '
        'file_query="X" (NOT folder_abs_path).\n'
        ' - "delete X" / "trash X" / "remove X" → verb="delete", '
        'file_query or folder_abs_path depending on what X is.\n'
        ' - "rename X to Y" → verb="rename", new_name="Y".\n'
        ' - "move X to Y" / "put X in Y" → verb="move", '
        'destination_folder_abs_path=<Y\'s path>.\n'
        ' - "copy X to Y" / "duplicate X" → verb="copy".\n'
        ' - "create folder X" / "new folder X on Desktop" → verb="create", '
        'new_name="X", folder_abs_path=<parent location>.\n'
        ' - "send hi to Garima" / "text mom" → verb="send", contact_query, '
        'message_text.\n'
        ' - "search for X" / "google X" → verb="search", search_query="X".\n\n'
        'If the transcript has no clear verb (e.g. just "the X video which '
        'is in downloads"), set verb to "other" and explain in `notes` that '
        'no action verb was found — DO NOT guess "open" or any verb.\n\n'
        "Set fields to null when not present in the request. Never invent "
        "paths or app names not in the lists above — leave the field null "
        "and explain in `notes`."
    )

    # Build a one-turn `contents` shape for logging consistency with the
    # main /chat call.
    resolver_contents_for_log = [types.Content(
        role="user",
        parts=[types.Part.from_text(text=resolver_prompt)],
    )]
    log_gemini_request(
        call_label="resolver",
        model=MODEL_RESOLVER,
        system_prompt="",
        contents=resolver_contents_for_log,
        extra_meta={"transcript": transcript},
    )

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=MODEL_RESOLVER,
            contents=resolver_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        raw_response_text = response.text or "{}"
        log_gemini_response(call_label="resolver", response_text=raw_response_text)
        print(f"🔎 Resolver raw response: {raw_response_text[:400]}")
        parsed = json.loads(raw_response_text)
        if not isinstance(parsed, dict):
            return {}
    except Exception as e:
        print(f"⚠️ Intent resolver failed: {e}")
        return {}

    # ── Deterministic basename override ───────────────────────────────────────
    # The LLM resolver tends to match by *semantic similarity* to folder
    # purpose descriptions ("tryown_media has 'media assets' in its purpose,
    # so 'new media' must mean tryown_media"). That's wrong: the user's
    # spoken phrase should match the path's *basename*, not its description.
    # We use Python to do the basename match deterministically and override
    # the LLM's guess if Python finds a different (or any) match.
    raw_phrases = parsed.get("raw_phrases") or {}
    folder_match_phrase_candidates = [
        raw_phrases.get("folder"),
        raw_phrases.get("file"),
    ]
    folder_match_phrase_candidates = [p for p in folder_match_phrase_candidates if p]

    # ── Hallucination guard ───────────────────────────────────────────────────
    # If the LLM returned a folder_abs_path whose basename has no plausible
    # relation to what the user actually said, null it out. Catches the case
    # where Flash Lite picks a familiar-sounding wiki folder (e.g. `kincare`)
    # for a spoken phrase that isn't in the wiki at all (e.g. `Xcode folder`).
    proposed_folder_path = parsed.get("folder_abs_path")
    if proposed_folder_path and folder_match_phrase_candidates:
        if not any(
            basename_matches_phrase(proposed_folder_path, phrase)
            for phrase in folder_match_phrase_candidates
        ):
            existing_notes = parsed.get("notes") or ""
            parsed["folder_abs_path"] = None
            parsed["notes"] = (
                f"{existing_notes} [scrubbed hallucinated folder "
                f"{proposed_folder_path!r} — basename doesn't relate to "
                f"spoken phrase(s) {folder_match_phrase_candidates}]"
            ).strip()
    # The LLM sometimes mis-classifies a folder phrase as a file (or
    # vice-versa). Try the basename match against BOTH raw_phrases.folder
    # and raw_phrases.file. If either matches a real folder in the wiki,
    # promote it to folder_abs_path — that's the canonical pointer.

    for candidate_phrase in folder_match_phrase_candidates:
        deterministic_folder_match = fuzzy_match_folder_basename(
            candidate_phrase, folders_body
        )
        if deterministic_folder_match and deterministic_folder_match != parsed.get("folder_abs_path"):
            previous_choice = parsed.get("folder_abs_path")
            parsed["folder_abs_path"] = deterministic_folder_match
            existing_notes = parsed.get("notes") or ""
            parsed["notes"] = (
                f"{existing_notes} [overrode LLM folder choice "
                f"{previous_choice!r} with deterministic basename match "
                f"{deterministic_folder_match!r} for phrase {candidate_phrase!r}]"
            ).strip()
            # If we promoted a `file` phrase to a folder match, clear the
            # file_query — the user named a folder, not a file.
            if candidate_phrase == raw_phrases.get("file"):
                parsed["file_query"] = None
            break

    raw_app_phrase = raw_phrases.get("app")
    if raw_app_phrase:
        deterministic_app_match = fuzzy_match_app_name(raw_app_phrase, apps_body)
        # Only override if the LLM's choice isn't already in the apps list
        # OR Python's match looks better. We trust the LLM for app category
        # words ("editor" → Antigravity via Preferences) so don't override
        # when the raw phrase is clearly a category, not a name.
        category_words = {"editor", "ide", "browser", "terminal", "shell",
                          "messaging", "text", "message"}
        is_category_phrase = (
            normalize_for_match(raw_app_phrase) in category_words
        )
        if (
            deterministic_app_match
            and not is_category_phrase
            and deterministic_app_match != parsed.get("app_canonical_name")
        ):
            previous_choice = parsed.get("app_canonical_name")
            parsed["app_canonical_name"] = deterministic_app_match
            existing_notes = parsed.get("notes") or ""
            parsed["notes"] = (
                f"{existing_notes} [overrode LLM app choice "
                f"{previous_choice!r} with deterministic name match "
                f"{deterministic_app_match!r} for phrase {raw_app_phrase!r}]"
            ).strip()

    # ── Post-validation: scrub any invented values ────────────────────────────
    # Final safety net for paths/apps that aren't literally in the catalog.
    proposed_folder_path = parsed.get("folder_abs_path")
    if proposed_folder_path and proposed_folder_path not in folders_body:
        invented = proposed_folder_path
        parsed["folder_abs_path"] = None
        existing_notes = parsed.get("notes") or ""
        parsed["notes"] = (
            f"{existing_notes} [scrubbed invented path {invented!r} — not in folders list]"
        ).strip()

    proposed_app_name = parsed.get("app_canonical_name")
    if proposed_app_name and proposed_app_name not in apps_body:
        invented = proposed_app_name
        parsed["app_canonical_name"] = None
        existing_notes = parsed.get("notes") or ""
        parsed["notes"] = (
            f"{existing_notes} [scrubbed invented app {invented!r} — not in apps list]"
        ).strip()

    return parsed


def normalize_for_match(text: str) -> str:
    """Lowercase, strip every non-alphanumeric char. 'New-Media' == 'newmedia'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def fuzzy_match_folder_basename(query: str, folders_body: str) -> str | None:
    """
    Given a user-spoken folder phrase and the folders catalog body (one
    `  /abs/path — purpose` line per folder), return the path whose
    basename best matches the query. None if nothing reasonable matches.

    Match preference (lower score = better):
      0. exact basename match after normalization
      1. query is a substring of basename
      2. basename is a substring of query
      3. token overlap (each word in query appears in basename)
    """
    if not query or not folders_body:
        return None
    query_normalized = normalize_for_match(query)
    if not query_normalized:
        return None

    query_tokens = [normalize_for_match(t) for t in re.findall(r"\w+", query)]
    query_tokens = [t for t in query_tokens if len(t) >= 2]

    candidates: list[tuple[int, str]] = []
    for raw_line in folders_body.splitlines():
        line = raw_line.strip()
        if not line.startswith("/"):
            continue
        path_only = line.split(" — ", 1)[0].strip()
        basename = path_only.rsplit("/", 1)[-1]
        basename_normalized = normalize_for_match(basename)
        if not basename_normalized:
            continue
        if query_normalized == basename_normalized:
            candidates.append((0, path_only))
        elif query_normalized in basename_normalized:
            candidates.append((1, path_only))
        elif basename_normalized in query_normalized:
            candidates.append((2, path_only))
        elif query_tokens and all(tok in basename_normalized for tok in query_tokens):
            candidates.append((3, path_only))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def basename_matches_phrase(folder_absolute_path: str, spoken_phrase: str) -> bool:
    """
    Does the basename of `folder_absolute_path` plausibly correspond to
    what the user said in `spoken_phrase`? Same generous matching as
    `fuzzy_match_folder_basename` (ignore case, spaces, underscores,
    hyphens; substring or token-AND), just for a single known path.

    Used to detect when the LLM picked a folder whose name is unrelated
    to what the user actually said (e.g. user said "Xcode folder", LLM
    returned `/.../kincare` because it's a prominent project folder
    that's actually in the wiki). That's a hallucination — we want to
    null it out and let Pro handle the unknown folder properly.
    """
    if not folder_absolute_path or not spoken_phrase:
        return False
    basename = folder_absolute_path.rsplit("/", 1)[-1]
    basename_normalized = normalize_for_match(basename)
    phrase_normalized = normalize_for_match(spoken_phrase)
    if not basename_normalized or not phrase_normalized:
        return False
    if phrase_normalized == basename_normalized:
        return True
    if phrase_normalized in basename_normalized:
        return True
    if basename_normalized in phrase_normalized:
        return True
    phrase_tokens = [
        normalize_for_match(t)
        for t in re.findall(r"\w+", spoken_phrase)
        if len(t) >= 2
    ]
    phrase_tokens = [t for t in phrase_tokens if t]
    if phrase_tokens and all(token in basename_normalized for token in phrase_tokens):
        return True
    return False


def fuzzy_match_app_name(query: str, apps_body: str) -> str | None:
    """
    Same logic as fuzzy_match_folder_basename, but over the comma-separated
    apps list. Returns the canonical app name from the catalog.
    """
    if not query or not apps_body:
        return None
    query_normalized = normalize_for_match(query)
    if not query_normalized:
        return None

    # Apps section format ends with a comma-separated list. Pull just the
    # tail of the section (the actual app names) and split on commas.
    list_segment = apps_body.split("\n\n")[-1]
    app_names = [name.strip() for name in list_segment.split(",") if name.strip()]
    query_tokens = [normalize_for_match(t) for t in re.findall(r"\w+", query)]
    query_tokens = [t for t in query_tokens if len(t) >= 2]

    candidates: list[tuple[int, str]] = []
    for canonical_app in app_names:
        canonical_normalized = normalize_for_match(canonical_app)
        if not canonical_normalized:
            continue
        if query_normalized == canonical_normalized:
            candidates.append((0, canonical_app))
        elif query_normalized in canonical_normalized:
            candidates.append((1, canonical_app))
        elif canonical_normalized in query_normalized:
            candidates.append((2, canonical_app))
        elif query_tokens and all(tok in canonical_normalized for tok in query_tokens):
            candidates.append((3, canonical_app))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: (pair[0], len(pair[1])))
    return candidates[0][1]


def has_useful_resolution(resolution: dict) -> bool:
    """A resolution is worth injecting only if at least one entity was matched."""
    for field in ("app_canonical_name", "folder_abs_path", "contact_query"):
        if resolution.get(field):
            return True
    return False


def smart_find_applescript_close(text: str, content_start: int) -> int | None:
    """
    Find the closing `]` of an `[APPLESCRIPT:...]` tag.

    Smarter than Clicky's `AgenticTagParser.findMatchingClose` — this
    version skips `]` characters inside single or double quotes, so
    AppleScript payloads like `do shell script "echo '...arr[i]...'"`
    don't get terminated mid-string. (Clicky's parser breaks on those.)

    Honors `\\]` escapes the same way Clicky does, for compatibility
    with action tags written to that convention.
    """
    idx = content_start
    in_single_quote = False
    in_double_quote = False
    while idx < len(text):
        ch = text[idx]
        if ch == "\\":
            idx += 2  # skip escaped char
            continue
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif ch == "]" and not in_single_quote and not in_double_quote:
            return idx
        idx += 1
    return None


def execute_applescript_server_side(script: str) -> tuple[bool, str]:
    """
    Run an AppleScript via osascript from the proxy process. Returns
    `(success, message)` — message is stdout on success, stderr on
    failure, capped to keep logs readable.

    Why this exists: Clicky is sandboxed
    (`com.apple.security.app-sandbox = true`, only
    `files.user-selected.read-write`). Any AppleScript that touches
    `~/Downloads`, `~/Desktop`, etc. fails with `error -54` or
    `Operation not permitted`. The proxy is a regular Python process
    started from the user's Terminal — no sandbox — so anything Clicky
    can't do, the proxy can.
    """
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as run_error:
        return False, f"subprocess error: {run_error}"
    if completed.returncode == 0:
        return True, (completed.stdout or "").strip()[:300]
    return False, f"rc={completed.returncode}: {(completed.stderr or '').strip()[:300]}"


def make_streaming_applescript_interceptor():
    """
    Returns `(process_chunk, flush_remaining)` — a stateful streaming
    filter for Gemini's SSE output.

    `process_chunk(new_text)` is a generator: feed it text chunks as
    they arrive, and it yields the substrings safe to forward to
    Clicky's SSE stream (everything that's not inside an
    `[APPLESCRIPT:...]` tag). For each complete `[APPLESCRIPT:...]` tag,
    the function runs the script server-side and DOES NOT yield it —
    Clicky never sees the tag, so it doesn't try to run it itself
    (and fail to, due to the sandbox).

    `flush_remaining()` returns any text held back at the end of the
    stream — call it once after all chunks are processed.
    """
    APPLESCRIPT_TAG_PREFIX = "[APPLESCRIPT:"
    pending = ""
    intercepted_actions: list[dict] = []

    def process_chunk(new_text: str):
        nonlocal pending
        pending += new_text
        while True:
            tag_start_index = pending.find(APPLESCRIPT_TAG_PREFIX)
            if tag_start_index == -1:
                # No tag. Flush all except the last (len(prefix)-1) chars in
                # case a partial prefix is forming at the end (e.g. "[APPLESC").
                safe_flush_count = max(0, len(pending) - (len(APPLESCRIPT_TAG_PREFIX) - 1))
                if safe_flush_count > 0:
                    yield pending[:safe_flush_count]
                    pending = pending[safe_flush_count:]
                return
            if tag_start_index > 0:
                # Stream the text before the tag.
                yield pending[:tag_start_index]
                pending = pending[tag_start_index:]
                continue
            # Tag is at position 0; locate its close.
            content_start = len(APPLESCRIPT_TAG_PREFIX)
            close_index = smart_find_applescript_close(pending, content_start)
            if close_index is None:
                # Tag isn't complete yet; wait for more chunks.
                return
            script = pending[content_start:close_index]
            success, message = execute_applescript_server_side(script)
            single_line_script_preview = script.replace("\n", " ⏎ ")[:120]
            if success:
                print(f"⚡ Intercepted AppleScript ok: {single_line_script_preview!r}")
            else:
                print(f"⚠️ Intercepted AppleScript failed: {message!r}  script={single_line_script_preview!r}")
            intercepted_actions.append({
                "script": script,
                "success": success,
                "message": message,
            })
            pending = pending[close_index + 1:]
            # Loop to look for more tags.

    def flush_remaining() -> str:
        """
        End-of-stream tail. If we're still mid-tag (Gemini was truncated
        with an unclosed `[APPLESCRIPT:...]`), DROP the partial — never
        speak script syntax aloud as TTS. Otherwise return what's left.
        """
        nonlocal pending
        leftover = pending
        pending = ""
        if leftover.lstrip().startswith(APPLESCRIPT_TAG_PREFIX):
            print(
                f"⚠️ Stream ended inside an [APPLESCRIPT:...] tag — "
                f"dropping {len(leftover)}B of unclosed script (likely Gemini hit max_output_tokens). "
                f"Spoken response will note the truncation instead."
            )
            return ""
        return leftover

    def get_intercepted_actions() -> list[dict]:
        return intercepted_actions

    return process_chunk, flush_remaining, get_intercepted_actions


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()


@app.post("/chat")
async def handle_chat(request: Request):
    """
    Streaming vision + text.
    Body: { systemPrompt, userPrompt, images: [{base64, mimeType, label}],
            history: [{userPlaceholder, assistantResponse}], model? }
    Returns: SSE stream  data: {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
    """
    body = await request.json()
    # Force the .env-configured model, ignoring what the (pre-built) Swift
    # app hardcodes in the request body. Lets us upgrade the model without
    # rebuilding Clicky.app.
    requested_model = body.get("model")
    model = MODEL_TEXT
    if requested_model and requested_model != MODEL_TEXT:
        print(f"⚙️  Overriding requested model {requested_model!r} → {MODEL_TEXT!r} (.env wins)")
    system = body.get("systemPrompt", "") or body.get("system_instruction", {}).get("parts", [{}])[0].get("text", "")
    prompt = body.get("userPrompt", "")

    # Pre-extract the latest user transcript + recent turn history so the
    # graph-driven personal-context builder can see them before we call
    # build_kb_augmentation. Both forms (Vertex `contents` array, legacy
    # userPrompt+history) are flattened to {role, text} dicts here.
    early_transcript, early_history = _extract_transcript_and_history(body, prompt)

    # Augment the system prompt with apps catalog + folders wiki + personal
    # context. The April-22 Clicky binary doesn't add these — it sends only
    # its base prompt — so the proxy supplies them. Re-read on every call so
    # `kb_compile.py` runs and wiki re-indexes are picked up live.
    base_system_size = len(system)
    augmentation = build_kb_augmentation(
        transcript=early_transcript,
        history_messages=early_history,
    )
    if augmentation:
        system = system + augmentation
        print(
            f"📚 KB augmentation: base={base_system_size}B + "
            f"+{len(augmentation)}B = {len(system)}B total"
        )

    # Swift app sends pre-built Vertex AI `contents` array with inline images.
    # Rebuild as google-genai Content objects so the SDK can forward them.
    raw_contents = body.get("contents", [])
    contents = []
    for turn in raw_contents:
        role = turn.get("role", "user")
        parts = []
        for p in turn.get("parts", []):
            if "text" in p:
                parts.append(types.Part.from_text(text=p["text"]))
            elif "inlineData" in p:
                parts.append(types.Part.from_bytes(
                    data=base64.b64decode(p["inlineData"]["data"]),
                    mime_type=p["inlineData"].get("mimeType", "image/jpeg"),
                ))
        if parts:
            contents.append(types.Content(role=role, parts=parts))

    # Fallback: old Cloudflare Worker format (userPrompt + images + history)
    if not contents:
        images  = body.get("images", [])
        history = body.get("history", [])
        for turn in history:
            contents.append(types.Content(role="user",  parts=[types.Part.from_text(text=turn["userPlaceholder"])]))
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=turn["assistantResponse"])]))
        current_parts = []
        for img in images:
            current_parts.append(types.Part.from_bytes(
                data=base64.b64decode(img["base64"]),
                mime_type=img.get("mimeType", "image/jpeg"),
            ))
            current_parts.append(types.Part.from_text(text=img.get("label", "")))
        current_parts.append(types.Part.from_text(text=prompt))
        contents.append(types.Content(role="user", parts=current_parts))

    # gemini-2.5-pro doesn't support response_modalities — only set it for flash models
    use_modalities = "pro" not in model
    config = types.GenerateContentConfig(
        system_instruction=system,
        # 16384 fits multi-step tasks (writing a full C++ file, multi-tag
        # action sequences). 4096 was hitting MAX_TOKENS mid-string for
        # heredoc-style file writes, leaving Clicky with an unclosed
        # [APPLESCRIPT:...] tag that the interceptor couldn't process.
        max_output_tokens=16384,
        response_modalities=["TEXT"] if use_modalities else None,
    )

    # Extract last user text for logging
    last_prompt = prompt
    if raw_contents:
        for p in reversed(raw_contents[-1].get("parts", [])):
            if "text" in p and p["text"].strip():
                last_prompt = p["text"].strip()
                break
    print(f"💬 Chat: model={model} prompt={repr(last_prompt[:80])}")

    # ── LLM-based intent resolution (pre-pass) ────────────────────────────────
    # Hand the noisy transcript to a fast Flash Lite call along with the
    # apps and folders sections of the system prompt, get back a structured
    # JSON resolution, and inject it into the user's last turn so the main
    # model has the fuzzy-matched entities ready to use verbatim.
    resolution = {}
    if last_prompt and contents and contents[-1].role == "user":
        resolution = resolve_intent_with_llm(last_prompt, system)

        # Inject the resolution into the prompt for the main model. Gemini owns
        # the decision of what to do — including emitting action tags and
        # handling multi-step requests like "open X and write Y". The proxy
        # never short-circuits server-side, because that drops any intent the
        # shortcut doesn't understand (e.g. message_text after a verb=open).
        if has_useful_resolution(resolution):
            print(f"🎯 Resolved entities: {json.dumps(resolution, ensure_ascii=False)}")
            resolution_block = (
                "\n\n── PRE-RESOLVED ENTITIES ──\n"
                "An LLM resolver has already fuzzy-matched the named entities "
                "in this request against the INSTALLED APPS, KNOWN FOLDERS, "
                "and Preferences sections of your system prompt.\n\n"
                "Rules for using this:\n"
                "1. When a field has a non-null value, use it VERBATIM. Do not "
                "re-derive, re-spell, or re-search.\n"
                "2. When a field is null, the resolver could not confidently "
                "match. Either ask the user to clarify (preferred when the "
                "intent is ambiguous) or fall back to `find` / `mdfind` for "
                "that one entity only.\n"
                "3. The `notes` field explains the resolver's reasoning, "
                "including any invented values that were scrubbed.\n\n"
                f"{json.dumps(resolution, indent=2, ensure_ascii=False)}"
            )
            contents[-1].parts.append(types.Part.from_text(text=resolution_block))
        else:
            notes = resolution.get("notes") if isinstance(resolution, dict) else None
            print(f"🎯 Resolver: no entities matched"
                  + (f" — {notes}" if notes else ""))

    # Drop the screenshot from the request unless this turn actually needs
    # vision. Saves ~250KB bandwidth + image-input tokens per request when
    # the task is purely text-driven (open / delete / send / etc.).
    is_followup = is_screenshot_followup_turn(last_prompt)
    needs_vision = initial_request_likely_needs_vision(last_prompt)
    if not (is_followup or needs_vision):
        contents, stripped_image_count = strip_images_from_contents(contents)
        if stripped_image_count > 0:
            print(f"📸→🗑️  Stripped {stripped_image_count} image(s) from request "
                  f"(no vision keywords in transcript; Gemini can request via [SCREENSHOT] if needed)")
    elif is_followup:
        print(f"📸 Keeping image — screenshot-driven follow-up turn")
    elif needs_vision:
        print(f"📸 Keeping image — transcript references screen content")

    log_gemini_request(
        call_label="main",
        model=model,
        system_prompt=system,
        contents=contents,
        extra_meta={"last_user_text": last_prompt},
    )

    full_response = []
    process_chunk, flush_remaining, get_intercepted_actions = (
        make_streaming_applescript_interceptor()
    )
    last_finish_reason: list = [None]  # mutable cell for the closure

    def emit_sse_text(text_to_emit: str):
        """Wrap a plain string in the SSE chunk shape the Swift app expects."""
        payload = json.dumps({"candidates": [{"content": {"parts": [{"text": text_to_emit}]}}]})
        return f"data: {payload}\n\n"

    def sse_generator():
        try:
            for chunk in GEMINI_CLIENT.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                text = chunk.text
                if text:
                    full_response.append(text)
                    # Stream filter: yield text not inside [APPLESCRIPT:...] tags;
                    # execute scripts server-side and strip them from output.
                    for safe_text_segment in process_chunk(text):
                        yield emit_sse_text(safe_text_segment)
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        last_finish_reason[0] = candidate.finish_reason
                        if not text:
                            print(f"⚠️ Chat chunk: no text, finish_reason={candidate.finish_reason}")
            # End of stream: flush any text the interceptor was holding back
            # (e.g. text after the last tag, or a partial-prefix tail).
            leftover = flush_remaining()
            if leftover:
                yield emit_sse_text(leftover)
            # If Gemini truncated us at the token limit, the user needs to
            # know — otherwise the spoken response just trails off and they
            # have no idea why nothing happened. Append a clear status.
            finish_reason_value = last_finish_reason[0]
            finish_reason_name = (
                getattr(finish_reason_value, "name", None)
                or str(finish_reason_value or "")
            ).upper()
            if "MAX_TOKENS" in finish_reason_name:
                truncation_note = (
                    " hmm, i hit my response limit before finishing that. "
                    "say 'continue' and i'll pick up where i left off, or "
                    "break the task into smaller steps."
                )
                print(f"⚠️ Gemini finish_reason=MAX_TOKENS — appending truncation note to spoken output")
                yield emit_sse_text(truncation_note)
        except Exception as e:
            print(f"❌ Chat error: {e}")
            log_gemini_response(call_label="main", response_text=f"<<< ERROR: {e} >>>")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        else:
            response_text = "".join(full_response)
            intercepted = get_intercepted_actions()
            if intercepted:
                ok_count = sum(1 for a in intercepted if a["success"])
                fail_count = len(intercepted) - ok_count
                print(f"⚡ Intercepted {len(intercepted)} AppleScript action(s) — {ok_count} ok, {fail_count} failed (none reached Clicky)")
            log_gemini_response(
                call_label="main",
                response_text=response_text,
                extra_meta={
                    "intercepted_applescript_count": len(intercepted),
                    "intercepted_applescript_failures": [a for a in intercepted if not a["success"]],
                } if intercepted else None,
            )
        finally:
            response_text = "".join(full_response)
            print(f"🤖 Gemini: {repr(response_text[:200])}")

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.post("/tts")
async def handle_tts(request: Request):
    """
    Text → speech WAV.
    Body: { text, voiceName? }
    Returns: audio/wav bytes
    """
    body      = await request.json()
    text      = body.get("text", "")
    voice     = body.get("voiceName") or TTS_VOICE

    response = GEMINI_CLIENT.models.generate_content(
        model=MODEL_TTS,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )

    try:
        candidate = response.candidates[0]
        part = candidate.content.parts[0]
        if not part.inline_data or not part.inline_data.data:
            print(f"❌ TTS: empty audio. finish_reason={candidate.finish_reason}")
            return JSONResponse({"error": "empty audio"}, status_code=502)
        # The google-genai SDK returns raw bytes in inline_data.data (already decoded)
        pcm = part.inline_data.data
        wav = build_wav(pcm)
    except Exception as e:
        print(f"❌ TTS error: {e}  response={response}")
        return JSONResponse({"error": str(e)}, status_code=502)

    print(f"🔊 TTS: {len(text)} chars → {len(wav)//1024}KB WAV  voice={voice}")
    return Response(content=wav, media_type="audio/wav")


@app.post("/transcribe")
async def handle_transcribe(request: Request):
    """
    WAV bytes → transcript.
    Body: JSON { audio: base64, keyterms: [...] }  OR  raw audio/octet-stream (WAV)
    Returns: { transcript: string }
    """
    content_type = request.headers.get("content-type", "")
    keyterms: list[str] = []

    if "application/json" in content_type:
        body = await request.json()
        wav_bytes = base64.b64decode(body.get("audio", ""))
        keyterms = body.get("keyterms", [])
    else:
        # Legacy: raw WAV bytes
        wav_bytes = await request.body()

    # Build transcription prompt — prepend keyterms hint so Gemini recognises
    # app names, project names, and tech terms instead of mishearing them.
    if keyterms:
        hint = "Possible words that may appear: " + ", ".join(keyterms) + "."
        prompt = (
            f"{hint}\n"
            "Transcribe the speech in this audio clip exactly as spoken. "
            "Return only the transcript text, nothing else."
        )
    else:
        prompt = "Transcribe the speech in this audio clip. Return only the transcript text, nothing else."

    response = GEMINI_CLIENT.models.generate_content(
        model=MODEL_STT,
        contents=[
            prompt,
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
        ],
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )

    transcript = (response.text or "").strip()
    print(f"🎙️ STT: {len(wav_bytes)//1024}KB WAV → {repr(transcript[:80])}")
    return JSONResponse({"transcript": transcript})


@app.get("/health")
async def health():
    return {"ok": True}


if __name__ == "__main__":
    print("\n🚀 Clicky local proxy starting on http://localhost:8081\n")
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
