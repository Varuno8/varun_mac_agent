"""
Local proxy server — replaces the Cloudflare Worker.
Runs on http://localhost:8080 and proxies the three routes Clicky uses:

  POST /chat        → Vertex AI streaming vision (SSE)
  POST /tts         → Vertex AI Gemini TTS → WAV bytes
  POST /transcribe  → Vertex AI Gemini STT → { transcript }

Auth: service-account credentials from .env (same format as Agent_Swarm).
"""

import base64
import hashlib
import json
import logging
import re
import struct
import subprocess
import threading
import time
import traceback
import uuid
import wave
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from google.genai import types
from google import genai
from google.oauth2 import service_account

try:
    from google.cloud import bigquery as gcp_bigquery
except Exception as bq_import_err:  # noqa: BLE001 — keep proxy bootable without BQ
    gcp_bigquery = None
    print(f"⚠️ google-cloud-bigquery not importable: {bq_import_err}")

# Dual-track refactor: schema validation + tiered memory live in parallel
# modules so this file doesn't grow further. Guarded so a missing dep can't
# brick the existing proxy flow (legacy bracket-tag path is unaffected).
try:
    from agent_memory import core_memory_read_block, core_memory_seed_if_empty, wiki_bootstrap
    from agent_runtime import decode_bracket_tag
    from agent_wiki_summarizer import maybe_write_wiki
    _DUAL_TRACK_AVAILABLE = True
except Exception as dual_track_err:  # noqa: BLE001
    core_memory_read_block = lambda: ""  # noqa: E731
    core_memory_seed_if_empty = lambda: 0  # noqa: E731
    wiki_bootstrap = lambda: 0  # noqa: E731
    decode_bracket_tag = lambda tag: None  # noqa: E731
    maybe_write_wiki = lambda **kwargs: None  # noqa: E731
    _DUAL_TRACK_AVAILABLE = False
    print(f"⚠️ dual-track modules not importable (legacy path still works): {dual_track_err}")

# Pydantic-AI orchestrator (Phase 1 wiring). Imported lazily-guarded so a
# missing pydantic-ai install doesn't break the proxy — orchestrator hints
# just become a no-op when the import fails. Opt-in by env flag during
# rollout: set MICKY_ORCHESTRATOR=1 to enable. Default off until measured.
try:
    from agent_orchestrator import run_orchestrator_prepass as _run_orchestrator_prepass
    _ORCHESTRATOR_IMPORTABLE = True
except Exception as orchestrator_import_err:  # noqa: BLE001
    _run_orchestrator_prepass = None  # type: ignore[assignment]
    _ORCHESTRATOR_IMPORTABLE = False
    print(f"⚠️ orchestrator not importable (pre-pass disabled): {orchestrator_import_err}")

_ORCHESTRATOR_ENABLED = (
    _ORCHESTRATOR_IMPORTABLE
    and os.environ.get("MICKY_ORCHESTRATOR", "0").strip().lower() in ("1", "true", "yes", "on")
)


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

# Per-turn structured metrics — one JSON object per /chat request, appended
# as a single line. Designed for `tail -f`, `jq`, or quick pandas analysis
# to answer "where is time going" and (later) "what's the cache hit rate".
TURN_METRICS_LOG_PATH = Path(__file__).parent / "metrics.jsonl"

# Maximum size before either log rotates. The wire log writes the full system
# prompt (~70 KB) per request, so it grows fast — cap it at 20 MB and keep 3
# backups (80 MB total worst-case). Metrics lines are <1 KB each, so 10 MB
# holds roughly 10k turns.
GEMINI_DEBUG_LOG_MAX_BYTES = 20_000_000
TURN_METRICS_LOG_MAX_BYTES = 10_000_000
LOG_BACKUP_COUNT           = 3


def build_rotating_logger(logger_name: str, log_file_path: Path, maximum_bytes: int) -> logging.Logger:
    """
    Build (or return the existing) rotating-file logger writing to
    `log_file_path`. Each handler is attached only once even across module
    re-imports, so re-running the proxy doesn't accumulate duplicate handlers.
    Format is the bare message — callers are expected to pre-format their
    payloads (multi-line text for the wire log, single-line JSON for metrics).
    """
    rotating_logger = logging.getLogger(logger_name)
    if rotating_logger.handlers:
        return rotating_logger
    rotating_logger.setLevel(logging.INFO)
    rotating_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=maximum_bytes,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    rotating_handler.setFormatter(logging.Formatter("%(message)s"))
    rotating_logger.addHandler(rotating_handler)
    rotating_logger.propagate = False
    return rotating_logger


GEMINI_DEBUG_LOGGER = build_rotating_logger(
    logger_name="gemini_debug",
    log_file_path=GEMINI_DEBUG_LOG_PATH,
    maximum_bytes=GEMINI_DEBUG_LOG_MAX_BYTES,
)
TURN_METRICS_LOGGER = build_rotating_logger(
    logger_name="turn_metrics",
    log_file_path=TURN_METRICS_LOG_PATH,
    maximum_bytes=TURN_METRICS_LOG_MAX_BYTES,
)


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
print(f"📒 Full Gemini wire log: {GEMINI_DEBUG_LOG_PATH}  "
      f"(rotates at {GEMINI_DEBUG_LOG_MAX_BYTES // 1_000_000}MB × {LOG_BACKUP_COUNT} backups)")
print(f"   (tail -f this file to see every request/response in detail)")
print(f"📊 Per-turn metrics (JSONL): {TURN_METRICS_LOG_PATH}  "
      f"(rotates at {TURN_METRICS_LOG_MAX_BYTES // 1_000_000}MB × {LOG_BACKUP_COUNT} backups)")
print(f"   tail -f \"{TURN_METRICS_LOG_PATH}\" | jq .   for live request stats")


# ── BigQuery telemetry ────────────────────────────────────────────────────────
#
# Three append-only tables in dataset `micky_agent` (US, same project):
#   turns        — one row per /chat call (transcript, response, tokens, timing)
#   task_events  — task-slot transitions emitted by Gemini via [TASK_*] tags
#   errors       — anything that fails in any component
#
# Writes are best-effort, fire-and-forget on a small thread pool. A BQ outage
# never blocks the user-facing /chat response. The local metrics.jsonl file
# stays as the durable fallback.

BQ_DATASET = ENV.get("BQ_DATASET", "micky_agent").strip() or "micky_agent"
BQ_TABLE_TURNS = "turns"
BQ_TABLE_TASK_EVENTS = "task_events"
BQ_TABLE_ERRORS = "errors"
BQ_DISABLED = ENV.get("BQ_DISABLED", "").strip().lower() in ("1", "true", "yes", "on")

BQ_CLIENT = None
if gcp_bigquery is not None and not BQ_DISABLED:
    try:
        BQ_CLIENT = gcp_bigquery.Client(project=PROJECT_ID, credentials=CREDENTIALS)
        print(f"📡 BigQuery telemetry ready: {PROJECT_ID}.{BQ_DATASET}")
    except Exception as bq_init_err:  # noqa: BLE001
        print(f"⚠️ BigQuery client init failed (telemetry disabled): {bq_init_err}")
        BQ_CLIENT = None
else:
    print(f"📡 BigQuery telemetry disabled "
          f"(BQ_DISABLED={BQ_DISABLED}, lib_loaded={gcp_bigquery is not None})")

BQ_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bq-writer")
SESSION_ID = str(uuid.uuid4())
TURN_INDEX_LOCK = threading.Lock()
_TURN_INDEX = {"value": 0}


def next_turn_index() -> int:
    with TURN_INDEX_LOCK:
        index = _TURN_INDEX["value"]
        _TURN_INDEX["value"] = index + 1
        return index


def _bq_table_ref(table_name: str) -> str:
    return f"{PROJECT_ID}.{BQ_DATASET}.{table_name}"


# JSON-typed columns per table. `insert_rows_json` interprets a Python list as
# a REPEATED field and rejects it for non-repeated JSON columns, so we
# pre-serialize these to strings on the way in. Single source of truth — keep
# in sync with the `JSON` columns declared in /tmp/*_schema.json.
_BQ_JSON_COLUMNS: dict[str, frozenset[str]] = {
    BQ_TABLE_TURNS: frozenset({"response_action_tags", "extras"}),
    BQ_TABLE_TASK_EVENTS: frozenset(
        {"steps_done", "steps_remaining", "blockers", "extras"}
    ),
    BQ_TABLE_ERRORS: frozenset({"context", "extras"}),
}


def _stringify_json_columns(table_name: str, row: dict) -> dict:
    """Return a copy of `row` with all JSON-typed columns serialized to
    strings, leaving everything else untouched. None stays None — the column
    is NULLABLE so an absent value is fine."""
    json_columns = _BQ_JSON_COLUMNS.get(table_name, frozenset())
    if not json_columns:
        return row
    out = dict(row)
    for column_name in json_columns:
        value = out.get(column_name)
        if value is None or isinstance(value, str):
            continue
        try:
            out[column_name] = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as serialize_err:
            print(
                f"⚠️ BQ JSON column {table_name}.{column_name} unserializable, "
                f"sending null: {serialize_err}"
            )
            out[column_name] = None
    return out


def _bq_insert_sync(table_name: str, rows: list[dict]) -> None:
    """Synchronous insert. Runs on the BQ_EXECUTOR thread, never on the event loop."""
    if BQ_CLIENT is None or not rows:
        return
    prepared_rows = [_stringify_json_columns(table_name, row) for row in rows]
    try:
        insert_errors = BQ_CLIENT.insert_rows_json(_bq_table_ref(table_name), prepared_rows)
        if insert_errors:
            print(f"⚠️ BQ insert {table_name} returned per-row errors: {insert_errors}")
    except Exception as bq_insert_err:  # noqa: BLE001
        # Falling back to local stderr — we never raise from telemetry.
        print(f"⚠️ BQ insert {table_name} raised: {bq_insert_err}")


def bq_insert_async(table_name: str, rows: list[dict]) -> None:
    """Fire-and-forget enqueue onto the BQ writer pool.

    Errors are caught inside `_bq_insert_sync` so a future never blows up. We
    still wrap submit() because the executor may be saturated or shut down."""
    if BQ_CLIENT is None or not rows:
        return
    try:
        BQ_EXECUTOR.submit(_bq_insert_sync, table_name, list(rows))
    except Exception as submit_err:  # noqa: BLE001
        print(f"⚠️ BQ submit {table_name} failed: {submit_err}")


def report_error_to_bq(
    component: str,
    exception: BaseException | None = None,
    *,
    message: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    context: dict | None = None,
) -> None:
    """Best-effort error logging into the `errors` table."""
    error_row = {
        "ts": _now_utc_iso(),
        "session_id": session_id or SESSION_ID,
        "turn_id": turn_id,
        "component": component,
        "error_type": type(exception).__name__ if exception else None,
        "message": message or (str(exception) if exception else None),
        "stack": (
            "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            if exception else None
        ),
        "context": context if context else None,
        "extras": None,
    }
    bq_insert_async(BQ_TABLE_ERRORS, [error_row])


# ── Task slot (MemGPT-style persistent goal) ─────────────────────────────────
#
# A small JSON blob persisted to task_state.json. Injected into every system
# prompt so Gemini remembers what it was working on even when conversation
# history is trimmed. Gemini owns the state machine entirely — the proxy only
# stores what Gemini emits via these tags:
#
#   [TASK_UPDATE:{"goal":"...","steps_remaining":["..."]}]   start a task
#   [TASK_UPDATE:{"step_done":"..."}]                        log progress
#   [TASK_DONE]                                              clear the slot
#   [TASK_ABANDON:reason]                                    drop with note
#
# This honors the `Proxy stays minimal — Gemini is the brain` rule: the proxy
# never decides when a task starts or ends, it just persists the slot.

TASK_STATE_PATH = Path(__file__).parent / "task_state.json"


def _now_utc_iso() -> str:
    """Always UTC, microsecond precision, BigQuery TIMESTAMP-friendly."""
    return datetime.now(timezone.utc).isoformat()


def load_task_state() -> dict:
    if not TASK_STATE_PATH.exists():
        return {}
    try:
        body = TASK_STATE_PATH.read_text(encoding="utf-8").strip()
        return json.loads(body) if body else {}
    except Exception as task_load_err:  # noqa: BLE001
        print(f"⚠️ task_state.json unreadable, treating as empty: {task_load_err}")
        return {}


def save_task_state(state: dict) -> None:
    try:
        TASK_STATE_PATH.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as task_save_err:  # noqa: BLE001
        print(f"⚠️ task_state.json write failed: {task_save_err}")
        report_error_to_bq("task_state", task_save_err)


def build_task_slot_section() -> str:
    """Render the `── CURRENT TASK SLOT ──` block for the system prompt.

    Always present (even when empty) so Gemini knows the slot exists and the
    update protocol is available."""
    state = load_task_state()
    if not state or not state.get("goal"):
        body = "<no active task>"
    else:
        body = json.dumps(state, indent=2, ensure_ascii=False)
    return (
        "\n\n── CURRENT TASK SLOT ──\n"
        "This is your persistent task scratchpad. It survives conversation "
        "history compaction — read it on every turn to know what you were "
        "doing if the user said \"continue\" or interrupted you.\n\n"
        "Update it with bracket tags (stripped from speech, like APPLESCRIPT):\n"
        "  [TASK_UPDATE:{\"goal\":\"...\",\"steps_remaining\":[\"...\"]}]   "
        "start a new multi-step task\n"
        "  [TASK_UPDATE:{\"step_done\":\"...\"}]                        "
        "after finishing a step\n"
        "  [TASK_DONE]                                              "
        "task complete, clear the slot\n"
        "  [TASK_ABANDON:reason]                                    "
        "drop the task with a note\n\n"
        "When the user interrupts mid-task, decide whether to pause "
        "(do nothing — slot persists), abandon ([TASK_ABANDON:...]), or "
        "merge (treat the new request as the next step).\n\n"
        f"Current state:\n{body}\n"
    )


_TASK_UPDATE_RE = re.compile(r"\[TASK_UPDATE:(\{.*?\})\]", re.DOTALL)
_TASK_DONE_RE = re.compile(r"\[TASK_DONE\]")
_TASK_ABANDON_RE = re.compile(r"\[TASK_ABANDON(?::([^\]]*))?\]")


def apply_task_tags_to_state(response_text: str, turn_id: str) -> list[dict]:
    """Parse [TASK_*] tags out of a completed response, mutate task_state.json,
    and return a list of `task_events` rows ready for BigQuery.

    Order of operations matters: TASK_UPDATEs first (they may create or
    update), then TASK_DONE (clears), then TASK_ABANDON (also clears).
    Same response should not contain both DONE and ABANDON; if it does,
    DONE wins because the goal was met."""
    events: list[dict] = []
    state = load_task_state()
    now = _now_utc_iso()

    for match in _TASK_UPDATE_RE.finditer(response_text):
        raw_json = match.group(1)
        try:
            update = json.loads(raw_json)
        except json.JSONDecodeError as parse_err:
            events.append({
                "ts": now,
                "session_id": SESSION_ID,
                "turn_id": turn_id,
                "task_id": state.get("task_id") or str(uuid.uuid4()),
                "event_type": "update_parse_failed",
                "goal": state.get("goal"),
                "steps_done": None,
                "steps_remaining": None,
                "blockers": None,
                "notes": f"JSON parse error: {parse_err}; raw={raw_json[:200]}",
                "extras": None,
            })
            continue

        is_new_goal = (
            "goal" in update
            and update["goal"]
            and update["goal"] != state.get("goal")
        )
        if is_new_goal:
            state = {
                "task_id": str(uuid.uuid4()),
                "goal": update["goal"],
                "steps_done": [],
                "steps_remaining": list(update.get("steps_remaining") or []),
                "blockers": list(update.get("blockers") or []),
                "started_at": now,
                "last_updated": now,
            }
            events.append({
                "ts": now,
                "session_id": SESSION_ID,
                "turn_id": turn_id,
                "task_id": state["task_id"],
                "event_type": "created",
                "goal": state["goal"],
                "steps_done": state["steps_done"],
                "steps_remaining": state["steps_remaining"],
                "blockers": state["blockers"],
                "notes": None,
                "extras": None,
            })
            continue

        # Update of existing task — only meaningful if a task exists
        if not state.get("goal"):
            continue
        if "step_done" in update and update["step_done"]:
            state.setdefault("steps_done", []).append(
                {"step": update["step_done"], "ts": now}
            )
            try:
                state.get("steps_remaining", []).remove(update["step_done"])
            except (ValueError, AttributeError):
                pass
        if "steps_remaining" in update:
            state["steps_remaining"] = list(update["steps_remaining"] or [])
        if "blockers" in update:
            state["blockers"] = list(update["blockers"] or [])
        state["last_updated"] = now
        events.append({
            "ts": now,
            "session_id": SESSION_ID,
            "turn_id": turn_id,
            "task_id": state.get("task_id"),
            "event_type": "updated",
            "goal": state.get("goal"),
            "steps_done": state.get("steps_done", []),
            "steps_remaining": state.get("steps_remaining", []),
            "blockers": state.get("blockers", []),
            "notes": None,
            "extras": None,
        })

    if _TASK_DONE_RE.search(response_text) and state.get("goal"):
        events.append({
            "ts": now,
            "session_id": SESSION_ID,
            "turn_id": turn_id,
            "task_id": state.get("task_id"),
            "event_type": "completed",
            "goal": state.get("goal"),
            "steps_done": state.get("steps_done", []),
            "steps_remaining": state.get("steps_remaining", []),
            "blockers": state.get("blockers", []),
            "notes": None,
            "extras": None,
        })
        state = {}
    else:
        abandon_match = _TASK_ABANDON_RE.search(response_text)
        if abandon_match and state.get("goal"):
            reason = (abandon_match.group(1) or "").strip() or None
            events.append({
                "ts": now,
                "session_id": SESSION_ID,
                "turn_id": turn_id,
                "task_id": state.get("task_id"),
                "event_type": "abandoned",
                "goal": state.get("goal"),
                "steps_done": state.get("steps_done", []),
                "steps_remaining": state.get("steps_remaining", []),
                "blockers": state.get("blockers", []),
                "notes": reason,
                "extras": None,
            })
            state = {}

    save_task_state(state)
    return events


def parse_response_action_tags(response_text: str) -> list[str]:
    """Extract every bracket-tag from the spoken response. Used to populate
    the `response_action_tags` JSON column on `turns`. Best-effort only —
    a malformed tag becomes a single string entry, not an exception."""
    if not response_text:
        return []
    return re.findall(r"\[[A-Z_]+(?::[^\]]*)?\]", response_text)


def strip_action_tags_for_storage(response_text: str) -> str:
    """Best-effort: strip bracket tags so the stored `response_text` matches
    roughly what the user heard. We do not rebuild perfect TTS output here —
    just remove the obvious tag bodies."""
    if not response_text:
        return ""
    return re.sub(r"\[[A-Z_]+(?::[^\]]*?)?\]", "", response_text).strip()


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
    `[image: <mime>, ~<KB>]` rather than dumped. Built up as a single
    string and emitted in one logger call so the rotating handler sees an
    atomic record (and the file rotates cleanly mid-record-stream).
    """
    try:
        log_lines: list[str] = []
        log_lines.append("\n" + "=" * 88)
        log_lines.append(f"📤 → GEMINI  [{call_label}]  {datetime.now().isoformat(timespec='seconds')}")
        log_lines.append(f"model: {model}")
        if extra_meta:
            for meta_key, meta_value in extra_meta.items():
                log_lines.append(f"{meta_key}: {meta_value}")
        if system_prompt:
            log_lines.append(f"\n--- SYSTEM_INSTRUCTION ({len(system_prompt)} bytes) ---")
            log_lines.append(system_prompt)
        log_lines.append(f"\n--- CONTENTS ({len(contents)} turn(s)) ---")
        for turn_index, turn in enumerate(contents):
            role = getattr(turn, "role", "?")
            turn_parts = getattr(turn, "parts", []) or []
            log_lines.append(f"\n[turn {turn_index}, role={role}]")
            for part in turn_parts:
                if hasattr(part, "text") and part.text:
                    log_lines.append(f"  TEXT ({len(part.text)} bytes):")
                    for text_line in part.text.splitlines():
                        log_lines.append(f"    {text_line}")
                elif hasattr(part, "inline_data") and part.inline_data:
                    mime_type = part.inline_data.mime_type or "?"
                    size_in_kilobytes = len(part.inline_data.data) // 1024
                    log_lines.append(f"  IMAGE: mime={mime_type}, ~{size_in_kilobytes}KB")
                else:
                    log_lines.append("  PART (unknown type)")
        log_lines.append("=" * 88)
        GEMINI_DEBUG_LOGGER.info("\n".join(log_lines))
    except Exception as debug_log_error:
        print(f"⚠️ Debug log write failed: {debug_log_error}")


def log_gemini_response(call_label: str, response_text: str, extra_meta: dict | None = None) -> None:
    """Append the full Gemini response text to gemini_debug.log."""
    try:
        log_lines: list[str] = []
        log_lines.append(f"\n📥 ← GEMINI  [{call_label}]  {datetime.now().isoformat(timespec='seconds')}")
        if extra_meta:
            for meta_key, meta_value in extra_meta.items():
                log_lines.append(f"{meta_key}: {meta_value}")
        log_lines.append(f"--- RESPONSE ({len(response_text)} bytes) ---")
        for response_line in response_text.splitlines():
            log_lines.append(f"  {response_line}")
        log_lines.append("=" * 88)
        GEMINI_DEBUG_LOGGER.info("\n".join(log_lines))
    except Exception as debug_log_error:
        print(f"⚠️ Debug log write failed: {debug_log_error}")


def record_turn_metrics(turn_metrics: dict) -> None:
    """
    Emit one JSON line per /chat request to metrics.jsonl AND best-effort
    insert into BigQuery `turns`. JSONL stays as the durable fallback so we
    never lose a record to a transient BQ outage. `default=str` handles
    datetimes / Gemini enum types without crashing the request path.
    """
    try:
        TURN_METRICS_LOGGER.info(json.dumps(turn_metrics, default=str, ensure_ascii=False))
    except Exception as metrics_log_error:
        print(f"⚠️ Metrics log write failed: {metrics_log_error}")

    bq_insert_async(BQ_TABLE_TURNS, [turn_metrics_to_bq_row(turn_metrics)])


# Whitelist of `turns` columns we expect from `turn_metrics`. Any field on
# turn_metrics not in this map lands in `extras` (the JSON escape hatch),
# so adding new metrics in the proxy never breaks the BQ insert.
_TURN_BQ_COLUMNS = {
    "ts", "session_id", "turn_id", "turn_index",
    "model", "transcript", "transcript_chars", "history_turns",
    "response_text", "response_action_tags",
    "base_system_bytes", "augmentation_bytes", "system_total_bytes",
    "image_parts_in_request", "image_parts_kept", "image_parts_stripped",
    "vision_decision",
    "resolver_elapsed_ms", "resolver_prompt_tokens", "resolver_output_tokens",
    "resolver_useful", "resolver_skipped", "resolver_error",
    "main_ttft_ms", "main_total_ms",
    "main_prompt_tokens", "main_output_tokens",
    "main_cached_tokens", "main_total_tokens",
    "intercepted_applescripts", "intercepted_failures",
    "finish_reason", "max_tokens_truncated", "total_handler_ms", "error",
}


def turn_metrics_to_bq_row(turn_metrics: dict) -> dict:
    """Project `turn_metrics` onto the `turns` table schema. Unknown fields are
    stuffed into `extras` so we never lose data and never break on an unknown
    column. Timestamps are normalized to RFC3339 UTC."""
    row: dict = {}
    extras: dict = {}
    for key, value in turn_metrics.items():
        if key in _TURN_BQ_COLUMNS:
            row[key] = value
        else:
            extras[key] = value

    # Required fields — always emit even if unset, so the row is valid.
    raw_ts = row.get("ts") or _now_utc_iso()
    if isinstance(raw_ts, str) and "T" in raw_ts and "+" not in raw_ts and "Z" not in raw_ts:
        # Naïve ISO from datetime.now().isoformat() — annotate as UTC.
        raw_ts = raw_ts + "+00:00"
    row["ts"] = raw_ts
    row.setdefault("session_id", SESSION_ID)
    row.setdefault("turn_id", str(uuid.uuid4()))
    row["extras"] = extras or None
    return row


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


# ── KB section caches ────────────────────────────────────────────────────────
#
# Apps catalog, folders wiki, and the personal-context MD all change only
# when the user installs/removes apps, regenerates the wiki, or recompiles
# the KB. On every /chat we re-read them from disk and rebuild the strings,
# even though the bytes are usually identical to last turn.
#
# Cost of the rebuild itself is small (~5–30ms), but the bigger win is that
# byte-stable output makes Vertex AI's implicit prompt cache hit faster and
# more often. Empirically the cache only kicked in around turn 5; with the
# prefix locked it should hit from turn 2.
#
# Each cache entry is `(fingerprint, computed_text)`:
#   - apps fingerprint     = max st_mtime_ns across all scan directories
#   - folders fingerprint  = st_mtime_ns of laptop_wiki/index.md (0 if missing)
#   - personal-context fp  = st_mtime_ns of micky_context.md (0 if missing)
# When the fingerprint matches the last value, the cached string is reused.
_apps_section_cache: tuple[int, str] | None = None
_folders_section_cache: tuple[int, str] | None = None
_personal_context_md_cache: tuple[int, str] | None = None


def _apps_scan_dirs_mtime_fingerprint() -> int:
    """
    A single int that changes whenever ANY of the apps scan directories
    changes (file added/removed/renamed). On macOS APFS the directory's
    `st_mtime_ns` updates whenever its child entries change — exactly the
    invalidation signal we need to know when to re-scan.
    """
    max_mtime_ns = 0
    for directory_path in APPLICATION_SCAN_DIRECTORIES:
        try:
            stat_result = os.stat(directory_path)
        except OSError:
            continue
        if stat_result.st_mtime_ns > max_mtime_ns:
            max_mtime_ns = stat_result.st_mtime_ns
    return max_mtime_ns


def _file_mtime_ns_or_zero(file_path: Path) -> int:
    """File mtime as an int, or 0 if the file is missing — used as a cache
    key for the folder-wiki index and the personal-context MD. Returning 0
    on missing means "if the file appears later, recompute"; returning the
    same 0 across repeated misses means we cache the empty result too."""
    try:
        return os.stat(file_path).st_mtime_ns
    except OSError:
        return 0


def build_apps_section() -> str:
    """
    Build the `── INSTALLED APPS ──` section the resolver looks for.
    Cached by max mtime of the apps scan directories — the disk listdir
    only reruns when an app is installed/removed.
    """
    global _apps_section_cache
    cache_fingerprint = _apps_scan_dirs_mtime_fingerprint()
    if _apps_section_cache is not None and _apps_section_cache[0] == cache_fingerprint:
        return _apps_section_cache[1]
    apps_list = ", ".join(scan_installed_app_names())
    apps_section_text = (
        "\n\n── INSTALLED APPS ──\n"
        "these are the apps actually installed on this machine. when you "
        "write `open -a 'NAME'`, prefer one of these exact names. you may "
        "write a casual short form (the executor fuzzy-matches it back to a "
        "real bundle name), but using a name from this list is faster and "
        "more reliable. if the user asks for an app that isn't on this list, "
        "tell them it's not installed instead of guessing.\n\n"
        f"{apps_list}"
    )
    _apps_section_cache = (cache_fingerprint, apps_section_text)
    return apps_section_text


def build_folders_section() -> str:
    """
    Build the `── KNOWN FOLDERS ON THIS MAC ──` section by parsing the
    wiki index. Same shape as `laptopFoldersSection()` in the Swift
    source, so the existing resolver code works against either one.
    Cached by mtime of laptop_wiki/index.md.
    """
    global _folders_section_cache
    cache_fingerprint = _file_mtime_ns_or_zero(LAPTOP_WIKI_INDEX_PATH)
    if _folders_section_cache is not None and _folders_section_cache[0] == cache_fingerprint:
        return _folders_section_cache[1]

    if not LAPTOP_WIKI_INDEX_PATH.exists():
        _folders_section_cache = (cache_fingerprint, "")
        return ""
    try:
        raw_index = LAPTOP_WIKI_INDEX_PATH.read_text(encoding="utf-8")
    except OSError:
        _folders_section_cache = (cache_fingerprint, "")
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
        _folders_section_cache = (cache_fingerprint, "")
        return ""

    formatted_entries = "\n".join(
        f"  {path} — {purpose}" for path, purpose in folder_entries
    )
    folders_section_text = (
        "\n\n── KNOWN FOLDERS ON THIS MAC ──\n"
        "the user's notable folders, with one-line summaries (sourced from "
        "~/laptop_wiki/index.md, regenerated when the user adds or moves "
        "files). when the user mentions a folder by partial or casual name, "
        "match it against this list FIRST to find the real absolute path. "
        "only fall back to `find` if no entry below plausibly matches. "
        "always pass absolute paths to shell commands — never `~`.\n\n"
        f"{formatted_entries}"
    )
    _folders_section_cache = (cache_fingerprint, folders_section_text)
    return folders_section_text


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

    return _load_personal_context_md_cached()


def _load_personal_context_md_cached() -> str:
    """
    Read `micky_context.md` from disk and wrap it in the section header,
    cached by file mtime. Splitting this out from
    `build_personal_context_section` keeps the cache scoped to the MD path
    only — the graph path is per-transcript dynamic and must NOT be cached.
    """
    global _personal_context_md_cache
    cache_fingerprint = _file_mtime_ns_or_zero(MICKY_CONTEXT_PATH)
    if (
        _personal_context_md_cache is not None
        and _personal_context_md_cache[0] == cache_fingerprint
    ):
        return _personal_context_md_cache[1]

    if not MICKY_CONTEXT_PATH.exists():
        _personal_context_md_cache = (cache_fingerprint, "")
        return ""
    try:
        context_body = MICKY_CONTEXT_PATH.read_text(encoding="utf-8")
    except OSError:
        _personal_context_md_cache = (cache_fingerprint, "")
        return ""
    section_text = f"\n\n── PERSONAL CONTEXT ──\n{context_body}\n"
    _personal_context_md_cache = (cache_fingerprint, section_text)
    return section_text


# Behavior-rules section is a static string — built once at module load
# and reused on every /chat. Closes specific failure patterns observed in
# real sessions (premature [TASK_DONE], over-long AXCLICK labels, ignoring
# BLOCKED/REFUSED replies). Lives in the proxy because the running Clicky
# binary predates the agent_contract.md wiring on the Swift side, so
# updating the .md alone has no effect at runtime — injection here does.
BEHAVIOR_RULES_SECTION_TEXT = (
    "\n\n── BEHAVIOR RULES (CRITICAL — these reinforce the contract above) ──\n"
    "these rules close real failure patterns observed in production sessions. "
    "apply them strictly. they take precedence over anything earlier in the prompt.\n\n"
    "1. ABSOLUTE RULE — server-enforced.\n"
    "   [TASK_DONE] is FORBIDDEN in any response that contains [CLICK], "
    "[AXCLICK], [DBLCLICK], [RCLICK], [TYPE], or [APPLESCRIPT] without a "
    "[SCREENSHOT] appearing AFTER them in the same response.\n\n"
    "   This is a SYNTACTIC rule, not a semantic one. It applies regardless of:\n"
    "     - whether your spoken text claims success or failure\n"
    "     - whether you 'know' the click worked\n"
    "     - whether you've already retried multiple times (especially then)\n"
    "     - whether the user is impatient\n\n"
    "   The proxy ENFORCES this. Any [TASK_DONE] you emit in violation of "
    "the rule will be AUTOMATICALLY REWRITTEN to [SCREENSHOT] on the wire — "
    "Clicky will never see your [TASK_DONE], the loop will continue, and "
    "you'll be asked to verify on the next turn. You cannot escape the loop "
    "by claiming false completion.\n\n"
    "   Valid:   ... [AXCLICK:X] [WAIT:1000] [SCREENSHOT]    ← stop here\n"
    "   Valid:   (next turn, after seeing the screenshot)\n"
    "            'X is now playing.' [TASK_DONE]\n"
    "   Invalid: ... [AXCLICK:X] [WAIT:1000] [TASK_DONE]     ← will be rewritten\n\n"
    "2. AXCLICK labels must be SHORT and distinctive.\n"
    "   chrome, safari, and electron apps truncate accessibility labels in the "
    "AX tree. NEVER paste a long aria-label with pipe separators like "
    "'Tum Mile - Title Track | Emraan Hashmi, Soha Ali | Pritam | Neeraj Shridhar | Kumaar' "
    "— the AX tree only contains a fragment and the click will fail. pick the "
    "3–6 most distinctive words instead: [AXCLICK:Tum Mile Title Track]. if "
    "even the short form fails, take a [SCREENSHOT] and use [CLICK:x,y] with "
    "coordinates read off the image — DO NOT keep retrying long labels, the "
    "OCR fallback will then click on Clicky's own overlay text and break things.\n\n"
    "3. failed action ≠ task complete.\n"
    "   if the executor's reply contains 'BLOCKED:', 'REFUSED:', any failure "
    "note, or returns no visible change, you have NOT completed the step. do "
    "NOT emit [TASK_DONE]. take a [SCREENSHOT] (if you don't have a fresh one), "
    "pick a different approach, and try again. emitting [TASK_DONE] after a "
    "BLOCKED action gets stored as a 'success' in long-term memory and will "
    "poison future similar tasks.\n\n"
    "4. [SUBTASK_DONE] needs evidence too.\n"
    "   only mark a subtask done after a verifying screenshot has shown the "
    "expected state. don't mark steps done speculatively just because you "
    "emitted the action that should accomplish them.\n\n"
    "5. Don't surrender via [TASK_DONE].\n"
    "   if you've already retried the same action 2+ times without visible "
    "success, do NOT silently emit [TASK_DONE] to escape the loop. instead:\n"
    "     (a) take a [SCREENSHOT] and describe out loud what you actually see "
    "vs. what you expected — naming the gap forces you to think differently;\n"
    "     (b) try a fundamentally different approach (different tag, different "
    "coordinates, different app entirely);\n"
    "     (c) or ask the user a brief question with [SCREENSHOT]: 'i'm not "
    "seeing the mix start playing — can you check?'\n"
    "   admitted failure beats false completion every time. a wrong "
    "[TASK_DONE] gets stored as memory and will misguide you on the next "
    "similar task.\n"
)


def build_behavior_rules_section() -> str:
    """
    Static behavior-rules block injected into every system prompt. Returning
    a constant (rather than reading from a file) keeps the bytes byte-stable
    across turns — important for prompt-cache stability. To edit the rules,
    change `BEHAVIOR_RULES_SECTION_TEXT` and restart the proxy.
    """
    return BEHAVIOR_RULES_SECTION_TEXT


# Hash of the previous turn's augmentation. Lets us tell the metrics layer
# whether this turn's prefix bytes match the last one — if they do,
# Vertex's implicit prompt cache has a real shot at hitting. Observed
# `main_cached_tokens` should track this signal closely.
_previous_kb_augmentation_sha256: str | None = None


def build_kb_augmentation(
    transcript: str = "",
    history_messages: list[dict] | None = None,
) -> tuple[str, dict]:
    """
    Concatenated KB sections to append to whatever system prompt the
    binary sends. Each underlying section is mtime-cached, so repeat calls
    do not re-read disk unless the source files actually changed.

    Returns `(augmentation_text, build_stats)` where `build_stats` carries
    observability signals for the per-turn metrics line:
      - `prefix_stable`: True when the bytes match the previous call's
        bytes — Vertex's implicit prompt cache should hit hard on these
        turns. False = the prefix changed (KB file edited, apps installed,
        graph path active, etc.).
      - `augmentation_sha256_prefix`: first 12 hex chars of the hash, so
        a series of turns can be eyeballed in metrics.jsonl to confirm
        the prefix really stayed identical.

    `transcript` and `history_messages` are only consulted by the graph path
    (`build_personal_context_section`). The apps + folders sections are still
    full-catalog dumps — those stay deterministic and small enough that
    selective retrieval doesn't pay off yet.
    """
    global _previous_kb_augmentation_sha256
    # Behavior rules go FIRST so they sit immediately after the Swift-side
    # contract — maximum prominence, before the model has scanned past
    # several KB of catalog data.
    augmentation_text = (
        build_behavior_rules_section()
        + build_apps_section()
        + build_folders_section()
        + build_personal_context_section(
            transcript=transcript,
            history_messages=history_messages,
        )
    )
    augmentation_sha256 = hashlib.sha256(
        augmentation_text.encode("utf-8")
    ).hexdigest()
    prefix_was_stable = (
        _previous_kb_augmentation_sha256 is not None
        and _previous_kb_augmentation_sha256 == augmentation_sha256
    )
    _previous_kb_augmentation_sha256 = augmentation_sha256
    build_stats = {
        "prefix_stable": prefix_was_stable,
        "augmentation_sha256_prefix": augmentation_sha256[:12],
    }
    return augmentation_text, build_stats


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


def resolve_intent_with_llm(transcript: str, system_instruction: str) -> tuple[dict, dict]:
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

    Returns `(parsed_resolution, resolver_metrics)`. `parsed_resolution`
    is the JSON-shaped dict (empty if anything goes wrong, so the main
    call can proceed). `resolver_metrics` carries per-call timing and
    token counts — populated for both success and failure so the metrics
    log captures what the resolver cost even when its result was unused.
    """
    resolver_metrics: dict = {
        "elapsed_ms": None,
        "prompt_tokens": None,
        "output_tokens": None,
        "skipped": False,
        "error": None,
    }
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
        resolver_metrics["skipped"] = True
        resolver_metrics["error"] = "empty apps and folders sections"
        return {}, resolver_metrics

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

    resolver_started_at = time.perf_counter()
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
        resolver_metrics["elapsed_ms"] = int(
            (time.perf_counter() - resolver_started_at) * 1000
        )
        resolver_usage = getattr(response, "usage_metadata", None)
        if resolver_usage is not None:
            resolver_metrics["prompt_tokens"] = getattr(resolver_usage, "prompt_token_count", None)
            resolver_metrics["output_tokens"] = getattr(resolver_usage, "candidates_token_count", None)
        raw_response_text = response.text or "{}"
        log_gemini_response(call_label="resolver", response_text=raw_response_text)
        print(f"🔎 Resolver raw response: {raw_response_text[:400]}")
        parsed = json.loads(raw_response_text)
        if not isinstance(parsed, dict):
            return {}, resolver_metrics
    except Exception as resolver_error:
        resolver_metrics["elapsed_ms"] = int(
            (time.perf_counter() - resolver_started_at) * 1000
        )
        resolver_metrics["error"] = str(resolver_error)
        print(f"⚠️ Intent resolver failed: {resolver_error}")
        return {}, resolver_metrics

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

    return parsed, resolver_metrics


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

    Side effect: when the script activates/launches a GUI app, run a
    follow-up zoom so the app opens filling the visible screen instead
    of restoring its last (often tiny) saved window size. Disable via
    `MICKY_MAXIMIZE_OPENED_APPS=0`.
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
    if completed.returncode != 0:
        return False, f"rc={completed.returncode}: {(completed.stderr or '').strip()[:300]}"

    if os.environ.get("MICKY_MAXIMIZE_OPENED_APPS", "1") == "1":
        opened_app = _detect_activated_app_name(script)
        if opened_app:
            _maximize_app_window(opened_app)

    return True, (completed.stdout or "").strip()[:300]


_ACTIVATE_TELL_RE = re.compile(
    r'tell\s+application\s+"([^"]+)"\s+to\s+(?:activate|launch)',
    re.IGNORECASE,
)
_ACTIVATE_BLOCK_RE = re.compile(
    r'tell\s+application\s+"([^"]+)".*?\bactivate\b',
    re.IGNORECASE | re.DOTALL,
)
_OPEN_DASH_A_RE = re.compile(
    r'open\s+-a\s+["\']?([^"\'&;]+?)["\']?(?:\s|$|"|\')',
    re.IGNORECASE,
)


def _detect_activated_app_name(script: str) -> str | None:
    """Return the app name if `script` launches or activates a GUI app."""
    for pattern in (_ACTIVATE_TELL_RE, _ACTIVATE_BLOCK_RE):
        m = pattern.search(script)
        if m:
            name = m.group(1).strip()
            if name and name.lower() not in {"system events", "finder"}:
                return name
    m = _OPEN_DASH_A_RE.search(script)
    if m:
        return m.group(1).strip()
    return None


def _maximize_app_window(app_name: str) -> None:
    """Fire-and-forget: set the front window of `app_name` to fill the
    screen. macOS clamps oversized bounds to the visible frame, so we
    just pass a huge size and let the OS shrink it to the right thing.
    Wrapped in `try` blocks so apps without AX-controllable windows
    don't error out."""
    safe = app_name.replace('"', '\\"')
    zoom_script = (
        f'delay 0.5\n'
        f'tell application "System Events"\n'
        f'  tell process "{safe}"\n'
        f'    try\n'
        f'      set frontmost to true\n'
        f'      delay 0.2\n'
        f'      if (count of windows) > 0 then\n'
        f'        set position of window 1 to {{0, 0}}\n'
        f'        set size of window 1 to {{4000, 3000}}\n'
        f'      end if\n'
        f'    end try\n'
        f'  end tell\n'
        f'end tell\n'
    )
    try:
        subprocess.run(
            ["osascript", "-e", zoom_script],
            capture_output=True,
            timeout=5,
        )
        print(f"📐 Auto-maximized window of '{app_name}'")
    except Exception as zoom_error:
        print(f"⚠️ Auto-maximize failed for '{app_name}': {zoom_error}")


def make_streaming_action_interceptor():
    """
    Returns `(process_chunk, flush_remaining, get_intercepted_actions,
    get_rewritten_task_done_count)` — a stateful streaming filter for
    Gemini's SSE output. Two jobs:

    1. AppleScript interception (existing behavior). When a complete
       `[APPLESCRIPT:...]` tag is seen, run the script server-side and
       strip the tag from the output stream — Clicky never sees it, so
       it doesn't try to run it itself and fail to (due to the sandbox).

    2. [TASK_DONE] rewrite (new). When `[TASK_DONE]` appears in the same
       response after a UI-mutating action ([CLICK]/[AXCLICK]/[DBLCLICK]/
       [RCLICK]/[TYPE]/[APPLESCRIPT]) without a verifying [SCREENSHOT] in
       between, we rewrite the [TASK_DONE] to [SCREENSHOT]. This forces
       the agentic loop to verify before declaring success, even when
       Gemini's prompt-level rule-following slips.

       Why server-side: behavior rules in the system prompt help (we did
       that too), but Gemini still emits `[AXCLICK:X] [WAIT] [TASK_DONE]`
       in retry-fatigue scenarios. The proxy enforces the contract so
       quitting mid-task can't silently happen.

    State machine:
      `has_unverified_mutating_action` flips True on any UI-mutating tag,
      flips False on [SCREENSHOT]. [TASK_DONE] is rewritten iff this flag
      is True at the moment it's seen.

    `flush_remaining()` returns text held back at the end of the stream —
    call it once after all chunks are processed.

    `get_rewritten_task_done_count()` reports how many [TASK_DONE] tags
    were rewritten in this stream — surfaced into per-turn metrics so we
    can see how often the rule-level guidance failed and the proxy had
    to step in.
    """
    APPLESCRIPT_TAG_PREFIX = "[APPLESCRIPT:"
    TASK_DONE_TAG          = "[TASK_DONE]"
    SCREENSHOT_TAG         = "[SCREENSHOT]"
    UI_MUTATING_TAG_PREFIXES = (
        "[CLICK:", "[AXCLICK:", "[DBLCLICK:", "[RCLICK:", "[TYPE:",
    )
    # All prefixes the scanner watches for. The earliest occurrence in
    # `pending` wins — that's the next tag to handle.
    DETECTION_PREFIXES = (
        APPLESCRIPT_TAG_PREFIX,
        TASK_DONE_TAG,
        SCREENSHOT_TAG,
    ) + UI_MUTATING_TAG_PREFIXES
    # Hold back this many chars at chunk boundaries while waiting to see
    # if a partial prefix is going to complete into a real tag.
    MAX_PREFIX_LENGTH = max(len(prefix) for prefix in DETECTION_PREFIXES)

    pending = ""
    intercepted_actions: list[dict] = []
    has_unverified_mutating_action = False
    rewritten_task_done_count = 0

    def find_earliest_detection_prefix(text: str) -> tuple[int, str | None]:
        """Scan `text` for any prefix in DETECTION_PREFIXES and return the
        (position, prefix) of the EARLIEST occurrence — that's the next
        tag we need to handle. Returns (-1, None) if none are present."""
        earliest_position = -1
        earliest_prefix: str | None = None
        for candidate_prefix in DETECTION_PREFIXES:
            position = text.find(candidate_prefix)
            if position != -1 and (earliest_position == -1 or position < earliest_position):
                earliest_position = position
                earliest_prefix = candidate_prefix
        return earliest_position, earliest_prefix

    def process_chunk(new_text: str):
        nonlocal pending, has_unverified_mutating_action, rewritten_task_done_count
        pending += new_text
        while True:
            earliest_position, earliest_prefix = find_earliest_detection_prefix(pending)

            if earliest_prefix is None:
                # No detection prefix anywhere in pending. Flush all except
                # the last (MAX_PREFIX_LENGTH - 1) chars, in case a partial
                # prefix is forming at the end.
                safe_flush_count = max(0, len(pending) - (MAX_PREFIX_LENGTH - 1))
                if safe_flush_count > 0:
                    yield pending[:safe_flush_count]
                    pending = pending[safe_flush_count:]
                return

            if earliest_position > 0:
                # Stream the text BEFORE the tag — it's safe (no prefix in it).
                yield pending[:earliest_position]
                pending = pending[earliest_position:]
                continue

            # Tag prefix is at position 0. Handle by type.

            if earliest_prefix == APPLESCRIPT_TAG_PREFIX:
                # Phase 6: AppleScript runs on the Swift side via NSAppleScript
                # (AgenticActionExecutor.swift:266). The proxy used to execute
                # it here and strip the tag — left over from when the Swift
                # build had App Sandbox enabled. Sandbox is off now, so the
                # tag passes through unmodified. This restores "Gemini is the
                # brain / proxy is dumb plumbing" for the AppleScript path.
                #
                # We still need to buffer until the closing `]` because
                # AppleScript bodies can contain `]` characters that the
                # naive prefix scanner would mis-parse. smart_find_applescript_close
                # handles quoted strings + nested brackets.
                content_start = len(APPLESCRIPT_TAG_PREFIX)
                close_index = smart_find_applescript_close(pending, content_start)
                if close_index is None:
                    return  # tag not complete yet — wait for more chunks
                # AppleScripts can mutate UI in arbitrary ways — count as
                # an unverified mutation requiring a [SCREENSHOT] before done.
                has_unverified_mutating_action = True
                yield pending[: close_index + 1]
                pending = pending[close_index + 1:]
                continue

            if earliest_prefix == TASK_DONE_TAG:
                if has_unverified_mutating_action:
                    rewritten_task_done_count += 1
                    print(
                        f"⚠️ Rewriting premature [TASK_DONE] → [SCREENSHOT] "
                        f"— unverified UI mutation in same response. "
                        f"(Gemini tried to declare done without verifying.)"
                    )
                    yield SCREENSHOT_TAG
                    # Emitting a synthetic [SCREENSHOT] does verify, so
                    # clear the flag — any subsequent mutations would set
                    # it again and a later [TASK_DONE] would be rewritten too.
                    has_unverified_mutating_action = False
                else:
                    yield TASK_DONE_TAG
                pending = pending[len(TASK_DONE_TAG):]
                continue

            if earliest_prefix == SCREENSHOT_TAG:
                # Verification — clear the unverified flag and forward as-is.
                has_unverified_mutating_action = False
                yield SCREENSHOT_TAG
                pending = pending[len(SCREENSHOT_TAG):]
                continue

            # UI-mutating prefix: find its close, mark unverified, forward.
            # These tags don't have the quoted-content complexity of
            # APPLESCRIPT, so a plain `]` search is sufficient. ([TYPE:...]
            # supports `\]` escape for literal brackets, but the simple find
            # would stop at the first `]` — acceptable for the state-tracking
            # use case here, since we only need to know WHERE the tag ends.)
            close_index = pending.find("]", len(earliest_prefix))
            if close_index == -1:
                return  # tag not complete yet — wait for more chunks
            has_unverified_mutating_action = True
            yield pending[:close_index + 1]
            pending = pending[close_index + 1:]
            continue

    def flush_remaining() -> str:
        """
        End-of-stream tail. If we're still mid-[APPLESCRIPT:...] (Gemini
        was truncated with an unclosed tag), DROP the partial — never
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

    def get_rewritten_task_done_count() -> int:
        return rewritten_task_done_count

    return (
        process_chunk,
        flush_remaining,
        get_intercepted_actions,
        get_rewritten_task_done_count,
    )


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
    # Single per-turn metrics blob — populated as the handler progresses,
    # written as one JSON line to metrics.jsonl at end-of-stream (success
    # OR error). Every field stays nullable so a partial failure still
    # produces a parseable record.
    turn_started_at = time.perf_counter()
    turn_id = str(uuid.uuid4())
    turn_index = next_turn_index()
    turn_metrics: dict = {
        "ts": _now_utc_iso(),
        "session_id": SESSION_ID,
        "turn_id": turn_id,
        "turn_index": turn_index,
        "transcript": None,
        "response_text": None,
        "response_action_tags": None,
        "model": None,
        "transcript_chars": None,
        "history_turns": None,
        "base_system_bytes": None,
        "augmentation_bytes": None,
        "system_total_bytes": None,
        "kb_prefix_stable": None,
        "kb_augmentation_sha256_prefix": None,
        "image_parts_in_request": None,
        "image_parts_kept": None,
        "image_parts_stripped": None,
        "vision_decision": None,
        "resolver_elapsed_ms": None,
        "resolver_prompt_tokens": None,
        "resolver_output_tokens": None,
        "resolver_useful": False,
        "resolver_skipped": False,
        "resolver_error": None,
        "main_ttft_ms": None,
        "main_total_ms": None,
        "main_prompt_tokens": None,
        "main_output_tokens": None,
        "main_cached_tokens": None,
        "main_total_tokens": None,
        "intercepted_applescripts": 0,
        "intercepted_failures": 0,
        "rewritten_premature_task_done_count": 0,
        "finish_reason": None,
        "max_tokens_truncated": False,
        "error": None,
        "total_handler_ms": None,
    }

    body = await request.json()
    # Force the .env-configured model, ignoring what the (pre-built) Swift
    # app hardcodes in the request body. Lets us upgrade the model without
    # rebuilding Clicky.app.
    requested_model = body.get("model")
    model = MODEL_TEXT
    if requested_model and requested_model != MODEL_TEXT:
        print(f"⚙️  Overriding requested model {requested_model!r} → {MODEL_TEXT!r} (.env wins)")
    turn_metrics["model"] = model
    system_instruction_block = body.get("system_instruction", {}) or {}
    system_instruction_parts = system_instruction_block.get("parts") or []
    system_instruction_first_text = (
        system_instruction_parts[0].get("text", "")
        if system_instruction_parts and isinstance(system_instruction_parts[0], dict)
        else ""
    )
    system = body.get("systemPrompt", "") or system_instruction_first_text
    prompt = body.get("userPrompt", "")

    # Pre-extract the latest user transcript + recent turn history so the
    # graph-driven personal-context builder can see them before we call
    # build_kb_augmentation. Both forms (Vertex `contents` array, legacy
    # userPrompt+history) are flattened to {role, text} dicts here.
    early_transcript, early_history = _extract_transcript_and_history(body, prompt)
    turn_metrics["transcript_chars"] = len(early_transcript or "")
    turn_metrics["history_turns"] = len(early_history)
    turn_metrics["transcript"] = early_transcript or None

    # Augment the system prompt with apps catalog + folders wiki + personal
    # context. The April-22 Clicky binary doesn't add these — it sends only
    # its base prompt — so the proxy supplies them. Re-read on every call so
    # `kb_compile.py` runs and wiki re-indexes are picked up live.
    base_system_size = len(system)
    augmentation, kb_build_stats = build_kb_augmentation(
        transcript=early_transcript,
        history_messages=early_history,
    )
    if augmentation:
        system = system + augmentation
        print(
            f"📚 KB augmentation: base={base_system_size}B + "
            f"+{len(augmentation)}B = {len(system)}B total  "
            f"(prefix_stable={kb_build_stats['prefix_stable']} "
            f"sha={kb_build_stats['augmentation_sha256_prefix']})"
        )

    # Core memory (Letta-style) sits between the static KB augmentation and
    # the per-turn task slot. It mutates only when Gemini calls
    # core_memory_append, so its prefix is stable across most turns — keeps
    # the cache hit rate high on (base + KB + core_memory) while the task
    # slot at the tail absorbs the per-turn churn.
    if _DUAL_TRACK_AVAILABLE:
        core_memory_block = core_memory_read_block()
        if core_memory_block:
            system = system + "\n\n" + core_memory_block + "\n"
            turn_metrics["core_memory_bytes"] = len(core_memory_block)

    # Pydantic-AI orchestrator pre-pass (env-gated). Runs a small Flash-Lite
    # agent with tools for use_corsair (MCP API routing), search_wiki, and
    # core_memory_append before the main streaming Gemini call. The structured
    # hints get appended right before the task slot so the per-turn churn
    # sits at the tail (preserves the upstream prefix cache).
    #
    # Fails open: any exception inside the pre-pass returns empty hints and
    # the main /chat path proceeds unchanged. Skipped entirely on screenshot
    # follow-up turns (transcript was injected by the proxy itself with the
    # `[fresh screenshot attached...]` prefix and there's no fresh user intent).
    if _ORCHESTRATOR_ENABLED and early_transcript and not early_transcript.startswith("[fresh screenshot attached"):
        orchestrator_started_at = time.perf_counter()
        try:
            hints = await _run_orchestrator_prepass(
                transcript=early_transcript,
                gemini_client=GEMINI_CLIENT,
                model_name=MODEL_RESOLVER,
            )
            hints_block = hints.render_for_prompt()
        except Exception as orchestrator_err:  # noqa: BLE001 — fail open
            print(f"⚠️ orchestrator pre-pass error: {orchestrator_err}")
            hints_block = ""
            hints = None
        orchestrator_ms = int((time.perf_counter() - orchestrator_started_at) * 1000)
        turn_metrics["orchestrator_ms"] = orchestrator_ms
        if hints_block:
            system = system + "\n\n" + hints_block + "\n"
            turn_metrics["orchestrator_hints_bytes"] = len(hints_block)
            turn_metrics["orchestrator_corsair_calls"] = (
                len(hints.corsair_results) if hints else 0
            )
            turn_metrics["orchestrator_wiki_hits"] = (
                len(hints.wiki_snippets) if hints else 0
            )
            turn_metrics["orchestrator_memory_writes"] = (
                len(hints.core_memory_writes) if hints else 0
            )
            print(
                f"🧠 Orchestrator: +{len(hints_block)}B in {orchestrator_ms}ms  "
                f"(corsair={turn_metrics['orchestrator_corsair_calls']}, "
                f"wiki={turn_metrics['orchestrator_wiki_hits']}, "
                f"mem+={turn_metrics['orchestrator_memory_writes']})"
            )
        else:
            print(f"🧠 Orchestrator: empty hints  ({orchestrator_ms}ms)")

    # Task slot lives AFTER the (cacheable) KB augmentation so the prefix that
    # changes most often (the slot mutates per turn) sits at the tail of the
    # system prompt — preserving the prefix-cache hit on the static KB chunk.
    task_slot_section = build_task_slot_section()
    system = system + task_slot_section
    print(f"🎯 Task slot: +{len(task_slot_section)}B "
          f"(active={'yes' if load_task_state().get('goal') else 'no'})")

    turn_metrics["base_system_bytes"] = base_system_size
    turn_metrics["augmentation_bytes"] = len(augmentation) if augmentation else 0
    turn_metrics["system_total_bytes"] = len(system)
    # Prefix-stability signals: if `prefix_stable=True` and yet
    # `main_cached_tokens` is null on the next turn, something other than
    # the KB augmentation is mutating the system prompt and trashing the
    # cache. The hash prefix lets us eyeball a run of turns in metrics.jsonl
    # to confirm the bytes really stayed identical.
    turn_metrics["kb_prefix_stable"] = kb_build_stats["prefix_stable"]
    turn_metrics["kb_augmentation_sha256_prefix"] = kb_build_stats["augmentation_sha256_prefix"]

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
    resolution: dict = {}
    if last_prompt and contents and contents[-1].role == "user":
        resolution, resolver_metrics = resolve_intent_with_llm(last_prompt, system)
        turn_metrics["resolver_elapsed_ms"]  = resolver_metrics.get("elapsed_ms")
        turn_metrics["resolver_prompt_tokens"] = resolver_metrics.get("prompt_tokens")
        turn_metrics["resolver_output_tokens"] = resolver_metrics.get("output_tokens")
        turn_metrics["resolver_skipped"] = resolver_metrics.get("skipped", False)
        turn_metrics["resolver_error"] = resolver_metrics.get("error")

        # Inject the resolution into the prompt for the main model. Gemini owns
        # the decision of what to do — including emitting action tags and
        # handling multi-step requests like "open X and write Y". The proxy
        # never short-circuits server-side, because that drops any intent the
        # shortcut doesn't understand (e.g. message_text after a verb=open).
        if has_useful_resolution(resolution):
            turn_metrics["resolver_useful"] = True
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

    # Count image parts in the request so the metrics line shows what we
    # were given vs what we kept after the vision-stripping heuristic.
    image_parts_in_request = sum(
        1
        for content_item in contents
        for part in (getattr(content_item, "parts", None) or [])
        if hasattr(part, "inline_data") and part.inline_data
    )
    turn_metrics["image_parts_in_request"] = image_parts_in_request

    # Drop the screenshot from the request unless this turn actually needs
    # vision. Saves ~250KB bandwidth + image-input tokens per request when
    # the task is purely text-driven (open / delete / send / etc.).
    is_followup = is_screenshot_followup_turn(last_prompt)
    needs_vision = initial_request_likely_needs_vision(last_prompt)
    if not (is_followup or needs_vision):
        contents, stripped_image_count = strip_images_from_contents(contents)
        turn_metrics["vision_decision"] = "stripped"
        turn_metrics["image_parts_stripped"] = stripped_image_count
        turn_metrics["image_parts_kept"] = image_parts_in_request - stripped_image_count
        if stripped_image_count > 0:
            print(f"📸→🗑️  Stripped {stripped_image_count} image(s) from request "
                  f"(no vision keywords in transcript; Gemini can request via [SCREENSHOT] if needed)")
    elif is_followup:
        turn_metrics["vision_decision"] = "kept_followup"
        turn_metrics["image_parts_stripped"] = 0
        turn_metrics["image_parts_kept"] = image_parts_in_request
        print(f"📸 Keeping image — screenshot-driven follow-up turn")
    elif needs_vision:
        turn_metrics["vision_decision"] = "kept_keyword"
        turn_metrics["image_parts_stripped"] = 0
        turn_metrics["image_parts_kept"] = image_parts_in_request
        print(f"📸 Keeping image — transcript references screen content")

    log_gemini_request(
        call_label="main",
        model=model,
        system_prompt=system,
        contents=contents,
        extra_meta={"last_user_text": last_prompt},
    )

    full_response = []
    (
        process_chunk,
        flush_remaining,
        get_intercepted_actions,
        get_rewritten_task_done_count,
    ) = make_streaming_action_interceptor()
    last_finish_reason: list = [None]  # mutable cell for the closure

    def emit_sse_text(text_to_emit: str):
        """Wrap a plain string in the SSE chunk shape the Swift app expects."""
        payload = json.dumps({"candidates": [{"content": {"parts": [{"text": text_to_emit}]}}]})
        return f"data: {payload}\n\n"

    def sse_generator():
        # Per-stream timing. `main_started_at` is set just before the
        # generate_content_stream call so we don't fold prep work into TTFT.
        # `first_chunk_at` is set on the first chunk that actually carries
        # text — chunks with only finish_reason / usage_metadata don't count
        # for time-to-first-token from the user's perspective.
        main_started_at: float | None = None
        first_chunk_at: float | None = None
        final_usage_metadata = None

        # How many times to retry a RESOURCE_EXHAUSTED error before giving up.
        # This fires when the user starts a new task immediately after the
        # previous one (ctrl+option mid-stream): the old Gemini connection is
        # still finishing, Vertex sees two concurrent requests, and returns 429.
        # Waiting 4 seconds gives the old stream time to drain before we retry.
        resource_exhausted_retries_remaining = 1

        try:
            for _stream_attempt in range(resource_exhausted_retries_remaining + 1):
                try:
                    main_started_at = time.perf_counter()
                    for chunk in GEMINI_CLIENT.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=config,
                    ):
                        text = chunk.text
                        if text:
                            if first_chunk_at is None:
                                first_chunk_at = time.perf_counter()
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
                        # usage_metadata may appear on multiple chunks (some SDKs
                        # emit running totals); keep the latest non-empty one as
                        # the authoritative count.
                        chunk_usage_metadata = getattr(chunk, "usage_metadata", None)
                        if chunk_usage_metadata is not None:
                            final_usage_metadata = chunk_usage_metadata
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
                        turn_metrics["max_tokens_truncated"] = True
                        truncation_note = (
                            " hmm, i hit my response limit before finishing that. "
                            "say 'continue' and i'll pick up where i left off, or "
                            "break the task into smaller steps."
                        )
                        print(f"⚠️ Gemini finish_reason=MAX_TOKENS — appending truncation note to spoken output")
                        yield emit_sse_text(truncation_note)
                except Exception as stream_error:
                    error_string = str(stream_error).upper()
                    is_resource_exhausted = any(marker in error_string for marker in (
                        "RESOURCE_EXHAUSTED", "RATE LIMIT", "QUOTA",
                    ))
                    # When the user presses ctrl+option immediately after a task
                    # completes, the previous Gemini stream may still be finishing
                    # its HTTP connection. Vertex AI sees two concurrent requests
                    # and returns 429 RESOURCE_EXHAUSTED. Waiting 4 seconds gives
                    # the old stream time to drain before we retry once.
                    if is_resource_exhausted and _stream_attempt < resource_exhausted_retries_remaining:
                        print(f"⚠️ RESOURCE_EXHAUSTED on attempt {_stream_attempt + 1} — waiting 4s for old stream to drain, then retrying")
                        time.sleep(4)
                        continue
                    turn_metrics["error"] = str(stream_error)
                    print(f"❌ Chat error: {stream_error}")
                    log_gemini_response(call_label="main", response_text=f"<<< ERROR: {stream_error} >>>")
                    # If we hit a quota / context-length error, the spoken response
                    # will otherwise be silence — emit a clear sentence so the user
                    # knows what happened and what to do. The active task slot is
                    # preserved on disk, so the next turn can pick up where we left off.
                    if any(marker in error_string for marker in (
                        "RESOURCE_EXHAUSTED", "RATE LIMIT", "QUOTA",
                        "CONTEXT LENGTH", "INPUT TOO LONG", "TOO MANY TOKENS",
                    )):
                        yield emit_sse_text(
                            "i hit a usage limit on that one. give me a moment, "
                            "or say continue and i'll pick up where i left off."
                        )
                    else:
                        yield f"data: {json.dumps({'error': str(stream_error)})}\n\n"
                    break
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
                    break
        finally:
            response_text = "".join(full_response)
            print(f"🤖 Gemini: {repr(response_text[:200])}")

            # Finalize per-turn metrics. Done in `finally` so even an error
            # mid-stream still emits a record (with `error` set) — that way
            # the metrics file shows true request rate, not just successes.
            stream_ended_at = time.perf_counter()
            if main_started_at is not None:
                turn_metrics["main_total_ms"] = int(
                    (stream_ended_at - main_started_at) * 1000
                )
            if main_started_at is not None and first_chunk_at is not None:
                turn_metrics["main_ttft_ms"] = int(
                    (first_chunk_at - main_started_at) * 1000
                )
            if final_usage_metadata is not None:
                turn_metrics["main_prompt_tokens"] = getattr(final_usage_metadata, "prompt_token_count", None)
                turn_metrics["main_output_tokens"] = getattr(final_usage_metadata, "candidates_token_count", None)
                turn_metrics["main_total_tokens"]  = getattr(final_usage_metadata, "total_token_count", None)
                # `cached_content_token_count` only appears once we wire up
                # explicit prompt caching (next step). Defensive read so
                # this stays None until then instead of crashing.
                turn_metrics["main_cached_tokens"] = getattr(final_usage_metadata, "cached_content_token_count", None)
            intercepted_for_metrics = get_intercepted_actions()
            turn_metrics["intercepted_applescripts"] = len(intercepted_for_metrics)
            turn_metrics["intercepted_failures"] = sum(
                1 for action in intercepted_for_metrics if not action["success"]
            )
            turn_metrics["rewritten_premature_task_done_count"] = get_rewritten_task_done_count()
            finish_reason_value_final = last_finish_reason[0]
            turn_metrics["finish_reason"] = (
                getattr(finish_reason_value_final, "name", None)
                or (str(finish_reason_value_final) if finish_reason_value_final else None)
            )
            turn_metrics["total_handler_ms"] = int(
                (time.perf_counter() - turn_started_at) * 1000
            )

            # Capture full response + parsed action tags for telemetry. Strip
            # tags from the stored `response_text` so it roughly mirrors what
            # the user actually heard via TTS.
            full_response_text = "".join(full_response)
            turn_metrics["response_text"] = strip_action_tags_for_storage(full_response_text) or None
            response_tags = parse_response_action_tags(full_response_text) or None
            turn_metrics["response_action_tags"] = response_tags

            # Dual-track schema validation (additive, observability-only).
            # Decode each MacAction-shaped tag through the Pydantic schema so
            # we can see, in metrics.jsonl, how often Gemini emits something
            # the new typed contract would have rejected. Untyped tags
            # ([PLAN], [TASK_*], [CONFIRM], etc.) return None and don't count
            # as failures. Behavior to Swift is unchanged.
            if _DUAL_TRACK_AVAILABLE and response_tags:
                schema_ok = 0
                schema_unmodeled = 0  # tags this schema doesn't cover yet (PLAN, TASK_*, CONFIRM)
                schema_invalid = 0
                for tag in response_tags:
                    try:
                        decoded = decode_bracket_tag(tag)
                        if decoded is None:
                            schema_unmodeled += 1
                        else:
                            schema_ok += 1
                    except Exception:
                        schema_invalid += 1
                turn_metrics["schema_macaction_ok"] = schema_ok
                turn_metrics["schema_macaction_unmodeled"] = schema_unmodeled
                turn_metrics["schema_macaction_invalid"] = schema_invalid

            # Apply [TASK_*] tags AFTER the full response is in hand — JSON
            # bodies can be partial mid-stream. Mutates task_state.json and
            # returns one event row per state transition.
            try:
                task_event_rows = apply_task_tags_to_state(full_response_text, turn_id)
            except Exception as task_apply_err:  # noqa: BLE001
                print(f"⚠️ task tag application failed: {task_apply_err}")
                report_error_to_bq(
                    "task_state",
                    task_apply_err,
                    session_id=SESSION_ID,
                    turn_id=turn_id,
                )
                task_event_rows = []
            if task_event_rows:
                bq_insert_async(BQ_TABLE_TASK_EVENTS, task_event_rows)
                print(f"📝 Task slot: {len(task_event_rows)} event(s) → "
                      f"{[event['event_type'] for event in task_event_rows]}")

            # Phase 4: wiki summarizer. Fires only when this turn closed a
            # task ([TASK_DONE] in the response). Runs in the BQ thread pool
            # so it never blocks the SSE response. A cheap Flash Lite call
            # classifies wiki_worthy and, if so, appends a 1-3-sentence note
            # to the right topic file under ~/.clicky_wiki/. All errors swallowed.
            if (
                _DUAL_TRACK_AVAILABLE
                and response_tags
                and "[TASK_DONE]" in response_tags
            ):
                BQ_EXECUTOR.submit(
                    maybe_write_wiki,
                    gemini_client=GEMINI_CLIENT,
                    classifier_model=MODEL_RESOLVER,  # Flash Lite — cheap
                    user_transcript=turn_metrics.get("transcript") or "",
                    agent_response=full_response_text,
                    actions_taken=response_tags or [],
                )

            # If anything failed during the stream, also drop a row into the
            # errors table with whatever context we have.
            if turn_metrics.get("error"):
                report_error_to_bq(
                    "main_chat",
                    message=turn_metrics["error"],
                    session_id=SESSION_ID,
                    turn_id=turn_id,
                    context={
                        "model": turn_metrics.get("model"),
                        "transcript_chars": turn_metrics.get("transcript_chars"),
                        "history_turns": turn_metrics.get("history_turns"),
                        "finish_reason": turn_metrics.get("finish_reason"),
                    },
                )

            record_turn_metrics(turn_metrics)

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
    # Dual-track bootstrap: ensure core memory + wiki skeleton exist on disk
    # before any /chat call. Safe to run every boot — idempotent (no-op if
    # files already present).
    if _DUAL_TRACK_AVAILABLE:
        seeded = core_memory_seed_if_empty()
        wiki_files = wiki_bootstrap()
        if seeded:
            print(f"🧠 Seeded {seeded} core-memory facts → ~/.clicky_core_memory.json")
        if wiki_files:
            print(f"📚 Created {wiki_files} wiki skeleton files → ~/.clicky_wiki/")

    print("\n🚀 Clicky local proxy starting on http://localhost:8081\n")
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
