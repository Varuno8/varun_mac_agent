"""
Pydantic-AI Agent runtime for the dual-track architecture.

Phase 1: skeleton — model selection helper, mock MCP, tool stubs.
proxy.py does NOT import this yet. It will be wired in Phase 1.5 once
schemas + tools are battle-tested in isolation.
"""

from __future__ import annotations

from typing import Iterable

from agent_schemas import (
    CorsairCall,
    CorsairResult,
    MacAction,
    TaskAbandon,
    TaskDone,
    WikiSearchHit,
)
# Re-export memory paths + tier-2/4 implementations so callers have one entry point.
from agent_memory import (  # noqa: F401
    CORE_MEMORY_PATH,
    WIKI_ROOT,
    EPISODIC_PATH,
    core_memory_append,
    core_memory_read_block,
    core_memory_seed_if_empty,
    search_wiki,
    wiki_append,
    wiki_bootstrap,
    wiki_list,
)


# ── Model selection ──────────────────────────────────────────────────────────
#
# Per the approved plan: prefer Gemini 3.x Pro when available, fall back to
# 3.1 Flash Lite (the current working model). We probe at startup, cache the
# result. Resolver stays on Flash Lite regardless — it's a cheap pre-pass.

PREFERRED_MAIN_MODELS = (
    "gemini-3.1-pro",
    "gemini-3.0-pro",
    "gemini-3.1-flash-lite-preview",
)

FALLBACK_MAIN_MODEL = "gemini-3.1-flash-lite-preview"


def pick_available_gemini(client, candidates: Iterable[str] = PREFERRED_MAIN_MODELS) -> str:
    """Return the first model in `candidates` that the Vertex client can
    enumerate. Falls back to FALLBACK_MAIN_MODEL if probing fails."""

    try:
        available = {m.name.split("/")[-1] for m in client.models.list()}
    except Exception:
        return FALLBACK_MAIN_MODEL

    for name in candidates:
        if name in available:
            return name
    return FALLBACK_MAIN_MODEL


# ── MCP backends ─────────────────────────────────────────────────────────────
#
# Two backends share the same call_tool(integration, action, params) interface:
#
#   MockMCPServer    — always returns integration_not_available. Used when the
#                      bridge isn't configured or fails to spawn. Keeps the
#                      orchestrator pre-pass functional (Gemini falls back
#                      to vision/UI), so MICKY_ORCHESTRATOR=1 is usable even
#                      before any Corsair setup is done.
#
#   CorsairMCPBridge — spawns the Node.js clicky-mcp-bridge as a subprocess,
#                      speaks MCP stdio to it, and forwards use_corsair calls
#                      as run_script invocations. Active iff MICKY_MCP_BRIDGE
#                      env var points to the bridge directory.


import atexit
import asyncio
import json
import os
import shlex
import subprocess
import threading
from typing import Any


class MockMCPServer:
    """Stand-in for Corsair MCP. Every call reports integration_not_available
    so the routing path + fallback logic exercises end-to-end without a real
    MCP server installed."""

    def list_tools(self) -> list[dict]:
        return []

    def list_operations(self, plugin: str | None = None) -> list[str]:
        return []

    def get_schema(self, path: str) -> str:
        return ""

    def call_tool(self, integration: str, action: str, params: dict) -> CorsairResult:
        return CorsairResult(
            ok=False,
            error="integration_not_available",
            data={"integration": integration, "action": action},
        )

    def close(self) -> None:
        pass


class CorsairMCPBridge:
    """Real MCP client that spawns `npx tsx src/mcp_server.ts` in the
    clicky-mcp-bridge directory and speaks the MCP stdio protocol via the
    `mcp` Python SDK.

    Async-internally, sync-externally: a dedicated event loop runs in a
    background thread so callers (pydantic-ai tools, FastAPI handlers) can
    call `call_tool(...)` synchronously and from any thread without thinking
    about asyncio. This avoids the "tried to nest event loops" footgun.

    Lazy: the subprocess + MCP session aren't spawned until the first call.
    A failed connect leaves the bridge in a broken state and subsequent
    calls return integration_not_available — the orchestrator pre-pass
    keeps working and Gemini falls back to UI."""

    def __init__(self, bridge_dir: str):
        self._bridge_dir = bridge_dir
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._session: Any | None = None  # mcp.ClientSession after connect
        self._stdio_ctx: Any | None = None
        self._session_ctx: Any | None = None
        self._connect_lock = threading.Lock()
        self._connect_error: str | None = None

    # ── Event loop plumbing ──────────────────────────────────────────────────

    def _ensure_loop_running(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and self._loop.is_running():
            return self._loop

        ready = threading.Event()

        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            ready.set()
            loop.run_forever()

        self._loop_thread = threading.Thread(
            target=run_loop, name="mcp-bridge-loop", daemon=True
        )
        self._loop_thread.start()
        ready.wait(timeout=5)
        if self._loop is None:
            raise RuntimeError("mcp-bridge loop failed to start")
        return self._loop

    def _submit(self, coro):
        loop = self._ensure_loop_running()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    # ── MCP session lifecycle ────────────────────────────────────────────────

    async def _connect(self) -> None:
        """Spawn the Node bridge and complete the MCP handshake. Called once."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command="npx",
            args=["tsx", "src/mcp_server.ts"],
            cwd=self._bridge_dir,
            env={**os.environ},
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

    def _ensure_connected(self) -> bool:
        """Lazy connect on first call. Returns True if the session is
        usable, False if the bridge failed to come up — caller falls back
        to a sentinel error result."""
        if self._session is not None:
            return True
        if self._connect_error is not None:
            return False
        with self._connect_lock:
            if self._session is not None:
                return True
            if self._connect_error is not None:
                return False
            try:
                fut = self._submit(self._connect())
                fut.result(timeout=20)
                return True
            except Exception as e:  # noqa: BLE001 — capture for debugging
                self._connect_error = f"{type(e).__name__}: {e}"
                print(f"⚠️ MCP bridge connect failed: {self._connect_error}")
                return False

    def close(self) -> None:
        async def _close():
            try:
                if self._session_ctx is not None:
                    await self._session_ctx.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            try:
                if self._stdio_ctx is not None:
                    await self._stdio_ctx.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._session = None
            self._stdio_ctx = None
            self._session_ctx = None

        if self._loop is not None and self._loop.is_running():
            try:
                self._submit(_close()).result(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── Public API ───────────────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        if not self._ensure_connected():
            return []

        async def _list():
            return await self._session.list_tools()

        try:
            res = self._submit(_list()).result(timeout=10)
            return [{"name": t.name, "description": t.description} for t in res.tools]
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ MCP list_tools failed: {e}")
            return []

    def list_operations(self, plugin: str | None = None) -> list[str]:
        """Return every Corsair operation path the bridge exposes (e.g.
        'slack.api.channels.list', 'gmail.api.messages.send'). Optionally
        filtered to one plugin. Empty list on bridge failure."""
        if not self._ensure_connected():
            return []
        args: dict[str, Any] = {}
        if plugin:
            args["plugin"] = plugin

        async def _call():
            return await self._session.call_tool("list_operations", args)

        try:
            res = self._submit(_call()).result(timeout=10)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ MCP list_operations failed: {e}")
            return []
        for block in (res.content or []):
            text = getattr(block, "text", None)
            if text:
                return [line.strip() for line in text.splitlines() if line.strip()]
        return []

    def get_schema(self, path: str) -> str:
        """Return the input/output schema for one Corsair operation path
        as a TypeScript-style declaration. Empty string on bridge failure."""
        if not self._ensure_connected():
            return ""

        async def _call():
            return await self._session.call_tool("get_schema", {"path": path})

        try:
            res = self._submit(_call()).result(timeout=10)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ MCP get_schema failed: {e}")
            return ""
        for block in (res.content or []):
            text = getattr(block, "text", None)
            if text:
                return text
        return ""

    def call_tool(self, integration: str, action: str, params: dict) -> CorsairResult:
        """Invoke `<integration>.api.<action>(params)` on the Corsair instance
        via the bridge's `run_script` tool. The `action` may contain dots
        (e.g. `"channels.list"` → `corsair.slack.api.channels.list(...)`),
        which is necessary for nested-namespace Corsair APIs. Returns a
        CorsairResult so callers (orchestrator → Gemini) always see the
        same shape regardless of whether the bridge succeeded, errored,
        or wasn't available."""
        if not self._ensure_connected():
            return CorsairResult(
                ok=False,
                error=f"mcp_bridge_unavailable: {self._connect_error or 'not connected'}",
                data={"integration": integration, "action": action},
            )

        # Build a tiny JS snippet for run_script. We JSON-encode the params
        # so quoting / escapes match exactly what the JS engine expects.
        params_json = json.dumps(params or {})
        code = (
            f"const result = await corsair.{integration}.api.{action}"
            f"({params_json}); return result;"
        )

        async def _call():
            return await self._session.call_tool("run_script", {"code": code})

        try:
            res = self._submit(_call()).result(timeout=30)
        except Exception as e:  # noqa: BLE001 — surface to caller
            return CorsairResult(
                ok=False,
                error=f"mcp_call_failed: {type(e).__name__}: {e}",
                data={"integration": integration, "action": action},
            )

        # run_script returns a single text content block with JSON or an
        # error message. is_error == True ⇒ exception inside the script
        # (auth, network, validation, etc.).
        text = ""
        for block in (res.content or []):
            block_text = getattr(block, "text", None)
            if block_text:
                text = block_text
                break

        if getattr(res, "isError", False):
            return CorsairResult(
                ok=False,
                error=text or "unknown_corsair_error",
                data={"integration": integration, "action": action},
            )

        try:
            decoded = json.loads(text) if text else None
        except json.JSONDecodeError:
            decoded = {"raw": text}

        data_payload: dict
        if isinstance(decoded, dict):
            data_payload = decoded
        else:
            data_payload = {"value": decoded}
        return CorsairResult(ok=True, data=data_payload)


# ── use_corsair entry point ──────────────────────────────────────────────────


_DEFAULT_MCP: MockMCPServer | CorsairMCPBridge | None = None


def _get_mcp() -> MockMCPServer | CorsairMCPBridge:
    """Return the active MCP backend. Constructed lazily on first call.
    Env-driven selection:
      MICKY_MCP_BRIDGE=/path/to/clicky-mcp-bridge → real bridge
      (unset)                                     → MockMCPServer"""
    global _DEFAULT_MCP
    if _DEFAULT_MCP is not None:
        return _DEFAULT_MCP

    bridge_dir = os.environ.get("MICKY_MCP_BRIDGE", "").strip()
    if bridge_dir and os.path.isdir(bridge_dir):
        bridge = CorsairMCPBridge(bridge_dir)
        atexit.register(bridge.close)
        _DEFAULT_MCP = bridge
        print(f"🌉 MCP bridge configured: {bridge_dir} (lazy spawn on first use)")
    else:
        _DEFAULT_MCP = MockMCPServer()
    return _DEFAULT_MCP


def use_corsair(integration: str, action: str, params: dict | None = None) -> CorsairResult:
    """Route an API call through the configured MCP backend (Corsair bridge
    if MICKY_MCP_BRIDGE is set, mock otherwise). Failures surface to the
    caller so Gemini can re-plan — never auto-retried server-side."""
    return _get_mcp().call_tool(integration, action, params or {})


# ── Bracket-tag bridge ───────────────────────────────────────────────────────
#
# Until proxy.py is migrated to emit typed tool calls, we need a way to
# round-trip MacAction objects through the legacy bracket-tag wire format.
# encode = Python-side MacAction → bracket tag string (Swift wire);
# decode = bracket tag string (from Gemini's response) → validated MacAction.
# Phase 1 ships encode (used by future tools); decode helps validate
# existing Gemini output against the new schema during the cutover.


def encode_mac_action(action: MacAction) -> str:
    return action.to_bracket_tag()


def decode_bracket_tag(tag: str) -> MacAction | None:
    """Best-effort parse of a single bracket tag into a MacAction. Returns
    None for tags this schema doesn't model (e.g. [PLAN:...], [TASK_*],
    [CONFIRM:...]) — those stay in the prose layer for now."""
    tag = tag.strip()
    if not (tag.startswith("[") and tag.endswith("]")):
        return None
    body = tag[1:-1]

    if body == "SCREENSHOT":
        return MacAction(action_type="screenshot")

    name, _, rest = body.partition(":")
    name = name.strip().upper()

    if name == "CLICK":
        x, _, y = rest.partition(",")
        return MacAction(action_type="click", x=int(x), y=int(y))
    if name == "DBLCLICK":
        x, _, y = rest.partition(",")
        return MacAction(action_type="dblclick", x=int(x), y=int(y))
    if name == "RCLICK":
        x, _, y = rest.partition(",")
        return MacAction(action_type="rclick", x=int(x), y=int(y))
    if name == "AXCLICK":
        return MacAction(action_type="axclick", ax_label=rest)
    if name == "TYPE":
        return MacAction(action_type="type", text=rest)
    if name == "HOTKEY":
        return MacAction(action_type="hotkey", text=rest)
    if name == "APPLESCRIPT":
        return MacAction(action_type="applescript", text=rest)
    if name == "WAIT":
        return MacAction(action_type="wait", wait_ms=int(rest))
    if name == "SCROLL":
        parts = rest.split(",")
        direction = parts[0].strip()
        amount = int(parts[1]) if len(parts) > 1 else 1
        x = int(parts[2]) if len(parts) > 2 else None
        y = int(parts[3]) if len(parts) > 3 else None
        return MacAction(
            action_type="scroll",
            scroll_direction=direction,  # type: ignore[arg-type]
            scroll_amount=amount,
            x=x,
            y=y,
        )
    return None
