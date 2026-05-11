"""
Pydantic-AI orchestrator pre-pass.

Run BEFORE the main streaming /chat call in proxy.py. The orchestrator is a
small, fast pydantic-ai Agent (Flash Lite under the hood) with three tools:

  - search_wiki(query)              → read ~/.clicky_wiki/ snippets
  - use_corsair(integration, ...)   → API-first execution via Corsair MCP
  - core_memory_append(fact)        → write a durable fact to T2 core memory

It returns an `OrchestratorHints` object. proxy.py injects the rendered
hints as a `── ORCHESTRATOR HINTS ──` block in the system prompt so the
main streaming Gemini call (which speaks to the user and emits the bracket
tags Swift executes) sees structured pre-resolved data instead of having
to navigate UIs from scratch.

Why a pre-pass and not a full replacement of the streaming call:
  - The Swift agentic loop + bracket-tag wire format still works and is
    battle-tested; pulling that apart in one shot would destabilize the
    running app.
  - The pre-pass gives us the MCP-first behavior the user wants without
    breaking the fallback path (Gemini can still drive vision/OCR if
    Corsair has no integration or returns an error).
  - Per the "Proxy stays minimal" memory: the LLM still decides — the
    orchestrator merely runs the tools the LLM picks, then hands the
    results back to the main turn.

Designed to fail open: any error in the orchestrator returns empty hints
and the /chat call proceeds as if the pre-pass never ran.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from agent_memory import (
    core_memory_append as _core_memory_append_impl,
    core_memory_read_block,
    search_wiki as _search_wiki_impl,
    wiki_append as _wiki_append_impl,
    wiki_list,
)
from agent_runtime import _get_mcp, use_corsair as _use_corsair_impl
from agent_schemas import CorsairResult, WikiSearchHit


log = logging.getLogger("orchestrator")


# ── Structured output the orchestrator returns to proxy.py ────────────────────


class OrchestratorHints(BaseModel):
    """Everything the orchestrator pre-pass discovered. proxy.py renders
    this into the system prompt as `── ORCHESTRATOR HINTS ──`. Always safe
    to be empty — a totally vision-driven request will return an empty hints
    object and the main Gemini call proceeds unchanged."""

    summary: str = Field(
        default="",
        description=(
            "One-sentence note to the main Gemini turn explaining what the "
            "orchestrator did. Empty if nothing useful was found."
        ),
    )
    wiki_snippets: list[WikiSearchHit] = Field(default_factory=list)
    corsair_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Each entry: {integration, action, params, result_ok, "
            "result_data|result_error}. Surfaces both successes and failures "
            "so the main turn can re-plan when Corsair couldn't help."
        ),
    )
    core_memory_writes: list[str] = Field(
        default_factory=list,
        description="Facts the orchestrator appended to T2 core memory this turn.",
    )

    def render_for_prompt(self) -> str:
        """Format hints for inclusion in the Gemini system prompt. Returns
        an empty string when there's nothing to add so proxy.py can skip
        the section entirely (preserves prompt-cache prefix stability)."""
        if not (
            self.summary
            or self.wiki_snippets
            or self.corsair_results
            or self.core_memory_writes
        ):
            return ""

        parts: list[str] = ["── ORCHESTRATOR HINTS ──"]
        if self.summary:
            parts.append(f"note: {self.summary}")

        if self.corsair_results:
            parts.append("\nCorsair MCP results (use verbatim, do not re-fetch via UI):")
            for entry in self.corsair_results:
                tag = f"{entry.get('integration')}.{entry.get('action')}"
                if entry.get("result_ok"):
                    parts.append(f"  ✓ {tag} → {entry.get('result_data')!r}")
                else:
                    parts.append(f"  ✗ {tag} failed: {entry.get('result_error')}")

        if self.wiki_snippets:
            parts.append("\nRelevant wiki notes (already searched, do not search again):")
            for hit in self.wiki_snippets:
                parts.append(f"  • {hit.path}:\n{hit.content}")

        if self.core_memory_writes:
            parts.append("\nNewly stored core-memory facts:")
            for fact in self.core_memory_writes:
                parts.append(f"  + {fact}")

        parts.append("── END ORCHESTRATOR HINTS ──")
        return "\n".join(parts)


# ── Agent context ────────────────────────────────────────────────────────────


class OrchestratorDeps(BaseModel):
    """Per-turn state threaded through tool calls so tools can record their
    side effects without globals. Tools mutate this; the agent's final
    response is rendered from it."""

    model_config = {"arbitrary_types_allowed": True}

    transcript: str
    wiki_snippets: list[WikiSearchHit] = Field(default_factory=list)
    corsair_results: list[dict[str, Any]] = Field(default_factory=list)
    core_memory_writes: list[str] = Field(default_factory=list)


# ── Tool implementations (thin wrappers that record into deps) ───────────────


def _make_agent(model: GoogleModel) -> Agent[OrchestratorDeps, OrchestratorHints]:
    """Build the pydantic-ai Agent against an already-constructed Vertex
    `GoogleModel`. Caller is responsible for constructing the model with
    the proxy's existing service-account credentials so we don't load auth
    twice."""

    agent: Agent[OrchestratorDeps, OrchestratorHints] = Agent(
        model=model,
        deps_type=OrchestratorDeps,
        output_type=OrchestratorHints,
        system_prompt=_SYSTEM_PROMPT,
    )

    @agent.tool
    def search_wiki(ctx: RunContext[OrchestratorDeps], query: str) -> list[WikiSearchHit]:
        """Search the persistent agent wiki (~/.clicky_wiki/) for notes that
        match the query. Returns up to 3 markdown snippets ranked by filename
        match weight 3 + content match weight 1. Use this BEFORE attempting a
        task to recover prior debugging fixes, working AppleScripts, or known
        user preferences. Empty list means nothing relevant was indexed."""
        hits = _search_wiki_impl(query)
        ctx.deps.wiki_snippets.extend(hits)
        return hits

    @agent.tool
    def use_corsair(
        ctx: RunContext[OrchestratorDeps],
        integration: str,
        action: str,
        params: dict | None = None,
    ) -> CorsairResult:
        """Execute an API call through Corsair MCP (Slack, Drive, Gmail,
        Calendar, GitHub, Linear, Notion, etc.) instead of driving the UI.

        Use this WHENEVER the user's request maps to a service with an API.
        Vision/UI navigation is the fallback, not the default — APIs are
        ~1s and fractions of a cent vs. multiple screenshot iterations.

        `action` may be a dotted path for nested namespaces (e.g.
        'channels.list', 'messages.send') — call list_corsair_operations
        FIRST when unsure of the exact path.

        Returns a CorsairResult. On `ok=False` with error='integration_not_available'
        or rate-limit/auth errors, the main turn will fall back to UI nav."""
        result = _use_corsair_impl(integration, action, params or {})
        ctx.deps.corsair_results.append(
            {
                "integration": integration,
                "action": action,
                "params": params or {},
                "result_ok": result.ok,
                "result_data": result.data,
                "result_error": result.error,
            }
        )
        return result

    @agent.tool_plain
    def list_corsair_operations(plugin: str | None = None) -> list[str]:
        """List every Corsair API path the bridge currently exposes (e.g.
        'slack.api.channels.list'). Optionally scoped to one plugin. Use
        this BEFORE calling use_corsair to discover the right action path —
        avoids 'is not a function' errors from guessing method names."""
        return _get_mcp().list_operations(plugin)

    @agent.tool_plain
    def get_corsair_schema(path: str) -> str:
        """Get the TypeScript-style input/output schema for one Corsair API
        path. Use this to learn what `params` shape `use_corsair` expects
        for a specific operation. Returns empty string if the path is
        unknown or the bridge isn't connected."""
        return _get_mcp().get_schema(path)

    @agent.tool
    def core_memory_append(ctx: RunContext[OrchestratorDeps], fact: str) -> bool:
        """Persist a SHORT durable fact (one sentence, < 200 chars) about the
        user or their environment to T2 core memory (~/.clicky_core_memory.json).
        Lives in EVERY future system prompt — keep it tight and meaningful.

        Use for: "user prefers Discord", "Diagnxt is in prototype phase",
        "Garima is the partner's name".

        For LONGER notes (a paragraph about how a project is configured,
        a debugging workaround, a working AppleScript) use update_wiki_knowledge
        instead — those live in the wiki and don't bloat every prompt.

        Returns False if the fact was already stored."""
        wrote = _core_memory_append_impl(fact)
        if wrote:
            ctx.deps.core_memory_writes.append(fact)
        return wrote

    @agent.tool_plain
    def update_wiki_knowledge(filename: str, content: str) -> bool:
        """Append a longer note to the agent's persistent knowledge wiki at
        ~/.clicky_wiki/knowledge/<filename>. Unlike core_memory_append, this
        content does NOT live in every future prompt — it's only loaded
        when search_wiki retrieves it on a relevant turn.

        Use for: paragraph-level project notes ("Diagnxt uses high-precision
        OCR by default and stores results in /tmp/diagnxt/"), debugging
        workarounds, working AppleScripts for specific app states, or any
        durable lesson too long for core memory.

        `filename` should be a topic name like 'diagnxt' or 'tryown_dev_loop'
        — the .md extension is added automatically, and the file lives under
        knowledge/ (path-traversal blocked). Returns False if content was empty."""
        wiki_path = filename if filename.startswith('knowledge/') else f'knowledge/{filename}'
        return _wiki_append_impl(wiki_path, content)

    @agent.tool_plain
    def list_wiki_topics() -> list[str]:
        """List all wiki topic files (relative paths under ~/.clicky_wiki/).
        Call this to discover what topics exist before searching."""
        return wiki_list()

    @agent.tool_plain
    def read_core_memory() -> str:
        """Return the current T2 core-memory block verbatim. Useful when you
        need to check what's already remembered before appending."""
        return core_memory_read_block()

    return agent


# ── System prompt ────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are Micky's pre-pass orchestrator. You do NOT speak to the user; another
Gemini turn handles that. Your job is to do API-first execution and prior-knowledge
retrieval BEFORE the main turn runs, then hand it structured results.

Inputs: the user's freshly-transcribed sentence and (in the deps) a place to
record what you did.

Decision algorithm:
1. Does this task map to a service with an API (Slack, Drive, Calendar,
   Gmail, GitHub, Notion, Linear, Discord, Stripe, Tavily)?
   a. If unsure of the exact API path → call list_corsair_operations(plugin)
      first to discover what's actually exposed. Optionally call
      get_corsair_schema(path) to learn the param shape.
   b. Then call use_corsair(integration, action, params) with a
      dot-path action (e.g. 'channels.list', 'messages.send').
   c. If use_corsair fails (auth, not_available, rate-limit), record it
      and stop — the main turn will fall back to UI navigation.
2. Is this a task category we might have notes about (a project name, a
   debugging topic, an app the user has scripted before)? → search_wiki.
3. Did the user reveal a durable preference, fact, or context that future
   sessions should know? → core_memory_append (only for things worth
   remembering forever; skip transient details).

After tool use, emit a final OrchestratorHints object:
- summary: one short sentence telling the main turn what you did (or "")
- wiki_snippets, corsair_results, core_memory_writes: leave empty if no
  tool produced useful output

DO NOT:
- Speak to the user (the main Gemini turn does that)
- Emit bracket tags ([CLICK], [APPLESCRIPT], etc.) — that's the main turn's job
- Call tools speculatively if the transcript clearly doesn't need them
  (e.g. "open Notes" has no API to call and no wiki to search — return empty)
- Re-write existing core-memory facts; check read_core_memory first if unsure
"""


# ── Public entry point ───────────────────────────────────────────────────────


# Cache keyed by model name so we don't rebuild the Agent (and re-register
# its tools) on every /chat request. The GoogleModel is cheap to construct
# but cached too so we reuse the same provider across calls.
_AGENT_CACHE: dict[str, Agent[OrchestratorDeps, OrchestratorHints]] = {}
_MODEL_CACHE: dict[int, dict[str, GoogleModel]] = {}


def _build_vertex_model(gemini_client: Any, model_name: str) -> GoogleModel:
    """Construct (and cache) a pydantic-ai GoogleModel that reuses the
    proxy's existing Vertex client. Keyed by (client identity, model_name)
    so the same client object yields the same model object across calls."""
    by_client = _MODEL_CACHE.setdefault(id(gemini_client), {})
    cached = by_client.get(model_name)
    if cached is not None:
        return cached
    provider = GoogleProvider(client=gemini_client)
    model = GoogleModel(model_name=model_name, provider=provider)
    by_client[model_name] = model
    return model


def _agent_for(model: GoogleModel) -> Agent[OrchestratorDeps, OrchestratorHints]:
    """Return a cached Agent built against `model`. Cache key is the model
    name string since GoogleModel doesn't implement __hash__ but model
    names are unique per Vertex project."""
    key = model.model_name
    cached = _AGENT_CACHE.get(key)
    if cached is not None:
        return cached
    agent = _make_agent(model)
    _AGENT_CACHE[key] = agent
    return agent


async def run_orchestrator_prepass(
    *,
    transcript: str,
    gemini_client: Any,
    model_name: str,
) -> OrchestratorHints:
    """Run the orchestrator on a single transcript. Fails open: any
    exception returns empty hints so the main /chat call proceeds normally.

    `gemini_client` is the proxy's google-genai Client (with Vertex auth
    already wired). `model_name` should be Flash Lite or similar — this is
    a fast pre-pass, not the user-facing turn."""
    if not transcript or not transcript.strip():
        return OrchestratorHints()

    try:
        model = _build_vertex_model(gemini_client, model_name)
        agent = _agent_for(model)
        deps = OrchestratorDeps(transcript=transcript)

        result = await agent.run(
            user_prompt=transcript,
            deps=deps,
        )

        hints = result.output
        # Defensive: ensure the deps-recorded side effects are reflected on
        # the returned hints even if the model forgot to copy them.
        if not hints.wiki_snippets and deps.wiki_snippets:
            hints.wiki_snippets = deps.wiki_snippets
        if not hints.corsair_results and deps.corsair_results:
            hints.corsair_results = deps.corsair_results
        if not hints.core_memory_writes and deps.core_memory_writes:
            hints.core_memory_writes = deps.core_memory_writes
        return hints

    except Exception:  # noqa: BLE001 — fail open, log only
        log.exception("orchestrator pre-pass failed; returning empty hints")
        return OrchestratorHints()
