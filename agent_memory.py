"""
Tiered memory implementation for the dual-track architecture.

Five tiers, per the approved plan (text-system-role-curious-storm.md):

  T1 Working    — pydantic-ai message_history (handled by the Agent, not here)
  T2 Core       — ~/.clicky_core_memory.json   (Letta-style, agent-writeable)
  T3 Context    — micky_context.md             (compiled KB, existing, unchanged)
  T4 LLM Wiki   — ~/.clicky_wiki/**.md         (Karpathy archival, on-demand read)
  T5 Episodic   — ~/.micky_memory.json         (existing, slated for deprecation)

This module owns T2 and T4. T3 stays in proxy.py's existing assembly path.
T5 is read/written by proxy.py today; left as-is until Phase 6 audit.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from agent_schemas import CoreMemoryFact, WikiSearchHit

# Filesystem locations for the memory tiers. Owned by this module so we
# avoid a circular import with agent_runtime (which re-exports from here).
CORE_MEMORY_PATH = Path.home() / ".clicky_core_memory.json"
WIKI_ROOT = Path.home() / ".clicky_wiki"
EPISODIC_PATH = Path.home() / ".micky_memory.json"  # existing, owned by proxy.py


# ── Token estimator ──────────────────────────────────────────────────────────
#
# Core memory has a hard cap (~300 tokens). We don't want to import tiktoken
# just for this — Gemini doesn't use BPE the same way anyway. A rough
# character-to-token ratio (4 chars/token for English) is good enough for
# eviction policy. Conservative: slightly overestimate to evict sooner.


def _rough_token_count(text: str) -> int:
    return max(1, len(text) // 3)


# ── T2: Core memory (Letta-style) ────────────────────────────────────────────

CORE_MEMORY_TOKEN_CAP = 300


def _load_core_facts() -> list[CoreMemoryFact]:
    try:
        raw = json.loads(CORE_MEMORY_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [CoreMemoryFact.model_validate(item) for item in raw if isinstance(item, dict)]


def _save_core_facts(facts: list[CoreMemoryFact]) -> None:
    CORE_MEMORY_PATH.write_text(json.dumps(
        [f.model_dump() for f in facts],
        indent=2,
        ensure_ascii=False,
    ))


def core_memory_read_block() -> str:
    """Render the full core memory as a prompt-ready block. Empty string
    if no facts. Goes AFTER the static layers (base + context md) so cache
    prefixes survive `core_memory_append`."""
    facts = _load_core_facts()
    if not facts:
        return ""
    lines = ["── CORE MEMORY (persistent facts about you and the user) ──"]
    for f in facts:
        prefix = "★ " if f.pinned else "• "
        lines.append(f"{prefix}{f.fact}")
    lines.append("── END CORE MEMORY ──")
    return "\n".join(lines)


def core_memory_append(fact: str, pinned: bool = False) -> bool:
    """Add a fact. Evicts oldest non-pinned fact if total exceeds token cap.
    Returns True if the fact was stored, False if it was a duplicate (no-op)."""
    fact = fact.strip()
    if not fact:
        return False

    facts = _load_core_facts()

    # Dedup: case-insensitive equality
    for existing in facts:
        if existing.fact.lower() == fact.lower():
            return False

    facts.append(CoreMemoryFact(fact=fact, pinned=pinned))

    # Evict oldest non-pinned facts until under cap
    def total_tokens(fs: list[CoreMemoryFact]) -> int:
        return sum(_rough_token_count(f.fact) for f in fs)

    while total_tokens(facts) > CORE_MEMORY_TOKEN_CAP:
        for i, f in enumerate(facts):
            if not f.pinned:
                facts.pop(i)
                break
        else:
            break  # all pinned — give up rather than drop pinned

    _save_core_facts(facts)
    return True


def core_memory_seed_if_empty() -> int:
    """Seed core memory with facts derived from the user's prompt the first
    time this runs. Returns count of facts seeded. No-op if file exists."""
    if CORE_MEMORY_PATH.exists():
        return 0

    seed: list[CoreMemoryFact] = [
        CoreMemoryFact(fact="user is Varun (garimatyagi268@gmail.com)", pinned=True),
        CoreMemoryFact(fact="user prefers VS Code", pinned=False),
        CoreMemoryFact(fact="user is actively building TryOwn with Next.js 16", pinned=False),
        CoreMemoryFact(fact="Diagnxt is in prototype phase and has not been trained on real data", pinned=False),
        CoreMemoryFact(fact="prefer APIs (Corsair MCP) over UI/OCR navigation when both are possible", pinned=True),
    ]
    _save_core_facts(seed)
    return len(seed)


# ── T4: LLM Wiki (Karpathy archival) ─────────────────────────────────────────

WIKI_TOPIC_SKELETON = [
    "projects/clicky.md",
    "projects/tryown.md",
    "projects/diagnxt.md",
    "preferences/coding_style.md",
    "preferences/communication.md",
    "debugging/macos_permissions.md",
    "debugging/applescript_quirks.md",
    "people/garima.md",
    "tools/corsair_integrations.md",
]

WIKI_MAX_HITS = 3
WIKI_SNIPPET_MAX_CHARS = 4000  # cap per-file content returned to LLM


def _sanitize_wiki_path(path: str) -> Path:
    """Resolve `path` against WIKI_ROOT, refuse anything that escapes the root."""
    candidate = (WIKI_ROOT / path).resolve()
    root = WIKI_ROOT.resolve()
    if not str(candidate).startswith(str(root)):
        raise ValueError(f"wiki path escapes root: {path!r}")
    if candidate.suffix != ".md":
        candidate = candidate.with_suffix(".md")
    return candidate


def wiki_bootstrap() -> int:
    """Create ~/.clicky_wiki/ + empty topic skeleton if absent. Returns
    count of files created (0 if everything already exists)."""
    WIKI_ROOT.mkdir(parents=True, exist_ok=True)
    created = 0
    for rel in WIKI_TOPIC_SKELETON:
        path = _sanitize_wiki_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"# {path.stem}\n\n")
            created += 1
    return created


def search_wiki(query: str) -> list[WikiSearchHit]:
    """Grep over filenames + content under ~/.clicky_wiki/. Returns up to
    WIKI_MAX_HITS hits, scored by (filename match weight 3) + (content
    match weight 1). Per-file content truncated to WIKI_SNIPPET_MAX_CHARS."""
    if not WIKI_ROOT.exists():
        return []

    terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) >= 3]
    if not terms:
        return []

    scored: list[tuple[int, Path, str]] = []
    for md in WIKI_ROOT.rglob("*.md"):
        try:
            content = md.read_text(errors="replace")
        except OSError:
            continue
        name_lc = md.name.lower()
        content_lc = content.lower()
        score = 0
        for t in terms:
            score += 3 * name_lc.count(t)
            score += content_lc.count(t)
        if score > 0:
            scored.append((score, md, content))

    scored.sort(key=lambda x: -x[0])
    hits: list[WikiSearchHit] = []
    for _, path, content in scored[:WIKI_MAX_HITS]:
        rel = str(path.relative_to(WIKI_ROOT))
        snippet = content[:WIKI_SNIPPET_MAX_CHARS]
        if len(content) > WIKI_SNIPPET_MAX_CHARS:
            snippet += f"\n\n[truncated — file is {len(content)} chars]"
        hits.append(WikiSearchHit(path=rel, content=snippet))
    return hits


def wiki_append(path: str, content: str) -> bool:
    """Append `content` to ~/.clicky_wiki/<path>.md. Creates the file (and
    parent dirs) if missing. Returns True on success."""
    target = _sanitize_wiki_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    body = content.strip()
    if not body:
        return False

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    block = f"\n## {timestamp}\n\n{body}\n"

    with target.open("a", encoding="utf-8") as fh:
        fh.write(block)
    return True


def wiki_list() -> list[str]:
    """List all wiki files (relative paths). Used by the agent to discover
    what topics exist before searching."""
    if not WIKI_ROOT.exists():
        return []
    return sorted(
        str(p.relative_to(WIKI_ROOT))
        for p in WIKI_ROOT.rglob("*.md")
    )
