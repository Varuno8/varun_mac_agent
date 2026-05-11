"""
Post-task LLM Wiki summarizer.

Runs after `[TASK_DONE]` is observed in a Gemini response. Uses Flash Lite
(cheap, fast) to classify whether the task was wiki-worthy and, if so,
extract a structured {wiki_path, content} to append.

Per the approved plan:
- wiki_worthy: bool classifier gates ALL writes (avoid pollution)
- by-topic flat files (no time buckets, no cross-links)
- writes go through agent_memory.wiki_append (path-traversal blocked)

Designed to be called in a background thread from proxy.py — never blocks
the /chat response. Failures are logged, never raised back to caller.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field

from agent_memory import wiki_append, wiki_list, WIKI_TOPIC_SKELETON


log = logging.getLogger("wiki_summarizer")


# ── Classifier output schema ─────────────────────────────────────────────────


class WikiWorthyVerdict(BaseModel):
    """Flash Lite's structured judgement on a finished task."""

    wiki_worthy: bool = Field(
        description=(
            "True if this task taught the agent something durable: a new "
            "preference, a debugging fix, a working AppleScript pattern, a "
            "fact about a person/project. False for trivial actions (open "
            "Slack, send a one-off message) or already-known patterns."
        )
    )
    wiki_path: str | None = Field(
        default=None,
        description=(
            "Relative path under ~/.clicky_wiki/, e.g. 'projects/clicky.md' "
            "or 'debugging/macos_permissions.md'. Required if wiki_worthy."
        ),
    )
    content: str | None = Field(
        default=None,
        description=(
            "The note to append (markdown, 1-3 sentences). Should be dense "
            "and self-contained — the agent reads it later with no other "
            "context. Required if wiki_worthy."
        ),
    )
    reason: str = Field(
        description="One sentence explaining the verdict (for telemetry)."
    )


CLASSIFIER_SYSTEM_PROMPT = """\
You decide if a just-completed agent task left behind a durable lesson worth
saving to the agent's persistent knowledge wiki. You return JSON matching
the WikiWorthyVerdict schema.

WRITE (wiki_worthy=true) when the task revealed:
- A new preference of the user (e.g. "prefers Discord over Slack for team chats")
- A debugging fix or workaround (e.g. "screen capture failed until I enabled the new accessibility permission")
- A working AppleScript / Hotkey / sequence for a specific app
- A fact about a project, person, or tool the agent didn't already know
- A trap or pitfall to avoid next time

SKIP (wiki_worthy=false) when:
- The task was a trivial one-shot (open app, send a one-line message)
- The pattern was already obvious from the system prompt
- Nothing surprising happened

Use the existing wiki structure when possible — prefer appending to:
- projects/<name>.md for project-specific facts
- preferences/*.md for user preferences
- debugging/<topic>.md for fixes and workarounds
- people/<name>.md for facts about contacts
- tools/<name>.md for tool/integration notes

If the right file doesn't exist, propose a new sensible path. Keep the
content to 1-3 dense sentences. No headings, no fluff."""


def _format_classifier_input(
    user_transcript: str,
    agent_response: str,
    actions_taken: list[str],
) -> str:
    existing_files = wiki_list() or list(WIKI_TOPIC_SKELETON)
    actions_summary = ", ".join(actions_taken[:8]) or "(none recorded)"
    return f"""\
TASK TRANSCRIPT
User said: {user_transcript or '(empty)'}
Agent replied: {agent_response or '(empty)'}
Actions taken: {actions_summary}

EXISTING WIKI FILES (prefer appending to these):
{chr(10).join(existing_files)}

Decide: wiki_worthy?
"""


# ── Main entry point ─────────────────────────────────────────────────────────


def maybe_write_wiki(
    *,
    gemini_client,
    classifier_model: str,
    user_transcript: str,
    agent_response: str,
    actions_taken: list[str],
) -> WikiWorthyVerdict | None:
    """Classify the just-finished task and, if worthy, append to the wiki.

    Returns the verdict for telemetry, or None if the classifier call
    failed entirely. Never raises — wiki writes are best-effort.

    Caller should invoke this in a background thread (e.g. via BQ_EXECUTOR
    or a fresh ThreadPoolExecutor) so it doesn't block the /chat response.
    """
    try:
        prompt = _format_classifier_input(
            user_transcript=user_transcript,
            agent_response=agent_response,
            actions_taken=actions_taken,
        )

        # google-genai supports response_schema for structured JSON output.
        # We call directly (not via pydantic-ai) so this module stays
        # framework-independent and can be reused outside the agent loop.
        from google.genai import types as genai_types

        response = gemini_client.models.generate_content(
            model=classifier_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=CLASSIFIER_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=WikiWorthyVerdict,
                temperature=0.2,
            ),
        )

        raw = response.text
        if not raw:
            log.warning("wiki classifier returned empty response")
            return None

        verdict = WikiWorthyVerdict.model_validate_json(raw)

        if verdict.wiki_worthy:
            if not (verdict.wiki_path and verdict.content):
                log.warning(
                    "wiki classifier said wiki_worthy=true but omitted "
                    "wiki_path or content; skipping write. reason=%s",
                    verdict.reason,
                )
                return verdict
            ok = wiki_append(verdict.wiki_path, verdict.content)
            log.info(
                "📚 wiki append: path=%s ok=%s reason=%s",
                verdict.wiki_path, ok, verdict.reason,
            )
        else:
            log.info("📚 wiki skip: %s", verdict.reason)

        return verdict

    except Exception:  # noqa: BLE001 — best-effort, never break the loop
        log.exception("wiki_summarizer failed")
        return None
