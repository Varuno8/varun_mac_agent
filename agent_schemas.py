"""
Pydantic schemas for the dual-track architecture.

These types are the contract between:
  - Gemini's tool calls (validated by pydantic-ai)
  - The Swift executor (consumes legacy bracket-tag wire format)

Phase 1 deliverable: schemas exist, are importable, and round-trip cleanly
to the existing bracket-tag wire format. proxy.py is NOT yet using them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


ActionType = Literal[
    "click",        # [CLICK:x,y]
    "axclick",      # [AXCLICK:label]
    "dblclick",     # [DBLCLICK:x,y]
    "rclick",       # [RCLICK:x,y]
    "type",         # [TYPE:text]
    "hotkey",       # [HOTKEY:cmd+s]
    "scroll",       # [SCROLL:direction,amount,x,y]
    "applescript",  # [APPLESCRIPT:source]
    "screenshot",   # [SCREENSHOT]
    "wait",         # [WAIT:ms]
]


class MacAction(BaseModel):
    """One physical action for Swift to execute. Schema-validated in Python,
    encoded to the legacy bracket-tag wire format before crossing to Swift
    so AgenticActionExecutor.swift sees no change."""

    action_type: ActionType
    x: int | None = None
    y: int | None = None
    text: str | None = Field(default=None, description="For type / applescript / hotkey")
    ax_label: str | None = Field(default=None, description="For axclick")
    scroll_direction: Literal["up", "down", "left", "right"] | None = None
    scroll_amount: int | None = None
    wait_ms: int | None = None

    @model_validator(mode="after")
    def _shape_matches_action(self) -> "MacAction":
        a = self.action_type
        if a == "click" or a == "dblclick" or a == "rclick":
            if self.x is None or self.y is None:
                raise ValueError(f"{a} requires x and y")
        elif a == "axclick":
            if not self.ax_label:
                raise ValueError("axclick requires ax_label")
        elif a == "type":
            if not self.text:
                raise ValueError("type requires text")
        elif a == "hotkey":
            if not self.text:
                raise ValueError("hotkey requires text (e.g. 'cmd+s')")
        elif a == "applescript":
            if not self.text:
                raise ValueError("applescript requires text (the script source)")
        elif a == "scroll":
            if self.scroll_direction is None or self.scroll_amount is None:
                raise ValueError("scroll requires scroll_direction and scroll_amount")
        elif a == "wait":
            if self.wait_ms is None:
                raise ValueError("wait requires wait_ms")
        return self

    def to_bracket_tag(self) -> str:
        """Encode to the legacy wire format Swift's AgenticActionExecutor expects."""
        a = self.action_type
        if a == "click":
            return f"[CLICK:{self.x},{self.y}]"
        if a == "dblclick":
            return f"[DBLCLICK:{self.x},{self.y}]"
        if a == "rclick":
            return f"[RCLICK:{self.x},{self.y}]"
        if a == "axclick":
            return f"[AXCLICK:{self.ax_label}]"
        if a == "type":
            return f"[TYPE:{self.text}]"
        if a == "hotkey":
            return f"[HOTKEY:{self.text}]"
        if a == "scroll":
            x_part = f",{self.x}" if self.x is not None else ""
            y_part = f",{self.y}" if self.y is not None else ""
            xy = f"{x_part}{y_part}"
            return f"[SCROLL:{self.scroll_direction},{self.scroll_amount}{xy}]"
        if a == "applescript":
            return f"[APPLESCRIPT:{self.text}]"
        if a == "screenshot":
            return "[SCREENSHOT]"
        if a == "wait":
            return f"[WAIT:{self.wait_ms}]"
        raise ValueError(f"unknown action_type {a!r}")


class TaskDone(BaseModel):
    """Agent declares the user's request is complete. Replaces the [TASK_DONE]
    server-side rewrite (proxy.py:1832). Terminates the pydantic-ai loop."""

    summary: str = Field(description="One-sentence summary of what was accomplished")


class TaskAbandon(BaseModel):
    """Agent gives up on the task (e.g. permission denied, integration missing,
    repeatedly failed). Carries the reason so the user gets feedback."""

    reason: str


class CorsairCall(BaseModel):
    """An API call routed through Corsair MCP. Phase 3 wires this to a real
    MCP server; Phase 1 has a MockMCPServer that returns integration_not_available."""

    integration: str = Field(description="e.g. 'slack', 'github', 'gmail', 'gcal'")
    action: str = Field(description="e.g. 'send_message', 'list_prs'")
    params: dict = Field(default_factory=dict)


class CorsairResult(BaseModel):
    """Standardized result envelope returned to Gemini so failures are
    re-plannable (per the 'Gemini decides' principle)."""

    ok: bool
    data: dict | None = None
    error: str | None = None
    retry_after_s: int | None = None


class CoreMemoryFact(BaseModel):
    """A single Letta-style fact in ~/.clicky_core_memory.json."""

    fact: str = Field(min_length=2, max_length=400)
    pinned: bool = False  # pinned facts never evicted when at cap


class WikiSearchHit(BaseModel):
    """One result from search_wiki(query) — a markdown file at ~/.clicky_wiki/path."""

    path: str
    content: str
