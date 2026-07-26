"""Portable state passed between the harness loop and every agent backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HarnessContext:
    """Everything an agent needs to know about the run so far.

    ``messages`` is the full conversation history (list of ``{"role", "content"}``
    dicts). ``scratchpad`` holds transient per-step notes any backend can read or
    write. ``artifacts`` holds named outputs produced along the way (files, JSON
    blobs, generated code). ``tools_state`` records which tools/agents have been
    invoked and with what result, so a supervisor can avoid redundant calls.
    """

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    tools_state: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: Any) -> None:
        self.messages.append({"role": role, "content": content})

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "scratchpad": self.scratchpad,
            "artifacts": self.artifacts,
            "tools_state": self.tools_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessContext":
        return cls(
            session_id=data["session_id"],
            messages=data.get("messages", []),
            scratchpad=data.get("scratchpad", {}),
            artifacts=data.get("artifacts", {}),
            tools_state=data.get("tools_state", {}),
        )
