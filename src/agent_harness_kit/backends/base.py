"""The interface every agent backend (LangGraph, CrewAI, Strands, raw LLM) implements.

This module has zero third-party dependencies — it only uses the stdlib. Every
concrete backend (``claude_backend``, ``langgraph_backend``, ...) depends on
this module, but this module never depends on them, so importing it (and by
extension importing ``agent_harness_kit`` itself) never requires any
framework SDK to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from agent_harness_kit.core.context import HarnessContext

#: The three states a backend can leave the harness loop in after one step.
StepStatus = Literal["continue", "done", "error"]


@dataclass
class StepResult:
    """What a backend returns after taking one step.

    Attributes:
        status: Drives the loop. ``"continue"`` means the harness should keep
            running — if this result came from the supervisor, ``next_agent``
            names which registered agent to invoke next. ``"done"`` ends the
            run (or, for a sub-agent, ends that agent's turn and hands control
            back to the supervisor). ``"error"`` signals a failure the loop
            should treat as terminal.
        next_agent: When ``status == "continue"`` and this result came from
            the supervisor, the name of the registered agent to call next.
            Ignored for results returned by a non-supervisor agent.
        payload: Arbitrary data to hand off. When the supervisor asks to
            continue, this is passed to the next agent via
            ``context.scratchpad["dispatch_payload"]``. When a backend
            signals ``"done"``, this is typically the agent's final output.
        message: An optional human-readable message — e.g. an error
            explanation, or a short status note. When present it is what
            gets recorded in ``context.messages``; when absent, ``payload``
            is used instead.
    """

    status: StepStatus
    next_agent: str | None = None
    payload: Any = None
    message: str | None = None


@runtime_checkable
class AgentBackend(Protocol):
    """Adapter wrapping a specific framework (or a raw LLM) behind one method.

    Any object with a compatible ``run_step`` method satisfies this protocol
    structurally — you don't need to subclass it.
    """

    def run_step(self, context: HarnessContext) -> StepResult:
        """Take one step given the current harness context, return a StepResult."""
        ...
