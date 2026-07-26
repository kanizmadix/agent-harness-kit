"""agent_harness_kit — a portable orchestration loop + supervisor pattern.

This top-level import is intentionally cheap: it only pulls in ``core`` and
``backends.base``, neither of which depend on any third-party package. It
does NOT import ``backends.claude_backend``, ``backends.langgraph_backend``,
or ``backends.crewai_backend`` — those are optional and framework-specific,
and are meant to be imported explicitly by whoever actually wants to use
them, e.g.::

    from agent_harness_kit.backends.claude_backend import ClaudeBackend

``StrandsBackend`` and ``OpenAIBackend`` ARE exported below, at the caller's
request, alongside the other top-level names. This is safe for the same
reason ``ClaudeBackend`` importing cleanly is safe: both only import their
underlying SDK (``strands`` / ``openai``) lazily, inside ``__init__``, so
importing the module here never requires either package to be installed —
only constructing a backend instance without the package (and without
passing your own ``client=``) does.

This means ``import agent_harness_kit`` always succeeds, with or without
``anthropic``/``langgraph``/``crewai``/``strands-agents``/``openai`` installed.
"""

from agent_harness_kit.backends.base import AgentBackend, StepResult
from agent_harness_kit.backends.openai_backend import OpenAIBackend
from agent_harness_kit.backends.strands_backend import StrandsBackend
from agent_harness_kit.core.context import HarnessContext
from agent_harness_kit.core.loop import HarnessLoop
from agent_harness_kit.core.memory import InMemoryProvider, MemoryProvider, SQLiteMemoryProvider

__all__ = [
    "HarnessLoop",
    "HarnessContext",
    "AgentBackend",
    "StepResult",
    "MemoryProvider",
    "InMemoryProvider",
    "SQLiteMemoryProvider",
    "StrandsBackend",
    "OpenAIBackend",
]

__version__ = "0.1.0"
