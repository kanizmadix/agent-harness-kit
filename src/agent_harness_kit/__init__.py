"""A portable orchestration loop and supervisor pattern.

The top-level package exports the zero-dependency core contracts plus
``OpenAIBackend`` and ``StrandsBackend``. Their modules are safe to import:
optional SDKs are loaded lazily only when a backend is constructed. Claude,
LangGraph, and CrewAI backends remain explicit submodule imports.

As a result, ``import agent_harness_kit`` succeeds whether or not any optional
provider or framework SDK is installed.
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

__version__ = "0.1.1"
