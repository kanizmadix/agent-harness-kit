"""Adapter stub for wrapping a compiled LangGraph graph as an AgentBackend.

LangGraph is not a dependency of this package — importing this module never
requires ``langgraph`` to be installed. The ``langgraph`` import only
happens inside ``__init__``, and only to fail loudly with an actionable
``ImportError`` if you try to actually construct a ``LangGraphBackend``
without the package installed.
"""

from __future__ import annotations

from typing import Any

from agent_harness_kit.backends.base import StepResult
from agent_harness_kit.core.context import HarnessContext


class LangGraphBackend:
    """Wraps a compiled LangGraph graph (the result of ``StateGraph(...).compile()``).

    TODO for a real integration — fill in ``run_step``:

      1. **Translate context -> graph input state.** LangGraph's conventional
         supervisor pattern (see the "Multi-agent supervisor" tutorial in the
         LangGraph docs) expects a state dict with a ``messages`` key holding
         a list of LangChain ``BaseMessage`` objects. Convert
         ``context.messages`` (``[{"role": ..., "content": ...}]``) into
         ``HumanMessage`` / ``AIMessage`` instances, and pull anything the
         graph needs out of ``context.scratchpad["dispatch_payload"]``
         (set by ``HarnessLoop`` right before this agent's ``run_step`` is
         called) as an additional state key.

      2. **Invoke the graph.**
         ``result = self.graph.invoke({"messages": lc_messages, ...})``
         Use ``self.graph.stream(...)`` instead if you want to surface
         incremental output (e.g. forwarding chunks to a UI) rather than
         blocking for the full run.

      3. **Map the graph's final state back onto a StepResult.** Typically
         ``result["messages"][-1]`` is the graph's final answer — put its
         content in ``StepResult.message`` (or ``.payload`` if it's
         structured data) and set ``status="done"``.

      4. **Nesting note.** If this graph itself implements a LangGraph
         "supervisor" node that fans out to LangGraph sub-agents internally,
         you don't need to expose those sub-agents to ``HarnessLoop`` at
         all — from the harness's point of view this backend is one opaque
         agent. Only build a ``HarnessLoop``-level supervisor/multi-agent
         split if you want the harness (rather than LangGraph) to own the
         routing decisions.
    """

    def __init__(self, graph: Any) -> None:
        """
        Args:
            graph: A compiled LangGraph graph (``StateGraph(...).compile()``).
        """
        try:
            import langgraph  # noqa: F401  (presence check only; not used directly)
        except ImportError as exc:
            raise ImportError(
                "LangGraphBackend requires the 'langgraph' package, which is not "
                "installed. Install it with `pip install \"agent-harness-kit[langgraph]\"` "
                "or `pip install langgraph`."
            ) from exc
        self.graph = graph

    def run_step(self, context: HarnessContext) -> StepResult:
        """# TODO: implement the real LangGraph invocation described above."""
        raise NotImplementedError(
            "LangGraphBackend.run_step is a stub. Translate context.messages / "
            "context.scratchpad['dispatch_payload'] into the graph's input state, "
            "call self.graph.invoke(...), and map the result onto a StepResult. "
            "See this class's docstring for the full integration sketch."
        )
