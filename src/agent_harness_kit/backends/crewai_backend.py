"""Adapter stub for wrapping a CrewAI Crew as an AgentBackend.

CrewAI is not a dependency of this package — importing this module never
requires ``crewai`` to be installed. The ``crewai`` import only happens
inside ``__init__``, and only to fail loudly with an actionable
``ImportError`` if you try to actually construct a ``CrewAIBackend``
without the package installed.
"""

from __future__ import annotations

from typing import Any

from agent_harness_kit.backends.base import StepResult
from agent_harness_kit.core.context import HarnessContext


class CrewAIBackend:
    """Wraps a configured CrewAI ``Crew`` (sequential or hierarchical process).

    TODO for a real integration — fill in ``run_step``:

      1. **Build the kickoff inputs.** CrewAI crews run via
         ``Crew(agents=[...], tasks=[...], process=Process.sequential | Process.hierarchical)``.
         Tasks reference ``{placeholder}`` variables in their descriptions
         that get filled from the ``inputs`` dict passed to ``kickoff``. Pull
         that dict from ``context.scratchpad["dispatch_payload"]`` (set by
         ``HarnessLoop`` right before this agent's ``run_step`` is called) —
         typically the supervisor's payload already *is* that inputs dict, or
         is close enough to pass through with minor renaming.

      2. **Invoke the crew.**
         ``crew_output = self.crew.kickoff(inputs=dispatch_payload)``
         For a hierarchical process the crew has its own internal manager
         agent doing sub-task delegation — from ``HarnessLoop``'s point of
         view this whole crew is still just one opaque agent backend.

      3. **Map ``CrewOutput`` back onto a StepResult.** ``crew_output.raw``
         is the final text output; ``crew_output.tasks_output`` holds
         per-task results if you want to surface intermediate work in
         ``context.tools_state`` as well. A crew run is expected to fully
         complete its task in one ``kickoff`` call, so this backend should
         always return ``status="done"`` (never ``"continue"`` — CrewAI has
         no notion of pausing a kickoff to hand control back to the caller
         mid-run).

      4. **Errors.** ``kickoff`` raises on failure; catch and translate into
         ``StepResult(status="error", message=str(exc))`` so ``HarnessLoop``
         can surface it as a ``HarnessError`` rather than an unhandled
         exception from deep inside a third-party call.
    """

    def __init__(self, crew: Any) -> None:
        """
        Args:
            crew: A configured ``crewai.Crew`` instance.
        """
        try:
            import crewai  # noqa: F401  (presence check only; not used directly)
        except ImportError as exc:
            raise ImportError(
                "CrewAIBackend requires the 'crewai' package, which is not "
                "installed. Install it with `pip install \"agent-harness-kit[crewai]\"` "
                "or `pip install crewai`."
            ) from exc
        self.crew = crew

    def run_step(self, context: HarnessContext) -> StepResult:
        """# TODO: implement the real Crew.kickoff invocation described above."""
        raise NotImplementedError(
            "CrewAIBackend.run_step is a stub. Call self.crew.kickoff(inputs=...) "
            "using context.scratchpad['dispatch_payload'], then wrap the CrewOutput "
            "in a StepResult(status='done', ...). See this class's docstring for the "
            "full integration sketch."
        )
