"""Adapter stub for wrapping a Strands Agent (AWS Bedrock AgentCore-compatible)
as an AgentBackend.

Strands Agents (the ``strands-agents`` PyPI package, imported as ``strands``)
is AWS's open-source, model-driven agent SDK. Agents built with it can be
deployed directly to Bedrock AgentCore Runtime, but the SDK itself has no
required dependency on AWS — it works against any supported model provider.

Strands is not a dependency of this package — importing this module never
requires ``strands`` to be installed. The ``strands`` import only happens
inside ``__init__``, and only to fail loudly with an actionable
``ImportError`` if you try to actually construct a ``StrandsBackend``
without the package installed.
"""

from __future__ import annotations

from typing import Any

from agent_harness_kit.backends.base import StepResult
from agent_harness_kit.core.context import HarnessContext


class StrandsBackend:
    """Wraps a native Strands ``Agent`` instance.

    TODO for a real integration — fill in ``run_step``:

      1. **Build the invocation input.** A Strands agent is invoked simply
         by calling it: ``result = agent("some prompt string")``, or with
         structured content via ``agent(prompt=..., ...)`` depending on SDK
         version. Build the prompt from
         ``context.scratchpad["dispatch_payload"]`` (set by ``HarnessLoop``
         right before this agent's ``run_step`` is called), plus optionally
         the recent turns of ``context.messages`` if the agent should see
         conversational history rather than just the latest dispatch.

      2. **Invoke the agent.**
         ``result = self.agent(prompt)``
         This runs the Strands agent's own internal model-driven loop
         (tool calls, reasoning steps, etc. are handled *inside* Strands) and
         returns an ``AgentResult``-like object once the agent produces a
         final response. For streaming, Strands exposes an async streaming
         entry point (``agent.stream_async(prompt)``) if you want incremental
         output instead of blocking for the full result.

      3. **Map the result back onto a StepResult.** The final assistant text
         is normally reachable via the result's ``message`` (or
         ``str(result)``, which Strands typically implements to return the
         final text) — put it in ``StepResult.message`` and set
         ``status="done"``. If the agent exposes structured output (e.g. via
         a Pydantic ``output_model``), prefer putting that structured value
         in ``StepResult.payload`` instead.

      4. **AgentCore Runtime note.** If this Strands agent is deployed behind
         AWS Bedrock AgentCore Runtime rather than run in-process, swap the
         direct ``self.agent(prompt)`` call for an AgentCore invocation (e.g.
         via the ``bedrock-agentcore`` client, calling the deployed runtime's
         invoke endpoint with the same prompt/session shape) — the rest of
         this class's contract (build input from the harness context, map
         the response back onto a ``StepResult``) is unchanged either way.

      5. **Errors.** Catch exceptions from the call and translate them into
         ``StepResult(status="error", message=str(exc))`` so ``HarnessLoop``
         can surface them as a ``HarnessError`` rather than letting a
         third-party exception escape ``run_step`` uncaught.
    """

    def __init__(self, agent: Any) -> None:
        """
        Args:
            agent: A native ``strands.Agent`` instance (already configured
                with its model, tools, and system prompt).
        """
        try:
            import strands  # noqa: F401  (presence check only; not used directly)
        except ImportError as exc:
            raise ImportError(
                "StrandsBackend requires the 'strands-agents' package, which is not "
                "installed. Install it with `pip install \"agent-harness-kit[strands]\"` "
                "or `pip install strands-agents`."
            ) from exc
        self.agent = agent

    def run_step(self, context: HarnessContext) -> StepResult:
        """# TODO: implement the real Strands Agent invocation described above."""
        raise NotImplementedError(
            "StrandsBackend.run_step is a stub. Call self.agent(prompt) using a "
            "prompt built from context.scratchpad['dispatch_payload'], then wrap "
            "the AgentResult in a StepResult(status='done', ...). See this class's "
            "docstring for the full integration sketch, including the Bedrock "
            "AgentCore Runtime deployment variant."
        )
