"""A real, working AgentBackend that wraps the Anthropic Claude API.

Usable as either the supervisor (asked to emit a routing decision as JSON)
or a plain sub-agent (asked to produce free-text or a tool call as its
output). This is deliberately a *single* completion call per ``run_step`` —
the agentic looping (deciding whether to call this backend again) is the
``HarnessLoop``'s job, not this class's.

The ``anthropic`` package is imported lazily, inside ``__init__``, so that
importing this module — or ``agent_harness_kit`` as a whole — never requires
``anthropic`` to be installed. It's only required if you actually construct
a ``ClaudeBackend``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent_harness_kit.backends.base import StepResult
from agent_harness_kit.core.context import HarnessContext

_SUPERVISOR_INSTRUCTIONS = """\
You are the supervisor of a multi-agent system. Given the conversation so far \
and the set of available agents, decide what happens next.

Available agents: {agents}

Respond with ONLY a JSON object, no other text, matching one of:
  {{"status": "continue", "next_agent": "<agent name>", "payload": <any JSON value>}}
  {{"status": "done", "message": "<final answer for the user>"}}
"""


class ClaudeBackend:
    """Wraps ``anthropic.Anthropic`` behind the ``AgentBackend`` protocol.

    Pass ``is_supervisor=True`` to have the model emit routing decisions
    (parsed either from a ``tool_use`` block, if ``tools`` are configured, or
    from a JSON object in the response text) instead of a plain completion.

    Requires the ``anthropic`` package (``pip install "agent-harness-kit[anthropic]"``
    or plain ``pip install anthropic``) and an API key, read from the
    ``ANTHROPIC_API_KEY`` environment variable by default.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        system_prompt: str = "",
        is_supervisor: bool = False,
        agent_names: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        client: Any = None,
    ) -> None:
        """
        Args:
            model: Claude model id to call.
            system_prompt: Base system prompt. When ``is_supervisor=True``,
                the routing instructions are appended to this.
            is_supervisor: Whether this backend should be prompted to emit a
                routing decision (``status``/``next_agent``/``payload``)
                rather than a plain answer.
            agent_names: Names of the agents this supervisor may route to —
                only used to render the system prompt when ``is_supervisor``.
            tools: Optional tool definitions (Anthropic Messages API shape)
                to pass on every call, enabling tool-use-capable completions.
            api_key: Explicit API key. If omitted, falls back to the
                ``ANTHROPIC_API_KEY`` environment variable (read by the SDK
                client itself if also omitted here).
            max_tokens: Max tokens for the completion.
            client: An already-constructed ``anthropic.Anthropic``-compatible
                client. Supplying this skips the lazy import entirely — this
                is how tests inject a fake client with no network access and
                no ``anthropic`` package required.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.is_supervisor = is_supervisor
        self.agent_names = agent_names or []
        self.tools = tools
        self.max_tokens = max_tokens

        if client is not None:
            self._client = client
        else:
            import anthropic  # lazy import: only required when actually instantiated

            self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def _system(self) -> str:
        if not self.is_supervisor:
            return self.system_prompt
        instructions = _SUPERVISOR_INSTRUCTIONS.format(agents=", ".join(self.agent_names) or "(none registered)")
        return f"{self.system_prompt}\n\n{instructions}".strip()

    def _build_messages(self, context: HarnessContext) -> list[dict[str, Any]]:
        """Translate ``HarnessContext.messages`` into Messages API turns.

        The harness records messages under whatever role produced them
        (``"user"``, an agent's name, ``"assistant"``, ...) but the Messages
        API only knows ``user``/``assistant``. Everything that isn't
        ``"assistant"`` is folded into a ``user`` turn.
        """
        return [
            {
                "role": "assistant" if m["role"] == "assistant" else "user",
                "content": str(m["content"]),
            }
            for m in context.messages
        ]

    def run_step(self, context: HarnessContext) -> StepResult:
        """Make one tool-use-capable completion call and translate it into a StepResult."""
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system(),
            messages=self._build_messages(context),
        )
        if self.tools:
            kwargs["tools"] = self.tools

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_use_block = None
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use" and tool_use_block is None:
                tool_use_block = block
        text = "".join(text_parts).strip()

        if not self.is_supervisor:
            if tool_use_block is not None:
                payload = {"tool_name": tool_use_block.name, "tool_input": tool_use_block.input}
                return StepResult(status="done", payload=payload, message=text or None)
            return StepResult(status="done", payload=text, message=text or None)

        # Supervisor: expect a routing decision, either as a tool call or as JSON text.
        if tool_use_block is not None:
            decision = dict(tool_use_block.input)
        else:
            try:
                decision = json.loads(text)
            except json.JSONDecodeError:
                return StepResult(
                    status="error",
                    message=f"Supervisor did not return a valid routing decision: {text!r}",
                )

        status = decision.get("status")
        if status not in ("continue", "done", "error"):
            return StepResult(
                status="error",
                message=f"Supervisor returned an unrecognized status: {status!r}",
            )

        return StepResult(
            status=status,
            next_agent=decision.get("next_agent"),
            payload=decision.get("payload"),
            message=decision.get("message"),
        )
