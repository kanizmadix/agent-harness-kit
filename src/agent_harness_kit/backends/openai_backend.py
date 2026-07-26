"""A real, working AgentBackend that wraps the OpenAI Chat Completions API.

Usable as either the supervisor (asked to emit a routing decision as JSON)
or a plain sub-agent (asked to produce free-text or a tool call as its
output). Like ``ClaudeBackend``, this is deliberately a *single* completion
call per ``run_step`` — the agentic looping (deciding whether to call this
backend again) is the ``HarnessLoop``'s job, not this class's.

Why this is fully implemented rather than a TODO stub (unlike
``LangGraphBackend``/``CrewAIBackend``/``StrandsBackend``): those three wrap
frameworks that own their *own* internal agent loop (a compiled graph, a
crew's ``kickoff``, a Strands agent's internal tool-calling loop) —
translating a generic ``HarnessContext`` into their bespoke invocation shape
is a real integration decision best left to whoever is wiring in the actual
graph/crew/agent object. OpenAI's Chat Completions API, by contrast, is a
single stateless request/response call with (almost) the same shape as the
Anthropic Messages API that ``ClaudeBackend`` already wraps: a list of
role/content turns, an optional system prompt, one completion, and — for the
supervisor role — a JSON routing decision parsed out of the reply. Writing
that stub's docstring would have taken about as much effort as writing the
real thing, so this class is a second fully working reference
implementation instead of a stub.

The ``openai`` package is imported lazily, inside ``__init__``, so that
importing this module — or ``agent_harness_kit`` as a whole — never requires
``openai`` to be installed. It's only required if you actually construct an
``OpenAIBackend`` without passing your own ``client=``.
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


class OpenAIBackend:
    """Wraps ``openai.OpenAI`` behind the ``AgentBackend`` protocol.

    Pass ``is_supervisor=True`` to have the model emit routing decisions
    (parsed either from a tool/function call, if ``tools`` are configured, or
    from a JSON object in the response text) instead of a plain completion.

    Requires the ``openai`` package (``pip install "agent-harness-kit[openai]"``
    or plain ``pip install openai``) and an API key, read from the
    ``OPENAI_API_KEY`` environment variable by default.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
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
            model: OpenAI model id to call (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
            system_prompt: Base system prompt. When ``is_supervisor=True``,
                the routing instructions are appended to this.
            is_supervisor: Whether this backend should be prompted to emit a
                routing decision (``status``/``next_agent``/``payload``)
                rather than a plain answer.
            agent_names: Names of the agents this supervisor may route to —
                only used to render the system prompt when ``is_supervisor``.
            tools: Optional tool/function definitions (Chat Completions
                ``tools=`` shape) to pass on every call, enabling
                function-calling completions.
            api_key: Explicit API key. If omitted, falls back to the
                ``OPENAI_API_KEY`` environment variable (read by the SDK
                client itself if also omitted here).
            max_tokens: Max tokens for the completion (passed to the API as
                ``max_completion_tokens``).
            client: An already-constructed ``openai.OpenAI``-compatible
                client. Supplying this skips the lazy import entirely — this
                is how tests inject a fake client with no network access and
                no ``openai`` package required (mirrors ``ClaudeBackend``).
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
            import openai  # lazy import: only required when actually instantiated

            self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def _system(self) -> str:
        if not self.is_supervisor:
            return self.system_prompt
        instructions = _SUPERVISOR_INSTRUCTIONS.format(agents=", ".join(self.agent_names) or "(none registered)")
        return f"{self.system_prompt}\n\n{instructions}".strip()

    def _build_messages(self, context: HarnessContext) -> list[dict[str, Any]]:
        """Translate ``HarnessContext.messages`` into Chat Completions turns.

        The harness records messages under whatever role produced them
        (``"user"``, an agent's name, ``"assistant"``, ...) but Chat
        Completions only knows ``system``/``user``/``assistant``/``tool``.
        Everything that isn't ``"assistant"`` is folded into a ``user`` turn.
        """
        messages: list[dict[str, Any]] = []
        system = self._system()
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(
            {
                "role": "assistant" if m["role"] == "assistant" else "user",
                "content": str(m["content"]),
            }
            for m in context.messages
        )
        return messages

    def run_step(self, context: HarnessContext) -> StepResult:
        """Make one completion call and translate it into a StepResult."""
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=self._build_messages(context),
        )
        if self.tools:
            kwargs["tools"] = self.tools

        response = self._client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        text = (message.content or "").strip()
        tool_calls = getattr(message, "tool_calls", None) or []
        tool_call = tool_calls[0] if tool_calls else None

        if not self.is_supervisor:
            if tool_call is not None:
                payload = {
                    "tool_name": tool_call.function.name,
                    "tool_input": json.loads(tool_call.function.arguments),
                }
                return StepResult(status="done", payload=payload, message=text or None)
            return StepResult(status="done", payload=text, message=text or None)

        # Supervisor: expect a routing decision, either as a tool call or as JSON text.
        if tool_call is not None:
            decision = json.loads(tool_call.function.arguments)
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
