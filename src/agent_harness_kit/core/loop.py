"""The supervisor loop: ask the supervisor what to do next, run it, persist, repeat.

This module is intentionally framework-agnostic — it only talks to the
``MemoryProvider`` and ``AgentBackend`` interfaces, never to a specific
framework SDK. All framework-specific logic belongs in ``backends/*``.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from agent_harness_kit.backends.base import AgentBackend, StepResult
from agent_harness_kit.core.context import HarnessContext
from agent_harness_kit.core.memory import MemoryProvider

#: The two error policies HarnessLoop supports when a backend's ``run_step``
#: raises, or when the supervisor names an agent that was never registered.
ErrorPolicy = Literal["stop", "retry"]


class HarnessError(RuntimeError):
    """Raised when a backend (supervisor or agent) reports status='error'."""


class MaxStepsExceededError(RuntimeError):
    """Raised by ``run_supervised`` when ``raise_on_max_steps=True`` and the
    supervisor never signals ``status='done'`` within ``max_steps`` steps."""


class HarnessLoop:
    """Runs the supervisor pattern over a registered set of agent backends.

    On each iteration the supervisor backend is asked what to do next via
    ``supervisor.run_step(context)``. Its ``StepResult`` either:

    - signals ``status="done"``, ending the run, or
    - signals ``status="continue"`` with a ``next_agent``, in which case that
      registered backend's own ``run_step`` is invoked and its result is
      merged into the context before looping back to the supervisor.

    The whole run is bounded by ``max_steps`` so a misbehaving supervisor
    can't loop forever.

    Error policy
    ------------
    Two distinct failure modes are handled the same way, via the ``on_error``
    policy:

    1. A backend's ``run_step`` (the supervisor's, or a dispatched agent's)
       raises an exception instead of returning a ``StepResult``.
    2. The supervisor names a ``next_agent`` that was never registered via
       ``register_agent`` (previously a raw, uncaught ``KeyError``).

    In both cases the loop never lets the exception escape ``run_supervised``.
    Instead it records a structured entry in ``context.scratchpad["errors"]``
    (a list of ``{"step", "error", "attempt"}`` dicts — the same scratchpad
    convention already used for ``"objective"``, ``"dispatch_payload"``, and
    ``"max_steps_exceeded"``), then applies one of two policies:

    - ``on_error="stop"`` (the default): give up after the first failure, set
      ``context.scratchpad["status"] = "failed"`` and
      ``context.scratchpad["failed_step"]`` to the name of the step that
      failed ("supervisor" or the agent's registered name), persist the
      context, and return it — no exception is raised. This is the safer
      default: a broken backend fails fast and visibly instead of silently
      burning through steps or crashing the caller's process.
    - ``on_error="retry"``: re-invoke the *same* step up to ``max_retries``
      additional times (``max_retries + 1`` attempts total) before giving up
      with the same "stop" outcome described above. Every failed attempt —
      including ones that are later retried successfully — appends its own
      entry to ``context.scratchpad["errors"]``, so a caller can see exactly
      how many times, and how, a step failed even if the run ultimately
      succeeded.

    Either way, a caller distinguishes a normal completion from a failed one
    by checking ``context.scratchpad.get("status") == "failed"`` — mirroring
    the existing ``context.scratchpad.get("max_steps_exceeded")`` check for
    the step-budget case. ``run_supervised`` itself never raises for these
    two failure modes; it only raises (unchanged) for ``ValueError`` (missing
    ``next_agent`` on a "continue" decision) and ``HarnessError`` (a backend
    *deliberately* reporting ``status="error"``), both of which represent a
    different kind of failure — a malformed decision, not a crash.
    """

    def __init__(
        self,
        memory: MemoryProvider,
        supervisor: AgentBackend,
        agents: dict[str, AgentBackend] | None = None,
        max_steps: int = 25,
        raise_on_max_steps: bool = False,
        on_error: ErrorPolicy = "stop",
        max_retries: int = 2,
    ) -> None:
        """
        Args:
            memory: Provider used to load/save the ``HarnessContext`` for a
                session, so a run can resume across process restarts.
            supervisor: The backend consulted each iteration for routing
                decisions.
            agents: Mapping of agent name -> backend, the set the supervisor
                is allowed to dispatch to. Can also be built up incrementally
                with ``register_agent``.
            max_steps: Hard ceiling on supervisor iterations before the loop
                gives up rather than spinning forever.
            raise_on_max_steps: If True, hitting ``max_steps`` without the
                supervisor signaling ``"done"`` raises ``MaxStepsExceededError``.
                If False (the default), the loop instead returns the context
                with ``context.scratchpad["max_steps_exceeded"] = True`` set,
                so callers can inspect and decide what to do.
            on_error: What to do when a backend's ``run_step`` raises, or the
                supervisor names an unregistered agent. ``"stop"`` (default)
                ends the run on the first failure; ``"retry"`` retries the
                same step up to ``max_retries`` more times first. See the
                class docstring's "Error policy" section for the full
                contract.
            max_retries: Extra attempts allowed for a failing step when
                ``on_error="retry"``. Ignored when ``on_error="stop"``.
        """
        self.memory = memory
        self.supervisor = supervisor
        self.max_steps = max_steps
        self.raise_on_max_steps = raise_on_max_steps
        self.on_error = on_error
        self.max_retries = max_retries
        self._agents: dict[str, AgentBackend] = dict(agents) if agents else {}

    def register_agent(self, name: str, backend: AgentBackend) -> None:
        """Register (or replace) an agent backend under ``name``."""
        self._agents[name] = backend

    def run_supervised(self, objective: str, input: Any, session_id: str) -> HarnessContext:
        """Load/create the context for ``session_id``, run it to completion.

        Seeds the context with ``objective`` (in ``scratchpad["objective"]``)
        and ``input`` (as a ``user`` message), then loops: ask the supervisor
        what to do next, run that agent, merge its result, save, repeat —
        until the supervisor signals ``status="done"``, ``max_steps`` is hit,
        or the error policy gives up (see the class docstring's "Error
        policy" section).
        """
        context = self.memory.load(session_id)
        context.scratchpad["objective"] = objective
        context.add_message("user", input)

        for _ in range(self.max_steps):
            decision = self._invoke_with_error_policy(
                lambda: self.supervisor.run_step(context), context, who="supervisor"
            )
            if decision is None:
                self.memory.save(session_id, context)
                return context
            self._require_not_error(decision, who="supervisor")

            if decision.status == "done":
                self._record_result(context, "assistant", decision)
                self.memory.save(session_id, context)
                return context

            # status == "continue": dispatch to the named agent.
            if decision.next_agent is None:
                raise ValueError(
                    "Supervisor returned status='continue' without a next_agent"
                )

            next_agent = decision.next_agent
            context.scratchpad["dispatch_payload"] = decision.payload
            result = self._invoke_with_error_policy(
                lambda: self._dispatch(next_agent, context), context, who=next_agent
            )
            if result is None:
                self.memory.save(session_id, context)
                return context
            self._require_not_error(result, who=next_agent)

            context.tools_state[next_agent] = {
                "status": result.status,
                "payload": result.payload,
                "message": result.message,
            }
            self._record_result(context, next_agent, result)
            self.memory.save(session_id, context)

        # Ran out of steps without the supervisor ever signaling "done".
        if self.raise_on_max_steps:
            raise MaxStepsExceededError(
                f"Supervisor did not finish within {self.max_steps} steps"
            )
        context.scratchpad["max_steps_exceeded"] = True
        self.memory.save(session_id, context)
        return context

    def _dispatch(self, name: str, context: HarnessContext) -> StepResult:
        """Look up ``name`` and run it — raises ``KeyError`` if unregistered.

        Split out from ``run_supervised`` so the lookup itself goes through
        the same ``on_error`` policy as an exception raised from inside
        ``run_step`` (see the class docstring's "Error policy" section).
        """
        if name not in self._agents:
            raise KeyError(f"No agent registered under name {name!r}")
        return self._agents[name].run_step(context)

    def _invoke_with_error_policy(
        self,
        step: Callable[[], StepResult],
        context: HarnessContext,
        *,
        who: str,
    ) -> StepResult | None:
        """Call ``step()``, applying ``self.on_error``/``self.max_retries`` if it raises.

        Returns the ``StepResult`` on success. Returns ``None`` if the step
        never succeeds — either immediately (``on_error="stop"``) or after
        exhausting ``max_retries`` (``on_error="retry"``) — in which case the
        caller must stop the loop and return the context as-is; this method
        has already recorded the failure in ``context.scratchpad``.
        """
        attempts = self.max_retries + 1 if self.on_error == "retry" else 1
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                return step()
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any backend failure
                last_exc = exc
                context.scratchpad.setdefault("errors", []).append(
                    {"step": who, "error": f"{type(exc).__name__}: {exc}", "attempt": attempt}
                )

        context.scratchpad["status"] = "failed"
        context.scratchpad["failed_step"] = who
        context.add_message("system", f"{who} failed after {attempts} attempt(s): {last_exc}")
        return None

    @staticmethod
    def _require_not_error(result: StepResult, *, who: str) -> None:
        if result.status == "error":
            raise HarnessError(result.message or f"{who} reported status='error'")

    @staticmethod
    def _record_result(context: HarnessContext, role: str, result: StepResult) -> None:
        """Append the human-visible half of a StepResult as a message.

        Prefers ``message`` (meant to be human-readable); falls back to
        ``payload`` when there's no message, and skips recording anything at
        all when neither is set.
        """
        content = result.message if result.message is not None else result.payload
        if content is not None:
            context.add_message(role, content)
