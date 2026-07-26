"""Shared fakes for testing the harness without any real LLM/network calls."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent_harness_kit.backends.base import StepResult
from agent_harness_kit.core.context import HarnessContext


@dataclass
class ScriptedBackend:
    """Replays a fixed sequence of StepResults, one per call to run_step.

    Once the sequence is exhausted, the last StepResult is repeated forever
    (useful for asserting a supervisor that never says "done" is correctly
    stopped by ``max_steps``).
    """

    steps: list[StepResult]
    calls: list[HarnessContext] = field(default_factory=list)
    _index: int = 0

    def run_step(self, context: HarnessContext) -> StepResult:
        self.calls.append(context)
        result = self.steps[self._index]
        self._index = min(self._index + 1, len(self.steps) - 1)
        return result


class EchoBackend:
    """A trivial sub-agent (worker) that echoes the dispatch payload back, then is done."""

    def run_step(self, context: HarnessContext) -> StepResult:
        payload = context.scratchpad.get("dispatch_payload")
        return StepResult(status="done", payload=f"echo:{payload}", message=f"echo:{payload}")


@dataclass
class RaisingBackend:
    """A backend whose run_step raises for its first ``fail_times`` calls.

    Used to exercise ``HarnessLoop``'s ``on_error``/``max_retries`` policy
    without any real backend. After ``fail_times`` failures, subsequent calls
    return ``then`` instead of raising (leave ``then`` unset if the test's
    attempt budget never reaches past ``fail_times``).
    """

    exc: Exception
    fail_times: int = 10**9
    then: StepResult | None = None
    calls: int = 0

    def run_step(self, context: HarnessContext) -> StepResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        assert self.then is not None, "RaisingBackend called past fail_times with no 'then' set"
        return self.then


@pytest.fixture
def scripted_backend():
    return ScriptedBackend


@pytest.fixture
def echo_backend():
    return EchoBackend()
