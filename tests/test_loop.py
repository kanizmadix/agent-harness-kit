import pytest

from agent_harness_kit.backends.base import StepResult
from agent_harness_kit.core.loop import HarnessError, HarnessLoop, MaxStepsExceededError
from agent_harness_kit.core.memory import InMemoryProvider
from tests.conftest import EchoBackend


def test_supervisor_done_immediately_returns_context(scripted_backend):
    supervisor = scripted_backend(steps=[StepResult(status="done", message="final answer")])
    loop = HarnessLoop(memory=InMemoryProvider(), supervisor=supervisor)

    context = loop.run_supervised("objective", "input", "s1")

    assert context.messages[-1] == {"role": "assistant", "content": "final answer"}


def test_context_seeded_with_objective_and_input(scripted_backend):
    supervisor = scripted_backend(steps=[StepResult(status="done", message="ok")])
    loop = HarnessLoop(memory=InMemoryProvider(), supervisor=supervisor)

    context = loop.run_supervised("build a thing", "please help", "s1")

    assert context.scratchpad["objective"] == "build a thing"
    assert context.messages[0] == {"role": "user", "content": "please help"}


def test_supervisor_dispatches_to_registered_agent_via_dict(scripted_backend):
    supervisor = scripted_backend(
        steps=[
            StepResult(status="continue", next_agent="worker", payload="do the thing"),
            StepResult(status="done", message="wrapped up"),
        ]
    )
    loop = HarnessLoop(
        memory=InMemoryProvider(),
        supervisor=supervisor,
        agents={"worker": EchoBackend()},
    )

    context = loop.run_supervised("objective", "input", "s1")

    assert context.tools_state["worker"]["payload"] == "echo:do the thing"
    assert {"role": "worker", "content": "echo:do the thing"} in context.messages
    assert context.messages[-1] == {"role": "assistant", "content": "wrapped up"}


def test_supervisor_dispatches_to_agent_registered_after_construction(scripted_backend):
    supervisor = scripted_backend(
        steps=[
            StepResult(status="continue", next_agent="worker", payload="go"),
            StepResult(status="done", message="done"),
        ]
    )
    loop = HarnessLoop(memory=InMemoryProvider(), supervisor=supervisor)
    loop.register_agent("worker", EchoBackend())

    context = loop.run_supervised("objective", "input", "s1")

    assert context.messages[-1] == {"role": "assistant", "content": "done"}


def test_unregistered_agent_raises_keyerror(scripted_backend):
    supervisor = scripted_backend(
        steps=[StepResult(status="continue", next_agent="ghost", payload=None)]
    )
    loop = HarnessLoop(memory=InMemoryProvider(), supervisor=supervisor)

    with pytest.raises(KeyError):
        loop.run_supervised("objective", "input", "s1")


def test_continue_without_next_agent_raises_valueerror(scripted_backend):
    supervisor = scripted_backend(steps=[StepResult(status="continue", next_agent=None)])
    loop = HarnessLoop(memory=InMemoryProvider(), supervisor=supervisor)

    with pytest.raises(ValueError):
        loop.run_supervised("objective", "input", "s1")


def test_supervisor_error_status_raises_harness_error(scripted_backend):
    supervisor = scripted_backend(steps=[StepResult(status="error", message="boom")])
    loop = HarnessLoop(memory=InMemoryProvider(), supervisor=supervisor)

    with pytest.raises(HarnessError, match="boom"):
        loop.run_supervised("objective", "input", "s1")


def test_max_steps_returns_context_with_flag_by_default(scripted_backend):
    supervisor = scripted_backend(
        steps=[StepResult(status="continue", next_agent="worker", payload="x")]
    )
    loop = HarnessLoop(
        memory=InMemoryProvider(),
        supervisor=supervisor,
        agents={"worker": EchoBackend()},
        max_steps=3,
    )

    context = loop.run_supervised("objective", "input", "s1")

    assert len(supervisor.calls) == 3
    assert context.scratchpad["max_steps_exceeded"] is True


def test_max_steps_raises_when_configured(scripted_backend):
    supervisor = scripted_backend(
        steps=[StepResult(status="continue", next_agent="worker", payload="x")]
    )
    loop = HarnessLoop(
        memory=InMemoryProvider(),
        supervisor=supervisor,
        agents={"worker": EchoBackend()},
        max_steps=2,
        raise_on_max_steps=True,
    )

    with pytest.raises(MaxStepsExceededError):
        loop.run_supervised("objective", "input", "s1")


def test_memory_persists_context_across_loop_runs(scripted_backend):
    memory = InMemoryProvider()
    supervisor = scripted_backend(steps=[StepResult(status="done", message="first run done")])
    loop = HarnessLoop(memory=memory, supervisor=supervisor)
    loop.run_supervised("objective", "first input", "s1")

    supervisor2 = scripted_backend(steps=[StepResult(status="done", message="second run done")])
    loop2 = HarnessLoop(memory=memory, supervisor=supervisor2)
    context = loop2.run_supervised("objective", "second input", "s1")

    contents = [m["content"] for m in context.messages]
    assert "first input" in contents
    assert "second input" in contents
