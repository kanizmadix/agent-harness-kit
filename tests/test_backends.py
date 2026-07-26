"""Tests for the backend lazy-import contract and the Claude backend.

These tests must pass whether or not langgraph/crewai/strands-agents are
installed: the whole point of the lazy-import design is that importing the
adapter modules never requires the framework, and constructing the adapter
only fails (with a clear ImportError) when the framework really is missing.

The Claude backend test injects a fake ``client`` object so no real network
call or ``anthropic`` package is required.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_harness_kit.core.context import HarnessContext


@pytest.mark.parametrize(
    "module_name,class_name,package_hint",
    [
        ("agent_harness_kit.backends.langgraph_backend", "LangGraphBackend", "langgraph"),
        ("agent_harness_kit.backends.crewai_backend", "CrewAIBackend", "crewai"),
        ("agent_harness_kit.backends.strands_backend", "StrandsBackend", "strands-agents"),
    ],
)
def test_stub_backend_module_imports_without_the_framework(module_name, class_name, package_hint):
    """Importing the module never requires the underlying framework package."""
    module = importlib.import_module(module_name)
    assert hasattr(module, class_name)


@pytest.mark.parametrize(
    "module_name,class_name,import_name",
    [
        ("agent_harness_kit.backends.langgraph_backend", "LangGraphBackend", "langgraph"),
        ("agent_harness_kit.backends.crewai_backend", "CrewAIBackend", "crewai"),
        ("agent_harness_kit.backends.strands_backend", "StrandsBackend", "strands"),
    ],
)
def test_stub_backend_raises_import_error_when_framework_missing(module_name, class_name, import_name):
    """Constructing the adapter raises a clear ImportError iff the framework isn't installed."""
    module = importlib.import_module(module_name)
    backend_cls = getattr(module, class_name)

    try:
        importlib.import_module(import_name)
    except ImportError:
        with pytest.raises(ImportError):
            backend_cls(object())
    else:
        pytest.skip(f"{import_name} is installed in this environment; nothing to assert here")


def test_stub_backend_run_step_is_not_implemented():
    """run_step is a documented TODO stub, not a silent no-op."""
    from agent_harness_kit.backends.crewai_backend import CrewAIBackend

    try:
        import crewai  # noqa: F401
    except ImportError:
        pytest.skip("crewai not installed; construction itself already raises ImportError")

    backend = CrewAIBackend(crew=object())
    with pytest.raises(NotImplementedError):
        backend.run_step(HarnessContext(session_id="s1"))


# --- ClaudeBackend, using a fake client (no network, no `anthropic` package) ---


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeToolUseBlock:
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _FakeMessage:
    content: list[Any] = field(default_factory=list)


class _FakeMessagesResource:
    def __init__(self, response: _FakeMessage):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response: _FakeMessage):
        self.messages = _FakeMessagesResource(response)


def test_claude_backend_plain_agent_returns_text_as_done():
    from agent_harness_kit.backends.claude_backend import ClaudeBackend

    fake_client = _FakeAnthropicClient(_FakeMessage(content=[_FakeTextBlock(text="hello there")]))
    backend = ClaudeBackend(client=fake_client)

    context = HarnessContext(session_id="s1")
    context.add_message("user", "hi")

    result = backend.run_step(context)

    assert result.status == "done"
    assert result.payload == "hello there"
    assert result.message == "hello there"
    assert fake_client.messages.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_claude_backend_supervisor_parses_json_routing_decision():
    from agent_harness_kit.backends.claude_backend import ClaudeBackend

    decision_json = '{"status": "continue", "next_agent": "worker", "payload": "go do it"}'
    fake_client = _FakeAnthropicClient(_FakeMessage(content=[_FakeTextBlock(text=decision_json)]))
    backend = ClaudeBackend(client=fake_client, is_supervisor=True, agent_names=["worker"])

    context = HarnessContext(session_id="s1")
    context.add_message("user", "start")

    result = backend.run_step(context)

    assert result.status == "continue"
    assert result.next_agent == "worker"
    assert result.payload == "go do it"


def test_claude_backend_supervisor_parses_tool_use_routing_decision():
    from agent_harness_kit.backends.claude_backend import ClaudeBackend

    tool_block = _FakeToolUseBlock(
        name="route",
        input={"status": "done", "message": "all finished"},
    )
    fake_client = _FakeAnthropicClient(_FakeMessage(content=[tool_block]))
    backend = ClaudeBackend(
        client=fake_client,
        is_supervisor=True,
        tools=[{"name": "route", "input_schema": {"type": "object"}}],
    )

    context = HarnessContext(session_id="s1")
    context.add_message("user", "start")

    result = backend.run_step(context)

    assert result.status == "done"
    assert result.message == "all finished"


def test_claude_backend_supervisor_invalid_json_becomes_error_status():
    from agent_harness_kit.backends.claude_backend import ClaudeBackend

    fake_client = _FakeAnthropicClient(_FakeMessage(content=[_FakeTextBlock(text="not json at all")]))
    backend = ClaudeBackend(client=fake_client, is_supervisor=True)

    context = HarnessContext(session_id="s1")
    context.add_message("user", "start")

    result = backend.run_step(context)

    assert result.status == "error"
    assert "not json at all" in result.message
