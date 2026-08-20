# agent-harness-kit

A tiny, provider-neutral **agent interoperability harness** for Python: one `Protocol`, a JSON-serializable shared context, pluggable memory, and a supervisor shell that can coordinate mixed existing agents without making their framework your application boundary.

> **Why use it?** It gives developers one small, provider-neutral supervisor, context, and memory contract around agents they already have—without forcing the entire application into one vendor or framework.

> **Release status:** `0.1.0` is the published release. This working tree is a documentation-only **`0.1.1` candidate**; it is not yet published.

## What it does

`agent-harness-kit` gives independently built agents one narrow contract:

```python
run_step(context: HarnessContext) -> StepResult
```

The `HarnessLoop` asks a supervisor which registered agent should run next, passes the same serializable `HarnessContext` through every step, records results through a replaceable `MemoryProvider`, and stops on completion or a configured step limit.

**Concrete benefits**

- Wrap raw model calls and agents from different frameworks behind one small interface.
- Keep messages, scratch data, artifacts, and agent results in portable JSON-compatible state.
- Swap in-memory and SQLite persistence without changing orchestration code.
- Test routing and context behavior with deterministic fake backends and no API calls.
- Avoid core framework or model SDK dependencies; install only the extras you use.

**Ideal for** prototypes, portfolio projects, framework migration spikes, small internal tools, and teams that already have heterogeneous agents but need a simple shared supervisor/session boundary. It is also useful as reference code for learning the supervisor pattern.

**Not a replacement for** full graph runtimes, durable execution, built-in tracing or evaluations, built-in guardrails, large tool ecosystems, or managed deployment platforms. If you need those capabilities, use a framework or platform that provides them and optionally place this harness around a suitable boundary. The included LangGraph, CrewAI, and Strands adapters remain **integration templates** whose `run_step` methods must be implemented for your objects.

## Architecture

```text
 Existing agents and model calls
 ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
 │ Claude/API │  │ OpenAI/API │  │ LangGraph* │  │ CrewAI* … │
 └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
       └────────────── adapters / wrappers ──────────────┘
                              │
                  AgentBackend Protocol
        run_step(HarnessContext) -> StepResult
                              │
                    ┌─────────▼─────────┐
                    │    HarnessLoop    │
                    │ supervisor routes│
                    └───────┬───────────┘
                            │ read / merge / save
                    ┌───────▼───────────┐
                    │  HarnessContext   │
                    │ JSON-serializable │
                    └───────┬───────────┘
                            │
              ┌─────────────▼─────────────┐
              │ MemoryProvider Protocol   │
              │ in-memory · SQLite · yours│
              └───────────────────────────┘

 * LangGraph, CrewAI, and Strands adapters are templates in v0.1.x.
```

Framework-specific code stays in `agent_harness_kit/backends/`. The core package uses only the Python standard library. Top-level imports remain safe because backend SDK imports are deferred until backend construction.

## How it compares

This is a scope comparison, not a claim that the projects are interchangeable.

| Project | What official documentation emphasizes | Where `agent-harness-kit` differs |
|---|---|---|
| **agent-harness-kit** | A tiny `AgentBackend` protocol, serializable context, pluggable memory, and a supervisor loop. Zero third-party core runtime dependencies; provider-neutral; designed to wrap mixed existing agents. | Intentionally lacks a graph runtime, durable execution, built-in tracing/evals/guardrails, a tool catalog, and a deployment platform. |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | A low-level runtime for long-running stateful agents, with durable execution, persistence, streaming, and human-in-the-loop control. | Use LangGraph when execution graphs, resumability, and stateful production runtime behavior are primary. The harness is a much smaller interoperability shell and can wrap a graph only after its template adapter is completed. |
| [CrewAI](https://docs.crewai.com/en/introduction) | **Crews** organize collaborating agents; **Flows** provide stateful, event-driven workflow structure and execution control. | CrewAI provides higher-level agent/team and workflow concepts. The harness provides only a narrow backend boundary, shared context, memory protocol, and supervisor loop. |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | Agent loops with tools, agent handoffs, input/output guardrails, [sessions](https://openai.github.io/openai-agents-python/sessions/), and built-in [tracing](https://openai.github.io/openai-agents-python/tracing/). | The SDK is more batteries-included for constructing and operating agents. The harness supplies no native tools, guardrails, or tracing and focuses on wrapping backends through a provider-neutral protocol. |
| [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview) | The direct AutoGen/Semantic Kernel successor: agents, session state, explicit functional/graph workflows, middleware, telemetry/integrations, and MCP tool connections. | Microsoft Agent Framework is a broad application framework. The harness is deliberately smaller and has no workflow engine, middleware pipeline, or MCP client. |
| [Google ADK](https://google.github.io/adk-docs/) | A model- and deployment-agnostic agent framework with tools, context management, multi-agent orchestration, graph workflows, evaluation, and production deployment paths. | ADK covers the agent development lifecycle. The harness is only a portable coordination shell and does not reproduce ADK's tool, evaluation, graph, or deployment capabilities. |

> Comparison content is paraphrased from the linked official documentation for clarity and licensing compliance. Capabilities evolve; consult those sources when making a framework decision.

## Core contracts

`HarnessContext` is the state passed between steps: the conversation (`messages`), transient `scratchpad`, named `artifacts`, and `tools_state` records. It is a plain dataclass with `to_dict()` / `from_dict()` serialization.

`StepResult` is the only vocabulary crossing the loop/backend boundary:

```python
@dataclass
class StepResult:
    status: Literal["continue", "done", "error"]
    next_agent: str | None = None   # supervisor only: who runs next
    payload: Any = None             # next input or final output
    message: str | None = None      # human-readable display note
```

`MemoryProvider` defines `save(session_id, context)` and `load(session_id)`. `InMemoryProvider` and stdlib-backed `SQLiteMemoryProvider` are included.

## Registering backends

```python
from agent_harness_kit import HarnessLoop, InMemoryProvider
from agent_harness_kit.backends.claude_backend import ClaudeBackend

supervisor = ClaudeBackend(
    model="claude-opus-4-8",
    is_supervisor=True,
    agent_names=["researcher", "writer"],
)

loop = HarnessLoop(
    memory=InMemoryProvider(),
    supervisor=supervisor,
    agents={
        "researcher": ClaudeBackend(
            model="claude-opus-4-8",
            system_prompt="You research topics thoroughly.",
        ),
        "writer": ClaudeBackend(
            model="claude-opus-4-8",
            system_prompt="You write clear, concise prose.",
        ),
    },
    max_steps=25,
)

context = loop.run_supervised(
    objective="Draft a short blog post about SQLite",
    input="Focus on why it is a good fit for small apps.",
    session_id="session-1",
)
print(context.messages[-1])
```

Agents can also be registered incrementally:

```python
loop = HarnessLoop(memory=InMemoryProvider(), supervisor=supervisor)
loop.register_agent("researcher", researcher_backend)
loop.register_agent("writer", writer_backend)
```

Swap `InMemoryProvider()` for `SQLiteMemoryProvider("agents.db")` to persist JSON context across process restarts without changing the loop.

## Writing an adapter

Anything with `run_step(self, context: HarnessContext) -> StepResult` satisfies `AgentBackend`; no base class is required because it is a `typing.Protocol`.

`ClaudeBackend` and `OpenAIBackend` are complete reference implementations for single-completion steps and supervisor routing. `LangGraphBackend`, `CrewAIBackend`, and `StrandsBackend` are explicit templates: their constructors validate/lazily import the corresponding SDK, but `run_step` raises `NotImplementedError` until you add application-specific context and result mapping.

The harness can sit above multiple framework workers, wrap one whole framework workflow as an opaque worker, or mix both approaches. It never attempts to reproduce LangGraph graph execution, CrewAI's crew/flow model, or Strands' model-driven tool loop.

## Install

Install the currently published package from PyPI:

```bash
python -m pip install agent-harness-kit
```

Install only the optional SDK needed by a backend:

```bash
python -m pip install "agent-harness-kit[anthropic]"   # complete ClaudeBackend
python -m pip install "agent-harness-kit[openai]"      # complete OpenAIBackend
python -m pip install "agent-harness-kit[langgraph]"   # LangGraph template
python -m pip install "agent-harness-kit[crewai]"      # CrewAI template
python -m pip install "agent-harness-kit[strands]"     # Strands template
python -m pip install "agent-harness-kit[all]"         # every optional SDK
```

The base install has **zero third-party runtime dependencies**. Credentials are never bundled; provide them through the SDK/client or environment used by your selected backend.

For local development:

```bash
python -m pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

Tests use fake in-process supervisors/workers, fake Anthropic and OpenAI clients, and a synthetic Strands module. They require no credentials or network access and pass whether optional framework SDKs are installed or not.

## Sources and competitive positioning

Official documentation used for the capability summary:

- [LangGraph overview: durable execution, persistence, and human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/overview)
- [CrewAI introduction: Crews and Flows](https://docs.crewai.com/en/introduction)
- [OpenAI Agents SDK overview](https://openai.github.io/openai-agents-python/), [sessions](https://openai.github.io/openai-agents-python/sessions/), and [tracing](https://openai.github.io/openai-agents-python/tracing/)
- [Microsoft Agent Framework overview: successor positioning, workflows, middleware, and MCP clients](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview)
- [Google Agent Development Kit overview: models, multi-agent/graph workflows, and deployment](https://google.github.io/adk-docs/)

These links describe those projects, not endorsements or compatibility guarantees. All comparison prose above is paraphrased from official documentation.

## v0.1.1 candidate release notes

**Status: unreleased candidate.** This version bump is necessary because the README embedded in already-published PyPI `0.1.0` metadata cannot be changed in place.

- Reframes the project around its intentionally small interoperability scope and ideal use cases.
- Adds a sourced, current comparison with major agent frameworks and explicit non-goals.
- Adds a polished static-site demo that simulates supervisor routing, agent state, shared context, memory, logs, and final output entirely in the browser.
- Clarifies that Claude/OpenAI backends are complete references while LangGraph/CrewAI/Strands adapters remain templates.
- Changes package/runtime metadata from `0.1.0` to `0.1.1`; no core runtime behavior or dependency surface changes.
