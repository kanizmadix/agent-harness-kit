# agent-harness-kit

A portable **agent harness**: a supervisor + orchestration loop that sits on
top of whatever multi-agent framework (or raw LLM) you're already using —
[LangGraph](https://github.com/langchain-ai/langgraph),
[CrewAI](https://github.com/crewAIInc/crewAI), AWS Bedrock
AgentCore/[Strands](https://github.com/strands-agents/sdk-python), or a
direct Claude/OpenAI call.

## Design: the harness is the orchestration shell, not the model

`agent_harness_kit` deliberately does **not** know anything about prompts,
tokens, or a specific model provider. It knows about exactly three things:

- **`HarnessContext`** — the state passed between every step: the full
  conversation (`messages`), a scratchpad for transient per-step notes,
  named `artifacts` produced along the way, and a `tools_state` record of
  which agents ran and what they returned. Plain dataclass, JSON-serializable
  via `to_dict()` / `from_dict()`.
- **`AgentBackend`** — the one method every adapter implements:
  `run_step(context: HarnessContext) -> StepResult`. Whether that adapter is
  wrapping a raw Claude call, a compiled LangGraph graph, a CrewAI crew, or a
  Strands agent is invisible to the harness — it only ever sees this one
  method.
- **`HarnessLoop`** — the supervisor loop itself: ask the supervisor backend
  what to do next, run the agent it names, merge the result back into the
  context, save, repeat until the supervisor says it's done (or `max_steps`
  is hit).

Everything framework-specific lives in `agent_harness_kit/backends/` and
nowhere else. `core/` never imports a backend module, and the top-level
`agent_harness_kit/__init__.py` never imports anything beyond `core` and
`backends.base` — so `import agent_harness_kit` always works, with **zero**
hard dependencies, regardless of whether `anthropic`, `langgraph`, `crewai`,
or `strands-agents` happen to be installed.

## The supervisor pattern

```
                 ┌─────────────────────────┐
   HarnessLoop → │ supervisor.run_step(ctx) │ → StepResult(status, next_agent, payload, message)
                 └─────────────────────────┘
                              │
              status="continue", next_agent="worker_x"
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ agents["worker_x"]       │ → StepResult
                 │   .run_step(ctx)         │
                 └─────────────────────────┘
                              │
              merged into ctx.messages / ctx.tools_state, saved via memory
                              │
                              ▼
                     loop back to supervisor
                  (until status="done" or max_steps)
```

`StepResult` is the only vocabulary that crosses the boundary between the
loop and a backend:

```python
@dataclass
class StepResult:
    status: Literal["continue", "done", "error"]
    next_agent: str | None = None   # supervisor only: who to call next
    payload: Any = None             # data handed to the next agent, or the final output
    message: str | None = None      # human-readable note; preferred for display
```

## Registering a backend

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
        "researcher": ClaudeBackend(model="claude-opus-4-8", system_prompt="You research topics thoroughly."),
        "writer": ClaudeBackend(model="claude-opus-4-8", system_prompt="You write clear, concise prose."),
    },
    max_steps=25,
)

context = loop.run_supervised(
    objective="Draft a short blog post about SQLite",
    input="Focus on why it's a good fit for small apps.",
    session_id="session-1",
)
print(context.messages[-1])
```

Agents can also be registered incrementally instead of passed as a dict up
front:

```python
loop = HarnessLoop(memory=InMemoryProvider(), supervisor=supervisor)
loop.register_agent("researcher", researcher_backend)
loop.register_agent("writer", writer_backend)
```

Swap `InMemoryProvider()` for `SQLiteMemoryProvider("agents.db")` to persist
runs (as JSON, via stdlib `sqlite3`) across process restarts — same
interface, no code changes elsewhere.

## Writing your own backend

Anything with a `run_step(self, context: HarnessContext) -> StepResult`
method satisfies `AgentBackend` — no base class required, since it's a
`typing.Protocol`. `backends/langgraph_backend.py`, `backends/crewai_backend.py`,
and `backends/strands_backend.py` are adapter **stubs**: they define the
constructor shape (accepting a compiled LangGraph graph / CrewAI `Crew` /
Strands `Agent`), lazily `import` the corresponding package only inside
`__init__` (so importing the module itself never requires the framework to
be installed — instantiating it does, with a clear `ImportError` if it's
missing), and leave `run_step` raising `NotImplementedError` with a detailed
docstring on exactly what real integration code needs to go there.
`backends/claude_backend.py` is the one fully-working reference
implementation, usable as both supervisor and plain sub-agent, wrapping the
Anthropic Python SDK (also lazily imported) for a single tool-use-capable
completion call per step.

## How this composes with LangGraph / CrewAI / AgentCore's own orchestration

This harness does **not** replace a framework's built-in orchestration — it
composes with it at whatever granularity you choose:

- **Whole framework as one opaque agent.** Point `LangGraphBackend` at a
  compiled graph that *itself* implements a full LangGraph supervisor
  pattern internally (its own sub-agent fan-out, its own routing). From
  `HarnessLoop`'s perspective, that's just one `AgentBackend` that happens to
  do a lot of work in a single `run_step` call. Same idea for a CrewAI
  hierarchical `Crew`, or a Strands agent with its own tool-use loop.
- **Harness as the top-level supervisor, framework agents as workers.**
  Register several backends — some `ClaudeBackend`, some `LangGraphBackend`,
  some `CrewAIBackend` — under one `HarnessLoop`, and let its supervisor
  decide which one handles the next step. This is useful when you want one
  consistent memory/session model (`HarnessContext` + `MemoryProvider`)
  across agents built in different frameworks that don't otherwise share
  state.
- **Mix of both.** A LangGraph graph can itself be one of several backends
  registered under the harness, while also containing its own internal
  sub-graph supervisor — the two orchestration layers just need to agree on
  where the boundary is (typically: the harness's `HarnessContext` maps to
  that graph's top-level invocation state).

The harness never tries to replicate LangGraph's graph execution, CrewAI's
task/process model, or Strands's model-driven tool loop — it only provides a
thin, consistent shell (context + memory + a supervisor loop) around
whichever of those you're already using.

## Install

Install the zero-dependency core package from PyPI:

```bash
python -m pip install agent-harness-kit
```

Framework-specific extras pull in only the SDK you need:

```bash
python -m pip install "agent-harness-kit[anthropic]"   # ClaudeBackend
python -m pip install "agent-harness-kit[openai]"      # OpenAIBackend
python -m pip install "agent-harness-kit[langgraph]"   # LangGraphBackend
python -m pip install "agent-harness-kit[crewai]"      # CrewAIBackend
python -m pip install "agent-harness-kit[strands]"     # StrandsBackend
python -m pip install "agent-harness-kit[all]"         # every optional backend SDK
```

`ClaudeBackend` and `OpenAIBackend` are complete reference implementations.
The LangGraph, CrewAI, and Strands adapters currently provide constructor and
integration templates whose `run_step` methods must be completed for your
framework objects; installing their extras supplies the corresponding SDKs.

The base install has **zero third-party runtime dependencies**—only the
Python standard library. API credentials are never bundled; provide the key
required by the backend you choose through its SDK or environment variable.

For local development from a clone:

```bash
python -m pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

Tests use fake in-process backends (a scripted supervisor, an echo worker,
and a fake Anthropic client injected into `ClaudeBackend`) — no API key and
no network access required, and they pass whether or not any optional
framework SDK is installed.
