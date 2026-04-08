<div align="center">

# AGENT SNOOP 🔍

**Lightweight agent observability for any AI agent framework. AgentSnoop evaluates your agent's traces and tells you how to improve its accuracy.**

[![PyPI](https://img.shields.io/pypi/v/agent_snoop)](https://pypi.org/project/agent_snoop/)
[![Python](https://img.shields.io/pypi/pyversions/agent_snoop)](https://pypi.org/project/agent_snoop/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/sarthakrastogi/agent-snoop?style=social)](https://github.com/sarthakrastogi/agent-snoop/stargazers)
[![Downloads](https://img.shields.io/pypi/dm/agent_snoop)](https://pypi.org/project/agent_snoop/)

<p>
  <a href="https://liten.tech/traces">
    <img src="https://img.shields.io/badge/Open%20Dashboard-liten.tech-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0zIDEzaDh2LTJIM3Yyem0wIDRoOHYtMkgzdjJ6bTAtOGg4VjdIM3Yyek0xMyAxN2g4di0yaC04djJ6bTAtNGg4di0yaC04djJ6bTAtNGg4VjdoLTh2MnoiLz48L3N2Zz4=" alt="Open Dashboard" />
  </a>
  &nbsp;
  <a href="https://pypi.org/project/agent-snoop/">
    <img src="https://img.shields.io/badge/Read%20the%20Docs-PyPI-f59e0b?style=for-the-badge&logo=python&logoColor=white" alt="Read the Docs" />
  </a>
</p>

<br />

> Two lines of code. Every step, every token, every millisecond — captured automatically.

<br />

</div>

---

## What does it do?

`agent_snoop` sits alongside your agent and records everything — then lets you **evaluate** it:

| What | Details |
|------|---------|
| **Steps** | Every LLM call, tool invocation, and agent action |
| **Inputs / outputs** | What went in and what came out at every step |
| **Tool calls** | Name, arguments, return value |
| **Token usage** | Per-step and aggregated for the full run |
| **Timing** | Start time, end time, and duration at every level |
| **Full trajectory** | Ordered list of all steps for the entire invocation |
| **Evals** | Accuracy, latency, tool usage patterns, harmful content, and more |
| **Issue detection** | What went wrong and how to fix it |

Everything is stored as a single document per invocation — easy to query, easy to diff, easy to debug.

---

## Up and running in 3 minutes

### Step 1 — Install

```bash
pip install agent-snoop[mongo,langgraph]
```

### Step 2 — Pick your storage

**Option A — Your own MongoDB (full data ownership)**

```bash
export MONGODB_URI="mongodb+srv://user:password@cluster.example.mongodb.net/"
```

agent_snoop automatically creates an `agentsnoop_db` database and a `traces` collection. Your data never leaves your infrastructure. Connect liten.tech to visualise it without moving a byte.

**Option B — liten.tech API key (zero infra, instant dashboard)**

1. Sign up at [liten.tech](https://liten.tech/auth/login?screen_hint=signup&returnTo=/traces)
2. Go to **Dashboard → Settings → API Keys** and create a new key
3. Export it:

```bash
export AGENTSNOOP_API_KEY="as_your_key_here"
```

> If you set both, `AGENTSNOOP_API_KEY` takes priority.

### Step 3 — Add two lines to your agent

```python
import agent_snoop
from agent_snoop.integrations.langgraph import AgentSnoopCallbackHandler

# Reads MONGODB_URI or AGENTSNOOP_API_KEY automatically
tracer = agent_snoop.init(agent_name="my-agent", framework="langgraph")

query = "Your question here"

handler = AgentSnoopCallbackHandler(
    handle=tracer.trace(input=query)
)

result = await graph.ainvoke(
    {"messages": [HumanMessage(content=query)]},
    config={"callbacks": [handler]},
)

handler.on_chain_end_final(result)
```

Open [liten.tech/traces](https://liten.tech/traces) to see your traces. That's it.

---

## View your traces on liten.tech

<div align="center">

<a href="https://liten.tech/traces">
  <img src="https://img.shields.io/badge/Open%20Dashboard-%E2%86%92-6366f1?style=for-the-badge" alt="Open Dashboard" />
</a>

</div>

- **If you used Option A (your MongoDB):** go to **Settings → Connect Database** and paste the same URI. liten.tech reads your traces directly — your data stays where it is.
- **If you used Option B (API key):** your traces are already there. Just sign in.

---

## See it in action

### Step-by-step trace timeline

Every node, tool call, token count, and timing — captured automatically and visualised in one place.

<img src="docs/media/trace.png" alt="Trace timeline showing every LLM call and tool invocation in sequence" width="100%" />

### Built-in evaluation

AgentSnoop scores your agent's performance across dozens of dimensions: accuracy, latency, tool usage patterns, safety, and more. Know whether your agent is actually improving between runs.

<img src="docs/media/eval_report.png" alt="Dashboard showing evaluation results" width="100%" />

### Automatic issue detection

AgentSnoop finds problems in your agent's behavior before they reach users — and tells you exactly what went wrong and how to fix it.

<img src="docs/media/issues_found.png" alt="Dashboard showing issues found by an eval" width="100%" />

### Your data, your infra

Bring your own MongoDB (local, Atlas, or any hosted provider) — or use a liten.tech API key for zero-infra storage. Connect your database in seconds:

| Connect via MongoDB URI | Connect via API Key |
|---|---|
| <img src="docs/media/connect_with_mongodb_uri_modal.png" alt="Connect with MongoDB URI" width="100%" /> | <img src="docs/media/connect_with_api_key_modal.png" alt="Connect with API Key" width="100%" /> |

### Debug regressions before they reach production

Compare runs side by side. Filter by tag, agent name, or time range. Spot the exact step where latency spiked or a tool returned garbage.

<img src="docs/media/tracedetails.png" alt="Side-by-side comparison of two agent runs" width="100%" />

---

## Integration styles

### Callback-based (recommended for LangGraph)

Captures every node, tool call, and LLM invocation in real time:

```python
import agent_snoop
from agent_snoop.integrations.langgraph import AgentSnoopCallbackHandler
from langchain_core.messages import HumanMessage

tracer = agent_snoop.init(agent_name="my-agent", framework="langgraph")
query = "What caused the 2008 financial crisis?"

handler = AgentSnoopCallbackHandler(handle=tracer.trace(input=query, tags=["prod"]))

result = await graph.ainvoke(
    {"messages": [HumanMessage(content=query)]},
    config={"callbacks": [handler]},
)

handler.on_chain_end_final(result)
```

### Post-run (zero agent changes)

Run your graph exactly as before, then hand the output to agent_snoop:

```python
from agent_snoop.integrations.langgraph import parse_langgraph_output

result = await graph.ainvoke({"messages": [HumanMessage(content=query)]})
trace = parse_langgraph_output(result, input=query, agent_name="my-agent")
tracer.log_trace(trace)
```

### Manual context manager (full control)

```python
with tracer.trace(input=query, tags=["prod"]) as t:
    result = my_agent.run(query)
    t.set_output(result)
    t.set_metadata(user_id="u123")
```

---

## What gets stored

Each trace is a single MongoDB document:

```json
{
  "_id": "550e8400-e29b-...",
  "agent_name": "my-research-agent",
  "framework": "langgraph",
  "input": "What caused the 2008 financial crisis?",
  "output": "The 2008 financial crisis was caused by...",
  "status": "success",
  "started_at": "2024-01-15T10:23:00Z",
  "ended_at": "2024-01-15T10:23:04Z",
  "duration_ms": 4021,
  "total_token_usage": { "prompt_tokens": 812, "completion_tokens": 234, "total_tokens": 1046 },
  "tags": ["prod"],
  "steps": [
    {
      "step_index": 0,
      "step_type": "llm_call",
      "node_name": "researcher",
      "duration_ms": 1823,
      "token_usage": { "prompt_tokens": 412, "completion_tokens": 134, "total_tokens": 546 },
      "tool_calls": [
        {
          "tool_name": "web_search",
          "tool_input": { "query": "2008 financial crisis causes" },
          "tool_output": "...",
          "duration_ms": 341
        }
      ]
    }
  ]
}
```

---

## Framework support

| Framework | Status |
|-----------|--------|
| LangGraph | ✅ Supported (callback + post-run) |
| AutoGen   | 🔜 Coming soon |
| CrewAI    | 🔜 Coming soon |

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=sarthakrastogi/agent-snoop&type=Date)](https://star-history.com/#sarthakrastogi/agent-snoop&Date)

---

## Contributing

Contributions are welcome! Please open an issue or PR. If you're adding a new framework integration, check the `agent_snoop/integrations/` folder for examples.

---

## License

MIT
</div>