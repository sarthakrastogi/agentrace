# AGENT SNOOP 🔍

> Lightweight agent observability for any AI agent framework. Capture every step, tool call, and LLM invocation — store them in your own MongoDB or on liten.tech.

[![PyPI](https://img.shields.io/pypi/v/agent_snoop)](https://pypi.org/project/agent_snoop/)
[![Python](https://img.shields.io/pypi/pyversions/agent_snoop)](https://pypi.org/project/agent_snoop/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What does it do?

`agent_snoop` sits alongside your agent and records:

| What | Details |
|------|---------|
| **Steps** | Every LLM call, tool invocation, and agent action |
| **Inputs / outputs** | What went in and what came out at every step |
| **Tool calls** | Name, arguments, return value |
| **Token usage** | Per-step and aggregated for the full run |
| **Timing** | Start time, end time, and duration at every level |
| **Full trajectory** | Ordered list of all steps for the entire invocation |

Everything is stored as a single document per invocation, making it easy to query, visualise, and debug.

---

## Installation

```bash
pip install agent-snoop[mongo,langgraph]
```

---

## Quick start — 3 minutes

### Step 1 — Pick your storage

You have two options. Choose one (or both):

**Option A — Store traces in your own MongoDB (full data ownership)**

If you already have a MongoDB instance (local, Atlas, or any hosted provider):

```bash
export MONGODB_URI="mongodb+srv://user:password@cluster.example.mongodb.net/"
```

agent_snoop will automatically create an `agentsnoop_db` database and a `traces` collection. Your data never leaves your infrastructure.

**Option B — Store traces on liten.tech (zero infra, instant dashboard)**

1. Sign up at [liten.tech/signup](https://liten.tech/signup)
2. Go to **Dashboard → Settings → API Keys** and create a new key
3. Export it:

```bash
export AGENTSNOOP_API_KEY="as_your_key_here"
```

> If you set both, `AGENTSNOOP_API_KEY` takes priority.

---

### Step 2 — View your traces on liten.tech

Go to [liten.tech/traces](https://liten.tech/traces) and connect your storage:

- **If you used Option A (your MongoDB):** go to **Settings → Connect Database** and paste the same URI. liten.tech will read traces directly from your database — your data stays where it is.
- **If you used Option B (API key):** your traces are already there. Just sign in.

---

### Step 3 — Add two lines to your agent code

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

That's it. Open [liten.tech/traces](https://liten.tech/traces) to see your traces.

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

## License

MIT