# Tutorial: Tracing a LangGraph Agent with agent_snoop

In this tutorial you'll build a multi-tool Apple customer support bot using LangGraph and Gemini, wire up agent_snoop to record every step, and view the traces on liten.tech — in under 15 minutes.

By the end you'll have:
- A branching LangGraph agent that routes between tool calls and direct answers
- Full observability: every LLM call, tool invocation, token count, and duration captured
- Traces visible in the liten.tech dashboard (or in your own MongoDB — your choice)

---

## Prerequisites

- Python 3.9+
- A Gemini API key (`GEMINI_API_KEY_1`)
- Either a MongoDB URI **or** a liten.tech account (see Step 1)

---

## Step 1 — Choose where your traces are stored

You have two options. Pick the one that fits your setup.

### Option A — Your own MongoDB

If you have a MongoDB instance (local, Atlas, Railway, etc.):

```bash
export MONGODB_URI="mongodb+srv://user:password@cluster.example.mongodb.net/"
```

agent_snoop will create the `agentsnoop_db` database and `traces` collection automatically. Your data never leaves your infrastructure.

To view traces on liten.tech, go to **Settings → Connect Database** and paste the same URI. liten.tech reads directly from your database — nothing is copied.

> 📸 *[screenshot: liten.tech Settings → Connect Database modal]*

### Option B — liten.tech hosted storage

1. Sign up at [liten.tech/signup](https://liten.tech/signup)
2. Go to **Dashboard → Settings → API Keys** and click **Create new key**

> 📸 *[screenshot: API key creation UI on liten.tech]*

3. Copy the key and export it:

```bash
export AGENTSNOOP_API_KEY="as_your_key_here"
```

Traces are sent to liten.tech's infrastructure and appear in your dashboard immediately. Nothing to manage.

> If you set both env vars, `AGENTSNOOP_API_KEY` takes priority.

---

## Step 2 — Install dependencies

```bash
pip install agent-snoop[mongo,langgraph] langchain-google-genai
```

---

## Step 3 — Understand the agent we're building

The bot has a simple branching graph:

```
              ┌──────────┐
   START ──▶  │  router  │
              └────┬─────┘
       ┌───────────┴───────────┐
       │ has tool calls        │ no tool calls
       ▼                       ▼
 ┌───────────┐           ┌──────────┐
 │ tool_node │──▶ router │ (answer) │──▶ END
 └───────────┘           └──────────┘
```

The `router` node calls the Gemini LLM. If the LLM decides to use a tool, execution goes to `tool_node`, then loops back to `router`. When the LLM produces a final answer without tool calls, the graph ends.

The bot has four tools:

| Tool | What it does |
|------|-------------|
| `lookup_device_info(model)` | Returns specs and support status for an Apple device |
| `check_warranty(serial_number)` | Returns warranty and AppleCare coverage status |
| `get_troubleshooting_steps(issue)` | Returns a step-by-step fix guide |
| `check_repair_pricing(model, issue)` | Returns estimated repair costs |

---

## Step 4 — Define the tools

```python
from langchain_core.tools import tool

@tool
def lookup_device_info(model: str) -> str:
    """Look up specifications and support status for an Apple device model."""
    devices = {
        "iphone 15 pro": "iPhone 15 Pro | Chip: A17 Pro | Status: Fully supported.",
        "macbook pro 14 m3": "MacBook Pro 14-inch (M3) | Chip: Apple M3 | Status: Fully supported.",
        # ... full dict in the example file
    }
    key = model.lower().strip()
    for k, v in devices.items():
        if all(word in key for word in k.split()):
            return v
    return f"Device '{model}' not found. Visit apple.com/support."

@tool
def check_warranty(serial_number: str) -> str:
    """Check warranty and AppleCare coverage status using a serial number."""
    warranty_db = {
        "F9K": "iPhone 15 Pro | AppleCare+: Active (expires 14 Oct 2025)",
        "C02": "MacBook Pro 14-inch (M3) | AppleCare+: Not purchased",
    }
    prefix = serial_number.upper()[:3]
    return warranty_db.get(prefix, f"Serial '{serial_number}' not found.")

# ... get_troubleshooting_steps and check_repair_pricing defined similarly

TOOLS = [lookup_device_info, check_warranty, get_troubleshooting_steps, check_repair_pricing]
```

---

## Step 5 — Build the LangGraph agent

```python
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

SYSTEM_PROMPT = (
    "You are an Apple Authorized Support Advisor. "
    "Always use the available tools to look up accurate information. "
    "Never guess — use the tools."
)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def router_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    # Prepend the system prompt on the first turn
    if not any(getattr(m, "type", None) == "system" for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY_1"),
    ).bind_tools(TOOLS)

    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"

tool_node = ToolNode(TOOLS)

def build_graph() -> Any:
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("tools", tool_node)
    builder.set_entry_point("router")
    builder.add_conditional_edges("router", should_continue, {"tools": "tools", "end": END})
    builder.add_edge("tools", "router")
    return builder.compile()
```

---

## Step 6 — Wire up agent_snoop

This is the only agent_snoop-specific code you need. Everything else above is your normal agent code.

```python
import agent_snoop
from agent_snoop.integrations.langgraph import AgentSnoopCallbackHandler

# Reads AGENTSNOOP_API_KEY or MONGODB_URI from the environment automatically.
# Prints a clear banner telling you exactly where traces are going.
tracer = agent_snoop.init(
    agent_name="apple-support-bot",
    framework="langgraph",
    default_tags=["apple-support", "tutorial"],
)
```

When you run this, you'll see one of these banners in your terminal:

**If AGENTSNOOP_API_KEY is set:**
```
────────────────────────────────────────────────────────────────────
📡  agent_snoop — sending traces to liten.tech
────────────────────────────────────────────────────────────────────
  Storage   : liten.tech hosted
  Dashboard : https://liten.tech/traces

  ✅  Your traces will appear in your dashboard within seconds.
────────────────────────────────────────────────────────────────────
```

**If MONGODB_URI is set:**
```
────────────────────────────────────────────────────────────────────
🗄️   agent_snoop — saving traces to your MongoDB
────────────────────────────────────────────────────────────────────
  URI       : mongodb+srv://user:***@cluster.example.mongodb.net/
  Database  : agentsnoop_db
  Collection: traces

  ✅  Traces are stored in YOUR database — full data ownership.
  👉  View them at: https://liten.tech/traces
────────────────────────────────────────────────────────────────────
```

**If neither is set:**
```
────────────────────────────────────────────────────────────────────
⚠️   agent_snoop — no storage configured
────────────────────────────────────────────────────────────────────
  Traces will NOT be persisted.

  Option A — liten.tech (easiest):
    export AGENTSNOOP_API_KEY=as_...

  Option B — Your own MongoDB:
    export MONGODB_URI=mongodb+srv://...
────────────────────────────────────────────────────────────────────
```

---

## Step 7 — Run a query with tracing

```python
async def run_query(graph, tracer, query: str):
    # Create a TraceHandle for this invocation
    handle = tracer.trace(
        input=query,
        tags=["tutorial"],
        metadata={"query_index": 1},
    )

    # Pass the handle to the callback handler
    handler = AgentSnoopCallbackHandler(handle=handle)

    # Run the graph exactly as you normally would — just add callbacks
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"callbacks": [handler]},
    )

    # Finalise the trace (sets output, marks status=success, flushes to storage)
    handler.on_chain_end_final(result)

    return result["messages"][-1].content
```

The callback handler is transparent — it observes the graph without changing its behaviour. Your agent code doesn't need to know about tracing at all.

---

## Step 8 — Run the full example

```python
import asyncio

async def main():
    graph = build_graph()
    tracer = agent_snoop.init(
        agent_name="apple-support-bot",
        framework="langgraph",
        default_tags=["apple-support"],
    )

    queries = [
        "My serial number is F9KXYZ123 — check my warranty and tell me about my device.",
        "My iPhone 15 Pro screen cracked. What are my options and the repair cost?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        answer = await run_query(graph, tracer, query)
        print(f"Answer: {answer}")
        await asyncio.sleep(2)  # be polite to the API

asyncio.run(main())
```

---

## Step 9 — View traces on liten.tech

Open [liten.tech/traces](https://liten.tech/traces).

> 📸 *[screenshot: liten.tech dashboard showing trace list]*

Click any trace to see the full trajectory:

> 📸 *[screenshot: trace detail view showing steps, tool calls, token usage]*

Each trace shows:
- The original user query and final answer
- Every step in order: which node ran, what went in, what came out
- Tool calls with their inputs and outputs
- Token usage per step and totalled across the run
- Timing at every level

---

## What a captured trace looks like

For the query `"My serial number is F9KXYZ123 — check my warranty and tell me about my device?"`, agent_snoop captures a trajectory like this:

```
Step 0  llm_call      router       → decides to call check_warranty + lookup_device_info
Step 1  tool_call     check_warranty(serial_number="F9KXYZ123")  → warranty info
Step 2  tool_call     lookup_device_info(model="iPhone 15 Pro")  → device specs
Step 3  llm_call      router       → synthesises tool results into final answer
```

The full MongoDB document for this run:

```json
{
  "_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent_name": "apple-support-bot",
  "framework": "langgraph",
  "input": "My serial number is F9KXYZ123 — check my warranty and tell me about my device.",
  "output": "Your iPhone 15 Pro (serial F9KXYZ...) has an active AppleCare+ plan...",
  "status": "success",
  "started_at": "2024-01-15T10:23:00Z",
  "ended_at": "2024-01-15T10:23:03Z",
  "duration_ms": 3241,
  "total_token_usage": {
    "prompt_tokens": 1240,
    "completion_tokens": 312,
    "total_tokens": 1552
  },
  "tags": ["apple-support", "tutorial"],
  "steps": [
    {
      "step_index": 0,
      "step_type": "llm_call",
      "node_name": "ChatGoogleGenerativeAI",
      "duration_ms": 1102,
      "tool_calls": [
        { "tool_name": "check_warranty",    "tool_input": { "serial_number": "F9KXYZ123" } },
        { "tool_name": "lookup_device_info","tool_input": { "model": "iPhone 15 Pro" } }
      ]
    },
    {
      "step_index": 1,
      "step_type": "tool_call",
      "node_name": "check_warranty",
      "input": "F9KXYZ123",
      "output": "Serial: F9K... | AppleCare+: Active (expires 14 Oct 2025)",
      "duration_ms": 2
    },
    {
      "step_index": 2,
      "step_type": "tool_call",
      "node_name": "lookup_device_info",
      "input": "iPhone 15 Pro",
      "output": "iPhone 15 Pro | Chip: A17 Pro | Status: Fully supported.",
      "duration_ms": 1
    },
    {
      "step_index": 3,
      "step_type": "llm_call",
      "node_name": "ChatGoogleGenerativeAI",
      "duration_ms": 2136,
      "token_usage": { "prompt_tokens": 1240, "completion_tokens": 312, "total_tokens": 1552 }
    }
  ]
}
```

---

## Querying traces directly in MongoDB

If you're using your own MongoDB, you can query traces with standard MongoDB commands:

```javascript
// Most recent 10 traces
db.traces.find().sort({ started_at: -1 }).limit(10)

// All failed traces
db.traces.find({ status: "error" })

// Traces that used the check_warranty tool
db.traces.find({ "steps.tool_calls.tool_name": "check_warranty" })

// Average duration across all runs
db.traces.aggregate([
  { $group: { _id: "$agent_name", avg_ms: { $avg: "$duration_ms" } } }
])
```

---

## Key concepts

| Concept | Description |
|---------|-------------|
| `agent_snoop.init()` | One-line setup. Reads env vars, picks the right backend, prints a banner. |
| `tracer.trace(input=...)` | Creates a `TraceHandle` for one agent invocation. |
| `AgentSnoopCallbackHandler` | Attached to LangGraph's `callbacks` list. Observes every node and tool call. |
| `handler.on_chain_end_final(result)` | Closes the trace, writes the final output, flushes to storage. |
| `TraceHandle` | Returned by `tracer.trace()`. You can call `.set_metadata()`, `.add_tag()`, `.set_output()` on it directly. |

---

## Next steps

- Add `tags` to every trace to filter by environment: `tracer.trace(input=q, tags=["prod"])`
- Add `session_id` to group multiple traces from the same user session: `tracer.trace(input=q, session_id=user_id)`
- Try the post-run integration (no callback changes needed): `parse_langgraph_output(result, input=q)`