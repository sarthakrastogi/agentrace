"""
LangGraph integration example — three styles, pick what fits your workflow.

Run with:
    pip install agent-snoop[mongo,langgraph]
    python examples/langgraph_example.py
"""

import asyncio
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

from agent_snoop import AgentTracer
from agent_snoop.storage.mongodb import MongoDBBackend

# ---------------------------------------------------------------------------
# Minimal LangGraph agent
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
llm = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.set_entry_point("chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Setup tracer
# ---------------------------------------------------------------------------

storage = MongoDBBackend(MONGO_URI, database="agent_snoop_demo")
tracer = AgentTracer(
    storage=storage,
    agent_name="langgraph-chatbot",
    framework="langgraph",
    default_tags=["demo"],
)
graph = build_graph()


# ---------------------------------------------------------------------------
# Style 1: One-liner — tracer.langgraph_config() (RECOMMENDED)
# The callback handler captures every step automatically.
# ---------------------------------------------------------------------------


async def run_style1(user_query: str):
    print("\n=== Style 1: langgraph_config (automatic callback) ===")
    config = tracer.langgraph_config(
        input=user_query,
        metadata={"style": "callback"},
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_query)]},
        config=config,
    )
    print("Answer:", result["messages"][-1].content)
    print("(trace saved automatically)")


# ---------------------------------------------------------------------------
# Style 2: Post-run — parse output after the fact (SIMPLEST for existing code)
# No changes to your agent code at all.
# ---------------------------------------------------------------------------


async def run_style2(user_query: str):
    print("\n=== Style 2: post-run parse (zero agent changes) ===")
    from agent_snoop.integrations.langgraph import parse_langgraph_output

    result = await graph.ainvoke({"messages": [HumanMessage(content=user_query)]})
    print("Answer:", result["messages"][-1].content)

    trace = parse_langgraph_output(
        output=result,
        input=user_query,
        agent_name="langgraph-chatbot",
    )
    tracer.log_trace(trace)
    print(f"Trace {trace.trace_id} saved.")


# ---------------------------------------------------------------------------
# Style 3: Manual context manager (most control)
# ---------------------------------------------------------------------------


async def run_style3(user_query: str):
    print("\n=== Style 3: manual context manager (full control) ===")
    async with tracer.trace(input=user_query, tags=["manual"]) as t:
        result = await graph.ainvoke({"messages": [HumanMessage(content=user_query)]})
        t.set_output(result["messages"][-1].content)
        t.set_metadata(graph_version="v1")

    print(f"Answer: {result['messages'][-1].content}")
    print(f"Trace ID: {t.trace_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    query = "What is the capital of New Zealand, and what is it known for?"
    await run_style1(query)
    await run_style2(query)
    await run_style3(query)


if __name__ == "__main__":
    asyncio.run(main())
