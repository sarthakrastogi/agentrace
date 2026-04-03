# agent_snoop/core/models.py

"""
Core data models for agent_snoop.

All trace data is represented as Pydantic models for validation,
serialisation, and easy MongoDB storage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StepType(str, Enum):
    """Categorises what kind of work happened in a step."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    AGENT_ACTION = "agent_action"
    ROUTER = "router"
    HUMAN_INPUT = "human_input"
    OTHER = "other"


class TraceStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Tool call
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """Represents a single tool / function call made by the agent."""

    tool_name: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    tool_output: Optional[Any] = None
    error: Optional[str] = None
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: Optional[datetime] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return None

    def model_dump_mongo(self) -> Dict[str, Any]:
        d = self.model_dump(mode="json")
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        return d


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Step (one node / action in the agent graph)
# ---------------------------------------------------------------------------


class Step(BaseModel):
    """
    A single step in an agent's execution trajectory.

    One Step corresponds to one node execution in LangGraph, one action
    in ReAct, one agent turn in AutoGen, etc.
    """

    step_id: str = Field(default_factory=_new_id)
    step_index: int = 0
    step_type: StepType = StepType.OTHER
    node_name: Optional[str] = None  # e.g. LangGraph node name

    # Input / output
    input: Optional[Any] = None
    output: Optional[Any] = None

    # LLM details
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None  # raw message list
    token_usage: Optional[TokenUsage] = None

    # Tool calls made during this step
    tool_calls: List[ToolCall] = Field(default_factory=list)

    # Timing
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: Optional[datetime] = None

    # Arbitrary metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return None

    def model_dump_mongo(self) -> Dict[str, Any]:
        d = self.model_dump(mode="json")
        d["tool_calls"] = [tc.model_dump_mongo() for tc in self.tool_calls]
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        return d


# ---------------------------------------------------------------------------
# Trace (one full agent invocation)
# ---------------------------------------------------------------------------


class Trace(BaseModel):
    """
    A complete trace of a single agent invocation.

    Contains the full trajectory (list of Steps) plus top-level metadata
    about the run.
    """

    trace_id: str = Field(default_factory=_new_id)
    session_id: Optional[str] = None  # group multiple traces together
    agent_name: str = "agent"
    framework: Optional[str] = None  # e.g. "langgraph", "autogen"

    # The question / task the user gave
    input: Optional[Any] = None
    # The final answer / output
    output: Optional[Any] = None

    status: TraceStatus = TraceStatus.RUNNING
    error: Optional[str] = None

    # Full step-by-step trajectory
    steps: List[Step] = Field(default_factory=list)

    # Aggregated token usage across all steps
    total_token_usage: Optional[TokenUsage] = None

    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: Optional[datetime] = None

    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return None

    def aggregate_token_usage(self) -> TokenUsage:
        """Sum token usage across all steps."""
        total = TokenUsage()
        for step in self.steps:
            if step.token_usage:
                total.prompt_tokens += step.token_usage.prompt_tokens
                total.completion_tokens += step.token_usage.completion_tokens
                total.total_tokens += step.token_usage.total_tokens
        return total

    def model_dump_mongo(self) -> Dict[str, Any]:
        d = self.model_dump(mode="json")
        d["steps"] = [s.model_dump_mongo() for s in self.steps]
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        # Use trace_id as the Mongo _id for easy lookups
        d["_id"] = self.trace_id
        return d
