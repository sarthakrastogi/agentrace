"""Tests for agent_snoop core models."""

from datetime import datetime, timezone

import pytest

from agent_snoop.core.models import (
    Step,
    StepType,
    ToolCall,
    TokenUsage,
    Trace,
    TraceStatus,
)


def test_trace_defaults():
    trace = Trace()
    assert trace.status == TraceStatus.RUNNING
    assert trace.steps == []
    assert trace.trace_id  # non-empty UUID string


def test_step_duration_ms():
    step = Step()
    assert step.duration_ms is None
    step.ended_at = datetime.now(timezone.utc)
    assert step.duration_ms is not None
    assert step.duration_ms >= 0


def test_trace_aggregate_token_usage():
    trace = Trace()
    trace.steps = [
        Step(
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            )
        ),
        Step(
            token_usage=TokenUsage(
                prompt_tokens=5, completion_tokens=10, total_tokens=15
            )
        ),
        Step(),  # no token usage
    ]
    total = trace.aggregate_token_usage()
    assert total.prompt_tokens == 15
    assert total.completion_tokens == 30
    assert total.total_tokens == 45


def test_trace_model_dump_mongo_has_id():
    trace = Trace(agent_name="test")
    doc = trace.model_dump_mongo()
    assert doc["_id"] == trace.trace_id
    assert doc["agent_name"] == "test"


def test_tool_call_duration():
    tc = ToolCall(tool_name="search")
    assert tc.duration_ms is None
    tc.ended_at = datetime.now(timezone.utc)
    assert isinstance(tc.duration_ms, float)


def test_step_type_enum():
    step = Step(step_type=StepType.LLM_CALL)
    dumped = step.model_dump(mode="json")
    assert dumped["step_type"] == "llm_call"
