"""Tests for AgentTracer using a simple in-memory storage backend."""

import pytest

from agent_snoop import AgentTracer, Trace, TraceStatus
from agent_snoop.storage.base import StorageBackend


class InMemoryBackend(StorageBackend):
    """Minimal in-memory backend for testing."""

    def __init__(self):
        self.traces: dict[str, Trace] = {}

    def save_trace(self, trace: Trace) -> None:
        self.traces[trace.trace_id] = trace

    def get_trace(self, trace_id: str):
        return self.traces.get(trace_id)


@pytest.fixture
def storage():
    return InMemoryBackend()


@pytest.fixture
def tracer(storage):
    return AgentTracer(storage=storage, agent_name="test-agent")


def test_trace_context_manager_success(tracer, storage):
    with tracer.trace(input="hello") as t:
        t.set_output("world")

    assert len(storage.traces) == 1
    trace = list(storage.traces.values())[0]
    assert trace.status == TraceStatus.SUCCESS
    assert trace.input == "hello"
    assert trace.output == "world"
    assert trace.ended_at is not None
    assert trace.duration_ms is not None


def test_trace_context_manager_exception(tracer, storage):
    with pytest.raises(ValueError):
        with tracer.trace(input="bad") as t:
            raise ValueError("something went wrong")

    trace = list(storage.traces.values())[0]
    assert trace.status == TraceStatus.ERROR
    assert "ValueError" in trace.error


def test_log_trace_directly(tracer, storage):
    trace = Trace(agent_name="direct", input="q", output="a")
    tracer.log_trace(trace)

    assert trace.trace_id in storage.traces
    saved = storage.traces[trace.trace_id]
    assert saved.status == TraceStatus.SUCCESS


def test_default_tags_applied(storage):
    tracer = AgentTracer(storage=storage, agent_name="x", default_tags=["prod", "v2"])
    with tracer.trace() as t:
        pass

    trace = list(storage.traces.values())[0]
    assert "prod" in trace.tags
    assert "v2" in trace.tags


def test_metadata_set(tracer, storage):
    with tracer.trace() as t:
        t.set_metadata(user_id="u123", env="staging")

    trace = list(storage.traces.values())[0]
    assert trace.metadata["user_id"] == "u123"
    assert trace.metadata["env"] == "staging"


def test_storage_error_does_not_raise(storage):
    """Tracer must not propagate storage errors to the caller."""

    class BrokenBackend(StorageBackend):
        def save_trace(self, trace):
            raise RuntimeError("DB is down")

    tracer = AgentTracer(storage=BrokenBackend(), agent_name="safe")
    # Should not raise
    with tracer.trace(input="x") as t:
        t.set_output("y")
