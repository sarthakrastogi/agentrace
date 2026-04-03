# agent_snoop/core/tracer.py

"""
AgentTracer — the central object users interact with.
"""

from __future__ import annotations

import asyncio
import logging
import traceback as tb
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent_snoop.core.models import Step, Trace, TraceStatus
from agent_snoop.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class TraceHandle:
    def __init__(self, tracer: "AgentTracer", trace: Trace) -> None:
        self._tracer = tracer
        self.trace = trace

    # sync context manager
    def __enter__(self) -> "TraceHandle":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._finalise(exc_type, exc_val, exc_tb)
        self._tracer._flush(self.trace)

    # async context manager — flush is fire-and-forget
    async def __aenter__(self) -> "TraceHandle":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._finalise(exc_type, exc_val, exc_tb)
        asyncio.ensure_future(self._tracer._async_flush(self.trace))

    def _finalise(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.trace.status = TraceStatus.ERROR
            self.trace.error = "".join(tb.format_exception(exc_type, exc_val, exc_tb))
        else:
            if self.trace.status == TraceStatus.RUNNING:
                self.trace.status = TraceStatus.SUCCESS
        self.trace.ended_at = datetime.now(timezone.utc)
        self.trace.total_token_usage = self.trace.aggregate_token_usage()

    # helpers
    def set_output(self, output: Any) -> None:
        self.trace.output = output

    def set_status(self, status: TraceStatus) -> None:
        self.trace.status = status

    def add_tag(self, *tags: str) -> None:
        self.trace.tags.extend(tags)

    def set_metadata(self, **kwargs: Any) -> None:
        self.trace.metadata.update(kwargs)

    def add_step(self, step: Step) -> None:
        step.step_index = len(self.trace.steps)
        self.trace.steps.append(step)

    @property
    def trace_id(self) -> str:
        return self.trace.trace_id


class AgentTracer:
    """
    Main entry-point for agent_snoop.

    Parameters
    ----------
    storage      : a StorageBackend instance (e.g. MongoDBBackend)
    agent_name   : label stored on every trace
    framework    : e.g. "langgraph"
    default_tags : applied to every trace from this tracer
    """

    def __init__(
        self,
        storage: StorageBackend,
        agent_name: str = "agent",
        framework: Optional[str] = None,
        default_tags: Optional[List[str]] = None,
    ) -> None:
        self.storage = storage
        self.agent_name = agent_name
        self.framework = framework
        self.default_tags: List[str] = default_tags or []

    def trace(
        self,
        input: Optional[Any] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceHandle:
        """
        Start a new trace.  Use as sync or async context manager::

            async with tracer.trace(input=query) as t:
                result = await agent.arun(query)
                t.set_output(result)
        """
        trace = Trace(
            agent_name=self.agent_name,
            framework=self.framework,
            input=input,
            session_id=session_id,
            tags=list(self.default_tags) + (tags or []),
            metadata=metadata or {},
        )
        return TraceHandle(self, trace)

    def log_trace(self, trace: Trace) -> None:
        """Directly log a pre-built Trace (e.g. from post-run parser)."""
        if trace.status == TraceStatus.RUNNING:
            trace.status = TraceStatus.SUCCESS
        if trace.ended_at is None:
            trace.ended_at = datetime.now(timezone.utc)
        trace.total_token_usage = trace.aggregate_token_usage()
        self._flush(trace)

    def langgraph_config(
        self,
        input: Optional[Any] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns a LangGraph config dict with the agent_snoop callback wired up.

            result = await graph.ainvoke(state, config=tracer.langgraph_config(input=q))
        """
        from agent_snoop.integrations.langgraph import AgentSnoopCallbackHandler

        handle = self.trace(
            input=input, session_id=session_id, tags=tags, metadata=metadata
        )
        handler = AgentSnoopCallbackHandler(handle)
        return {"callbacks": [handler]}

    # ------------------------------------------------------------------
    # Internal flush helpers
    # ------------------------------------------------------------------

    def _flush(self, trace: Trace) -> None:
        """
        Sync flush — but if a loop is already running, schedule async
        so we never block the event loop.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_flush(trace))
        except RuntimeError:
            # No running loop — plain sync save is fine
            try:
                self.storage.save_trace(trace)
            except Exception:
                logger.exception(
                    "agent_snoop: failed to save trace %s — continuing silently.",
                    trace.trace_id,
                )

    async def _async_flush(self, trace: Trace) -> None:
        """
        Async flush — runs the DB write in a thread executor so the
        event loop is never blocked, even with a sync pymongo driver.
        """
        try:
            await self.storage.async_save_trace(trace)
            logger.debug("agent_snoop: saved trace %s", trace.trace_id)
        except Exception:
            logger.exception(
                "agent_snoop: failed to async save trace %s — continuing silently.",
                trace.trace_id,
            )
