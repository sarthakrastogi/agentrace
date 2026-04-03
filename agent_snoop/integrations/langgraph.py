# agent_snoop/integrations/langgraph.py
"""
LangGraph integration for agent_snoop.

Two integration styles are supported:

Style 1 — Callback (live, step-by-step)
----------------------------------------
Wire up the callback handler via ``tracer.langgraph_config()``:

    config = tracer.langgraph_config(input=user_query)
    result = await graph.ainvoke({"messages": [...]}, config=config)

The handler automatically captures every node start/end, tool calls,
and LLM invocations, and flushes to storage when the graph finishes.

Style 2 — Post-run (simplest possible integration)
---------------------------------------------------
Run your graph however you like, then pass the result to the parser:

    from agent_snoop.integrations.langgraph import parse_langgraph_output

    result = await graph.ainvoke({"messages": [...]})
    trace = parse_langgraph_output(
        output=result,
        input=user_query,
        agent_name="my-agent",
    )
    tracer.log_trace(trace)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from agent_snoop.core.models import (
    Step,
    StepType,
    ToolCall,
    TokenUsage,
    Trace,
    TraceStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback handler (live tracing)
# ---------------------------------------------------------------------------


class AgentSnoopCallbackHandler:
    """
    LangChain / LangGraph callback handler that builds a Trace in real-time.

    Compatible with ``langchain_core.callbacks.BaseCallbackHandler``.
    You don't need to subclass it explicitly because LangGraph accepts
    any object with the right methods.
    """

    ignore_chain = False
    raise_error = True
    ignore_chat_model = False
    ignore_llm = False
    ignore_tool = False
    run_inline = True
    ignore_agent = False
    ignore_chain_end = False
    ignore_chain_end_final = False

    def __init__(self, handle: Any) -> None:
        """
        Parameters
        ----------
        handle:
            A :class:`~agent_snoop.core.tracer.TraceHandle` from
            ``AgentTracer.trace()``.
        """
        self._handle = handle
        self._active_steps: Dict[str, Step] = {}  # run_id -> Step
        self._active_tool_calls: Dict[str, ToolCall] = {}  # run_id -> ToolCall

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        step = Step(
            step_type=StepType.LLM_CALL,
            node_name=_get_name(serialized),
            input=prompts,
            started_at=datetime.now(timezone.utc),
        )
        self._active_steps[str(run_id)] = step

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        # messages is a list-of-lists; flatten and serialize each message
        flat_messages = [_safe_serialize(m) for batch in messages for m in batch]
        step = Step(
            step_type=StepType.LLM_CALL,
            node_name=_get_name(serialized),
            messages=flat_messages,
            started_at=datetime.now(timezone.utc),
        )
        self._active_steps[str(run_id)] = step

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        step = self._active_steps.pop(str(run_id), None)
        if step is None:
            return
        step.ended_at = datetime.now(timezone.utc)
        step.output = _safe_serialize(response)
        # Extract token usage if available
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage") or response.llm_output.get(
                "usage"
            )
            if usage:
                step.token_usage = _parse_token_usage(usage)
        self._handle.add_step(step)

    def on_llm_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._active_steps.pop(str(run_id), None)
        if step is None:
            return
        step.ended_at = datetime.now(timezone.utc)
        step.error = str(error)
        self._handle.add_step(step)

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        tool_call = ToolCall(
            tool_name=_get_name(serialized),
            tool_input={"input": input_str},
            started_at=datetime.now(timezone.utc),
        )
        self._active_tool_calls[str(run_id)] = tool_call

        # Also create a wrapping step for the tool
        step = Step(
            step_type=StepType.TOOL_CALL,
            node_name=_get_name(serialized),
            input=input_str,
            started_at=datetime.now(timezone.utc),
        )
        self._active_steps[str(run_id)] = step

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_call = self._active_tool_calls.pop(str(run_id), None)
        step = self._active_steps.pop(str(run_id), None)
        now = datetime.now(timezone.utc)

        if tool_call:
            tool_call.tool_output = _safe_serialize(output)
            tool_call.ended_at = now

        if step:
            step.ended_at = now
            step.output = _safe_serialize(output)
            if tool_call:
                step.tool_calls.append(tool_call)
            self._handle.add_step(step)

    def on_tool_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_call = self._active_tool_calls.pop(str(run_id), None)
        step = self._active_steps.pop(str(run_id), None)
        now = datetime.now(timezone.utc)

        if tool_call:
            tool_call.error = str(error)
            tool_call.ended_at = now

        if step:
            step.ended_at = now
            step.error = str(error)
            if tool_call:
                step.tool_calls.append(tool_call)
            self._handle.add_step(step)

    # ------------------------------------------------------------------
    # Chain / node callbacks (LangGraph nodes show up here)
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        # Only track top-level chain or named nodes, ignore deeply nested chains
        name = _get_name(serialized)
        if name in ("RunnableSequence", "RunnableLambda", ""):
            return
        step = Step(
            step_type=StepType.AGENT_ACTION,
            node_name=name,
            input=_safe_serialize(inputs),
            started_at=datetime.now(timezone.utc),
        )
        self._active_steps[str(run_id)] = step

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._active_steps.pop(str(run_id), None)
        if step is None:
            return
        step.ended_at = datetime.now(timezone.utc)
        step.output = _safe_serialize(outputs)
        self._handle.add_step(step)

    def on_chain_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._active_steps.pop(str(run_id), None)
        if step is None:
            return
        step.ended_at = datetime.now(timezone.utc)
        step.error = str(error)
        self._handle.add_step(step)

    # ------------------------------------------------------------------
    # Final flush — called when the top-level graph run finishes
    # ------------------------------------------------------------------

    def on_chain_end_final(self, output: Any) -> None:
        """Call this manually after graph.ainvoke() if not using context manager."""
        self._handle.set_output(_safe_serialize(output))
        self._handle.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Post-run parser (Style 2 — no callbacks needed)
# ---------------------------------------------------------------------------


def parse_langgraph_output(
    output: Any,
    input: Optional[Any] = None,
    agent_name: str = "agent",
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Trace:
    """
    Build a :class:`~agent_snoop.core.models.Trace` from LangGraph's final
    output state dict.

    This is the simplest integration path — run your graph normally and
    then pass the result here::

        result = await graph.ainvoke({"messages": [user_message]})
        trace = parse_langgraph_output(result, input=user_query)
        tracer.log_trace(trace)

    Parameters
    ----------
    output:
        The dict returned by ``graph.invoke()`` / ``graph.ainvoke()``.
    input:
        The original user input (for display purposes).
    agent_name:
        Name of the agent.
    session_id:
        Optional session grouping ID.
    tags / metadata:
        Arbitrary labels and key-value pairs.

    Returns
    -------
    Trace
        A complete, ready-to-log trace. The trajectory is reconstructed
        from the messages in the output state.
    """
    trace = Trace(
        agent_name=agent_name,
        framework="langgraph",
        input=input,
        session_id=session_id,
        tags=tags or [],
        metadata=metadata or {},
        status=TraceStatus.SUCCESS,
        ended_at=datetime.now(timezone.utc),
    )

    # Reconstruct steps from the message history in the output state
    messages = _extract_messages(output)
    for i, msg in enumerate(messages):
        step = _message_to_step(msg, i)
        if step:
            trace.steps.append(step)

    # Best-effort: set the final output to the last AI message content
    trace.output = _extract_final_output(output)
    trace.total_token_usage = trace.aggregate_token_usage()
    return trace


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_name(serialized: Any) -> str:
    if not serialized:
        return ""
    return serialized.get("name") or (serialized.get("id") or [""])[-1] or ""


def _safe_serialize(obj: Any) -> Any:
    """Best-effort conversion to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    # LangChain message objects
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return _safe_serialize(obj.__dict__)
    return str(obj)


def _parse_token_usage(usage: Any) -> TokenUsage:
    if isinstance(usage, dict):
        return TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
    return TokenUsage()


def _extract_messages(output: Any) -> List[Any]:
    """Pull the messages list out of a LangGraph state dict."""
    if isinstance(output, dict):
        return output.get("messages", [])
    if hasattr(output, "messages"):
        return list(output.messages)
    return []


def _message_to_step(msg: Any, index: int) -> Optional[Step]:
    """Convert a LangChain message object to a Step."""
    try:
        msg_type = type(msg).__name__  # HumanMessage, AIMessage, ToolMessage, etc.
        content = getattr(msg, "content", str(msg))
        tool_calls_raw = getattr(msg, "tool_calls", []) or []

        if "AI" in msg_type or "Assistant" in msg_type:
            step_type = StepType.LLM_CALL
        elif "Tool" in msg_type or "Function" in msg_type:
            step_type = StepType.TOOL_CALL
        elif "Human" in msg_type or "User" in msg_type:
            step_type = StepType.HUMAN_INPUT
        else:
            step_type = StepType.OTHER

        tool_calls = []
        for tc in tool_calls_raw:
            if isinstance(tc, dict):
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.get("name", "unknown"),
                        tool_input=tc.get("args", {}),
                    )
                )

        return Step(
            step_index=index,
            step_type=step_type,
            node_name=msg_type,
            output=content,
            tool_calls=tool_calls,
        )
    except Exception:
        logger.debug("agent_snoop: could not parse message %s", msg, exc_info=True)
        return None


def _extract_final_output(output: Any) -> Optional[str]:
    """Extract the last AI message's text content as the final answer."""
    messages = _extract_messages(output)
    for msg in reversed(messages):
        msg_type = type(msg).__name__
        if "AI" in msg_type or "Assistant" in msg_type:
            content = getattr(msg, "content", None)
            if content and isinstance(content, str):
                return content
    return None
