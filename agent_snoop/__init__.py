# agent_snoop/__init__.py

"""
agent_snoop — lightweight agent observability SDK.

Quick start
-----------
The simplest setup is one line.  agent_snoop reads your environment
automatically and picks the right storage backend:

    import agent_snoop
    tracer = agent_snoop.init()

Storage selection (in priority order):

    1. AGENTSNOOP_API_KEY  → traces sent to liten.tech (hosted, zero infra)
    2. MONGODB_URI / MONGO_URI / MONGODB_URL / DATABASE_URL  → saved to your
       own MongoDB under the ``agentsnoop_db`` database
    3. Neither set → a warning is printed and traces are silently dropped

You can also configure storage explicitly:

    from agent_snoop.storage.mongodb import MongoDBBackend
    from agent_snoop.storage.liten_backend import LitenBackend

    tracer = agent_snoop.init(storage=MongoDBBackend(uri="mongodb://..."))
    tracer = agent_snoop.init(storage=LitenBackend(api_key="sk-..."))
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from agent_snoop.core.models import (  # noqa: F401 — re-exported for convenience
    Step,
    StepType,
    Trace,
    TraceStatus,
    TokenUsage,
    ToolCall,
)
from agent_snoop.core.tracer import AgentTracer
from agent_snoop.storage.base import StorageBackend, NoOpBackend

__all__ = [
    "init",
    "AgentTracer",
    "Step",
    "StepType",
    "Trace",
    "TraceStatus",
    "TokenUsage",
    "ToolCall",
]

logger = logging.getLogger(__name__)

# Environment variable names we probe for a MongoDB URI (in priority order).
_MONGO_ENV_VARS = [
    "MONGODB_URI",
    "MONGO_URI",
    "MONGODB_URL",
    "MONGO_URL",
    "DATABASE_URL",
]

# Liten.tech dashboard base URL — used in log messages.
_LITEN_DASHBOARD = "https://liten.tech/traces"
_LITEN_SIGNUP = "https://liten.tech"
_LITEN_DOCS = "https://liten.tech/docs/sdk"

# ─────────────────────────────────────────────────────────────────────────────


def init(
    *,
    agent_name: str = "agent",
    framework: Optional[str] = None,
    default_tags: Optional[list] = None,
    storage: Optional[StorageBackend] = None,
    # Explicit overrides (skip env-var lookup)
    api_key: Optional[str] = None,
    mongo_uri: Optional[str] = None,
) -> AgentTracer:
    """
    Initialise agent_snoop and return a ready-to-use :class:`AgentTracer`.

    Parameters
    ----------
    agent_name:
        A human-readable label for this agent (stored on every trace).
    framework:
        e.g. ``"langgraph"``, ``"autogen"``.  Set automatically by
        framework integrations.
    default_tags:
        Tags applied to every trace produced by this tracer.
    storage:
        Pass an explicit :class:`~agent_snoop.storage.base.StorageBackend`
        to skip auto-detection entirely.
    api_key:
        liten.tech API key.  Overrides ``AGENTSNOOP_API_KEY`` env var.
    mongo_uri:
        MongoDB connection string.  Overrides all ``MONGODB_URI`` env vars.

    Returns
    -------
    AgentTracer
        Fully configured tracer, ready to use.
    """
    if storage is not None:
        # Caller supplied their own backend — use it directly.
        logger.debug("agent_snoop: using caller-supplied storage backend.")
        return AgentTracer(
            storage=storage,
            agent_name=agent_name,
            framework=framework,
            default_tags=default_tags,
        )

    resolved_backend = _auto_detect_backend(api_key=api_key, mongo_uri=mongo_uri)
    return AgentTracer(
        storage=resolved_backend,
        agent_name=agent_name,
        framework=framework,
        default_tags=default_tags,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal auto-detection
# ─────────────────────────────────────────────────────────────────────────────


def _auto_detect_backend(
    api_key: Optional[str],
    mongo_uri: Optional[str],
) -> StorageBackend:
    """
    Probe environment variables and return the appropriate backend.

    Priority:
        1. Explicit api_key / mongo_uri arguments
        2. AGENTSNOOP_API_KEY  → LitenBackend
        3. MONGODB_URI (or variants) → MongoDBBackend
        4. Neither → NoOpBackend + warning
    """

    # ── 1. Explicit api_key ──────────────────────────────────────────────────
    resolved_key = api_key or os.environ.get("AGENTSNOOP_API_KEY", "").strip()
    if resolved_key:
        return _build_liten_backend(resolved_key)

    # ── 2. Explicit / env-var MongoDB URI ────────────────────────────────────
    resolved_uri = mongo_uri
    if not resolved_uri:
        for var in _MONGO_ENV_VARS:
            val = os.environ.get(var, "").strip()
            if val:
                resolved_uri = val
                logger.debug("agent_snoop: picked up MongoDB URI from %s.", var)
                break

    if resolved_uri:
        return _build_mongo_backend(resolved_uri)

    # ── 3. Nothing configured ────────────────────────────────────────────────
    _warn_no_storage()
    return NoOpBackend()


def _build_liten_backend(api_key: str) -> StorageBackend:
    from agent_snoop.storage.liten_backend import LitenBackend

    backend = LitenBackend(api_key=api_key)

    _banner(
        "📡  agent_snoop — sending traces to liten.tech",
        [
            "  Storage   : liten.tech hosted (Anthropic-managed MongoDB)",
            f"  Dashboard : {_LITEN_DASHBOARD}",
            "",
            "  ✅  Your traces will appear in your dashboard within seconds.",
            "  ℹ️   Traces are stored on liten.tech's infrastructure.",
            "       To keep data in your own database, use a MongoDB URI instead.",
            f"       Docs: {_LITEN_DOCS}",
        ],
    )
    return backend


def _build_mongo_backend(uri: str) -> StorageBackend:
    from agent_snoop.storage.mongodb import MongoDBBackend, AGENTSNOOP_DB, _redact_uri

    try:
        backend = MongoDBBackend(uri=uri)
    except ConnectionError as exc:
        _banner(
            "⚠️   agent_snoop — MongoDB connection failed",
            [
                f"  URI       : {_redact_uri(uri)}",
                "",
                str(exc),
                "",
                "  Traces will NOT be persisted until the connection is fixed.",
                f"  Docs: {_LITEN_DOCS}",
            ],
            level=logging.WARNING,
        )
        return NoOpBackend()

    _banner(
        "🗄️   agent_snoop — saving traces to your MongoDB",
        [
            f"  URI       : {_redact_uri(uri)}",
            f"  Database  : {AGENTSNOOP_DB}",
            "  Collection: traces",
            "",
            "  ✅  Traces are stored in YOUR database — full data ownership.",
            f"  👉  View them at: {_LITEN_DASHBOARD}",
            "       (add the same MongoDB URI in Settings → Connect Database)",
        ],
    )
    return backend


def _warn_no_storage() -> None:
    _banner(
        "⚠️  No storage configured for AgentSnoop",
        [
            "  Traces will NOT be saved.",
            "",
            "  Choose one of the following options:",
            "",
            "  Option A — Use liten.tech (easiest, no infra needed):",
            f"    1. Sign up at {_LITEN_SIGNUP}",
            "    2. Copy your API key from the dashboard",
            "    3. Set the env var:  AGENTSNOOP_API_KEY=sk-...",
            "       ✅  Traces stored securely on liten.tech",
            "",
            "  Option B — Use your own MongoDB (full data ownership):",
            "    1. Set the env var:  MONGODB_URI=mongodb+srv://...",
            "       ✅  Data stays in your own database",
            f"       Then connect it to liten.tech: {_LITEN_DASHBOARD}",
            "",
            f"  Docs: {_LITEN_DOCS}",
        ],
        level=logging.WARNING,
    )


def _banner(title: str, lines: list, level: int = logging.INFO) -> None:
    """Emit a clearly delineated log block so it stands out in terminal output."""
    width = 68
    border = "─" * width
    body = "\n".join(f"  {line}" for line in lines)
    message = f"\n  {border}\n  {title}\n  {border}\n{body}\n  {border}"
    logging.getLogger("agent_snoop").log(level, message)
