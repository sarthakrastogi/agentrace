# agent_snoop/storage/mongodb.py

"""
MongoDB storage backend for agent_snoop.

Requires ``pymongo``:

    pip install agent_snoop[mongo]
    # or
    pip install pymongo
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent_snoop.core.models import Trace
from agent_snoop.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# The database agent_snoop always writes to in the user's MongoDB.
AGENTSNOOP_DB = "agentsnoop_db"
AGENTSNOOP_COLLECTION = "traces"


class MongoDBBackend(StorageBackend):
    """
    Stores agent traces in the user's own MongoDB instance.

    The database is always ``agentsnoop_db`` and the collection ``traces``,
    so the user knows exactly where to look and what to point liten.tech at.

    Parameters
    ----------
    uri:
        MongoDB connection string.
    client_kwargs:
        Extra keyword arguments forwarded to ``pymongo.MongoClient``.
    """

    def __init__(self, uri: str, **client_kwargs: Any) -> None:
        try:
            from pymongo import MongoClient, ASCENDING, DESCENDING
        except ImportError as exc:
            raise ImportError(
                "pymongo is required for MongoDBBackend.\n"
                "Install it with:  pip install pymongo\n"
                "              or:  pip install agent_snoop[mongo]"
            ) from exc

        # Use a short timeout for the startup ping so bad URIs fail fast.
        client_kwargs.setdefault("serverSelectionTimeoutMS", 5000)
        self._client = MongoClient(uri, **client_kwargs)

        # Eagerly verify the connection so misconfiguration is caught immediately.
        try:
            self._client.admin.command("ping")
        except Exception as exc:
            self._client.close()
            # Extract just the human-readable part — pymongo errors are very verbose.
            short_reason = str(exc).split("Timeout:")[0].strip().rstrip(",")
            raise ConnectionError(
                f"agent_snoop: could not reach MongoDB at {_redact_uri(uri)}\n"
                f"  Reason: {short_reason}\n\n"
                f"  Check your URI, credentials, and network/IP-allowlist settings."
            ) from exc

        self._db = self._client[AGENTSNOOP_DB]
        self._col = self._db[AGENTSNOOP_COLLECTION]
        self._ensure_indexes(ASCENDING, DESCENDING)

    # ------------------------------------------------------------------
    # Index setup
    # ------------------------------------------------------------------

    def _ensure_indexes(self, ASCENDING: int, DESCENDING: int) -> None:
        try:
            self._col.create_index([("agent_name", ASCENDING)])
            self._col.create_index([("session_id", ASCENDING)])
            self._col.create_index([("started_at", DESCENDING)])
            self._col.create_index([("status", ASCENDING)])
            self._col.create_index([("tags", ASCENDING)])
        except Exception:
            logger.warning(
                "agent_snoop: could not create MongoDB indexes.", exc_info=True
            )

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save_trace(self, trace: Trace) -> None:
        doc = trace.model_dump_mongo()
        self._col.replace_one({"_id": trace.trace_id}, doc, upsert=True)
        logger.debug("agent_snoop: trace %s written to MongoDB.", trace.trace_id)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        doc = self._col.find_one({"_id": trace_id})
        return self._doc_to_trace(doc) if doc else None

    def list_traces(
        self,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Trace]:
        query: Dict[str, Any] = {}
        if agent_name:
            query["agent_name"] = agent_name
        if session_id:
            query["session_id"] = session_id
        cursor = self._col.find(query).sort("started_at", -1).skip(skip).limit(limit)
        return [self._doc_to_trace(doc) for doc in cursor]

    def get_latest_traces(self, n: int = 10) -> List[Trace]:
        cursor = self._col.find().sort("started_at", -1).limit(n)
        return [self._doc_to_trace(doc) for doc in cursor]

    def delete_trace(self, trace_id: str) -> bool:
        result = self._col.delete_one({"_id": trace_id})
        return result.deleted_count > 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_to_trace(doc: Dict[str, Any]) -> Trace:
        doc = dict(doc)
        if "_id" in doc and "trace_id" not in doc:
            doc["trace_id"] = str(doc.pop("_id"))
        elif "_id" in doc:
            doc.pop("_id")
        return Trace.model_validate(doc)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MongoDBBackend":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def _redact_uri(uri: str) -> str:
    """Replace the password in a MongoDB URI with *** for safe logging."""
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(uri)
        if parsed.password:
            netloc = parsed.netloc.replace(parsed.password, "***")
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return "<uri>"
