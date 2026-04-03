# agent_snoop/storage/liten_backend.py

"""
Liten.tech hosted storage backend.

Sends traces to the liten.tech API so they appear in the user's dashboard
without requiring them to expose their own MongoDB to the internet.

Set the ``AGENTSNOOP_API_KEY`` environment variable (or pass ``api_key``
directly) to activate this backend.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Optional

from agent_snoop.core.models import Trace
from agent_snoop.storage.base import StorageBackend

logger = logging.getLogger(__name__)

LITEN_API_URL = "https://liten.tech/api/trace/save"
_TIMEOUT_SECONDS = 10


class LitenBackend(StorageBackend):
    """
    Sends traces to liten.tech via HTTPS.

    Uses only the standard library (``urllib``) so there are zero extra
    dependencies.

    Parameters
    ----------
    api_key:
        Your liten.tech API key.  If omitted, read from ``AGENTSNOOP_API_KEY``.
    api_url:
        Override the endpoint (useful for self-hosted liten or testing).
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = LITEN_API_URL,
    ) -> None:
        if not api_key:
            raise ValueError(
                "agent_snoop: an API key is required for LitenBackend.\n"
                "Set the AGENTSNOOP_API_KEY environment variable or pass api_key=."
            )
        self._api_key = api_key
        self._api_url = api_url

        # Verify the key is at least structurally plausible before the first trace.
        logger.debug("agent_snoop: LitenBackend ready → %s", self._api_url)

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save_trace(self, trace: Trace) -> None:
        payload = trace.model_dump_mongo()
        # _id is a MongoDB-ism; the API uses trace_id as the canonical key.
        payload.pop("_id", None)

        body = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            self._api_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "agent-snoop-sdk/1.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                status = resp.status
                if status not in (200, 201):
                    body_text = resp.read().decode("utf-8", errors="replace")
                    logger.warning(
                        "agent_snoop: unexpected status %s from liten.tech: %s",
                        status,
                        body_text,
                    )
                else:
                    logger.debug(
                        "agent_snoop: trace %s accepted by liten.tech (HTTP %s).",
                        trace.trace_id,
                        status,
                    )
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401:
                raise PermissionError(
                    "agent_snoop: invalid or expired AGENTSNOOP_API_KEY.\n"
                    "Check your key at https://liten.tech/traces"
                ) from exc
            logger.error(
                "agent_snoop: HTTP %s saving trace %s to liten.tech: %s",
                exc.code,
                trace.trace_id,
                body_text,
            )
        except urllib.error.URLError as exc:
            logger.error(
                "agent_snoop: could not reach liten.tech (%s). "
                "Trace %s was NOT saved. Check your network connection.",
                exc.reason,
                trace.trace_id,
            )
        except Exception:
            logger.exception(
                "agent_snoop: unexpected error saving trace %s to liten.tech.",
                trace.trace_id,
            )
