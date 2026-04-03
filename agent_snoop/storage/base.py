# agent_snoop/storage/base.py

"""
Abstract base class for storage backends.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from agent_snoop.core.models import Trace

_executor = ThreadPoolExecutor(thread_name_prefix="agent-snoop-storage")


class StorageBackend(ABC):
    """
    Interface that every storage backend must implement.

    Only ``save_trace`` is required.
    """

    @abstractmethod
    def save_trace(self, trace: Trace) -> None:
        """Persist a completed trace (synchronous)."""
        ...

    async def async_save_trace(self, trace: Trace) -> None:
        """
        Persist a completed trace without blocking the event loop.

        Default implementation runs ``save_trace`` in a thread-pool executor.
        Override in subclasses that have a native async driver.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self.save_trace, trace)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        raise NotImplementedError(f"{type(self).__name__} does not support get_trace")

    def list_traces(
        self,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Trace]:
        raise NotImplementedError(f"{type(self).__name__} does not support list_traces")

    def delete_trace(self, trace_id: str) -> bool:
        raise NotImplementedError(
            f"{type(self).__name__} does not support delete_trace"
        )


class NoOpBackend(StorageBackend):
    """
    A backend that does nothing — used when no storage is configured.
    Traces are silently dropped after the warning is shown at init time.
    """

    def save_trace(self, trace: Trace) -> None:
        pass
