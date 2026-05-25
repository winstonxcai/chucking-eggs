"""Helpers for supervised background asyncio tasks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


def create_logged_task(coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task:
    """Create a task and log unhandled exceptions.

    Fire-and-forget tasks are still allowed in this small app, but their
    failures should be visible in logs instead of disappearing into the event
    loop.
    """
    task = asyncio.create_task(coro, name=name)

    def _log_result(done: asyncio.Task) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Background task failed: %s", name)

    task.add_done_callback(_log_result)
    return task
