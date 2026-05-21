"""Metrics emission for Dart training.

One ``MetricsWriter`` per learner process fans out to one or more
``MetricsBackend`` sinks. Today the only backend is JSONL on disk; W&B or
TensorBoard can be added by implementing the same protocol.

Every row is tagged with ``_schema`` so older runs remain parseable when the
metric set evolves. Bump ``METRICS_SCHEMA_VERSION`` when fields are renamed
or removed; additive changes do not require a bump.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

METRICS_SCHEMA_VERSION = 1


class MetricsBackend(Protocol):
    """Sink for a single structured metrics row."""

    def write(self, row: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...


class JsonlBackend:
    """Append rows as newline-delimited JSON to ``path``."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("a", buffering=1)  # line-buffered

    def write(self, row: Mapping[str, Any]) -> None:
        self._fp.write(json.dumps(row) + "\n")

    def close(self) -> None:
        self._fp.flush()
        self._fp.close()


class MetricsWriter:
    """Fan out structured rows to one or more backends.

    Accepts dataclass instances or plain mappings; stamps every emitted row
    with the current schema version.
    """

    def __init__(self, backends: list[MetricsBackend]) -> None:
        self._backends = backends

    def write(self, row: Any) -> None:
        if dataclasses.is_dataclass(row):
            payload = dataclasses.asdict(row)
        else:
            payload = dict(row)
        payload["_schema"] = METRICS_SCHEMA_VERSION
        for b in self._backends:
            b.write(payload)

    def close(self) -> None:
        for b in self._backends:
            b.close()


def jsonl_writer(path: Path) -> MetricsWriter:
    """Convenience factory: writer with a single JSONL backend."""
    return MetricsWriter([JsonlBackend(path)])


__all__ = [
    "METRICS_SCHEMA_VERSION",
    "MetricsBackend",
    "JsonlBackend",
    "MetricsWriter",
    "jsonl_writer",
]
