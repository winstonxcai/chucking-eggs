"""Shared-memory inference server for distributed Dart actors.

Submodules:
- ``batching`` — shared-memory buffer types, field layout, lifecycle helpers
- ``client``   — actor-side IPC client (InferenceClient, build_client)
- ``server``   — batched GPU inference server (InferenceServer, run_server)
"""

from .batching import (
    NO_VERSION,
    ACTION_SIZE,
    STATE_SIZE,
    InferenceProtocolError,
    InferenceTimeoutError,
    RequestDesc,
    SharedBufferMeta,
    SharedBuffers,
    allocate_shared_buffers,
    attach_shared_buffers,
    release_shared_buffers,
)
from .client import InferenceArgs, InferenceClient, build_client
from .server import InferenceServer, run_server

__all__ = [
    "InferenceArgs",
    "InferenceClient",
    "InferenceServer",
    "run_server",
    "build_client",
    "SharedBuffers",
    "SharedBufferMeta",
    "RequestDesc",
    "allocate_shared_buffers",
    "attach_shared_buffers",
    "release_shared_buffers",
    "InferenceTimeoutError",
    "InferenceProtocolError",
    "STATE_SIZE",
    "ACTION_SIZE",
    "NO_VERSION",
]
