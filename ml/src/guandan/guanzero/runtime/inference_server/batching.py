"""Shared-memory buffer layout and lifecycle for the inference IPC protocol.

Wire format: tiny ``RequestDesc`` (~24 B) through ``mp.Queue`` referencing
slots in preallocated shared-memory ``state_buf`` / ``action_buf`` blocks.
Per-actor response slot in shared mem + ``mp.Event`` for wakeup.

Layout (matches encoder.py output ordering — see _STATE_OFFSETS):

    State (per-request, 7 fields, 3226 bytes)
      own_hand                  108
      others_hand               108
      recent_action_each_player 432   (4 × 108)
      played_cards_others       324   (3 × 108)
      remaining_counts_others    81   (3 × 27)
      level                      13
      history                  2160   (20 × 108)

    Per-action (variable K, 2 fields, 117 bytes)
      behavior                    9
      candidate_action          108
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from ...model.encoding.base_encoder import ENCODE_CHANNEL_SHAPES


# Sentinel for samples produced without an active server (epsilon shortcut,
# CPU fallback). Treated by buffer analytics as "unknown / not-server".
NO_VERSION = -1


class InferenceTimeoutError(RuntimeError):
    pass


class InferenceProtocolError(RuntimeError):
    pass


# ─── Field element layout ─────────────────────────────────────────────────────


def _channel_elements(name: str) -> int:
    """Number of uint8 elements in one row of channel ``name``."""
    return math.prod(ENCODE_CHANNEL_SHAPES[name])


_STATE_FIELDS: tuple[tuple[str, int], ...] = (
    ("own_hand",                  _channel_elements("own_hand")),
    ("others_hand",               _channel_elements("others_hand")),
    ("recent_action_each_player", _channel_elements("recent_action_each_player")),
    ("played_cards_others",       _channel_elements("played_cards_others")),
    ("remaining_counts_others",   _channel_elements("remaining_counts_others")),
    ("level",                     _channel_elements("level")),
    ("behavior",                  _channel_elements("behavior")),
    ("history",                   _channel_elements("history")),
)
_ACTION_FIELDS: tuple[tuple[str, int], ...] = (
    ("candidate_action",          _channel_elements("candidate_action")),
)

STATE_SIZE  = sum(n for _, n in _STATE_FIELDS)
ACTION_SIZE = sum(n for _, n in _ACTION_FIELDS)

_STATE_OFFSETS: dict[str, tuple[int, int]] = {}
_off = 0
for _name, _n in _STATE_FIELDS:
    _STATE_OFFSETS[_name] = (_off, _off + _n)
    _off += _n
_ACTION_OFFSETS: dict[str, tuple[int, int]] = {}
_off = 0
for _name, _n in _ACTION_FIELDS:
    _ACTION_OFFSETS[_name] = (_off, _off + _n)
    _off += _n
del _off, _name, _n


# ─── Shared-memory data types ─────────────────────────────────────────────────


@dataclass
class SharedBufferMeta:
    """Names + shapes of the shared-memory blocks. Pickled into child
    processes so they can attach to the same blocks the parent allocated."""
    state_buf_name:    str
    action_buf_name:   str
    response_buf_name: str
    num_slots:         int
    max_actions:       int
    n_actors:          int


@dataclass
class SharedBuffers:
    """Per-process attached views of the three shared-memory blocks."""
    state_buf:        np.ndarray              # uint8 [num_slots, STATE_SIZE]
    action_buf:       np.ndarray              # uint8 [num_slots, max_actions, ACTION_SIZE]
    response_buf:     np.ndarray              # uint32 [n_actors, 3] (req_id, chosen, version)
    free_slots:       Any                     # mp.Queue[int]
    request_queue:    Any                     # mp.Queue[RequestDesc]
    events:           list                    # list[mp.Event], length n_actors
    weights_version:  Any | None = None       # mp.Value('Q') or None
    _shm_handles:     list = field(default_factory=list)


@dataclass
class RequestDesc:
    """Tiny descriptor pickled through ``request_queue``. ~24 bytes."""
    slot:       int
    actor_id:   int
    request_id: int
    seat:       int
    n_actions:  int


# ─── Lifecycle helpers ────────────────────────────────────────────────────────


def allocate_shared_buffers(
    num_slots:   int,
    max_actions: int,
    n_actors:    int,
    ctx,                     # mp.context (spawn)
) -> tuple[SharedBuffers, SharedBufferMeta]:
    """Parent-side: allocate all shared blocks + queues + events."""
    shm_state    = shared_memory.SharedMemory(create=True, size=num_slots * STATE_SIZE)
    shm_action   = shared_memory.SharedMemory(
        create=True, size=num_slots * max_actions * ACTION_SIZE
    )
    shm_response = shared_memory.SharedMemory(create=True, size=n_actors * 3 * 4)

    state_buf    = np.ndarray((num_slots, STATE_SIZE),               dtype=np.uint8,  buffer=shm_state.buf)
    action_buf   = np.ndarray((num_slots, max_actions, ACTION_SIZE), dtype=np.uint8,  buffer=shm_action.buf)
    response_buf = np.ndarray((n_actors, 3),                         dtype=np.uint32, buffer=shm_response.buf)

    state_buf.fill(0)
    action_buf.fill(0)
    response_buf.fill(0)

    free_slots    = ctx.Queue(maxsize=num_slots)
    for i in range(num_slots):
        free_slots.put(i)
    request_queue = ctx.Queue(maxsize=num_slots)
    events        = [ctx.Event() for _ in range(n_actors)]

    bufs = SharedBuffers(
        state_buf=state_buf, action_buf=action_buf, response_buf=response_buf,
        free_slots=free_slots, request_queue=request_queue, events=events,
        _shm_handles=[shm_state, shm_action, shm_response],
    )
    meta = SharedBufferMeta(
        state_buf_name=shm_state.name, action_buf_name=shm_action.name,
        response_buf_name=shm_response.name,
        num_slots=num_slots, max_actions=max_actions, n_actors=n_actors,
    )
    return bufs, meta


def attach_shared_buffers(
    meta:           SharedBufferMeta,
    free_slots,
    request_queue,
    events,
    weights_version=None,
) -> SharedBuffers:
    """Child-side: re-attach to the parent's shared blocks by name."""
    shm_state    = shared_memory.SharedMemory(name=meta.state_buf_name)
    shm_action   = shared_memory.SharedMemory(name=meta.action_buf_name)
    shm_response = shared_memory.SharedMemory(name=meta.response_buf_name)

    state_buf    = np.ndarray((meta.num_slots, STATE_SIZE),                    dtype=np.uint8,  buffer=shm_state.buf)
    action_buf   = np.ndarray((meta.num_slots, meta.max_actions, ACTION_SIZE), dtype=np.uint8,  buffer=shm_action.buf)
    response_buf = np.ndarray((meta.n_actors, 3),                              dtype=np.uint32, buffer=shm_response.buf)

    return SharedBuffers(
        state_buf=state_buf, action_buf=action_buf, response_buf=response_buf,
        free_slots=free_slots, request_queue=request_queue, events=events,
        weights_version=weights_version,
        _shm_handles=[shm_state, shm_action, shm_response],
    )


def release_shared_buffers(bufs: SharedBuffers, unlink: bool) -> None:
    """Detach (and optionally unlink — parent only) the shared-memory blocks."""
    for shm in bufs._shm_handles:
        try:
            shm.close()
        except Exception:
            pass
        if unlink:
            try:
                shm.unlink()
            except Exception:
                pass


__all__ = [
    "NO_VERSION",
    "InferenceTimeoutError",
    "InferenceProtocolError",
    "STATE_SIZE",
    "ACTION_SIZE",
    "_STATE_FIELDS",
    "_ACTION_FIELDS",
    "_STATE_OFFSETS",
    "_ACTION_OFFSETS",
    "SharedBufferMeta",
    "SharedBuffers",
    "RequestDesc",
    "allocate_shared_buffers",
    "attach_shared_buffers",
    "release_shared_buffers",
]
