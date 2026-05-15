"""Actor-side IPC client for the shared-memory inference server."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from .batching import (
    NO_VERSION,
    InferenceProtocolError,
    InferenceTimeoutError,
    RequestDesc,
    SharedBufferMeta,
    SharedBuffers,
    _ACTION_OFFSETS,
    _STATE_OFFSETS,
    attach_shared_buffers,
)


class InferenceArgs(TypedDict):
    """Shared-memory handles passed from the training process to each actor.

    Produced by ``train.py`` after ``allocate_shared_buffers``; consumed by
    ``build_client`` inside each spawned actor subprocess.
    """
    meta:          SharedBufferMeta
    free_slots:    Any   # mp.Queue[int]
    request_queue: Any   # mp.Queue[RequestDesc]
    events:        Any   # list[mp.Event]


def build_client(
    actor_id:       int,
    inference_args: InferenceArgs,
    timeout_s:      float,
    max_actions:    int,
) -> "InferenceClient":
    """Construct an ``InferenceClient`` inside a spawned actor subprocess.

    Reattaches to shared-memory blocks here because spawn-context children
    do not inherit parent memory mappings.
    """
    bufs = attach_shared_buffers(
        meta            = inference_args["meta"],
        free_slots      = inference_args["free_slots"],
        request_queue   = inference_args["request_queue"],
        events          = inference_args["events"],
        weights_version = inference_args.get("weights_version"),
    )
    return InferenceClient(
        actor_id    = actor_id,
        bufs        = bufs,
        timeout_s   = timeout_s,
        max_actions = max_actions,
    )


class InferenceClient:
    """Actor-side client for the shared-memory inference server.

    Payload travels through preallocated shared-memory slots referenced by
    tiny ``RequestDesc`` descriptors on ``request_queue``.
    """

    def __init__(
        self,
        actor_id:     int,
        bufs:         SharedBuffers,
        timeout_s:    float = 5.0,
        max_actions:  int = 320,
    ) -> None:
        self.actor_id    = actor_id
        self.bufs        = bufs
        self.timeout_s   = timeout_s
        self.max_actions = max_actions
        self._next_req_id = 0

    def _pack_state(self, slot: int, encoded: dict) -> None:
        """Copy the shared-state channels of one decision into state_buf[slot]."""
        row = self.bufs.state_buf[slot]
        for name, (lo, hi) in _STATE_OFFSETS.items():
            arr = encoded[name]
            np.copyto(row[lo:hi], arr.reshape(-1).astype(np.uint8, copy=False))

    def _pack_actions(self, slot: int, encoded_list: list[dict], K: int) -> None:
        """Copy K per-action vectors into action_buf[slot, :K]."""
        block = self.bufs.action_buf[slot, :K]
        for k in range(K):
            row = block[k]
            for name, (lo, hi) in _ACTION_OFFSETS.items():
                arr = encoded_list[k][name].reshape(-1)
                np.copyto(row[lo:hi], arr.astype(np.uint8, copy=False))

    def submit(self, seat: int, encoded_list: list[dict]) -> tuple[int, int]:
        """Send one inference request through shared mem. Blocks for response."""
        K = len(encoded_list)
        if K == 0:
            raise ValueError("encoded_list must be non-empty")
        assert K <= self.max_actions, (
            f"K={K} exceeds inference_max_actions={self.max_actions}; "
            f"either bump max_actions or have caller truncate identically to local path"
        )
        evt = self.bufs.events[self.actor_id]
        assert not evt.is_set(), (
            f"actor {self.actor_id}: stale event signal — invariant violated "
            f"(one in-flight request per actor)"
        )

        slot = self.bufs.free_slots.get()
        try:
            self._pack_state(slot, encoded_list[0])
            self._pack_actions(slot, encoded_list, K)

            req_id = self._next_req_id
            self._next_req_id = (self._next_req_id + 1) & 0xFFFFFFFF

            desc = RequestDesc(slot, self.actor_id, req_id, seat, K)
            self.bufs.request_queue.put(desc)

            if not evt.wait(timeout=self.timeout_s):
                raise InferenceTimeoutError(
                    f"actor {self.actor_id} timed out waiting for inference response"
                )
            evt.clear()

            stored_req_id, chosen, version = self.bufs.response_buf[self.actor_id]
            if int(stored_req_id) != req_id:
                raise InferenceProtocolError(
                    f"actor {self.actor_id}: expected request_id={req_id}, "
                    f"got {int(stored_req_id)}"
                )
            return int(chosen), int(version)
        finally:
            self.bufs.free_slots.put(slot)

    def latest_known_version(self) -> int:
        """O(1) read of the last server version stamped into our response slot."""
        version_in_slot = int(self.bufs.response_buf[self.actor_id][2])
        if version_in_slot > 0:
            return version_in_slot
        if self.bufs.weights_version is not None:
            return int(self.bufs.weights_version.value)
        return NO_VERSION


__all__ = [
    "InferenceArgs",
    "InferenceClient",
    "build_client",
]
