"""Shared GPU inference server for distributed GuanZero actors.

Phase 1 implementation (mp.Queue PoC):
    Actors send (seat, encoded_list) requests via mp.Queue. The server
    drains a hybrid (size + row-count + timeout) batch policy, groups
    requests by seat, runs forward on the 4 q-nets, and returns chosen
    action indices via per-actor reply queues.

    No shared memory yet — every payload pickles through mp.Queue. This
    isolates the batching/numerics concerns before we optimize IPC.

Phase 2 (planned, not in this file yet):
    Replace the pickled `encoded_list` in each request with a slot-id
    pointing into preallocated `state_buf` / `action_buf` shared-memory
    blocks. The client/server API stays the same; only the wire format
    changes.

API — actor side:
    client = InferenceClient(actor_id, request_queue, response_queue)
    chosen_idx, policy_version = client.submit(seat, encoded_list)

API — server side:
    server = InferenceServer(cfg, q_nets, request_queue, response_queues,
                             stop_event, weights_lock=None,
                             weights_buf=None, weights_version=None)
    server.server_loop()  # runs until stop_event.is_set()

Numerical equivalence:
    For matching weights and the same encoded_list, the server's chosen
    action index is identical to the local-CPU `actor._argmax_q` path on
    CPU (exact match) and matches up to floating-point near-ties on CUDA
    (see tests/guanzero/test_inference_server.py).
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch

from .buffer import collate_encoded
from .q_network import GuanZeroQNet


logger = logging.getLogger("guanzero.inference_server")


# Sentinel returned by the epsilon shortcut / CPU fallback when no server
# version is meaningful for the produced sample. Treated by buffer analytics
# as "unknown / not-server".
NO_VERSION = -1


@dataclass
class _Request:
    """Phase-1 wire format: full pickled payload through mp.Queue.

    actor_id and request_id form a unique key so responses can be routed
    back to the right actor; actor blocks on its dedicated response_queue
    until the server pushes a reply matching its request_id.
    """
    actor_id:     int
    request_id:   int
    seat:         int
    encoded_list: list[dict]   # length K, will be pickled


@dataclass
class _Response:
    request_id:   int
    chosen_idx:   int
    version:      int


# ─── Actor-side client ───────────────────────────────────────


class InferenceClient:
    """Actor-side client. One per actor process.

    `submit()` blocks until the server returns the chosen action index.
    The actor must own a unique `actor_id` and a private `response_queue`
    that only the server writes to.
    """

    def __init__(
        self,
        actor_id:        int,
        request_queue,                          # mp.Queue
        response_queue,                         # mp.Queue (private to this actor)
        weights_version_view=None,              # mp.Value('Q') — read-only handle
        timeout_s:       float = 5.0,
    ) -> None:
        self.actor_id        = actor_id
        self.request_queue   = request_queue
        self.response_queue  = response_queue
        self._weights_version_view = weights_version_view
        self.timeout_s       = timeout_s
        self._next_req_id    = 0

    def submit(self, seat: int, encoded_list: list[dict]) -> tuple[int, int]:
        """Send one inference request, block on the response.

        Returns ``(chosen_idx, policy_version)`` where ``policy_version`` is
        the server's local weight version at time of forward.
        """
        if len(encoded_list) == 0:
            raise ValueError("encoded_list must be non-empty")

        req_id = self._next_req_id
        self._next_req_id += 1

        req = _Request(
            actor_id=self.actor_id,
            request_id=req_id,
            seat=seat,
            encoded_list=encoded_list,
        )
        self.request_queue.put(req)

        # Block on response; the server writes only to our response_queue,
        # so we don't need request_id matching beyond a defensive sanity check.
        try:
            resp = self.response_queue.get(timeout=self.timeout_s)
        except queue.Empty as e:
            raise InferenceTimeoutError(
                f"actor {self.actor_id} timed out waiting for inference response"
            ) from e

        if resp.request_id != req_id:
            raise InferenceProtocolError(
                f"actor {self.actor_id}: expected request_id={req_id}, got {resp.request_id}"
            )
        return resp.chosen_idx, resp.version

    def latest_known_version(self) -> int:
        """Best-effort read of the learner's weight-publish counter for the
        epsilon shortcut. Returns NO_VERSION if no shared counter is wired.
        """
        if self._weights_version_view is None:
            return NO_VERSION
        return int(self._weights_version_view.value)


class InferenceTimeoutError(RuntimeError):
    pass


class InferenceProtocolError(RuntimeError):
    pass


# ─── Server-side ─────────────────────────────────────────────


class InferenceServer:
    """One-process GPU inference server.

    Drains the request queue under a hybrid batching policy:
      - flush when len(batch) >= max_requests
      - flush when sum(K_i) >= max_action_rows
      - flush when elapsed_ms since first request >= timeout_ms

    Batches are then grouped by seat and run through the 4 position q-nets.
    Per-request argmax over the K rows for that request gives the chosen index.

    Optional weight-version refresh: pass `weights_lock`, `weights_buf`,
    `weights_version` and a `weight_specs` list to enable shared-memory
    weight reload. Phase 1 leaves these None and reloads only on cold start.
    """

    def __init__(
        self,
        q_nets:                  Mapping[int, GuanZeroQNet],
        request_queue,                                            # mp.Queue
        response_queues:         dict[int, "queue.Queue"],        # actor_id -> mp.Queue
        stop_event,                                               # mp.Event
        device:                  str | torch.device = "cpu",
        max_requests:            int = 32,
        max_action_rows:         int = 4096,
        timeout_ms:              float = 1.0,
        use_bf16:                bool = False,
        # Optional weight refresh wiring (Phase 4+):
        weights_lock=None,
        weights_buf:             Optional[torch.Tensor] = None,
        weights_version=None,
        weight_specs:            Optional[list] = None,
    ) -> None:
        self.device          = torch.device(device)
        self.q_nets          = {p: q_nets[p].to(self.device).eval() for p in range(4)}
        self.request_queue   = request_queue
        self.response_queues = response_queues
        self.stop_event      = stop_event
        self.max_requests    = max_requests
        self.max_action_rows = max_action_rows
        self.timeout_s       = timeout_ms / 1000.0
        self.use_bf16        = use_bf16 and self.device.type == "cuda"

        self._weights_lock    = weights_lock
        self._weights_buf     = weights_buf
        self._weights_version = weights_version
        self._weight_specs    = weight_specs
        self._local_version   = 0   # bumps every time we reload from weights_buf

        # Per-batch metric accumulators (for jsonl logging by caller)
        self.metrics: dict[str, float] = {
            "n_batches":      0.0,
            "n_requests":     0.0,
            "n_action_rows":  0.0,
            "forward_ms_sum": 0.0,
        }

    # ── batching loop ──────────────────────────────────────

    def server_loop(self) -> None:
        """Main loop. Returns when stop_event is set or queue closes."""
        logger.info("InferenceServer started on device=%s, local_version=%d",
                    self.device, self._local_version)
        while not self.stop_event.is_set():
            self._maybe_reload_weights()
            batch = self._drain_batch()
            if not batch:
                continue
            self._process_batch(batch)
        logger.info("InferenceServer exiting; processed %d batches, %d requests",
                    int(self.metrics["n_batches"]), int(self.metrics["n_requests"]))

    def _drain_batch(self) -> list[_Request]:
        """Block briefly for the first request, then drain by hybrid policy."""
        try:
            first = self.request_queue.get(timeout=0.5)
        except queue.Empty:
            return []
        batch:  list[_Request] = [first]
        rows                   = len(first.encoded_list)
        deadline               = time.perf_counter() + self.timeout_s
        while (
            len(batch) < self.max_requests
            and rows < self.max_action_rows
            and time.perf_counter() < deadline
        ):
            try:
                req = self.request_queue.get_nowait()
                batch.append(req)
                rows += len(req.encoded_list)
            except queue.Empty:
                # short yield to let actors enqueue more work
                time.sleep(0.0001)
        return batch

    # ── inference ──────────────────────────────────────────

    def _process_batch(self, batch: list[_Request]) -> None:
        t0 = time.perf_counter()
        chosen = self._infer_batch(batch)
        forward_ms = (time.perf_counter() - t0) * 1000.0

        # Scatter responses
        for req, idx in zip(batch, chosen):
            resp = _Response(req.request_id, int(idx), self._local_version)
            try:
                self.response_queues[req.actor_id].put(resp)
            except KeyError:
                logger.warning("no response_queue for actor_id=%d (dropped)",
                               req.actor_id)

        # Metrics
        rows = sum(len(r.encoded_list) for r in batch)
        self.metrics["n_batches"]      += 1
        self.metrics["n_requests"]     += len(batch)
        self.metrics["n_action_rows"]  += rows
        self.metrics["forward_ms_sum"] += forward_ms

    @torch.no_grad()
    def _infer_batch(self, batch: list[_Request]) -> list[int]:
        """Group by seat, run one forward per seat, return per-request argmax.

        Phase 1 reuses ``collate_encoded`` for simplicity — Phase 2 replaces
        this with a fast-path that reads directly from shared-memory views
        and uses ``repeat_interleave`` to expand state across K rows.
        """
        # Group by seat
        by_seat: dict[int, list[tuple[int, _Request]]] = {p: [] for p in range(4)}
        for original_idx, req in enumerate(batch):
            by_seat[req.seat].append((original_idx, req))

        chosen: list[int] = [0] * len(batch)

        for seat, items in by_seat.items():
            if not items:
                continue

            # Flatten all K-rows from all requests for this seat into one collate
            flat_encoded: list[dict] = []
            sizes: list[int] = []
            for _, req in items:
                flat_encoded.extend(req.encoded_list)
                sizes.append(len(req.encoded_list))

            batch_dict = collate_encoded(flat_encoded, device=self.device)

            ctx = (
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.use_bf16 else nullcontext()
            )
            with ctx:
                q = self.q_nets[seat](batch_dict)              # shape [sum_K]

            # Per-request argmax over its K-row slice
            offset = 0
            for (original_idx, _), K in zip(items, sizes):
                segment = q[offset : offset + K]
                chosen[original_idx] = int(segment.argmax().item())
                offset += K

        return chosen

    # ── weights ────────────────────────────────────────────

    def _maybe_reload_weights(self) -> None:
        """Phase 4+ hook. Phase 1 server is started with frozen weights."""
        if self._weights_version is None or self._weights_buf is None:
            return
        if self._weights_version.value <= self._local_version:
            return
        if self._weight_specs is None:
            return
        with self._weights_lock:
            for spec in self._weight_specs:
                t = self.q_nets[spec.seat].state_dict()[spec.name]
                t.copy_(
                    self._weights_buf[spec.offset : spec.offset + spec.numel].view(spec.shape)
                )
            self._local_version = int(self._weights_version.value)
        logger.debug("server weights reloaded to version=%d", self._local_version)


# ─── Process entry point ─────────────────────────────────────


def server_loop_entry(
    cfg_dict:       dict,
    q_net_kwargs:   dict,
    request_queue,
    response_queues,
    stop_event,
    initial_state_dicts: dict[int, dict] | None = None,
    weights_lock=None,
    weights_buf=None,
    weights_version=None,
    weight_specs=None,
) -> None:
    """Top-level subprocess entry. Picklable for spawn context.

    Builds q_nets fresh in the child process (MPS/CUDA contexts can't be
    forked), optionally seeds them from `initial_state_dicts`, then runs
    the server loop.
    """
    import torch
    from .q_network import init_position_nets

    torch.set_num_threads(1)

    cfg_device       = cfg_dict.get("inference_device", "cpu")
    max_requests     = cfg_dict.get("inference_batch_max_requests", 32)
    max_action_rows  = cfg_dict.get("inference_batch_max_action_rows", 4096)
    timeout_ms       = cfg_dict.get("inference_batch_timeout_ms", 1.0)
    use_bf16         = cfg_dict.get("use_bf16_learner", False) and cfg_device == "cuda"

    q_nets = init_position_nets(**q_net_kwargs)
    if initial_state_dicts is not None:
        for p in range(4):
            q_nets[p].load_state_dict(initial_state_dicts[p])

    server = InferenceServer(
        q_nets=q_nets,
        request_queue=request_queue,
        response_queues=response_queues,
        stop_event=stop_event,
        device=cfg_device,
        max_requests=max_requests,
        max_action_rows=max_action_rows,
        timeout_ms=timeout_ms,
        use_bf16=use_bf16,
        weights_lock=weights_lock,
        weights_buf=weights_buf,
        weights_version=weights_version,
        weight_specs=weight_specs,
    )
    server.server_loop()


__all__ = [
    "InferenceClient",
    "InferenceServer",
    "InferenceTimeoutError",
    "InferenceProtocolError",
    "server_loop_entry",
    "NO_VERSION",
]
