"""Shared GPU inference server for distributed GuanZero actors.

Two implementations live side by side, sharing the same actor-facing API:

  Phase 1 — InferenceClient + InferenceServer
      Wire format: pickled `encoded_list` through `mp.Queue`. Per-actor reply
      queue. Used for unit tests (numerics + batching policy validation) and
      as the simplest implementation reference.

  Phase 2 — SharedInferenceClient + SharedInferenceServer
      Wire format: tiny `RequestDesc` (~24 B) through `mp.Queue` referencing
      slots in preallocated shared-memory `state_buf` / `action_buf` blocks.
      Per-actor response slot in shared mem + `mp.Event` for wakeup. Slot
      ownership is held by the actor for the entire submit lifetime (acquire
      → write → publish desc → wait response → read response → return slot).

Both paths produce numerically identical chosen action indices for matching
q-net weights — see tests/guanzero/test_inference_server.py.

Layout reference (matches encoder.py output ordering — see _STATE_OFFSETS):

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

import dataclasses
import logging
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from multiprocessing import shared_memory
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


# ─── Field byte layout (must match encoder.py output) ────────


# (key, byte length per row); order matters — these are the offsets the
# server uses to slice flat state/action tensors back into named channels.
_STATE_FIELDS: tuple[tuple[str, int], ...] = (
    ("own_hand",                  108),
    ("others_hand",               108),
    ("recent_action_each_player", 4 * 108),
    ("played_cards_others",       3 * 108),
    ("remaining_counts_others",   3 * 27),
    ("level",                     13),
    ("history",                   20 * 108),
)
_ACTION_FIELDS: tuple[tuple[str, int], ...] = (
    ("behavior",                  9),
    ("candidate_action",          108),
)
# Shape per field (used to reshape the sliced flat tensor back to its 2-D form).
_FIELD_SHAPES: dict[str, tuple[int, ...]] = {
    "own_hand":                  (108,),
    "others_hand":               (108,),
    "recent_action_each_player": (4, 108),
    "played_cards_others":       (3, 108),
    "remaining_counts_others":   (3, 27),
    "level":                     (13,),
    "history":                   (20, 108),
    "behavior":                  (9,),
    "candidate_action":          (108,),
}

STATE_SIZE  = sum(n for _, n in _STATE_FIELDS)    # 3226
ACTION_SIZE = sum(n for _, n in _ACTION_FIELDS)   # 117

# Cumulative offsets — built once at import for fast packing/unpacking.
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
del _off, _name, _n  # leak hygiene


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


# ─── Phase 2: shared-memory buffers + descriptor queue ────────


@dataclass
class SharedBufferMeta:
    """Names + shapes of the shared-memory blocks. Pickled into child
    processes so they can attach to the same blocks the parent allocated.

    The mp.Queue / mp.Event / mp.Value objects are passed separately —
    they marshal themselves through spawn-context pickling.
    """
    state_buf_name:    str
    action_buf_name:   str
    response_buf_name: str
    num_slots:         int
    max_actions:       int
    n_actors:          int


@dataclass
class SharedBuffers:
    """Per-process attached views. Each subprocess builds one of these from
    the SharedBufferMeta + the queue/event handles passed in args.

    Keep the SharedMemory objects alive on the instance so the OS doesn't
    free the mappings while we hold numpy views into them.
    """
    state_buf:        np.ndarray              # uint8 [num_slots, STATE_SIZE]
    action_buf:       np.ndarray              # uint8 [num_slots, max_actions, ACTION_SIZE]
    response_buf:     np.ndarray              # uint32 [n_actors, 3] (req_id, chosen, version)
    free_slots:       "queue.Queue"           # mp.Queue[int] — owned by parent
    request_queue:    "queue.Queue"           # mp.Queue[RequestDesc]
    events:           list                    # list[mp.Event], length n_actors
    weights_version: object | None = None     # mp.Value('Q') or None
    # Internal: keep the shared-memory handles alive
    _shm_handles:     list = field(default_factory=list)


@dataclass
class RequestDesc:
    """Tiny descriptor pickled through `request_queue`. ~24 bytes."""
    slot:       int
    actor_id:   int
    request_id: int
    seat:       int
    n_actions:  int


def allocate_shared_buffers(
    num_slots:    int,
    max_actions:  int,
    n_actors:     int,
    ctx,                                       # mp.context (spawn)
) -> tuple[SharedBuffers, SharedBufferMeta]:
    """Parent-side: allocate all shared blocks + queues + events. Returns the
    attached views plus the meta blob to pass to children.
    """
    # 1. Shared-memory blocks
    state_size_bytes    = num_slots * STATE_SIZE
    action_size_bytes   = num_slots * max_actions * ACTION_SIZE
    response_size_bytes = n_actors * 3 * 4   # u32

    shm_state    = shared_memory.SharedMemory(create=True, size=state_size_bytes)
    shm_action   = shared_memory.SharedMemory(create=True, size=action_size_bytes)
    shm_response = shared_memory.SharedMemory(create=True, size=response_size_bytes)

    state_buf  = np.ndarray((num_slots, STATE_SIZE),
                            dtype=np.uint8,  buffer=shm_state.buf)
    action_buf = np.ndarray((num_slots, max_actions, ACTION_SIZE),
                            dtype=np.uint8,  buffer=shm_action.buf)
    response_buf = np.ndarray((n_actors, 3),
                              dtype=np.uint32, buffer=shm_response.buf)

    # Zero-init so a stale read on startup doesn't pretend to be a valid response.
    state_buf.fill(0)
    action_buf.fill(0)
    response_buf.fill(0)

    # 2. Free-slot pool — preload all slot ids
    free_slots = ctx.Queue(maxsize=num_slots)
    for i in range(num_slots):
        free_slots.put(i)

    # 3. Request queue + per-actor events
    request_queue = ctx.Queue(maxsize=num_slots)   # bounded by total in-flight slots
    events        = [ctx.Event() for _ in range(n_actors)]

    bufs = SharedBuffers(
        state_buf=state_buf,
        action_buf=action_buf,
        response_buf=response_buf,
        free_slots=free_slots,
        request_queue=request_queue,
        events=events,
        _shm_handles=[shm_state, shm_action, shm_response],
    )
    meta = SharedBufferMeta(
        state_buf_name    = shm_state.name,
        action_buf_name   = shm_action.name,
        response_buf_name = shm_response.name,
        num_slots         = num_slots,
        max_actions       = max_actions,
        n_actors          = n_actors,
    )
    return bufs, meta


def attach_shared_buffers(
    meta:           SharedBufferMeta,
    free_slots,                                  # mp.Queue from parent
    request_queue,                               # mp.Queue from parent
    events,                                      # list[mp.Event] from parent
    weights_version=None,
) -> SharedBuffers:
    """Child-side: re-attach to the parent's shared blocks by name."""
    shm_state    = shared_memory.SharedMemory(name=meta.state_buf_name)
    shm_action   = shared_memory.SharedMemory(name=meta.action_buf_name)
    shm_response = shared_memory.SharedMemory(name=meta.response_buf_name)

    state_buf  = np.ndarray((meta.num_slots, STATE_SIZE),
                            dtype=np.uint8,  buffer=shm_state.buf)
    action_buf = np.ndarray((meta.num_slots, meta.max_actions, ACTION_SIZE),
                            dtype=np.uint8,  buffer=shm_action.buf)
    response_buf = np.ndarray((meta.n_actors, 3),
                              dtype=np.uint32, buffer=shm_response.buf)

    return SharedBuffers(
        state_buf=state_buf,
        action_buf=action_buf,
        response_buf=response_buf,
        free_slots=free_slots,
        request_queue=request_queue,
        events=events,
        weights_version=weights_version,
        _shm_handles=[shm_state, shm_action, shm_response],
    )


def release_shared_buffers(bufs: SharedBuffers, unlink: bool) -> None:
    """Detach (and optionally unlink — parent only) the shared-memory blocks.
    Call `unlink=True` from the parent process exactly once at shutdown.
    """
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


# ── Actor side ──────────────────────────────────────────────


class SharedInferenceClient:
    """Phase 2 actor-side client. Same `submit(seat, encoded_list)` API as
    `InferenceClient`; payload travels through preallocated shared-memory
    slots referenced by tiny `RequestDesc` descriptors on `request_queue`.
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
        """Copy the 7 shared-state channels of one decision into state_buf[slot].
        Encoder produces float32 0/1; we cast to uint8 on the fly (exact)."""
        row = self.bufs.state_buf[slot]                # uint8 view, length 3226
        for name, (lo, hi) in _STATE_OFFSETS.items():
            arr = encoded[name]                         # any shape, ndarray
            flat = arr.reshape(-1)                      # length matches (hi-lo)
            np.copyto(row[lo:hi], flat.astype(np.uint8, copy=False))

    def _pack_actions(self, slot: int, encoded_list: list[dict], K: int) -> None:
        """Copy the K per-action vectors into action_buf[slot, :K]."""
        block = self.bufs.action_buf[slot, :K]         # [K, 117] uint8 view
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

        slot = self.bufs.free_slots.get()              # blocks if pool empty

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
            # Actor owns the slot for its entire lifetime; return it now.
            self.bufs.free_slots.put(slot)

    def latest_known_version(self) -> int:
        """O(1) read of the last server version stamped into our response slot.
        Falls back to the shared mp.Value if no response has arrived yet."""
        version_in_slot = int(self.bufs.response_buf[self.actor_id][2])
        if version_in_slot > 0:
            return version_in_slot
        if self.bufs.weights_version is not None:
            return int(self.bufs.weights_version.value)
        return NO_VERSION


# ── Server side ─────────────────────────────────────────────


class SharedInferenceServer:
    """Phase 2 server. Same hybrid batching policy as InferenceServer.

    Reads from `request_queue` (descriptors), gathers payloads directly from
    `state_buf` / `action_buf`, runs forward, writes responses to
    `response_buf[actor_id]`, and signals `events[actor_id]`.

    Server never returns slots to `free_slots` — actors do that on read.
    """

    def __init__(
        self,
        q_nets:           Mapping[int, GuanZeroQNet],
        bufs:             SharedBuffers,
        stop_event,
        device:           str | torch.device = "cpu",
        max_requests:     int = 32,
        max_action_rows:  int = 4096,
        timeout_ms:       float = 1.0,
        use_bf16:         bool = False,
        # Optional shared-mem weight refresh (Phase 4+)
        weights_lock=None,
        weights_buf:      Optional[torch.Tensor] = None,
        weights_version=None,
        weight_specs:     Optional[list] = None,
    ) -> None:
        self.device          = torch.device(device)
        self.q_nets          = {p: q_nets[p].to(self.device).eval() for p in range(4)}
        self.bufs            = bufs
        self.stop_event      = stop_event
        self.max_requests    = max_requests
        self.max_action_rows = max_action_rows
        self.timeout_s       = timeout_ms / 1000.0
        self.use_bf16        = use_bf16 and self.device.type == "cuda"

        self._weights_lock    = weights_lock
        self._weights_buf     = weights_buf
        self._weights_version = weights_version
        self._weight_specs    = weight_specs
        self._local_version   = 0

        self.metrics: dict[str, float] = {
            "n_batches":      0.0,
            "n_requests":     0.0,
            "n_action_rows":  0.0,
            "forward_ms_sum": 0.0,
        }

    # ── batching loop ──────────────────────────────────────

    def server_loop(self) -> None:
        logger.info("SharedInferenceServer started on device=%s", self.device)
        first_batch_seen = False
        last_metric_log = time.perf_counter()
        last_metric_snapshot = dict(self.metrics)
        try:
            while not self.stop_event.is_set():
                self._maybe_reload_weights()
                batch = self._drain_batch()
                if not batch:
                    continue
                self._process_batch(batch)
                if not first_batch_seen:
                    logger.info(
                        "first batch processed: requests=%d rows=%d forward_ms=%.2f",
                        len(batch),
                        sum(d.n_actions for d in batch),
                        self.metrics["forward_ms_sum"],
                    )
                    first_batch_seen = True
                # Periodic metric summary every ~5s so we can see contention live
                now = time.perf_counter()
                if now - last_metric_log >= 5.0:
                    dt = now - last_metric_log
                    d_b = self.metrics["n_batches"]      - last_metric_snapshot["n_batches"]
                    d_r = self.metrics["n_requests"]     - last_metric_snapshot["n_requests"]
                    d_a = self.metrics["n_action_rows"]  - last_metric_snapshot["n_action_rows"]
                    d_f = self.metrics["forward_ms_sum"] - last_metric_snapshot["forward_ms_sum"]
                    rps  = d_r / dt if dt > 0 else 0
                    rows = d_a / dt if dt > 0 else 0
                    avg_b = d_r / d_b if d_b else 0
                    avg_f = d_f / d_b if d_b else 0
                    logger.info(
                        "interval: %.0f reqs/s  %.0f rows/s  avg_batch=%.1f reqs (%.0f rows)  forward=%.1fms",
                        rps, rows, avg_b, (d_a / d_b if d_b else 0), avg_f,
                    )
                    last_metric_log = now
                    last_metric_snapshot = dict(self.metrics)
        finally:
            logger.info("SharedInferenceServer exiting; %d batches, %d requests, %d rows",
                        int(self.metrics["n_batches"]), int(self.metrics["n_requests"]),
                        int(self.metrics["n_action_rows"]))

    def _drain_batch(self) -> list[RequestDesc]:
        try:
            first = self.bufs.request_queue.get(timeout=0.5)
        except queue.Empty:
            return []
        batch: list[RequestDesc] = [first]
        rows = first.n_actions
        deadline = time.perf_counter() + self.timeout_s
        while (
            len(batch) < self.max_requests
            and rows < self.max_action_rows
            and time.perf_counter() < deadline
        ):
            try:
                desc = self.bufs.request_queue.get_nowait()
                batch.append(desc)
                rows += desc.n_actions
            except queue.Empty:
                time.sleep(0.0001)
        return batch

    # ── inference ──────────────────────────────────────────

    def _process_batch(self, batch: list[RequestDesc]) -> None:
        t0 = time.perf_counter()
        chosen = self._infer_batch(batch)
        forward_ms = (time.perf_counter() - t0) * 1000.0

        # Scatter responses + signal events
        for desc, idx in zip(batch, chosen):
            self.bufs.response_buf[desc.actor_id, 0] = np.uint32(desc.request_id)
            self.bufs.response_buf[desc.actor_id, 1] = np.uint32(idx)
            self.bufs.response_buf[desc.actor_id, 2] = np.uint32(self._local_version)
            self.bufs.events[desc.actor_id].set()

        rows = sum(d.n_actions for d in batch)
        self.metrics["n_batches"]      += 1
        self.metrics["n_requests"]     += len(batch)
        self.metrics["n_action_rows"]  += rows
        self.metrics["forward_ms_sum"] += forward_ms

    @torch.no_grad()
    def _infer_batch(self, descs: list[RequestDesc]) -> list[int]:
        """Group by seat, gather state/action from shared mem, expand state
        on GPU via repeat_interleave, run forward, per-request argmax."""
        by_seat: dict[int, list[tuple[int, RequestDesc]]] = {p: [] for p in range(4)}
        for original_idx, desc in enumerate(descs):
            by_seat[desc.seat].append((original_idx, desc))

        chosen: list[int] = [0] * len(descs)

        for seat, items in by_seat.items():
            if not items:
                continue

            slots = [d.slot for _, d in items]
            Ks    = [d.n_actions for _, d in items]

            # Gather state rows (one per request) and per-action rows (sum_K).
            # NumPy fancy index returns a copy; for the actions we slice+concat.
            state_np  = self.bufs.state_buf[slots]                  # [B, 3226] uint8
            action_np = np.concatenate(
                [self.bufs.action_buf[d.slot, :d.n_actions] for _, d in items],
                axis=0,
            )                                                       # [sum_K, 117] uint8

            state   = torch.from_numpy(state_np).to(self.device, non_blocking=True).float()
            actions = torch.from_numpy(action_np).to(self.device, non_blocking=True).float()

            # On-GPU expansion: state[i] repeated K_i times → [sum_K, 3226]
            repeats = torch.tensor(Ks, device=self.device, dtype=torch.long)
            state_rows = torch.repeat_interleave(state, repeats, dim=0)

            batch_dict = self._unpack_to_qnet_dict(state_rows, actions)

            ctx = (
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.use_bf16 else nullcontext()
            )
            with ctx:
                q = self.q_nets[seat](batch_dict)                   # [sum_K]

            offset = 0
            for (original_idx, _), K in zip(items, Ks):
                segment = q[offset : offset + K]
                chosen[original_idx] = int(segment.argmax().item())
                offset += K

        return chosen

    def _unpack_to_qnet_dict(
        self,
        state_rows: torch.Tensor,                # [N, 3226] float
        actions:    torch.Tensor,                # [N, 117]  float
    ) -> dict[str, torch.Tensor]:
        """Slice the flat byte tensors into the named-channel dict that
        GuanZeroQNet.forward consumes. Pure tensor offset arithmetic — no
        Python iteration over rows."""
        N = state_rows.shape[0]
        out: dict[str, torch.Tensor] = {}
        for name, (lo, hi) in _STATE_OFFSETS.items():
            shape = (N,) + _FIELD_SHAPES[name]
            out[name] = state_rows[:, lo:hi].reshape(*shape)
        for name, (lo, hi) in _ACTION_OFFSETS.items():
            shape = (N,) + _FIELD_SHAPES[name]
            out[name] = actions[:, lo:hi].reshape(*shape)
        return out

    # ── weights ────────────────────────────────────────────

    def _maybe_reload_weights(self) -> None:
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


def shared_server_loop_entry(
    cfg_dict:           dict,
    q_net_kwargs:       dict,
    meta:               SharedBufferMeta,
    free_slots,
    request_queue,
    events,
    stop_event,
    initial_state_dicts: dict[int, dict] | None = None,
    weights_lock=None,
    weights_buf=None,
    weights_version=None,
    weight_specs=None,
    weight_dir=None,                   # Phase 4: disk-based weight refresh
) -> None:
    """Top-level shared-mem server entry point. Picklable for spawn."""
    import os
    import sys
    import logging as _logging
    import torch
    from .q_network import init_position_nets

    # Wire server logger to stdout so its messages reach `modal run` output
    # alongside the learner's. Mirrors the GUANZERO_STREAM_LOGS hook used by
    # the learner subprocess (see learner.py).
    _logging.getLogger("guanzero.inference_server").handlers.clear()
    if os.environ.get("GUANZERO_STREAM_LOGS") == "1":
        _h = _logging.StreamHandler(sys.stdout)
        _h.setFormatter(_logging.Formatter(
            "%(asctime)s [%(levelname)-5s] [server] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        _h.setLevel(_logging.INFO)
        _slog = _logging.getLogger("guanzero.inference_server")
        _slog.addHandler(_h)
        _slog.setLevel(_logging.INFO)

    torch.set_num_threads(1)

    cfg_device              = cfg_dict.get("inference_device", "cpu")
    max_requests            = cfg_dict.get("inference_batch_max_requests", 32)
    max_action_rows         = cfg_dict.get("inference_batch_max_action_rows", 4096)
    timeout_ms              = cfg_dict.get("inference_batch_timeout_ms", 5.0)
    weight_refresh_s        = cfg_dict.get("inference_weight_refresh_s", 5.0)
    use_bf16                = cfg_dict.get("use_bf16_learner", False) and cfg_device == "cuda"

    bufs = attach_shared_buffers(
        meta=meta,
        free_slots=free_slots,
        request_queue=request_queue,
        events=events,
        weights_version=weights_version,
    )

    q_nets = init_position_nets(**q_net_kwargs)
    if initial_state_dicts is not None:
        for p in range(4):
            q_nets[p].load_state_dict(initial_state_dicts[p])

    server = SharedInferenceServer(
        q_nets=q_nets,
        bufs=bufs,
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

    # ── Phase 4 disk-based weight refresh thread ──
    # Polls weight_dir/latest.txt on a background thread; reloads q-nets when
    # the on-disk version is newer than what the server holds. Keeps the
    # shared-mem path open for Phase 5 (which will short-circuit this).
    refresh_stop = threading.Event()

    def _disk_refresh_loop():
        from pathlib import Path
        from .learner import load_latest_weights
        wd = Path(weight_dir) if weight_dir else None
        last_version = -1
        while not stop_event.is_set() and not refresh_stop.is_set():
            try:
                if wd is not None:
                    ver, state_dicts = load_latest_weights(wd)
                    if ver is not None and ver > last_version:
                        for p in range(4):
                            sd = state_dicts[p]
                            q_nets[p].load_state_dict(sd)
                            q_nets[p].to(cfg_device).eval()
                        # Bump server's local_version so future responses tag samples.
                        server._local_version = ver
                        last_version = ver
                        logger.info("server reloaded weights from disk: version=%d", ver)
            except Exception as e:
                logger.warning("disk weight refresh failed: %s", e)
            # Sleep with stop check
            for _ in range(int(weight_refresh_s * 10)):
                if stop_event.is_set() or refresh_stop.is_set():
                    break
                time.sleep(0.1)

    if weight_dir is not None:
        refresher = threading.Thread(target=_disk_refresh_loop, daemon=True,
                                     name="server_disk_refresh")
        refresher.start()
    else:
        refresher = None

    try:
        server.server_loop()
    finally:
        refresh_stop.set()
        if refresher is not None:
            refresher.join(timeout=2.0)
        # Detach shared-mem in this child; parent unlinks at shutdown.
        release_shared_buffers(bufs, unlink=False)


__all__ = [
    "InferenceClient",
    "InferenceServer",
    "SharedInferenceClient",
    "SharedInferenceServer",
    "SharedBuffers",
    "SharedBufferMeta",
    "RequestDesc",
    "allocate_shared_buffers",
    "attach_shared_buffers",
    "release_shared_buffers",
    "InferenceTimeoutError",
    "InferenceProtocolError",
    "server_loop_entry",
    "shared_server_loop_entry",
    "STATE_SIZE",
    "ACTION_SIZE",
    "NO_VERSION",
]
