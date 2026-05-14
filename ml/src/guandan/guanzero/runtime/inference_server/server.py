"""Batched GPU inference server for distributed GuanZero actors."""

from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import nullcontext
from typing import Mapping, Optional

import numpy as np
import torch

from ...model.encoding.base_encoder import ENCODE_CHANNEL_SHAPES
from ...model.q_network import GuanZeroQNet
from ...utils.profiler import PhaseProfiler
from .batching import (
    RequestDesc,
    SharedBufferMeta,
    SharedBuffers,
    _ACTION_OFFSETS,
    _STATE_OFFSETS,
    attach_shared_buffers,
    release_shared_buffers,
)


logger = logging.getLogger("guanzero.inference_server")


class InferenceServer:
    """Batched GPU inference server.

    Reads from ``request_queue``, gathers payloads from shared memory, runs
    batched forward, writes responses to ``response_buf[actor_id]``, and
    signals ``events[actor_id]``.
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

        import os as _os
        self.profiler = PhaseProfiler(
            enabled=(_os.environ.get("GUANZERO_SERVER_PROFILE") == "1"),
            device=self.device,
        )
        self._loop_t0 = time.perf_counter()

    # ── main loop ──────────────────────────────────────────────────────────

    def server_loop(self) -> None:
        logger.info("InferenceServer started on device=%s", self.device)
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
            logger.info(
                "InferenceServer exiting; %d batches, %d requests, %d rows",
                int(self.metrics["n_batches"]), int(self.metrics["n_requests"]),
                int(self.metrics["n_action_rows"]),
            )
            if self.profiler.enabled:
                wall_s = time.perf_counter() - self._loop_t0
                report = self.profiler.report(
                    wall_s=wall_s,
                    n_events=int(self.metrics["n_batches"]),
                    event_label="batches",
                )
                for line in report.splitlines():
                    logger.info(line)

    def _drain_batch(self) -> list[RequestDesc]:
        prof = self.profiler
        try:
            with prof.time("drain_wait"):
                first = self.bufs.request_queue.get(timeout=0.5)
        except queue.Empty:
            return []
        with prof.time("drain_batch"):
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

    # ── inference ──────────────────────────────────────────────────────────

    def _process_batch(self, batch: list[RequestDesc]) -> None:
        prof = self.profiler
        t0 = time.perf_counter()
        chosen = self._infer_batch(batch)
        forward_ms = (time.perf_counter() - t0) * 1000.0

        with prof.time("scatter+events"):
            for desc, idx in zip(batch, chosen):
                self.bufs.response_buf[desc.actor_id, 0] = np.uint32(desc.request_id)
                self.bufs.response_buf[desc.actor_id, 1] = np.uint32(idx)
                self.bufs.response_buf[desc.actor_id, 2] = np.uint32(self._local_version)
                self.bufs.events[desc.actor_id].set()
        prof.add_count("scatter+events", len(batch))

        rows = sum(d.n_actions for d in batch)
        self.metrics["n_batches"]      += 1
        self.metrics["n_requests"]     += len(batch)
        self.metrics["n_action_rows"]  += rows
        self.metrics["forward_ms_sum"] += forward_ms
        prof.add_count("batches", 1)
        prof.add_count("requests", len(batch))
        prof.add_count("action_rows", rows)

    @torch.no_grad()
    def _infer_batch(self, descs: list[RequestDesc]) -> list[int]:
        """Group by seat, gather from shared mem, expand on GPU, per-request argmax."""
        prof = self.profiler

        with prof.time("group_by_seat"):
            by_seat: dict[int, list[tuple[int, RequestDesc]]] = {p: [] for p in range(4)}
            for original_idx, desc in enumerate(descs):
                by_seat[desc.seat].append((original_idx, desc))

        chosen: list[int] = [0] * len(descs)

        for seat, items in by_seat.items():
            if not items:
                continue

            slots = [d.slot for _, d in items]
            Ks    = [d.n_actions for _, d in items]

            with prof.time("collate_cpu"):
                state_np  = self.bufs.state_buf[slots]
                action_np = np.concatenate(
                    [self.bufs.action_buf[d.slot, :d.n_actions] for _, d in items],
                    axis=0,
                )

            with prof.time("h2d+unpack", sync=True):
                state   = torch.from_numpy(state_np).to(self.device, non_blocking=True).float()
                actions = torch.from_numpy(action_np).to(self.device, non_blocking=True).float()
                repeats = torch.tensor(Ks, device=self.device, dtype=torch.long)
                state_dict  = self._unpack_state_to_qnet_dict(state)
                action_dict = self._unpack_action_to_qnet_dict(actions)

            with prof.time("forward", sync=True):
                ctx = (
                    torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                    if self.use_bf16 else nullcontext()
                )
                with ctx:
                    q = self.q_nets[seat].forward_grouped(
                        state_dict, action_dict, repeats,
                    )

            with prof.time("argmax+d2h"):
                offset = 0
                for (original_idx, _), K in zip(items, Ks):
                    chosen[original_idx] = int(q[offset : offset + K].argmax().item())
                    offset += K
            prof.add_count("argmax+d2h", len(items))

        return chosen

    def _unpack_state_to_qnet_dict(self, state_rows: torch.Tensor) -> dict[str, torch.Tensor]:
        N = state_rows.shape[0]
        out: dict[str, torch.Tensor] = {}
        for name, (lo, hi) in _STATE_OFFSETS.items():
            out[name] = state_rows[:, lo:hi].reshape(N, *ENCODE_CHANNEL_SHAPES[name])
        return out

    def _unpack_action_to_qnet_dict(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        N = actions.shape[0]
        out: dict[str, torch.Tensor] = {}
        for name, (lo, hi) in _ACTION_OFFSETS.items():
            out[name] = actions[:, lo:hi].reshape(N, *ENCODE_CHANNEL_SHAPES[name])
        return out

    def _unpack_to_qnet_dict(
        self,
        state_rows: torch.Tensor,
        actions:    torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self._unpack_state_to_qnet_dict(state_rows)
        out.update(self._unpack_action_to_qnet_dict(actions))
        return out

    # ── weights ────────────────────────────────────────────────────────────

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


def run_server(
    cfg_dict:            dict,
    meta:                SharedBufferMeta,
    free_slots,
    request_queue,
    events,
    stop_event,
    initial_state_dicts: dict[int, dict] | None = None,
    initial_version:     int = 0,
    weights_lock=None,
    weights_buf=None,
    weights_version=None,
    weight_specs=None,
    weight_dir=None,
    server_log_path=None,
) -> None:
    """Top-level shared-mem server entry point. Picklable for spawn."""
    from pathlib import Path
    import torch
    from ...utils.logging_setup import setup_run_logging
    from ...model.q_network import init_seat_nets
    from ..weight_publish import load_latest_weights

    server_log_path = Path(server_log_path) if server_log_path else Path("inference_server.log")
    _slog, _ = setup_run_logging(
        server_log_path.parent,
        server_log_path.name,
        name="guanzero.inference_server",
    )
    _slog.info("entry reached")

    torch.set_num_threads(1)

    try:
        from ...config import TrainConfig
        cfg              = TrainConfig.from_flat_dict(cfg_dict)
        cfg_device       = cfg.inference.device
        max_requests     = cfg.inference.batch_max_requests
        max_action_rows  = cfg.inference.batch_max_action_rows
        timeout_ms       = cfg.inference.batch_timeout_ms
        weight_refresh_s = cfg.inference.weight_refresh_s
        use_bf16         = cfg.use_bf16_learner and cfg_device == "cuda"

        _slog.info("cfg.inference.device=%s", cfg_device)
        _slog.info("attaching shared buffers...")
        bufs = attach_shared_buffers(
            meta=meta, free_slots=free_slots,
            request_queue=request_queue, events=events,
            weights_version=weights_version,
        )

        _slog.info("building q_nets and moving to %s...", cfg_device)
        q_nets = init_seat_nets(cfg.qnet)
        if initial_state_dicts is not None:
            for p in range(4):
                q_nets[p].load_state_dict(initial_state_dicts[p])
        for p in range(4):
            q_nets[p] = q_nets[p].to(cfg_device).eval()
        _slog.info("q_nets ready on %s", cfg_device)
    except Exception:
        _slog.exception("SETUP FAILED")
        raise

    server = InferenceServer(
        q_nets=q_nets, bufs=bufs, stop_event=stop_event,
        device=cfg_device, max_requests=max_requests,
        max_action_rows=max_action_rows, timeout_ms=timeout_ms,
        use_bf16=use_bf16, weights_lock=weights_lock,
        weights_buf=weights_buf, weights_version=weights_version,
        weight_specs=weight_specs,
    )
    server._local_version = int(initial_version)

    refresh_stop = threading.Event()

    def _disk_refresh_loop():
        wd = Path(weight_dir) if weight_dir else None
        last_version = int(initial_version)
        while not stop_event.is_set() and not refresh_stop.is_set():
            try:
                if wd is not None:
                    snapshot = load_latest_weights(wd)
                    if snapshot is not None and snapshot.version > last_version:
                        for p in range(4):
                            q_nets[p].load_state_dict(snapshot.state_dicts[p])
                            q_nets[p].to(cfg_device).eval()
                        server._local_version = snapshot.version
                        last_version = snapshot.version
                        logger.info("server reloaded weights from disk: version=%d", snapshot.version)
            except Exception as e:
                logger.warning("disk weight refresh failed: %s", e)
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

    _slog.info("entering server_loop")
    try:
        server.server_loop()
    except Exception:
        _slog.exception("server_loop crashed")
        raise
    finally:
        _slog.info("server_loop exited; cleaning up")
        refresh_stop.set()
        if refresher is not None:
            refresher.join(timeout=2.0)
        release_shared_buffers(bufs, unlink=False)


__all__ = [
    "InferenceServer",
    "run_server",
]
