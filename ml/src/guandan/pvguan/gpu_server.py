"""Centralized GPU inference server for parallel rollout.

Workers send (state_actor, actions) and (state_critic) batches via mp.Queue
to a daemon thread in the main process, which holds the model on GPU and
runs batched forward passes. This replaces per-worker CPU inference.

Architecture:
    request_q (shared)        ── workers put requests
    reply_qs[worker_id]       ── workers get replies (one queue per worker)
    daemon thread in main     ── drains request_q, batches, scores on GPU,
                                 dispatches replies via reply_qs

The server thread polls request_q with a short timeout, accumulates pending
requests, and runs one batched forward per loop iteration. Latency is bounded
by max_wait_ms (default 1ms) and batch size by max_batch (default 256).
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import threading
import time
from typing import Any

import numpy as np
import torch

from .actor_critic import ActorCriticNet


_POISON = "__STOP__"


class GPUInferenceServer:
    """Holds model on GPU, services scoring requests from rollout workers.

    Use start() once at training startup, update_weights(net) after each PPO
    iter, and stop() at shutdown. The shared request_q and per-worker reply_qs
    are passed into rollout workers via collect_rollout_parallel.
    """

    def __init__(
        self,
        net: ActorCriticNet,
        device: torch.device,
        n_workers: int,
        ctx: Any,
        max_batch: int = 256,
        max_wait_ms: float = 1.0,
    ) -> None:
        self.device      = device
        self.n_workers   = n_workers
        self.max_batch   = max_batch
        self.max_wait_s  = max_wait_ms / 1000.0
        self._net        = net
        self._lock       = threading.Lock()  # guards _net.state_dict swap

        # Queues created from the spawn context so workers can pickle handles
        self.request_q: mp.Queue = ctx.Queue()
        self.reply_qs: list[mp.Queue] = [ctx.Queue() for _ in range(n_workers)]

        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        # Telemetry (incremented only by server thread, read after stop)
        self.actor_calls   = 0
        self.critic_calls  = 0
        self.actor_items   = 0
        self.critic_items  = 0
        self.batch_actor_sum = 0
        self.batch_critic_sum = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPUInferenceServer already started")
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-infer-server")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        # Send poison sentinels so a blocking get() can return promptly
        self._stop_evt.set()
        try:
            self.request_q.put(_POISON, timeout=1.0)
        except Exception:
            pass
        self._thread.join(timeout=5.0)
        self._thread = None

    def update_weights(self, net: ActorCriticNet) -> None:
        """Swap in fresh weights after a PPO step. Server thread sees the
        change at its next batch (no in-flight request gets stale weights
        because we hold the lock during forward passes)."""
        with self._lock:
            # Same-device, same-shape state dict → load_state_dict is fastest
            self._net.load_state_dict(net.state_dict())
            self._net.eval()

    # ─────────────────── server loop ───────────────────────────────────────

    def _run(self) -> None:
        actor_buf:  list[dict] = []
        critic_buf: list[dict] = []
        while not self._stop_evt.is_set():
            # Block on first message; once we have one, drain quickly
            try:
                first = self.request_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            if first == _POISON:
                break
            self._sort(first, actor_buf, critic_buf)

            # Drain whatever else is queued without waiting
            deadline = time.time() + self.max_wait_s
            while (len(actor_buf) + len(critic_buf)) < self.max_batch:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    msg = self.request_q.get(timeout=remaining)
                except _queue.Empty:
                    break
                if msg == _POISON:
                    self._stop_evt.set()
                    break
                self._sort(msg, actor_buf, critic_buf)

            # Score whatever we have
            if actor_buf:
                self._score_actor(actor_buf)
                actor_buf = []
            if critic_buf:
                self._score_critic(critic_buf)
                critic_buf = []

    @staticmethod
    def _sort(msg: dict, actor_buf: list, critic_buf: list) -> None:
        if msg["type"] == "actor":
            actor_buf.append(msg)
        else:
            critic_buf.append(msg)

    def _score_actor(self, batch: list[dict]) -> None:
        sizes = [m["sa"].shape[0] for m in batch]
        sa_np = np.concatenate([m["sa"] for m in batch], axis=0)
        ac_np = np.concatenate([m["ac"] for m in batch], axis=0)
        sa = torch.from_numpy(sa_np).to(self.device, non_blocking=True)
        ac = torch.from_numpy(ac_np).to(self.device, non_blocking=True)
        with self._lock, torch.no_grad():
            logits = self._net.score_actions(sa, ac).cpu().numpy()
        offset = 0
        for m, sz in zip(batch, sizes):
            self.reply_qs[m["wid"]].put({
                "rid":    m["rid"],
                "logits": logits[offset:offset + sz].copy(),
            })
            offset += sz
        self.actor_calls += 1
        self.actor_items += sum(sizes)
        self.batch_actor_sum += sum(sizes)

    def _score_critic(self, batch: list[dict]) -> None:
        sc_np = np.stack([m["sc"] for m in batch], axis=0)  # [B, CRITIC_DIM]
        sc = torch.from_numpy(sc_np).to(self.device, non_blocking=True)
        with self._lock, torch.no_grad():
            v = self._net.value(sc).cpu().numpy()  # [B]
        for m, vi in zip(batch, v):
            self.reply_qs[m["wid"]].put({
                "rid":   m["rid"],
                "value": float(vi),
            })
        self.critic_calls += 1
        self.critic_items += len(batch)
        self.batch_critic_sum += len(batch)

    # ─────────────────── stats ─────────────────────────────────────────────

    def stats(self) -> dict[str, float]:
        return {
            "actor_calls":   self.actor_calls,
            "actor_items":   self.actor_items,
            "actor_avg_batch":  self.actor_items / max(self.actor_calls, 1),
            "critic_calls":  self.critic_calls,
            "critic_items":  self.critic_items,
            "critic_avg_batch": self.critic_items / max(self.critic_calls, 1),
        }


class NetProxy:
    """Drop-in replacement for ActorCriticNet inside rollout workers.

    Forwards score_actions / policy_logits / value calls through mp.Queue to
    the GPUInferenceServer in the main process.
    """

    def __init__(
        self,
        request_q: mp.Queue,
        reply_q:   mp.Queue,
        worker_id: int,
    ) -> None:
        self.request_q = request_q
        self.reply_q   = reply_q
        self.worker_id = worker_id
        self._rid = 0

    def policy_logits(
        self,
        state_actor: torch.Tensor,
        actions:     torch.Tensor,
        legal_mask:  torch.Tensor,
    ) -> torch.Tensor:
        rid = self._rid
        self._rid += 1
        self.request_q.put({
            "type": "actor",
            "wid":  self.worker_id,
            "rid":  rid,
            "sa":   state_actor.cpu().numpy(),
            "ac":   actions.cpu().numpy(),
        })
        msg = self.reply_q.get()
        assert msg["rid"] == rid, f"out-of-order reply (rid={msg['rid']} expected={rid})"
        logits = torch.from_numpy(msg["logits"]).to(state_actor.device)
        return logits.masked_fill(~legal_mask, -1e9)

    def value(self, state_critic: torch.Tensor) -> torch.Tensor:
        rid = self._rid
        self._rid += 1
        # state_critic is [1, CRITIC_DIM]; we send the [CRITIC_DIM] slice
        self.request_q.put({
            "type": "critic",
            "wid":  self.worker_id,
            "rid":  rid,
            "sc":   state_critic.cpu().numpy().squeeze(0),
        })
        msg = self.reply_q.get()
        assert msg["rid"] == rid
        return torch.tensor([msg["value"]], device=state_critic.device)

    def eval(self) -> "NetProxy": return self
    def train(self, mode: bool = True) -> "NetProxy": return self
