"""Per-seat DMC learner (paper §4.2 faithful reproduction).

Four independent Q-networks — one per position — with separate Adam
optimizers and optional per-stream CUDA parallelism.  Owns no replay
buffer; the buffer is managed externally (by ``_SeatAdapter`` inside
``learner_loop``).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Mapping

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F

from ...constants import NUM_PLAYERS
from ...data.buffer import ReplayBuffer
from ...model.q_network import GuanZeroQNet
from ...utils.profiler import PhaseProfiler


class SeatLearner:
    """Per-seat DMC learner — one Q-net + Adam per position.

    Gradient math is independent per position (paper §4.2).  Streams allow
    all 4 forward+backward passes to overlap on CUDA when available.
    """

    def __init__(
        self,
        q_nets: Mapping[int, GuanZeroQNet],
        lr: float = 1e-4,
        device: torch.device | str = "cpu",
        use_bf16: bool = False,
        max_grad_norm: float = 10.0,
    ) -> None:
        self.device = torch.device(device)
        self.q_nets = {p: q_nets[p].to(self.device) for p in range(NUM_PLAYERS)}
        self.optims = {
            p: torch.optim.Adam(self.q_nets[p].parameters(), lr=lr, foreach=True)
            for p in range(NUM_PLAYERS)
        }
        if use_bf16 and self.device.type != "cuda":
            logger.warning("use_bf16=True requested but device=%s; BF16 disabled", self.device)
            self.use_bf16 = False
        else:
            self.use_bf16 = use_bf16
        self.max_grad_norm = max_grad_norm
        # One stream per position so CUDA can schedule all 4 forward+backward
        # passes concurrently. Not used on MPS (no multi-stream support).
        self.streams: dict[int, torch.cuda.Stream] | None = (
            {p: torch.cuda.Stream(device=self.device) for p in range(NUM_PLAYERS)}
            if self.device.type == "cuda" else None
        )
        self.prof = PhaseProfiler(
            enabled=os.environ.get("DART_LEARNER_PROFILE") == "1",
            device=self.device if self.device.type == "cuda" else None,
        )

    def state_dict(self) -> dict:
        """Return optimizer/runtime state needed for exact resume."""
        return {
            "optimizers": {
                p: optim.state_dict()
                for p, optim in self.optims.items()
            }
        }

    def load_state_dict(self, state: dict | None) -> None:
        """Restore optimizer/runtime state when present in a checkpoint."""
        if not state:
            return
        optimizers = state.get("optimizers", {})
        for p, optim_state in optimizers.items():
            self.optims[int(p)].load_state_dict(optim_state)

    def update(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
    ) -> dict[int, float]:
        """One gradient step per position, all 4 in parallel on CUDA.

        Collates all positions first so H2D copies can overlap, then launches
        forward+backward on separate streams.  Loss tensors are read after a
        single synchronize() to avoid per-position CPU stalls.
        """
        with self.prof.time("sample+h2d", sync=True):
            batches: dict[int, tuple] = {}
            for p in range(NUM_PLAYERS):
                res = buffer.sample_batch_for_player(p, batch_size, device=self.device)
                if res is not None:
                    batches[p] = res

        if not batches:
            return {}

        q_preds: dict[int, torch.Tensor] = {}
        targets_d: dict[int, torch.Tensor] = {}
        with self.prof.time("forward", sync=True):
            for p, (batch, targets) in batches.items():
                ctx = (torch.cuda.stream(self.streams[p])
                       if self.streams else contextlib.nullcontext())
                with ctx:
                    self.q_nets[p].train()
                    if self.use_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            q_preds[p] = self.q_nets[p](batch)
                    else:
                        q_preds[p] = self.q_nets[p](batch)
                targets_d[p] = targets

        loss_tensors: dict[int, torch.Tensor] = {}
        with self.prof.time("backward+step", sync=True):
            for p, q_pred in q_preds.items():
                ctx = (torch.cuda.stream(self.streams[p])
                       if self.streams else contextlib.nullcontext())
                with ctx:
                    if self.use_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = F.mse_loss(q_pred, targets_d[p])
                    else:
                        loss = F.mse_loss(q_pred, targets_d[p])
                    loss_tensors[p] = loss
                    if not torch.isfinite(targets_d[p]).all():
                        bad = int((~torch.isfinite(targets_d[p])).sum().item())
                        raise RuntimeError(
                            f"SeatLearner received {bad} non-finite targets for player {p}"
                        )
                    if not torch.isfinite(q_pred).all():
                        bad = int((~torch.isfinite(q_pred)).sum().item())
                        raise RuntimeError(
                            f"SeatLearner produced {bad} non-finite predictions for player {p}"
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"SeatLearner produced non-finite loss for player {p}: "
                            f"{float(loss.item())}"
                        )
                    self.optims[p].zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.q_nets[p].parameters(), self.max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        raise RuntimeError(
                            f"SeatLearner produced non-finite gradient norm for player {p}: "
                            f"{float(grad_norm.item())}"
                        )
                    self.optims[p].step()

        with self.prof.time("final_sync"):
            if self.streams:
                torch.cuda.synchronize(self.device)
        with self.prof.time("loss_item"):
            return {p: float(lt.item()) for p, lt in loss_tensors.items()}


__all__ = ["SeatLearner"]
