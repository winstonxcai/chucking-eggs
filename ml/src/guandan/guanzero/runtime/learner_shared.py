"""Shared-head DMC learner for role-aware training.

Supports both ``SharedHeadQNet`` (absolute-seat heads) and
``SharedTrickHeadQNet`` (trick-relative heads).  Uses a single shared trunk
with per-head output layers and one Adam optimizer over all parameters.

Owns no replay buffer; the buffer is managed externally (by
``_SharedAdapter`` inside ``learner_loop``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.buffer import RoleAwareReplayBuffer
from ..model.q_network import SharedHeadQNet, SharedTrickHeadQNet
from .loss_buckets import _emit_phase_aggregations


def _grad_norm_of(module: torch.nn.Module) -> torch.Tensor:
    total = torch.zeros((), device=next(module.parameters()).device)
    for p in module.parameters():
        if p.grad is not None:
            total = total + p.grad.detach().norm().pow(2)
    return total.sqrt()


class SharedHeadLearner:
    """Learner for a single shared-head Q-net with one shared optimizer.

    Supports both ``SharedHeadQNet`` (absolute-seat heads, ``seat_id`` field)
    and ``SharedTrickHeadQNet`` (trick-relative heads, ``trick_head_id``
    field).  The head field is auto-detected from the buffer at update time
    and drives both stratification and per-head metric naming.
    """

    def __init__(
        self,
        q_net: SharedHeadQNet | SharedTrickHeadQNet,
        lr: float = 1e-4,
        device: torch.device | str = "cpu",
        use_bf16: bool = False,
        max_grad_norm: float = 10.0,
    ) -> None:
        self.device = torch.device(device)
        self.q_net = q_net.to(self.device)
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr, foreach=True)
        self.use_bf16 = use_bf16 and self.device.type == "cuda"
        self.max_grad_norm = max_grad_norm

    def update(
        self,
        buffer: RoleAwareReplayBuffer,
        batch_size: int,
        replay_mix: dict | None = None,
        max_forced_k1_replay_frac: float = 1.0,
    ) -> dict | None:
        """One balanced gradient step, or None if any head bucket is cold."""
        sizes = buffer.size_by_seat()
        min_per_seat = batch_size // 4
        if min(sizes.values()) < min_per_seat:
            return None

        if max_forced_k1_replay_frac < 1.0:
            batch, targets, tags = buffer.sample_batch_balanced_k1_capped(
                batch_size, max_forced_k1_replay_frac, self.device, return_tags=True,
            )
        elif replay_mix:
            batch, targets, tags = buffer.sample_batch_stratified(
                batch_size, replay_mix, self.device, return_tags=True,
            )
        else:
            batch, targets, tags = buffer.sample_batch_balanced(
                batch_size, self.device, return_tags=True,
            )
        head_field = buffer.head_field
        head_ids = batch[head_field].long()
        self.q_net.train()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            preds = self.q_net(batch)
            loss = F.mse_loss(preds, targets)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_total = torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(),
            self.max_grad_norm,
        )
        self.opt.step()

        suffix = "seat" if head_field == "seat_id" else "trick_head"
        with torch.no_grad():
            metrics: dict = {
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm_total.item()),
                "grad_norm_trunk": float(_grad_norm_of(self.q_net.trunk).item()),
            }
            for k in range(4):
                mask = head_ids == k
                metrics[f"sample_count_{suffix}_{k}"] = float(mask.sum().item())
                if mask.any():
                    seat_preds = preds[mask]
                    seat_targets = targets[mask]
                    metrics[f"loss_{suffix}_{k}"] = float(F.mse_loss(seat_preds, seat_targets).item())
                    metrics[f"q_mean_{suffix}_{k}"] = float(seat_preds.mean().item())
                    metrics[f"q_std_{suffix}_{k}"] = float(seat_preds.std(unbiased=False).item())
                metrics[f"grad_norm_head_{k}"] = float(_grad_norm_of(self.q_net.heads[k]).item())
            _emit_phase_aggregations(metrics, tags, preds.detach(), targets.detach())
        return metrics


__all__ = ["SharedHeadLearner"]
