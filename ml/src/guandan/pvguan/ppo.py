"""PPO clipped objective with KL-to-warm-start regulariser.

Loss:
  L = -L_π_clip + c_V * L_V - c_H * H(π) + c_KL(t) * KL(π_warm || π_θ)

KL schedule:
  - Stays at kl_init during critic warm-up (actor frozen).
  - Decays linearly to 0 over kl_decay_actor_iters after actor unfreezes.

All probability computations use masked softmax over legal candidates only.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic import ActorCriticNet
from .buffer import PPOBatch


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over the legal subset. Illegal logits are already -1e9 in logits."""
    return F.log_softmax(logits, dim=-1)


def _masked_entropy(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean entropy over legal actions per decision. [N] → scalar."""
    probs = log_probs.exp()
    # zero out illegal slots so they don't contribute
    probs = probs * mask.float()
    ent = -(probs * log_probs.clamp(min=-1e9)).sum(dim=-1)  # [N]
    return ent.mean()


def _kl_warm_to_theta(
    log_warm: torch.Tensor,
    log_theta: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """KL(π_warm || π_θ) over legal actions. [N, K] → scalar.

    KL(p||q) = Σ p * (log p - log q)
    """
    p = log_warm.exp() * mask.float()
    kl = (p * (log_warm - log_theta)).sum(dim=-1)  # [N]
    return kl.clamp(min=0).mean()


class PPOConfig:
    def __init__(
        self,
        clip_eps:              float = 0.2,
        c_value:               float = 0.5,
        c_entropy:             float = 0.01,
        kl_init:               float = 0.05,
        kl_decay_actor_iters:  int   = 100,
        critic_warmup_iters:   int   = 30,
        lr:                    float = 3e-4,
        weight_decay:          float = 0.0,
        n_epochs:              int   = 4,
        mini_batch_size:       int   = 256,
        gamma:                 float = 1.0,
        lam:                   float = 0.95,
        temperature:           float = 1.0,
    ) -> None:
        self.clip_eps             = clip_eps
        self.c_value              = c_value
        self.c_entropy            = c_entropy
        self.kl_init              = kl_init
        self.kl_decay_actor_iters = kl_decay_actor_iters
        self.critic_warmup_iters  = critic_warmup_iters
        self.lr                   = lr
        self.weight_decay         = weight_decay
        self.n_epochs             = n_epochs
        self.mini_batch_size      = mini_batch_size
        self.gamma                = gamma
        self.lam                  = lam
        self.temperature          = temperature


class PPOTrainer:
    """Manages PPO updates, KL schedule, and critic warm-up phase."""

    def __init__(
        self,
        net: ActorCriticNet,
        cfg: PPOConfig,
        device: torch.device,
    ) -> None:
        self.net = net
        self.cfg = cfg
        self.device = device
        self._iter = 0
        self._actor_unfrozen_at: int | None = None

        # Frozen warm-start copy — used for KL regularisation
        self._warm_net = copy.deepcopy(net).to(device)
        for p in self._warm_net.parameters():
            p.requires_grad_(False)
        self._warm_net.eval()

        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    @property
    def actor_frozen(self) -> bool:
        return self._iter < self.cfg.critic_warmup_iters

    def _c_kl(self) -> float:
        if self.actor_frozen:
            return self.cfg.kl_init
        if self._actor_unfrozen_at is None:
            self._actor_unfrozen_at = self._iter
        iters_since_unfreeze = self._iter - self._actor_unfrozen_at
        if iters_since_unfreeze >= self.cfg.kl_decay_actor_iters:
            return 0.0
        frac = iters_since_unfreeze / self.cfg.kl_decay_actor_iters
        return self.cfg.kl_init * (1.0 - frac)

    def _set_actor_grad(self, requires_grad: bool) -> None:
        for p in self.net.actor_head.parameters():
            p.requires_grad_(requires_grad)

    def update(self, batch: PPOBatch) -> dict[str, float]:
        """Run n_epochs over the batch. Returns mean loss metrics."""
        N = batch.sampled_idx.shape[0]
        c_kl = self._c_kl()
        frozen = self.actor_frozen
        self._set_actor_grad(not frozen)

        metric_sums: dict[str, float] = {
            k: 0.0 for k in [
                "loss_total", "loss_policy", "loss_value",
                "loss_entropy", "loss_kl_warmstart",
                "clip_fraction", "approx_kl", "explained_variance_current",
            ]
        }
        n_updates = 0

        for _ in range(self.cfg.n_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.cfg.mini_batch_size):
                idx = perm[start: start + self.cfg.mini_batch_size]
                if idx.numel() == 0:
                    continue

                sa   = batch.state_actor[idx]    # [B, max_K, ACTOR_DIM]
                acts = batch.actions[idx]         # [B, max_K, ACTION_DIM]
                mask = batch.legal_mask[idx]      # [B, max_K]
                sidx = batch.sampled_idx[idx]     # [B]
                old_lp = batch.old_log_prob[idx]  # [B]
                sc   = batch.state_critic[idx]    # [B, CRITIC_DIM]
                ret  = batch.returns[idx]         # [B]
                adv  = batch.advantages[idx]      # [B]

                B, max_K, _ = sa.shape

                # Flatten candidate dim for actor head
                sa_flat   = sa.view(B * max_K, -1)
                acts_flat = acts.view(B * max_K, -1)
                logits_flat = self.net.score_actions(sa_flat, acts_flat) / self.cfg.temperature
                logits = logits_flat.view(B, max_K)
                logits = logits.masked_fill(~mask, -1e9)

                log_probs = F.log_softmax(logits, dim=-1)   # [B, max_K]

                # Log-prob of the sampled action
                new_log_prob = log_probs.gather(1, sidx.unsqueeze(1)).squeeze(1)  # [B]

                # Policy ratio and clip
                ratio = (new_log_prob - old_lp).exp()
                clip = self.cfg.clip_eps
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - clip, 1 + clip) * adv
                loss_pi = -torch.min(surr1, surr2).mean()

                # Value loss
                v_pred = self.net.value(sc)
                loss_v = F.mse_loss(v_pred, ret)

                # Entropy over legal actions
                ent = _masked_entropy(log_probs, mask)
                loss_h = -ent

                # KL(π_warm || π_θ)
                loss_kl = torch.tensor(0.0, device=self.device)
                if c_kl > 0:
                    with torch.no_grad():
                        warm_logits_flat = self._warm_net.score_actions(sa_flat, acts_flat) / self.cfg.temperature
                        warm_logits = warm_logits_flat.view(B, max_K).masked_fill(~mask, -1e9)
                        log_warm = F.log_softmax(warm_logits, dim=-1)
                    loss_kl = _kl_warm_to_theta(log_warm, log_probs, mask)

                total = loss_pi + self.cfg.c_value * loss_v + self.cfg.c_entropy * loss_h
                if c_kl > 0:
                    total = total + c_kl * loss_kl

                self.optimizer.zero_grad()
                total.backward()
                if frozen:
                    # Ensure actor gradients are truly zero during warmup
                    for p in self.net.actor_head.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    cf = (ratio - 1.0).abs().gt(clip).float().mean().item()
                    akl = ((ratio.log())).mean().item()
                    ev = 1.0 - (ret - v_pred.detach()).var() / (ret.var() + 1e-8)
                    ev = ev.item()

                metric_sums["loss_total"]          += total.item()
                metric_sums["loss_policy"]         += loss_pi.item()
                metric_sums["loss_value"]          += loss_v.item()
                metric_sums["loss_entropy"]        += ent.item()
                metric_sums["loss_kl_warmstart"]   += loss_kl.item()
                metric_sums["clip_fraction"]       += cf
                metric_sums["approx_kl"]           += akl
                metric_sums["explained_variance_current"] += ev
                n_updates += 1

        self._iter += 1
        if n_updates == 0:
            return metric_sums

        return {k: v / n_updates for k, v in metric_sums.items()} | {
            "c_kl_current": c_kl,
            "actor_frozen": float(frozen),
        }

    def actor_grad_norm(self) -> float:
        total = 0.0
        for p in self.net.actor_head.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def critic_grad_norm(self) -> float:
        total = 0.0
        for p in self.net.critic_head.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5
