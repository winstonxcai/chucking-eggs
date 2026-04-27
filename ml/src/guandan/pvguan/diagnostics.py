"""Per-iter telemetry for pvguan PPO runs.

Metrics are written to metrics.jsonl (one JSON line per iter).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .actor_critic import ActorCriticNet
from .buffer import PPOBatch
from .encoders import CRITIC_PRIV


class RunLogger:
    """Writes config.json, train.log, and metrics.jsonl to a run directory."""

    def __init__(self, run_dir: Path, config: dict) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self._metrics_path = run_dir / "metrics.jsonl"
        self._log_path = run_dir / "train.log"
        self._val_path = run_dir / "validation_eval.jsonl"
        self._snap_path = run_dir / "snapshot_eval.jsonl"
        self._start_time = time.time()

        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def log_iter(self, metrics: dict[str, Any]) -> None:
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Console summary
        it = metrics.get("iter", "?")
        dec = metrics.get("cumulative_decisions", 0)
        r = metrics.get("terminal_reward_mean", 0.0)
        H = metrics.get("entropy_legal", 0.0)
        ev = metrics.get("explained_variance_current", 0.0)
        cf = metrics.get("clip_fraction", 0.0)
        kl = metrics.get("approx_kl", 0.0)
        t_iter = metrics.get("iter_wall_s", 0.0)
        line = (
            f"iter {it:4d} | dec {dec/1e3:.0f}k | "
            f"r̄={r:+.3f} | H={H:.2f} | EV={ev:.2f} | "
            f"clip={cf:.2f} | KL={kl:.4f} | t={t_iter:.1f}s"
        )
        print(line)
        with open(self._log_path, "a") as f:
            f.write(line + "\n")

    def log_val(self, row: dict[str, Any]) -> None:
        with open(self._val_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def log_snapshot(self, row: dict[str, Any]) -> None:
        with open(self._snap_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def elapsed(self) -> float:
        return time.time() - self._start_time


def compute_policy_diagnostics(
    net: ActorCriticNet,
    batch: PPOBatch,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Entropy, top-1 prob, logit std, KL to warm-start (if available)."""
    net.eval()
    with torch.no_grad():
        N, max_K, _ = batch.state_actor.shape
        sa_flat = batch.state_actor.view(N * max_K, -1)
        ac_flat = batch.actions.view(N * max_K, -1)
        logits_flat = net.score_actions(sa_flat, ac_flat) / temperature
        logits = logits_flat.view(N, max_K).masked_fill(~batch.legal_mask, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp() * batch.legal_mask.float()

        ent = -(probs * log_probs.clamp(min=-1e9)).sum(dim=-1).mean().item()
        top1 = probs.max(dim=-1).values.mean().item()
        legal_logits = logits.masked_fill(~batch.legal_mask, float("nan"))
        logit_std = torch.nanmean(
            torch.tensor([
                legal_logits[i, batch.legal_mask[i]].std().item()
                for i in range(N)
            ])
        ).item()

    net.train()
    return {
        "entropy_legal": ent,
        "top1_prob": top1,
        "logit_std": float(logit_std),
    }


def compute_value_diagnostics(batch: PPOBatch) -> dict[str, float]:
    """Value and advantage statistics from the frozen rollout batch."""
    v = batch.v_old.cpu().numpy()
    ret = batch.returns.cpu().numpy()
    adv_pre = (batch.returns - batch.v_old).cpu().numpy()

    ev_old = float(1.0 - np.var(ret - v) / (np.var(ret) + 1e-8))
    return {
        "value_mean": float(v.mean()),
        "value_std":  float(v.std()),
        "value_min":  float(v.min()),
        "value_max":  float(v.max()),
        "return_mean": float(ret.mean()),
        "return_std":  float(ret.std()),
        "explained_variance_old": ev_old,
        "advantage_mean_pre_norm": float(adv_pre.mean()),
        "advantage_std_pre_norm":  float(adv_pre.std()),
    }


def pv_ac_priv_slots_zero(batch: PPOBatch) -> bool:
    """For PV-AC runs: asserts critic privileged slots are exactly zero."""
    priv = batch.state_critic[:, CRITIC_PRIV]
    return bool(priv.abs().max().item() == 0.0)


def critic_priv_weight_norms(
    net: ActorCriticNet,
    init_priv_weights: torch.Tensor,
) -> dict[str, float]:
    """L2 norm and delta-norm of critic first-layer privileged-slot weights."""
    # Privileged columns start at index 755 in the critic input
    w = net.critic_head.net[0].weight[:, CRITIC_PRIV]
    norm  = w.norm(2).item()
    delta = (w - init_priv_weights).norm(2).item()
    return {
        "critic_priv_weight_norm":       norm,
        "critic_priv_weight_delta_norm": delta,
    }


def critic_priv_sensitivity(
    net: ActorCriticNet,
    probe_states_with_priv: torch.Tensor,   # [P, CRITIC_DIM]
    probe_states_zero_priv: torch.Tensor,   # [P, CRITIC_DIM]
) -> float:
    """Mean |V(real_priv) - V(zero_priv)| on a fixed probe set."""
    net.eval()
    with torch.no_grad():
        v_real = net.value(probe_states_with_priv)
        v_zero = net.value(probe_states_zero_priv)
        sens = (v_real - v_zero).abs().mean().item()
    net.train()
    return sens
