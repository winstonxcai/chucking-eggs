"""Per-iter telemetry for pvguan PPO runs.

Metrics are written to metrics.jsonl (one JSON line per iter).
Console + file logging via the standard logging module.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .actor_critic import ActorCriticNet
from .buffer import PPOBatch
from .encoders import (
    ACTOR_DIM, ACTION_DIM,
    CRITIC_PRIV, OPP_STYLE_FEATURES,
    G1_OWN_HAND, G1_PARTNER_HAND, G1_TEAM_PLAYED, G1_OPP_PLAYED, G1_UNKNOWN,
)


class RunLogger:
    """Structured logging for a PPO training run.

    Files written to run_dir:
      config.json          — hyperparameters (written once at init)
      train.log            — timestamped human-readable log
      metrics.jsonl        — one JSON line per iter (all metrics)
      validation_eval.jsonl
      snapshot_eval.jsonl
    """

    def __init__(self, run_dir: Path, config: dict) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self._metrics_path = run_dir / "metrics.jsonl"
        self._val_path     = run_dir / "validation_eval.jsonl"
        self._snap_path    = run_dir / "snapshot_eval.jsonl"
        self._start_time   = time.time()

        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # ── Python logger ───────────────────────────────────────────────────
        log_name = f"train_pvguan.{run_dir.name}"
        self.log = logging.getLogger(log_name)
        self.log.setLevel(logging.DEBUG)
        self.log.handlers.clear()
        self.log.propagate = False

        fmt_file = logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fmt_con = logging.Formatter("%(message)s")

        fh = logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_file)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt_con)

        self.log.addHandler(fh)
        self.log.addHandler(ch)

    def log_iter(self, metrics: dict[str, Any]) -> None:
        # Structured record
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Human-readable summary line (file + console via logging)
        it     = metrics.get("iter", 0)
        dec    = metrics.get("cumulative_decisions", 0)
        total  = metrics.get("total_decisions", 0)
        r      = metrics.get("terminal_reward_mean", 0.0)
        H      = metrics.get("entropy_legal", 0.0)
        ev     = metrics.get("explained_variance_old", 0.0)
        cf     = metrics.get("clip_fraction", 0.0)
        kl     = metrics.get("approx_kl", 0.0)
        t_iter = metrics.get("iter_wall_s", 0.0)
        frozen = " [critic-warmup]" if metrics.get("actor_frozen") else ""
        pct    = f"{100*dec/total:.1f}%" if total else ""
        self.log.info(
            f"iter {it:4d} | {dec/1e3:.0f}k/{total/1e3:.0f}k dec ({pct}) | "
            f"r̄={r:+.3f} | H={H:.2f} | EV={ev:.2f} | "
            f"clip={cf:.2f} | KL={kl:.4f} | t={t_iter:.1f}s{frozen}"
        )

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
    chunk: int = 4096,
) -> dict[str, float]:
    """Entropy, top-1 prob, logit std, KL to warm-start (if available)."""
    net.eval()
    with torch.no_grad():
        N, max_K, _ = batch.state_actor.shape
        sa_flat = batch.state_actor.view(N * max_K, -1)
        ac_flat = batch.actions.view(N * max_K, -1)
        # Chunked forward pass to avoid OOM on large batches
        logits_parts = [
            net.score_actions(sa_flat[i:i+chunk], ac_flat[i:i+chunk])
            for i in range(0, N * max_K, chunk)
        ]
        logits_flat = torch.cat(logits_parts, dim=0) / temperature
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


def opp_style_weight_norms(
    net: ActorCriticNet,
    init_opp_style_weights: torch.Tensor,
) -> dict[str, float]:
    """L2 norm and delta-norm of actor first-layer opp-style-feature columns."""
    w = net.actor_head.net[0].weight[:, OPP_STYLE_FEATURES]
    norm  = w.norm(2).item()
    delta = (w - init_opp_style_weights).norm(2).item()
    return {
        "opp_style_weight_norm":       norm,
        "opp_style_weight_delta_norm": delta,
    }


# Ordered dict of actor weight-column groups for per-group delta-norm tracking.
# Keys are short names used as metric prefixes (e.g. "g1_own_hand_wdelta").
# Values are column slices into the actor first linear layer weight matrix
# (shape [hidden, ACTOR_DIM + ACTION_DIM]).
ACTOR_GROUPS: dict[str, slice] = {
    "g1_own_hand":     G1_OWN_HAND,                       # 60
    "g1_partner_hand": G1_PARTNER_HAND,                   # 60
    "g1_team_played":  G1_TEAM_PLAYED,                    # 120
    "g1_opp_played":   G1_OPP_PLAYED,                     # 120
    "g1_unknown":      G1_UNKNOWN,                        # 60
    "g2_seat_status":  slice(420, 460),                   # 40
    "g3_acting_ctx":   slice(460, 478),                   # 18
    "g4_active_trick": slice(478, 576),                   # 98
    "g5_last_action":  slice(576, 672),                   # 96
    "g6_move_history": slice(672, 755),                   # 83
    "g7_opp_style":    OPP_STYLE_FEATURES,                # 12
    "g8_behavior":     slice(ACTOR_DIM - 9, ACTOR_DIM),   # 9
    "action":          slice(ACTOR_DIM, ACTOR_DIM + ACTION_DIM),  # 198
}


def actor_group_weight_norms(
    net: ActorCriticNet,
    init_weights: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Per encoder-group normalized delta-norm for the actor first linear layer.

    For each group in ACTOR_GROUPS, computes delta_norm / sqrt(n_cols) so values
    are comparable across groups of different sizes.

    Returns {group_name}_wdelta for every group.
    """
    w = net.actor_head.net[0].weight  # [hidden, ACTOR_DIM + ACTION_DIM]
    result = {}
    for name, col_slice in ACTOR_GROUPS.items():
        w_group = w[:, col_slice]
        n_cols = col_slice.stop - col_slice.start
        delta = (w_group - init_weights[name]).norm(2).item() / (n_cols ** 0.5)
        result[f"{name}_wdelta"] = delta
    return result


def build_actor_group_init_weights(net: ActorCriticNet) -> dict[str, torch.Tensor]:
    """Snapshot actor first-layer column weights for each group. Call once at startup."""
    w = net.actor_head.net[0].weight
    return {
        name: w[:, col_slice].clone()
        for name, col_slice in ACTOR_GROUPS.items()
    }


def critic_priv_weight_norms(
    net: ActorCriticNet,
    init_priv_weights: torch.Tensor,
) -> dict[str, float]:
    """L2 norm and delta-norm of critic first-layer privileged-slot weights."""
    # Privileged columns start at index 767 in the critic input
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


def build_priv_probe_set(
    batch: PPOBatch,
    n_probe: int = 100,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n_probe critic states from a batch, return (with_priv, zero_priv).

    For PV-AC mode the privileged slots are already zero, so both tensors are
    identical (sensitivity will be ~0). For PV-PTIE the with_priv tensor has
    real opponent hands and zero_priv has slots[767:887] zeroed out.
    """
    N = batch.state_critic.shape[0]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randperm(N, generator=g)[: min(n_probe, N)]
    with_priv = batch.state_critic[idx].clone()
    zero_priv = with_priv.clone()
    zero_priv[:, CRITIC_PRIV] = 0.0
    return with_priv, zero_priv
