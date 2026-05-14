"""Per-batch metric stratification helpers for the learner.

Provides the bucket tensor builders and grid/marginal emitters that annotate
each gradient step's loss by phase, trick-role, opponent, epsilon, etc.
Consumed by SharedHeadLearner.update() and exposed here so analysis scripts
can reconstruct the same bucketing offline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.sample_tags import ACTION_CLASS_LOOKUP, OPP_GRID_TOP


PHASE_KEY_PREFIXES = (
    "phase_role_", "phase_pair_", "source_phase_", "opp_phase_",
    "action_phase_", "epsilon_", "is_pass_", "is_bomb_",
    "k_bucket_", "q_gap_", "team_", "reward_",
)


def _emit_grid(
    metrics: dict[str, float | None],
    key_prefix: str,
    axis_a: torch.Tensor,
    axis_b: torch.Tensor,
    n_a: int,
    n_b: int,
    preds: torch.Tensor,
    targets: torch.Tensor,
    min_n: int = 32,
) -> None:
    total = max(axis_a.shape[0], 1)
    for a in range(n_a):
        for b in range(n_b):
            mask = (axis_a == a) & (axis_b == b)
            n = int(mask.sum().item())
            cell = a * n_b + b
            metrics[f"{key_prefix}_{cell}_n"] = n
            metrics[f"{key_prefix}_{cell}_frac"] = n / total
            if n < min_n:
                metrics[f"{key_prefix}_{cell}_loss"] = None
            else:
                metrics[f"{key_prefix}_{cell}_loss"] = float(
                    F.mse_loss(preds[mask], targets[mask]).item()
                )


def _emit_marginal(
    metrics: dict[str, float | None],
    key_prefix: str,
    axis: torch.Tensor,
    n_levels: int,
    preds: torch.Tensor,
    targets: torch.Tensor,
    min_n: int = 32,
) -> None:
    total = max(axis.shape[0], 1)
    for a in range(n_levels):
        mask = axis == a
        n = int(mask.sum().item())
        metrics[f"{key_prefix}_{a}_n"] = n
        metrics[f"{key_prefix}_{a}_frac"] = n / total
        if n < min_n:
            metrics[f"{key_prefix}_{a}_loss"] = None
        else:
            metrics[f"{key_prefix}_{a}_loss"] = float(
                F.mse_loss(preds[mask], targets[mask]).item()
            )


def _k_bucket_tensor(k: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(k, dtype=torch.long)
    out[(k >= 2) & (k <= 5)] = 1
    out[(k >= 6) & (k <= 20)] = 2
    out[k > 20] = 3
    return out


def _q_gap_bucket_tensor(q: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(q, dtype=torch.long)
    nan_mask = torch.isnan(q)
    out[nan_mask] = 0
    out[~nan_mask & (q < 0.05)] = 1
    out[~nan_mask & (q >= 0.05) & (q < 0.20)] = 2
    out[~nan_mask & (q >= 0.20)] = 3
    return out


def _reward_bucket_tensor(r: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(r, dtype=torch.long)
    out[r <= -2] = 0
    out[(r > -2) & (r < 0)] = 1
    out[(r >= 0) & (r <= 1)] = 2
    out[r > 1] = 3
    return out


def _opp_grid_index_tensor(opp_id: torch.Tensor) -> torch.Tensor:
    """Map raw opponent_id to index-in-OPP_GRID_TOP; non-listed → -1 (excluded)."""
    out = torch.full_like(opp_id, -1, dtype=torch.long)
    for i, v in enumerate(OPP_GRID_TOP):
        out[opp_id == v] = i
    return out


def _emit_phase_aggregations(
    metrics: dict,
    tags: dict[str, torch.Tensor],
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    phase_self = tags["phase_self"].long()
    trick_role = tags["trick_role"].long()
    phase_partner = tags["phase_partner"].long()
    episode_mode = tags["episode_mode"].long()
    opponent_id = tags["opponent_id"].long()
    action_type = tags["action_type"].long()
    is_pass = tags["is_pass"].long()
    is_bomb = tags["is_bomb"].long()
    chosen_by_epsilon = tags["chosen_by_epsilon"].long()
    latest_team = tags["latest_team"].long()
    num_legal_actions = tags["num_legal_actions"].long()
    q_gap = tags["q_gap"].float()
    terminal_reward = tags["terminal_reward"].float()

    _emit_grid(metrics, "phase_role", phase_self, trick_role, 3, 3, preds, targets)
    _emit_grid(metrics, "phase_pair", phase_self, phase_partner, 3, 4, preds, targets)
    _emit_grid(metrics, "source_phase", episode_mode, phase_self, 3, 3, preds, targets)

    opp_idx = _opp_grid_index_tensor(opponent_id)
    keep = opp_idx >= 0
    if keep.any():
        _emit_grid(
            metrics, "opp_phase",
            opp_idx[keep], phase_self[keep],
            len(OPP_GRID_TOP), 3,
            preds[keep], targets[keep],
        )
    else:
        for a in range(len(OPP_GRID_TOP)):
            for b in range(3):
                cell = a * 3 + b
                metrics[f"opp_phase_{cell}_n"] = 0
                metrics[f"opp_phase_{cell}_frac"] = 0.0
                metrics[f"opp_phase_{cell}_loss"] = None

    lookup = torch.as_tensor(ACTION_CLASS_LOOKUP, dtype=torch.long, device=action_type.device)
    action_class = lookup[action_type.clamp(min=0, max=lookup.numel() - 1)]
    _emit_grid(metrics, "action_phase", action_class, phase_self, 6, 3, preds, targets)

    _emit_marginal(metrics, "epsilon", chosen_by_epsilon, 2, preds, targets)
    _emit_marginal(metrics, "is_pass", is_pass, 2, preds, targets)
    _emit_marginal(metrics, "is_bomb", is_bomb, 2, preds, targets)
    _emit_marginal(metrics, "k_bucket", _k_bucket_tensor(num_legal_actions), 4, preds, targets)
    _emit_marginal(metrics, "q_gap", _q_gap_bucket_tensor(q_gap), 4, preds, targets)
    _emit_marginal(metrics, "team", latest_team, 2, preds, targets)
    _emit_marginal(metrics, "reward", _reward_bucket_tensor(terminal_reward), 4, preds, targets)


__all__ = [
    "PHASE_KEY_PREFIXES",
    "_emit_grid",
    "_emit_marginal",
    "_k_bucket_tensor",
    "_q_gap_bucket_tensor",
    "_reward_bucket_tensor",
    "_opp_grid_index_tensor",
    "_emit_phase_aggregations",
]
