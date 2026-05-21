"""Per-batch metric stratification helpers for the learner.

Emitted metric keys are schema-backed and human-readable, e.g.
``phase_role_opening__leading_loss``. See ``loss_bucket_schema.py`` for the
axis labels and key contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...data.sample_tags import ACTION_CLASS_LOOKUP, OPP_GRID_TOP
from .loss_bucket_schema import (
    LOSS_BUCKET_GRIDS,
    LOSS_BUCKET_MARGINALS,
    LOSS_BUCKET_SCHEMA,
    PHASE_KEY_PREFIXES,
    GridSpec,
    MarginalSpec,
)


def _emit_grid(
    metrics: dict[str, float | None],
    spec: GridSpec,
    axis_a: torch.Tensor,
    axis_b: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    min_n: int = 32,
) -> None:
    total = max(axis_a.shape[0], 1)
    for a in range(len(spec.axis_a.labels)):
        for b in range(len(spec.axis_b.labels)):
            mask = (axis_a == a) & (axis_b == b)
            n = int(mask.sum().item())
            metrics[spec.key(a, b, "n")] = n
            metrics[spec.key(a, b, "frac")] = n / total
            if n < min_n:
                metrics[spec.key(a, b, "loss")] = None
            else:
                metrics[spec.key(a, b, "loss")] = float(
                    F.mse_loss(preds[mask], targets[mask]).item()
                )


def _emit_marginal(
    metrics: dict[str, float | None],
    spec: MarginalSpec,
    axis: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    min_n: int = 32,
) -> None:
    total = max(axis.shape[0], 1)
    for a in range(len(spec.axis.labels)):
        mask = axis == a
        n = int(mask.sum().item())
        metrics[spec.key(a, "n")] = n
        metrics[spec.key(a, "frac")] = n / total
        if n < min_n:
            metrics[spec.key(a, "loss")] = None
        else:
            metrics[spec.key(a, "loss")] = float(
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

    grids = {spec.name: spec for spec in LOSS_BUCKET_GRIDS}
    marginals = {spec.name: spec for spec in LOSS_BUCKET_MARGINALS}

    _emit_grid(metrics, grids["phase_role"], phase_self, trick_role, preds, targets)
    _emit_grid(metrics, grids["phase_pair"], phase_self, phase_partner, preds, targets)
    _emit_grid(metrics, grids["source_phase"], episode_mode, phase_self, preds, targets)

    opp_idx = _opp_grid_index_tensor(opponent_id)
    keep = opp_idx >= 0
    if keep.any():
        _emit_grid(
            metrics,
            grids["opp_phase"],
            opp_idx[keep], phase_self[keep],
            preds[keep], targets[keep],
        )
    else:
        _emit_grid(
            metrics,
            grids["opp_phase"],
            torch.empty(0, dtype=torch.long, device=opponent_id.device),
            torch.empty(0, dtype=torch.long, device=opponent_id.device),
            preds[:0], targets[:0],
        )

    lookup = torch.as_tensor(
        ACTION_CLASS_LOOKUP,
        dtype=torch.long,
        device=action_type.device,
    )
    action_class = lookup[action_type.clamp(min=0, max=lookup.numel() - 1)]
    _emit_grid(metrics, grids["action_phase"], action_class, phase_self, preds, targets)

    _emit_marginal(metrics, marginals["epsilon"], chosen_by_epsilon, preds, targets)
    _emit_marginal(metrics, marginals["is_pass"], is_pass, preds, targets)
    _emit_marginal(metrics, marginals["is_bomb"], is_bomb, preds, targets)
    _emit_marginal(
        metrics,
        marginals["k_bucket"],
        _k_bucket_tensor(num_legal_actions),
        preds,
        targets,
    )
    _emit_marginal(
        metrics,
        marginals["q_gap"],
        _q_gap_bucket_tensor(q_gap),
        preds,
        targets,
    )
    _emit_marginal(metrics, marginals["team"], latest_team, preds, targets)
    _emit_marginal(
        metrics,
        marginals["reward"],
        _reward_bucket_tensor(terminal_reward),
        preds,
        targets,
    )


__all__ = [
    "PHASE_KEY_PREFIXES",
    "LOSS_BUCKET_SCHEMA",
    "_emit_grid",
    "_emit_marginal",
    "_k_bucket_tensor",
    "_q_gap_bucket_tensor",
    "_reward_bucket_tensor",
    "_opp_grid_index_tensor",
    "_emit_phase_aggregations",
]
