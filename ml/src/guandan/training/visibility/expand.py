"""Expand a standard-rules checkpoint into a Tier 1 (partner-visible) model.

The zero-init trick: new columns for the partner hand feature are initialized
to zero, so on episode 0 the expanded model produces bit-identical Q-values
to the base checkpoint. Training then gradually learns to use partner info.

Verified dimensions (prod_03_29_11_51.pt):
  mlp.0.weight:      [1024, 833] → [1024, 893]
  hand_pred.0.weight: [512, 673] → [512, 733]
  All other layers:   same shape, direct copy.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..encoding import STATE_DIM  # 417
from .encoding import PARTNER_HAND_DIM, PARTNER_INSERT_POS, STATE_DIM_TIER1  # 60, 60, 477


def _expand_first_layer(
    old_w: torch.Tensor,
    d_state_old: int,
    d_partner: int,
    insert_pos: int,
) -> torch.Tensor:
    """Insert zero columns for partner hand into a first-layer weight matrix.

    Works for any first layer whose input is [state | ...rest...]:
      - mlp.0.weight:      [H, d_state + d_action + d_lstm]
      - hand_pred.0.weight: [H, d_state + d_lstm]

    The partner columns are inserted at `insert_pos` within the state portion.
    Everything after the state portion (action, lstm) shifts right by d_partner.
    """
    out_features, old_in = old_w.shape
    new_in = old_in + d_partner
    new_w = torch.zeros(out_features, new_in, dtype=old_w.dtype, device=old_w.device)

    # State portion: [0, d_state_old) in old → split at insert_pos
    new_w[:, :insert_pos] = old_w[:, :insert_pos]
    # new[:, insert_pos : insert_pos + d_partner] stays zero (partner hand)
    new_w[:, insert_pos + d_partner : d_state_old + d_partner] = old_w[:, insert_pos:d_state_old]

    # Everything after state (action + lstm) — direct copy, shifted
    new_w[:, d_state_old + d_partner :] = old_w[:, d_state_old:]

    return new_w


def expand_state_dict(
    sd: dict[str, torch.Tensor],
    d_state_old: int = STATE_DIM,
    d_partner: int = PARTNER_HAND_DIM,
    insert_pos: int = PARTNER_INSERT_POS,
) -> dict[str, torch.Tensor]:
    """Expand a single network's state dict for Tier 1.

    Returns a new dict (does not mutate the input).
    """
    new_sd = {}
    for key, tensor in sd.items():
        if key in ("mlp.0.weight", "hand_pred.0.weight"):
            new_sd[key] = _expand_first_layer(tensor, d_state_old, d_partner, insert_pos)
        else:
            new_sd[key] = tensor.clone()
    return new_sd


def expand_checkpoint(
    base_path: str | Path,
    output_path: str | Path,
    d_state_old: int = STATE_DIM,
    d_partner: int = PARTNER_HAND_DIM,
    insert_pos: int = PARTNER_INSERT_POS,
) -> dict:
    """Load a standard checkpoint, expand both lead/follow, save Tier 1 checkpoint.

    Returns the expanded checkpoint dict.
    """
    base_path = Path(base_path)
    output_path = Path(output_path)

    ckpt = torch.load(base_path, map_location="cpu", weights_only=True)

    # Expand both networks
    for key in ("lead", "follow"):
        old_sd = ckpt[key]

        # Print what we're expanding
        for param_name in ("mlp.0.weight", "hand_pred.0.weight"):
            old_shape = old_sd[param_name].shape
            new_in = old_shape[1] + d_partner
            print(f"  {key}.{param_name}: {list(old_shape)} → [{old_shape[0]}, {new_in}]")

        ckpt[key] = expand_state_dict(old_sd, d_state_old, d_partner, insert_pos)

    # Update metadata
    ckpt["tier1_expanded_from"] = str(base_path)
    ckpt["d_state_tier1"] = d_state_old + d_partner

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"Saved expanded checkpoint to {output_path}")

    return ckpt
