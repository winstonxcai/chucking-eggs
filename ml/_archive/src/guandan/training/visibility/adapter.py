"""Partner adapter: additive Q-value correction from partner's hand.

Frozen base Q-network produces Q_base. The adapter adds a learned
action-specific correction conditioned on partner hand + action only:

    Q_final = Q_base + adapter(partner_hand, action_enc)

Deliberately excludes hist_emb — Q_base already captures game history.
Excluding it prevents the adapter from learning global game-level shifts
(e.g. "this game is going badly → subtract 0.9 from all actions"), which
increases MSE without improving action ranking.

Output layer is zero-initialized so Q_final == Q_base at the start.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..encoding import ACTION_DIM
from .encoding import PARTNER_HAND_DIM


class PartnerAdapter(nn.Module):
    """Small MLP that predicts action-specific Q-adjustment from partner hand."""

    def __init__(
        self,
        d_partner: int = PARTNER_HAND_DIM,  # 60
        d_action: int = ACTION_DIM,          # 160
        d_hist: int = 256,                   # unused, kept for API compat
        hidden: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_partner + d_action, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init output layer so adapter starts at 0, preserving base policy
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        partner_hand: torch.Tensor,
        action_enc: torch.Tensor,
        hist_emb: torch.Tensor | None = None,  # ignored — Q_base already has this
    ) -> torch.Tensor:
        """(B, 60), (B, 160) -> (B,) action-specific Q-value adjustment."""
        x = torch.cat([partner_hand, action_enc], dim=-1)
        return self.net(x).squeeze(-1)

    def output_l2(
        self,
        partner_hand: torch.Tensor,
        action_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared output — used as regularization term to keep outputs small."""
        return (self.forward(partner_hand, action_enc) ** 2).mean()
