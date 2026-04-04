"""QMIX / WQMIX mixing networks.

TeamMixer: monotonic mixer (QMIX) — enforces ∂Q_team/∂Q_i ≥ 0 via abs() weights.
UnrestrictedMixer: non-monotonic mixer (WQMIX Q*) — no abs(), can express any function.

WQMIX uses both: TeamMixer for Q_tot (deployed), UnrestrictedMixer for Q* (training only).
The OW weighting down-weights overestimated actions (Q_tot > Q*), letting the monotonic
mixer avoid forcing up Q-values for actions that are bad for the team.
"""

from __future__ import annotations

import torch
import torch.nn as nn

D_GLOBAL = 310  # global state dimension (see encode_global_state)


class TeamMixer(nn.Module):
    """QMIX hypernetwork mixer.

    Takes per-agent Q-values [B, 2] and global state [B, D_GLOBAL],
    outputs team Q-value [B].

    Monotonicity enforced via torch.abs() on hypernetwork weights.
    """

    def __init__(self, embed_dim: int = 64, hypernet_hidden: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Hypernetwork 1: global_state → weights for first mixing layer [2 → embed_dim]
        self.hyper_w1 = nn.Sequential(
            nn.Linear(D_GLOBAL, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, 2 * embed_dim),
        )
        # Hypernetwork 1 bias
        self.hyper_b1 = nn.Linear(D_GLOBAL, embed_dim)

        # Hypernetwork 2: global_state → weights for second mixing layer [embed_dim → 1]
        self.hyper_w2 = nn.Sequential(
            nn.Linear(D_GLOBAL, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, embed_dim),
        )
        # Hypernetwork 2 bias (ELU so it can be negative)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(D_GLOBAL, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, q_vals: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_vals:       [B, 2] — Q-values for agents 0 and 2
            global_state: [B, D_GLOBAL]
        Returns:
            q_team: [B]
        """
        B = q_vals.size(0)
        q_vals = q_vals.unsqueeze(1)  # [B, 1, 2]

        # Layer 1: [B, 1, 2] × [B, 2, embed_dim] → [B, 1, embed_dim]
        w1 = torch.abs(self.hyper_w1(global_state)).view(B, 2, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(B, 1, self.embed_dim)
        hidden = torch.relu(torch.bmm(q_vals, w1) + b1)  # [B, 1, embed_dim]

        # Layer 2: [B, 1, embed_dim] × [B, embed_dim, 1] → [B, 1, 1]
        w2 = torch.abs(self.hyper_w2(global_state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(B, 1, 1)
        q_team = (torch.bmm(hidden, w2) + b2).view(B)  # [B]

        return q_team


class UnrestrictedMixer(nn.Module):
    """Unrestricted mixer for WQMIX Q* — no monotonicity constraint.

    Same architecture as TeamMixer but without torch.abs() on hypernetwork
    weights, allowing negative weights and non-monotonic value functions.
    Used only during training to approximate the true joint Q-value.
    """

    def __init__(self, embed_dim: int = 64, hypernet_hidden: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(D_GLOBAL, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, 2 * embed_dim),
        )
        self.hyper_b1 = nn.Linear(D_GLOBAL, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(D_GLOBAL, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(D_GLOBAL, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, q_vals: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """Same interface as TeamMixer: [B,2] + [B,D_GLOBAL] → [B]."""
        B = q_vals.size(0)
        q_vals = q_vals.unsqueeze(1)

        # No torch.abs() — weights can be negative
        w1 = self.hyper_w1(global_state).view(B, 2, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(B, 1, self.embed_dim)
        hidden = torch.relu(torch.bmm(q_vals, w1) + b1)

        w2 = self.hyper_w2(global_state).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(B, 1, 1)
        q_team = (torch.bmm(hidden, w2) + b2).view(B)

        return q_team
