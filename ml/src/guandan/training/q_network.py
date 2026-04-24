"""Bare MLP Q-network for Direction C (Jidan-distillation).

No LSTM, no GNN, no aux heads. Single forward pass: (state, action) -> Q.
Stripped from the archived QNetworkLSTM after LOGBOOK §9 (GNN dead) and §13
(aux heads collapse without selection pressure).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import ACTION_DIM


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QNetwork(nn.Module):
    """Q(state, action) -> scalar. 4-layer MLP with ReLU."""

    def __init__(
        self,
        d_state: int = 480,
        d_action: int = ACTION_DIM,
        hidden: int = 256,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.net = nn.Sequential(
            nn.Linear(d_state + d_action, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """state: [B, d_state], action: [B, d_action] -> Q: [B]."""
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class QValueNet(QNetwork):
    """QNetwork + separate state-only value head V(s) for AlphaZero training.

    The Q-trunk is identical to QNetwork (same weights, same forward contract).
    The V-trunk is zero-initialized so old jidan_policy.pt loads cleanly with
    strict=False — Q weights restore exactly, V weights stay at zero until
    gen-1 AZ training fills them in.
    """

    def __init__(
        self,
        d_state: int = 480,
        d_action: int = ACTION_DIM,
        hidden: int = 256,
    ):
        super().__init__(d_state, d_action, hidden)
        self.v_net = nn.Sequential(
            nn.Linear(d_state, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        for p in self.v_net.parameters():
            nn.init.zeros_(p)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        """state: [B, d_state] -> V: [B]."""
        return self.v_net(state).squeeze(-1)
