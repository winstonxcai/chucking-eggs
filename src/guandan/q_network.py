"""MLP Q-network: Q(state, action) → scalar value."""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import ACTION_DIM, STATE_DIM


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QNetwork(nn.Module):
    def __init__(
        self,
        d_state: int = STATE_DIM,
        d_action: int = ACTION_DIM,
        hidden: int = 256,
    ):
        super().__init__()
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
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)
