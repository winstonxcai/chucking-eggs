"""Ring buffer for QMIX trick-level training data."""

from __future__ import annotations

import numpy as np
import torch

from .mixing import D_GLOBAL


class QMIXBuffer:
    """Ring buffer storing trick-level data for QMIX mixer training.

    Each entry: (q_0, q_2, global_state, team_return)
    where q_0/q_2 are the Q-values chosen by agents 0 and 2 on their
    last decision within the trick, and team_return is the final
    episode reward for team {0, 2}.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.q_vals = np.zeros((capacity, 2), dtype=np.float32)       # [N, 2]
        self.global_states = np.zeros((capacity, D_GLOBAL), dtype=np.float32)  # [N, D_GLOBAL]
        self.returns = np.zeros(capacity, dtype=np.float32)            # [N]

    def push(
        self,
        q_0: float,
        q_2: float,
        global_state: np.ndarray,
        team_return: float,
    ) -> None:
        i = self.idx
        self.q_vals[i, 0] = q_0
        self.q_vals[i, 1] = q_2
        self.global_states[i] = global_state
        self.returns[i] = team_return
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | None = None) -> dict:
        indices = np.random.randint(0, self.size, size=batch_size)
        q_vals = torch.tensor(self.q_vals[indices])
        gs = torch.tensor(self.global_states[indices])
        returns = torch.tensor(self.returns[indices])
        if device is not None:
            q_vals = q_vals.to(device)
            gs = gs.to(device)
            returns = returns.to(device)
        return {"q_vals": q_vals, "global_state": gs, "return": returns}

    def __len__(self) -> int:
        return self.size
