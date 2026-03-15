"""Pre-allocated circular replay buffer with history support."""

from __future__ import annotations

import numpy as np
import torch

from .encoding import ACTION_DIM, D_MOVE, MAX_HISTORY, STATE_DIM


class ReplayBuffer:
    """Ring buffer with pre-allocated numpy arrays for state/action/history/return."""

    def __init__(
        self,
        capacity: int = 250_000,
        d_state: int = STATE_DIM,
        d_action: int = ACTION_DIM,
        d_move: int = D_MOVE,
        max_history: int = MAX_HISTORY,
    ):
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.states = np.zeros((capacity, d_state), dtype=np.float32)
        self.actions = np.zeros((capacity, d_action), dtype=np.float32)
        self.histories = np.zeros(
            (capacity, max_history, d_move), dtype=np.float32
        )
        self.hist_lens = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        history: np.ndarray,
        hist_len: int,
        mc_return: float,
    ) -> None:
        i = self.idx
        self.states[i] = state
        self.actions[i] = action
        seq_len = min(hist_len, self.histories.shape[1])
        self.histories[i, :seq_len] = history[:seq_len]
        self.histories[i, seq_len:] = 0.0
        self.hist_lens[i] = seq_len
        self.returns[i] = mc_return
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | None = None) -> dict:
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "state": torch.tensor(self.states[indices]),
            "action": torch.tensor(self.actions[indices]),
            "history": torch.tensor(self.histories[indices]),
            "hist_len": torch.tensor(self.hist_lens[indices]),  # keep as LongTensor
            "return": torch.tensor(self.returns[indices]),
        }
        if device is not None:
            batch["state"] = batch["state"].to(device)
            batch["action"] = batch["action"].to(device)
            batch["history"] = batch["history"].to(device)
            # hist_len stays on CPU for pack_padded_sequence
            batch["return"] = batch["return"].to(device)
        return batch

    def __len__(self) -> int:
        return self.size
