"""Ring buffer for QMIX trick-level training data (stores raw Q-net inputs)."""

from __future__ import annotations

import numpy as np
import torch

from .encoding import ACTION_DIM, D_GLOBAL, D_MOVE, MAX_HISTORY, STATE_DIM


class QMIXBuffer:
    """Ring buffer storing trick-level data for QMIX end-to-end training.

    Stores raw Q-network inputs per trick (NOT pre-computed Q-values), so
    Q-networks can be re-run with gradients during Phase B training.

    Two slots per trick: slot 0 = P0, slot 1 = P2.
    If a teammate didn't act in a trick, their slot's `active` flag is False.

    Memory at capacity=50K: ~780 MB.
    """

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        C = capacity
        self.states       = np.zeros((C, 2, STATE_DIM),           dtype=np.float32)
        self.actions      = np.zeros((C, 2, ACTION_DIM),          dtype=np.float32)
        self.histories    = np.zeros((C, 2, MAX_HISTORY, D_MOVE), dtype=np.float32)
        self.hist_lens    = np.ones ((C, 2),                      dtype=np.int64)   # min 1 for packing
        self.is_leading   = np.zeros((C, 2),                      dtype=np.bool_)
        self.active       = np.zeros((C, 2),                      dtype=np.bool_)
        self.global_states = np.zeros((C, D_GLOBAL),              dtype=np.float32)
        self.returns      = np.zeros(C,                           dtype=np.float32)

    def push(self, trick_record: dict, team_return: float) -> None:
        """Push one trick record.

        trick_record = {
            'transitions': {player_id: {state, action, history, hist_len, is_leading}},
            'global_state': np.ndarray [D_GLOBAL],
        }
        """
        i = self.idx
        self.active[i] = False
        self.states[i] = 0
        self.actions[i] = 0
        self.histories[i] = 0
        self.hist_lens[i] = 1

        for slot, player in enumerate((0, 2)):
            t = trick_record["transitions"].get(player)
            if t is None:
                continue
            self.states[i, slot]  = t["state"]
            self.actions[i, slot] = t["action"]
            T = min(t["hist_len"], MAX_HISTORY)
            if T > 0:
                self.histories[i, slot, :T] = t["history"][:T]
            self.hist_lens[i, slot]  = max(T, 1)
            self.is_leading[i, slot] = t["is_leading"]
            self.active[i, slot]     = True

        self.global_states[i] = trick_record["global_state"]
        self.returns[i] = team_return

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | None = None) -> dict:
        indices = np.random.randint(0, self.size, size=batch_size)

        def _t(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(arr[indices].copy())
            return t.to(device) if device is not None else t

        return {
            "states":       _t(self.states),        # [B, 2, STATE_DIM]
            "actions":      _t(self.actions),        # [B, 2, ACTION_DIM]
            "histories":    _t(self.histories),      # [B, 2, MAX_HISTORY, D_MOVE]
            "hist_lens":    _t(self.hist_lens),      # [B, 2]
            "is_leading":   _t(self.is_leading),     # [B, 2] bool
            "active":       _t(self.active),         # [B, 2] bool
            "global_state": _t(self.global_states),  # [B, D_GLOBAL]
            "return":       _t(self.returns),        # [B]
        }

    def __len__(self) -> int:
        return self.size
