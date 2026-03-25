"""Pre-allocated circular replay buffer with history support."""

from __future__ import annotations

import numpy as np
import torch

from .encoding import ACTION_DIM, D_MOVE, MAX_HISTORY, OPP_CARDS_DIM, STATE_DIM

GNN_EMB_DIM = 384  # 3 × 128 (hand_emb + action_emb + remain_emb)


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
        self.opponent_cards = np.zeros(
            (capacity, OPP_CARDS_DIM), dtype=np.float32
        )
        # Pre-computed GNN embeddings (384 = 3×128: hand + action + remain)
        self.gnn_emb = np.zeros((capacity, GNN_EMB_DIM), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        history: np.ndarray,
        hist_len: int,
        mc_return: float,
        opponent_cards: np.ndarray | None = None,
        gnn_emb: np.ndarray | None = None,
    ) -> None:
        i = self.idx
        self.states[i] = state
        self.actions[i] = action
        seq_len = min(hist_len, self.histories.shape[1])
        self.histories[i, :seq_len] = history[:seq_len]
        self.histories[i, seq_len:] = 0.0
        self.hist_lens[i] = seq_len
        self.returns[i] = mc_return
        if opponent_cards is not None:
            self.opponent_cards[i] = opponent_cards
        else:
            self.opponent_cards[i] = 0.0
        if gnn_emb is not None:
            self.gnn_emb[i] = gnn_emb
        else:
            self.gnn_emb[i] = 0.0
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | None = None) -> dict:
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "state": torch.tensor(self.states[indices]),
            "action": torch.tensor(self.actions[indices]),
            "history": torch.tensor(self.histories[indices]),
            "hist_len": torch.tensor(self.hist_lens[indices]),
            "return": torch.tensor(self.returns[indices]),
            "opponent_cards": torch.tensor(self.opponent_cards[indices]),
            "gnn_emb": torch.tensor(self.gnn_emb[indices]),
        }
        if device is not None:
            batch["state"] = batch["state"].to(device)
            batch["action"] = batch["action"].to(device)
            batch["history"] = batch["history"].to(device)
            # hist_len stays on CPU for pack_padded_sequence
            batch["return"] = batch["return"].to(device)
            batch["opponent_cards"] = batch["opponent_cards"].to(device)
            batch["gnn_emb"] = batch["gnn_emb"].to(device)
        return batch

    def clear(self) -> None:
        """Clear the buffer between curriculum stages."""
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size
