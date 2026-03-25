"""Pre-allocated circular replay buffer with history support."""

from __future__ import annotations

import numpy as np
import torch

from .encoding import ACTION_DIM, D_MOVE, MAX_HISTORY, OPP_CARDS_DIM, STATE_DIM


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

        # GNN hand data (optional — for use_gnn=True training)
        self._max_hand = 29  # 27 cards + margin
        self.hand_cards = np.zeros(
            (capacity, self._max_hand, 3), dtype=np.int32  # (rank, suit, deck)
        )
        self.hand_sizes = np.zeros(capacity, dtype=np.int32)
        self.action_card_mask = np.zeros(
            (capacity, self._max_hand), dtype=np.float32
        )

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        history: np.ndarray,
        hist_len: int,
        mc_return: float,
        opponent_cards: np.ndarray | None = None,
        hand_cards: np.ndarray | None = None,
        hand_size: int = 0,
        action_card_mask: np.ndarray | None = None,
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
        if hand_cards is not None:
            n = min(hand_size, self._max_hand)
            self.hand_cards[i] = 0
            self.hand_cards[i, :n] = hand_cards[:n]
            self.hand_sizes[i] = n
        else:
            self.hand_cards[i] = 0
            self.hand_sizes[i] = 0
        if action_card_mask is not None:
            self.action_card_mask[i] = 0.0
            n = min(len(action_card_mask), self._max_hand)
            self.action_card_mask[i, :n] = action_card_mask[:n]
        else:
            self.action_card_mask[i] = 0.0
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
            "opponent_cards": torch.tensor(self.opponent_cards[indices]),
            "hand_cards": torch.tensor(self.hand_cards[indices]),
            "hand_size": torch.tensor(self.hand_sizes[indices]),
            "action_card_mask": torch.tensor(self.action_card_mask[indices]),
        }
        if device is not None:
            batch["state"] = batch["state"].to(device)
            batch["action"] = batch["action"].to(device)
            batch["history"] = batch["history"].to(device)
            # hist_len stays on CPU for pack_padded_sequence
            batch["return"] = batch["return"].to(device)
            batch["opponent_cards"] = batch["opponent_cards"].to(device)
        return batch

    def clear(self) -> None:
        """Clear the buffer between curriculum stages."""
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size
