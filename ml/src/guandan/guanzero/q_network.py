"""GuanZero Q-network: LSTM(history) + flat MLP scorer.

Input is a batched dict (see ``buffer.collate_encoded``). Output is a 1-D
Q tensor of shape ``(B,)`` — one Q-value per (state, candidate-action) pair.
Action selection is argmax over Q across legal actions; there is no policy head.

Build four seat-independent copies with ``init_seat_nets(cfg)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import QNetConfig
from .encoder import CARD_ID_DIM, HISTORY_LEN, static_dim


class GuanZeroQNet(nn.Module):
    """LSTM history encoder → concat with flat per-step features → MLP scoring head.

    Input dict keys are defined by ``encoder.ENCODE_CHANNEL_KEYS``.
    Output shape: ``(B,)`` — one scalar Q-value per row.
    """

    def __init__(self, cfg: QNetConfig) -> None:
        super().__init__()
        self.use_oracle_others_hand = cfg.use_oracle_others_hand
        self.hidden_lstm = cfg.hidden_lstm

        self.history_lstm = nn.LSTM(
            input_size=CARD_ID_DIM,
            hidden_size=cfg.hidden_lstm,
            batch_first=True,
        )

        in_dim = static_dim(cfg.use_oracle_others_hand) + cfg.hidden_lstm
        layers: list[nn.Module] = []
        for i in range(cfg.n_mlp_layers):
            layers.append(nn.Linear(in_dim if i == 0 else cfg.hidden_mlp, cfg.hidden_mlp))
            layers.append(nn.ReLU(inplace=True))  # inplace=True safe: no autograd through ReLU output
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(cfg.hidden_mlp, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict with keys from ``ENCODE_CHANNEL_KEYS``, each ``(B, ...)``.
        Returns:
            Q-values of shape ``(B,)``.
        """
        hist = batch["history"]
        _, (h_n, _) = self.history_lstm(hist)
        z_hist = h_n.squeeze(0)   # num_layers=1, so h_n is (1, B, hidden); squeeze to (B, hidden)

        b = hist.shape[0]
        flat = torch.cat(
            [
                batch["own_hand"],
                batch["others_hand"],
                batch["recent_action_each_player"].reshape(b, -1),
                batch["played_cards_others"].reshape(b, -1),
                batch["remaining_counts_others"].reshape(b, -1),
                batch["level"],
                batch["behavior"],
                batch["candidate_action"],
                z_hist,
            ],
            dim=-1,
        )
        return self.mlp(flat).squeeze(-1)


def init_seat_nets(cfg: QNetConfig) -> dict[int, GuanZeroQNet]:
    """One Q-network per seat (paper §4.2).

    All four networks share the same architecture; they are trained
    independently on per-seat replay buffers.
    """
    return {p: GuanZeroQNet(cfg) for p in range(4)}


__all__ = ["GuanZeroQNet", "init_seat_nets"]
