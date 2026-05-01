"""GuanZero Q-network: LSTM(history) + flat MLP scorer.

Input is a *batched dict* (see ``buffer.collate``). Output is a 1-D Q
tensor of shape (B,) — one Q-value per (state, candidate-action) pair.
There is no policy head; action selection is argmax over Q across legal
actions, evaluated one-by-one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import CARD_ID_DIM, HISTORY_LEN, static_dim


class GuanZeroQNet(nn.Module):
    def __init__(
        self,
        hidden_lstm: int = 256,
        hidden_mlp: int = 1024,
        n_mlp_layers: int = 6,
        dropout: float = 0.0,
        use_oracle_others_hand: bool = True,
    ) -> None:
        super().__init__()
        self.use_oracle_others_hand = use_oracle_others_hand
        self.hidden_lstm = hidden_lstm

        self.history_lstm = nn.LSTM(
            input_size=CARD_ID_DIM,
            hidden_size=hidden_lstm,
            batch_first=True,
        )

        in_dim = static_dim(use_oracle_others_hand) + hidden_lstm
        layers: list[nn.Module] = []
        for i in range(n_mlp_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_mlp, hidden_mlp))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_mlp, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # history: (B, HISTORY_LEN, 108) -> last hidden state (B, hidden_lstm)
        hist = batch["history"]
        _, (h_n, _) = self.history_lstm(hist)
        z_hist = h_n[-1]

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


def init_position_nets(
    hidden_lstm: int = 256,
    hidden_mlp: int = 1024,
    n_mlp_layers: int = 6,
    dropout: float = 0.0,
    use_oracle_others_hand: bool = True,
) -> dict[int, GuanZeroQNet]:
    """One Q-network per seat (paper §4.2)."""
    return {
        p: GuanZeroQNet(
            hidden_lstm=hidden_lstm,
            hidden_mlp=hidden_mlp,
            n_mlp_layers=n_mlp_layers,
            dropout=dropout,
            use_oracle_others_hand=use_oracle_others_hand,
        )
        for p in range(4)
    }


__all__ = ["GuanZeroQNet", "init_position_nets", "HISTORY_LEN"]
