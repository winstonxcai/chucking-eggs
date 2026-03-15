"""Q-networks: MLP baseline and LSTM+MLP with history encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import ACTION_DIM, D_MOVE, STATE_DIM


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QNetwork(nn.Module):
    """Legacy MLP Q-network (Day 1-2)."""

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


class QNetworkLSTM(nn.Module):
    """Q(state, action, history) with LSTM history encoder.

    Inputs:
        state:       [B, d_state]   (417)
        action:      [B, d_action]  (160)
        history:     [B, T, d_move] (T up to MAX_HISTORY, each 83)
        history_len: [B]            (actual lengths for packing)
    Output:
        q_value:     [B]
    """

    def __init__(
        self,
        d_state: int = STATE_DIM,
        d_action: int = ACTION_DIM,
        d_move: int = D_MOVE,
        lstm_hidden: int = 128,
        hidden: int = 512,
        n_layers: int = 3,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=d_move,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        input_dim = d_state + d_action + lstm_hidden
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)

        # Initialize LSTM forget gate bias to 1 (helps learning)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                bias_size = param.size(0)
                param.data[bias_size // 4 : bias_size // 2].fill_(1.0)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor,
        history_len: torch.Tensor,
    ) -> torch.Tensor:
        lengths = history_len.clamp(min=1)

        if history.device.type == "mps":
            # MPS doesn't support pack_padded_sequence — run LSTM on
            # full padded sequence; h_n is the hidden state after the last
            # timestep (including padding, but zero-padding has minimal effect)
            _, (h_n, _) = self.lstm(history)
            history_emb = h_n.squeeze(0)  # [B, lstm_hidden]
        else:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                history, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            history_emb = h_n.squeeze(0)  # [B, lstm_hidden]

        x = torch.cat([state, action, history_emb], dim=-1)
        return self.mlp(x).squeeze(-1)
