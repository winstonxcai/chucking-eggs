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

    def encode_history(
        self, history: torch.Tensor, history_len: torch.Tensor,
    ) -> torch.Tensor:
        """Run LSTM on history sequences. [B, T, d_move], [B] → [B, lstm_hidden]."""
        lengths = history_len.clamp(min=1)

        if history.device.type == "mps":
            _, (h_n, _) = self.lstm(history)
            return h_n.squeeze(0)
        else:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                history, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            return h_n.squeeze(0)

    def forward_from_embedding(
        self, state: torch.Tensor, action: torch.Tensor, hist_emb: torch.Tensor,
    ) -> torch.Tensor:
        """MLP forward from pre-computed history embedding.
        [B, d_state], [B, d_action], [B, lstm_hidden] → [B]
        """
        x = torch.cat([state, action, hist_emb], dim=-1)
        return self.mlp(x).squeeze(-1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor,
        history_len: torch.Tensor,
    ) -> torch.Tensor:
        history_emb = self.encode_history(history, history_len)
        return self.forward_from_embedding(state, action, history_emb)
