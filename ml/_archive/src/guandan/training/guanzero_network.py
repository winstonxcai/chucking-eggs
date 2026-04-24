"""GuanZero network architecture (arXiv:2402.13582, Figure 3).

Architecture:
  history [5, 432] → LSTM(432, 128) → h_T [128]
  [non_history(1075) ‖ h_T(128) ‖ action(108)] = [1311]
  → Linear(1311, 512) → ReLU
  → Linear(512, 512)  → ReLU  (×4 more)
  → Linear(512, 1)             → Q(s, a)

~2M parameters total.

Key design: forward_fast() runs LSTM once per decision, then batches all
B action MLP forwards — avoids the O(B) LSTM overhead of naïve batching.
"""

from __future__ import annotations

import torch
import torch.nn as nn

D_NON_HISTORY = 1075
D_ACTION = 108
D_HISTORY_STEP = 432   # 4 players × 108 cards per round
LSTM_HIDDEN = 128
MLP_HIDDEN = 512
N_MLP_LAYERS = 6       # 5 hidden layers + 1 output layer


class GuanZeroNetwork(nn.Module):
    """Faithful reproduction of the GuanZero Q-network (Figure 3)."""

    def __init__(
        self,
        d_non_history: int = D_NON_HISTORY,
        d_action: int = D_ACTION,
        d_history_step: int = D_HISTORY_STEP,
        lstm_hidden: int = LSTM_HIDDEN,
        mlp_hidden: int = MLP_HIDDEN,
        n_mlp_layers: int = N_MLP_LAYERS,
    ) -> None:
        super().__init__()
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=d_history_step,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        # Initialize forget gate bias to 1.0 for better gradient flow
        nn.init.constant_(self.lstm.bias_hh_l0[lstm_hidden:2 * lstm_hidden], 1.0)
        nn.init.constant_(self.lstm.bias_ih_l0[lstm_hidden:2 * lstm_hidden], 1.0)

        mlp_input = d_non_history + lstm_hidden + d_action  # 1311
        layers: list[nn.Module] = []
        in_dim = mlp_input
        for _ in range(n_mlp_layers - 1):   # 5 hidden layers
            layers.append(nn.Linear(in_dim, mlp_hidden))
            layers.append(nn.ReLU())
            in_dim = mlp_hidden
        layers.append(nn.Linear(in_dim, 1))  # scalar Q output
        self.mlp = nn.Sequential(*layers)

    # ── Standard forward (for training, takes batched inputs) ───────────────

    def forward(
        self,
        non_history: torch.Tensor,   # [B, 1075]
        history: torch.Tensor,       # [B, 5, 432]
        hist_len: torch.Tensor,      # [B] — number of valid history steps
        action: torch.Tensor,        # [B, 108]
    ) -> torch.Tensor:               # [B]
        """Standard forward pass. Each sample gets its own LSTM call."""
        packed = nn.utils.rnn.pack_padded_sequence(
            history,
            hist_len.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)    # h_n: [1, B, 128]
        lstm_out = h_n.squeeze(0)           # [B, 128]
        x = torch.cat([non_history, lstm_out, action], dim=-1)  # [B, 1311]
        return self.mlp(x).squeeze(-1)     # [B]

    # ── LSTM-once forward (for inference, B actions share one history) ───────

    def encode_history(
        self,
        history: torch.Tensor,   # [5, 432] or [1, 5, 432]
        hist_len: torch.Tensor,  # [1] scalar
    ) -> torch.Tensor:           # [128]
        """Run LSTM on a single history sequence. Returns h_T [128]."""
        if history.dim() == 2:
            history = history.unsqueeze(0)   # [1, 5, 432]
        packed = nn.utils.rnn.pack_padded_sequence(
            history,
            hist_len.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)      # h_n: [1, 1, 128]
        return h_n.squeeze()                  # [128]

    def forward_fast(
        self,
        non_histories: torch.Tensor,  # [B, 1075] — one per action (behavior flags differ)
        history: torch.Tensor,        # [5, 432]  — shared across all actions
        hist_len: torch.Tensor,       # [1]
        actions: torch.Tensor,        # [B, 108]
    ) -> torch.Tensor:                # [B]
        """LSTM-once inference: run LSTM once, expand to B, batch MLP."""
        B = non_histories.shape[0]
        lstm_out = self.encode_history(history, hist_len)       # [128]
        lstm_expanded = lstm_out.unsqueeze(0).expand(B, -1)     # [B, 128]
        x = torch.cat([non_histories, lstm_expanded, actions], dim=-1)  # [B, 1311]
        return self.mlp(x).squeeze(-1)                          # [B]

    def forward_from_embedding(
        self,
        non_history: torch.Tensor,   # [B, 1075]
        action: torch.Tensor,        # [B, 108]
        lstm_emb: torch.Tensor,      # [B, 128]
    ) -> torch.Tensor:               # [B]
        """MLP-only forward given pre-computed LSTM embedding (for training)."""
        x = torch.cat([non_history, lstm_emb, action], dim=-1)  # [B, 1311]
        return self.mlp(x).squeeze(-1)                           # [B]

    def batch_encode_history(
        self,
        history: torch.Tensor,   # [B, 5, 432]
        hist_len: torch.Tensor,  # [B]
    ) -> torch.Tensor:           # [B, 128]
        """Run LSTM on a batch of histories (used during training)."""
        packed = nn.utils.rnn.pack_padded_sequence(
            history,
            hist_len.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)   # h_n: [1, B, 128]
        return h_n.squeeze(0)             # [B, 128]
