"""Q-networks: MLP baseline and LSTM+MLP with history encoder."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .encoding import ACTION_DIM, D_MOVE, OPP_CARDS_DIM, STATE_DIM

log = logging.getLogger(__name__)


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
        use_gnn: bool = False,
        gnn_out: int = 128,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.use_gnn = use_gnn
        self.gnn_out = gnn_out

        self.lstm = nn.LSTM(
            input_size=d_move,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Optional GNN for hand structure
        gnn_dims = 0
        if use_gnn:
            from .gnn import HandGNN
            self.hand_gnn = HandGNN(d_out=gnn_out)
            gnn_dims = 3 * gnn_out  # hand + action + remain embeddings

        input_dim = d_state + d_action + lstm_hidden + gnn_dims
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)

        # Auxiliary hand prediction head (training-only)
        self.hand_pred = nn.Sequential(
            nn.Linear(d_state + lstm_hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, OPP_CARDS_DIM),
        )

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

        When use_gnn=True, pads with zeros for missing GNN dims so the flat
        forward path still works (GNN contribution = 0).
        """
        x = torch.cat([state, action, hist_emb], dim=-1)
        if self.use_gnn:
            # Pad with zeros for GNN embedding dims (3 × gnn_out)
            B = x.size(0)
            zeros = torch.zeros(B, 3 * self.gnn_out, device=x.device)
            x = torch.cat([x, zeros], dim=-1)
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

    def predict_opponent_cards(
        self, state: torch.Tensor, hist_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict opponent card distribution. [B, d_state], [B, lstm_hidden] → [B, 60] sigmoid."""
        return torch.sigmoid(self.hand_pred(torch.cat([state, hist_emb], dim=-1)))

    def forward_with_aux(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor,
        history_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward returning both Q-value and hand prediction.
        Returns: (q_value [B], hand_pred [B, 60]).
        """
        hist_emb = self.encode_history(history, history_len)
        q = self.forward_from_embedding(state, action, hist_emb)
        hp = self.predict_opponent_cards(state, hist_emb)
        return q, hp


    def forward_amortized(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        hist_emb_batch: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        action_masks: torch.Tensor,
        remaining_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Amortized GNN: encode_nodes ONCE, pool B actions.

        Args:
            state_batch:     [B, d_state]
            action_batch:    [B, d_action]
            hist_emb_batch:  [B, lstm_hidden]  (pre-computed, shared)
            node_features:   [N, 23]           (single graph, shared)
            edge_index:      [2, E]
            edge_features:   [E, 8]
            action_masks:    [B, N]            (different per action)
            remaining_masks: [B, N]
        Returns: [B] Q-values
        """
        assert self.use_gnn
        B = state_batch.size(0)
        device = state_batch.device

        h_nodes, hand_emb = self.hand_gnn.encode_nodes(
            node_features.to(device), edge_index.to(device),
            edge_features.to(device))

        hand_emb_batch = hand_emb.unsqueeze(0).expand(B, -1)

        action_embs, remain_embs = [], []
        for i in range(B):
            a_emb, r_emb = self.hand_gnn.pool_action(
                h_nodes, action_masks[i].to(device),
                remaining_masks[i].to(device))
            action_embs.append(a_emb)
            remain_embs.append(r_emb)

        action_embs = torch.stack(action_embs)
        remain_embs = torch.stack(remain_embs)

        x = torch.cat([state_batch, action_batch, hist_emb_batch,
                        hand_emb_batch, action_embs, remain_embs], dim=-1)
        return self.mlp(x).squeeze(-1)


def load_compat(model: nn.Module, state_dict: dict) -> None:
    """Load state_dict with backward compatibility (strict=False)."""
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.info("New params (random init): %s",
                 list({k.split(".")[0] for k in missing}))


def load_with_gnn_expansion(model: QNetworkLSTM, old_state_dict: dict) -> None:
    """Load old checkpoint into GNN-augmented model.

    Copies MLP weights for shared dimensions, zero-inits GNN input columns.
    This makes the GNN contribution start at exactly zero — identical Q-values
    to the old checkpoint, with gradual GNN influence during training.
    """
    new_sd = model.state_dict()

    for key, old_tensor in old_state_dict.items():
        if key not in new_sd:
            log.info("Skipping unknown key: %s", key)
            continue
        new_tensor = new_sd[key]
        if old_tensor.shape == new_tensor.shape:
            new_sd[key] = old_tensor
        elif 'mlp.0.weight' in key:
            # First MLP layer expanded: [H, old_dim] → [H, new_dim]
            new_sd[key][:, :old_tensor.shape[1]] = old_tensor
            new_sd[key][:, old_tensor.shape[1]:] = 0.0
            log.info("Expanded %s: %s → %s", key,
                      tuple(old_tensor.shape), tuple(new_tensor.shape))
        else:
            log.warning("Shape mismatch for %s: %s vs %s",
                        key, tuple(old_tensor.shape), tuple(new_tensor.shape))

    model.load_state_dict(new_sd)
    gnn_loaded = [k for k in old_state_dict if 'hand_gnn' in k and k in new_sd]
    gnn_missing = [k for k in new_sd if 'hand_gnn' in k and k not in old_state_dict]
    if gnn_loaded:
        log.info("GNN params (loaded from checkpoint): %d tensors", len(gnn_loaded))
    if gnn_missing:
        log.info("GNN params (random init): %d tensors", len(gnn_missing))


def checkpoint_has_gnn(state_dict: dict) -> bool:
    """Check if a checkpoint state_dict has GNN-expanded MLP (input > 833)."""
    mlp0 = state_dict.get("mlp.0.weight")
    if mlp0 is None:
        return False
    return mlp0.shape[1] > STATE_DIM + ACTION_DIM + 128  # 833 = 417+160+256


def load_strip_gnn(model: QNetworkLSTM, old_state_dict: dict) -> None:
    """Load GNN-expanded checkpoint into non-GNN model, discarding dead GNN columns."""
    new_sd = model.state_dict()
    stripped = 0

    for key, old_tensor in old_state_dict.items():
        if 'hand_gnn' in key:
            stripped += 1
            continue
        if key not in new_sd:
            continue
        new_tensor = new_sd[key]
        if old_tensor.shape == new_tensor.shape:
            new_sd[key] = old_tensor
        elif 'mlp.0.weight' in key and old_tensor.shape[1] > new_tensor.shape[1]:
            # Take only the non-GNN columns
            new_sd[key] = old_tensor[:, :new_tensor.shape[1]]
            log.info("Stripped GNN columns from %s: %s → %s", key,
                     tuple(old_tensor.shape), tuple(new_tensor.shape))
        else:
            log.warning("Shape mismatch for %s: %s vs %s",
                        key, tuple(old_tensor.shape), tuple(new_tensor.shape))

    model.load_state_dict(new_sd)
    if stripped:
        log.info("Stripped %d GNN param tensors from checkpoint", stripped)
