"""Q-networks for DART training and GuanZero comparison.

``GuanZeroQNet`` is the per-seat comparison baseline. ``DartQNet`` is the
current role-aware, trick-relative shared-trunk model used by DART runs.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import QNetConfig
from .encoder import CARD_ID_DIM, HISTORY_LEN
from .encoding.base_encoder import static_dim


class _LSTMHistoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D] → [B, hidden_dim]
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)


class _TransformerHistoryEncoder(nn.Module):
    def __init__(
        self, input_dim: int, d_model: int, nhead: int, num_layers: int, ff_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(HISTORY_LEN, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D] → [B, d_model]
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        h = self.proj(x) + self.pos_embed(pos)  # [B, 20, d_model]
        h = self.encoder(h)                       # [B, 20, d_model]
        return h.mean(dim=1)                       # [B, d_model]


class GuanZeroQNet(nn.Module):
    """History encoder → concat with flat per-step features → MLP scoring head.

    Input dict keys are defined by ``encoder.ENCODE_CHANNEL_KEYS``.
    Output shape: ``(B,)`` — one scalar Q-value per row.
    """

    def __init__(self, cfg: QNetConfig) -> None:
        super().__init__()
        self.is_partner_visible = cfg.is_partner_visible
        self.hidden_lstm = cfg.hidden_lstm

        if cfg.history_encoder == "transformer":
            self.history_module: nn.Module = _TransformerHistoryEncoder(
                CARD_ID_DIM, cfg.hidden_lstm, cfg.transformer_nhead,
                cfg.transformer_layers, cfg.transformer_ff_dim, cfg.dropout,
            )
        else:
            self.history_module = _LSTMHistoryEncoder(CARD_ID_DIM, cfg.hidden_lstm)

        in_dim = static_dim(cfg.is_partner_visible) + cfg.hidden_lstm
        layers: list[nn.Module] = []
        for i in range(cfg.n_mlp_layers):
            layers.append(nn.Linear(in_dim if i == 0 else cfg.hidden_mlp, cfg.hidden_mlp))
            layers.append(nn.ReLU(inplace=True))  # inplace=True safe: no autograd through ReLU output
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(cfg.hidden_mlp, 1))
        self.mlp = nn.Sequential(*layers)

    def _flat_state_features(
        self,
        batch: dict[str, torch.Tensor],
        b: int,
    ) -> torch.Tensor:
        return torch.cat(
            [
                batch["own_hand"],
                batch["others_hand"],
                batch["recent_action_each_player"].reshape(b, -1),
                batch["played_cards_others"].reshape(b, -1),
                batch["remaining_counts_others"].reshape(b, -1),
                batch["level"],
                batch["behavior"],
            ],
            dim=-1,
        )

    def _flat_action_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["candidate_action"]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict with keys from ``ENCODE_CHANNEL_KEYS``, each ``(B, ...)``.
        Returns:
            Q-values of shape ``(B,)``.
        """
        hist = batch["history"]
        z_hist = self.history_module(hist)  # [B, hidden_lstm]

        b = hist.shape[0]
        flat = torch.cat(
            [self._flat_state_features(batch, b), self._flat_action_features(batch), z_hist],
            dim=-1,
        )
        return self.mlp(flat).squeeze(-1)

    def forward_grouped(
        self,
        state_batch: dict[str, torch.Tensor],
        action_batch: dict[str, torch.Tensor],
        repeats: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate actions when multiple actions share each state.

        ``state_batch`` has one row per decision, ``action_batch`` has
        ``sum(repeats)`` rows, and ``repeats[i]`` is the number of actions for
        state ``i``. This avoids re-running the history encoder once per
        candidate action.
        """
        hist = state_batch["history"]
        b = hist.shape[0]
        repeats = repeats.to(device=hist.device, dtype=torch.long)

        z_hist = self.history_module(hist)
        state_flat = self._flat_state_features(state_batch, b)
        if b == 1:
            return self._forward_grouped_single_state(
                state_flat,
                self._flat_action_features(action_batch),
                z_hist,
            )

        action_flat = self._flat_action_features(action_batch)
        first = self.mlp[0]
        if isinstance(first, nn.Linear):
            state_dim = state_flat.shape[-1]
            action_dim = action_flat.shape[-1]
            w = first.weight
            shared = F.linear(state_flat, w[:, :state_dim], first.bias)
            shared = shared + F.linear(z_hist, w[:, state_dim + action_dim :], None)
            shared_rows = torch.repeat_interleave(shared, repeats, dim=0)
            x = F.linear(action_flat, w[:, state_dim : state_dim + action_dim], None)
            x = x + shared_rows
            for layer in list(self.mlp.children())[1:]:
                x = layer(x)
            return x.squeeze(-1)

        state_rows = torch.repeat_interleave(state_flat, repeats, dim=0)
        z_hist_rows = torch.repeat_interleave(z_hist, repeats, dim=0)
        flat = torch.cat([state_rows, action_flat, z_hist_rows], dim=-1)
        return self.mlp(flat).squeeze(-1)

    def _forward_grouped_single_state(
        self,
        state_flat: torch.Tensor,   # [1, state_dim]
        action_flat: torch.Tensor,  # [K, action_dim]
        z_hist: torch.Tensor,       # [1, hidden_lstm]
    ) -> torch.Tensor:
        """Fast path for the actor's common one-decision grouped forward.

        The first MLP layer is linear, so the shared state/history contribution
        can be computed once and added to the per-action contribution. This is
        exactly equivalent to expanding state/history to K rows before the MLP,
        but avoids a large repeated first-layer matmul for local actor calls.

        Falls through to the expansion path when ``self.mlp[0]`` is not a plain
        nn.Linear (e.g. dynamic-quantized int8), since the weight-slicing trick
        only works on a regular dense linear.
        """
        first = self.mlp[0]
        K = action_flat.shape[0]
        if isinstance(first, nn.Linear):
            state_dim = state_flat.shape[-1]
            action_dim = action_flat.shape[-1]
            w = first.weight
            shared = F.linear(state_flat, w[:, :state_dim], first.bias)
            shared = shared + F.linear(z_hist, w[:, state_dim + action_dim :], None)
            x = F.linear(action_flat, w[:, state_dim : state_dim + action_dim], None)
            x = x + shared
            for layer in list(self.mlp.children())[1:]:
                x = layer(x)
        else:
            state_exp = state_flat.expand(K, -1)
            z_hist_exp = z_hist.expand(K, -1)
            x = self.mlp(torch.cat([state_exp, action_flat, z_hist_exp], dim=-1))
        return x.squeeze(-1)


def init_guanzero_nets(cfg: QNetConfig) -> dict[int, GuanZeroQNet]:
    """One Q-network per seat for the GuanZero comparison baseline.

    All four networks share the same architecture; they are trained
    independently on per-seat replay buffers.
    """
    return {p: GuanZeroQNet(cfg) for p in range(4)}


@dataclasses.dataclass(frozen=True)
class DartQNetConfig:
    """Hyperparameters for the Dart role-aware shared-trunk Q-network."""

    role_d_model: int = 128
    history_hidden: int = 256
    global_hidden: int = 128
    action_hidden: int = 128
    trunk_hidden: int = 1024
    trunk_layers: int = 4
    dropout: float = 0.0


class DartQNet(nn.Module):
    """Role-aware shared trunk with four trick-leader-relative Q heads.

    * No ``seat_emb`` — absolute seat id is dropped entirely. Heads specialize
      on tactical context (leader / 1st responder / across / last responder)
      rather than arbitrary seat labels.
    * ``player_blocks`` is (4, 256): each role gets the existing 252 public-info
      slots plus a 4-dim trick-position one-hot at [252:256].
    * Trunk input is 4*d + history_hidden + global_hidden + action_hidden
      (no `+ d` for seat_emb).
    * Heads are indexed by ``batch["trick_head_id"]``.
    """

    PLAYER_BLOCK_DIM: int = 256

    def __init__(self, cfg: DartQNetConfig) -> None:
        super().__init__()
        d = cfg.role_d_model
        self.role_emb = nn.Embedding(4, d)
        self.player_mlp = nn.Sequential(
            nn.Linear(self.PLAYER_BLOCK_DIM, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.history_lstm = nn.LSTM(
            input_size=CARD_ID_DIM + 4 + 1,
            hidden_size=cfg.history_hidden,
            batch_first=True,
        )
        # State branch: level(13) + own_hand(108) + partner_hand(108)
        # + others_hand(108) + behavior(9) = 346
        self.global_mlp = nn.Sequential(
            nn.Linear(13 + 3 * CARD_ID_DIM + 9, cfg.global_hidden),
            nn.ReLU(),
            nn.Linear(cfg.global_hidden, cfg.global_hidden),
            nn.ReLU(),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(CARD_ID_DIM, cfg.action_hidden),
            nn.ReLU(),
            nn.Linear(cfg.action_hidden, cfg.action_hidden),
            nn.ReLU(),
        )
        trunk_in = 4 * d + cfg.history_hidden + cfg.global_hidden + cfg.action_hidden
        layers: list[nn.Module] = []
        for i in range(cfg.trunk_layers):
            layers.append(nn.Linear(trunk_in if i == 0 else cfg.trunk_hidden, cfg.trunk_hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(cfg.trunk_hidden, 1) for _ in range(4)])

    def _state_features(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = batch["player_blocks"].shape[0]
        role_ids = torch.arange(4, device=batch["player_blocks"].device).expand(b, 4)
        z_p = (self.player_mlp(batch["player_blocks"].float()) + self.role_emb(role_ids)).flatten(1)
        hist_in = torch.cat(
            [
                batch["history_actions"].float(),
                batch["history_roles"].float(),
                batch["history_is_pass"].float(),
            ],
            dim=-1,
        )
        _, (h_n, _) = self.history_lstm(hist_in)
        z_hist = h_n[-1]
        z_g = self.global_mlp(
            torch.cat(
                [
                    batch["global_features"].float(),
                    batch["own_hand"].float(),
                    batch["partner_hand"].float(),
                    batch["others_hand"].float(),
                    batch["behavior"].float(),
                ],
                dim=-1,
            )
        )
        return z_p, z_hist, z_g

    def _action_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.action_mlp(batch["candidate_action"].float())

    def _route_heads(self, shared: torch.Tensor, head_id: torch.Tensor) -> torch.Tensor:
        all_q = torch.stack([head(shared).squeeze(-1) for head in self.heads], dim=1)
        return all_q.gather(1, head_id.long().view(-1, 1)).squeeze(1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return one Q-value per encoded role-aware state-action row."""
        z_p, z_hist, z_g = self._state_features(batch)
        z_a = self._action_features(batch)
        shared = self.trunk(torch.cat([z_p, z_hist, z_g, z_a], dim=-1))
        return self._route_heads(shared, batch["trick_head_id"])

    def forward_grouped(
        self,
        state_batch: dict[str, torch.Tensor],
        action_batch: dict[str, torch.Tensor],
        repeats: torch.Tensor,
    ) -> torch.Tensor:
        """Score flattened candidate actions while encoding each state once."""
        z_p, z_hist, z_g = self._state_features(state_batch)
        repeats = repeats.to(device=z_p.device, dtype=torch.long)
        z_a = self._action_features(action_batch)
        head_id = action_batch["trick_head_id"]
        if z_p.shape[0] == 1:
            return self._forward_grouped_single_state(z_p, z_hist, z_g, z_a, head_id)

        first = self.trunk[0]
        if isinstance(first, nn.Linear):
            state = torch.cat([z_p, z_hist, z_g], dim=-1)
            state_dim = state.shape[-1]
            w = first.weight
            shared = F.linear(state, w[:, :state_dim], first.bias)
            shared = torch.repeat_interleave(shared, repeats, dim=0)
            x = F.linear(z_a, w[:, state_dim:], None)
            x = x + shared
            for layer in list(self.trunk.children())[1:]:
                x = layer(x)
            return self._route_heads(x, head_id)

        z_p_rows = torch.repeat_interleave(z_p, repeats, dim=0)
        z_hist_rows = torch.repeat_interleave(z_hist, repeats, dim=0)
        z_g_rows = torch.repeat_interleave(z_g, repeats, dim=0)
        shared = self.trunk(torch.cat([z_p_rows, z_hist_rows, z_g_rows, z_a], dim=-1))
        return self._route_heads(shared, head_id)

    def _forward_grouped_single_state(
        self,
        z_p: torch.Tensor,
        z_hist: torch.Tensor,
        z_g: torch.Tensor,
        z_a: torch.Tensor,
        head_id: torch.Tensor,
    ) -> torch.Tensor:
        """Fast path for the actor's common one-decision grouped forward.

        The first trunk layer is linear, so the shared state contribution can
        be computed once and added to the per-action contribution.

        Falls through to the expansion path when ``self.trunk[0]`` is not a
        plain nn.Linear (e.g. dynamic-quantized int8).
        """
        first = self.trunk[0]
        K = z_a.shape[0]
        if isinstance(first, nn.Linear):
            state = torch.cat([z_p, z_hist, z_g], dim=-1)
            state_dim = state.shape[-1]
            w = first.weight
            shared = F.linear(state, w[:, :state_dim], first.bias)
            x = F.linear(z_a, w[:, state_dim:], None)
            x = x + shared
            for layer in list(self.trunk.children())[1:]:
                x = layer(x)
        else:
            z_p_exp = z_p.expand(K, -1)
            z_hist_exp = z_hist.expand(K, -1)
            z_g_exp = z_g.expand(K, -1)
            x = self.trunk(torch.cat([z_p_exp, z_hist_exp, z_g_exp, z_a], dim=-1))
        if head_id.device.type == "cpu" and head_id.numel() > 0:
            # Actor inference scores one decision at a time, so every
            # candidate row routes to the same head.
            return self.heads[int(head_id.reshape(-1)[0].item())](x).squeeze(-1)
        return self._route_heads(x, head_id)


__all__ = [
    "GuanZeroQNet",
    "init_guanzero_nets",
    "DartQNetConfig",
    "DartQNet",
]
