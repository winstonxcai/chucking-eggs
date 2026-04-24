"""ImpossibleBot — Tier 1 agent with partner hand visibility.

Uses a frozen base Q-network plus a small trained PartnerAdapter:
  Q_final(s, a) = Q_base(s, a) + PartnerAdapter(partner_hand, action_enc)

The base policy is preserved exactly; the adapter adds partner-awareness on top.
The UI must disclose that this bot sees its partner's cards.
"""

from __future__ import annotations

import numpy as np
import torch

from ..cards import Rank
from ..training.encoding import encode_action, encode_history
from ..training.q_network import QNetworkLSTM, get_device
from ..training.visibility.adapter import PartnerAdapter
from ..training.visibility.encoding import (
    PARTNER_INSERT_POS,
    PARTNER_HAND_DIM,
    STATE_DIM_TIER1,
    encode_state_tier1,
)
from .base import Agent


class ImpossibleBot(Agent):
    """Wraps frozen Tier 1 Q-networks with a trained PartnerAdapter."""

    def __init__(
        self,
        q_lead: QNetworkLSTM,
        q_follow: QNetworkLSTM,
        adapter_lead: PartnerAdapter,
        adapter_follow: PartnerAdapter,
        device: torch.device | None = None,
        level_rank: int = Rank.TWO,
    ):
        self.q_lead = q_lead
        self.q_follow = q_follow
        self.adapter_lead = adapter_lead
        self.adapter_follow = adapter_follow
        self.device = device if device is not None else get_device()
        self.level_rank = level_rank

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device | None = None,
        level_rank: int = Rank.TWO,
    ) -> "ImpossibleBot":
        """Load from a Tier 1 checkpoint (includes both base weights and adapter)."""
        dev = device if device is not None else get_device()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        lstm_hidden = ckpt.get("lstm_hidden", 256)
        mlp_hidden = ckpt.get("mlp_hidden", 1024)
        adapter_hidden = ckpt.get("adapter_hidden", 256)

        q_lead = QNetworkLSTM(d_state=STATE_DIM_TIER1, lstm_hidden=lstm_hidden, hidden=mlp_hidden).to(dev)
        q_follow = QNetworkLSTM(d_state=STATE_DIM_TIER1, lstm_hidden=lstm_hidden, hidden=mlp_hidden).to(dev)
        q_lead.load_state_dict(ckpt["lead"])
        q_follow.load_state_dict(ckpt["follow"])
        q_lead.eval()
        q_follow.eval()

        adapter_lead = PartnerAdapter(d_hist=lstm_hidden, hidden=adapter_hidden).to(dev)
        adapter_follow = PartnerAdapter(d_hist=lstm_hidden, hidden=adapter_hidden).to(dev)
        adapter_lead.load_state_dict(ckpt["adapter_lead"])
        adapter_follow.load_state_dict(ckpt["adapter_follow"])
        adapter_lead.eval()
        adapter_follow.eval()

        return cls(q_lead, q_follow, adapter_lead, adapter_follow, dev, level_rank)

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        is_leading = env.current_trick is None
        q_net = self.q_lead if is_leading else self.q_follow
        adapter = self.adapter_lead if is_leading else self.adapter_follow

        state_enc = encode_state_tier1(env, player)
        hand = env.hands[player]
        action_encs = np.array([encode_action(m, hand, self.level_rank) for m in legal])
        history, hist_len = encode_history(env, player, self.level_rank)

        with torch.no_grad():
            B = len(legal)
            s = torch.tensor(state_enc, device=self.device).unsqueeze(0).expand(B, -1)
            a = torch.tensor(action_encs, device=self.device)
            h = torch.tensor(history, device=self.device).unsqueeze(0).expand(B, -1, -1)
            hl = torch.tensor([hist_len], dtype=torch.long, device=self.device).expand(B)
            hist_emb = q_net.encode_history(h, hl)
            q_base = q_net.forward_from_embedding(s, a, hist_emb)
            partner_hand = s[:, PARTNER_INSERT_POS: PARTNER_INSERT_POS + PARTNER_HAND_DIM]
            q_total = q_base + adapter(partner_hand, a, hist_emb)
            idx = q_total.argmax().item()

        return legal[idx]
