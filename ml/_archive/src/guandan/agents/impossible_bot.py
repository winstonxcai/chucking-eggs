"""ImpossibleBot — Tier 1 agent with partner hand visibility.

Identical to RLAgentLSTM except uses encode_state_tier1 (477 dims)
which includes the partner's hand. The UI must disclose that this
bot sees its partner's cards.
"""

from __future__ import annotations

import numpy as np
import torch

from ..cards import Rank
from ..training.encoding import encode_action, encode_history
from ..training.q_network import get_device
from ..training.visibility.encoding import encode_state_tier1
from .base import Agent


class ImpossibleBot(Agent):
    """Wraps Tier 1 Q-networks with partner hand visibility."""

    def __init__(
        self,
        q_lead,
        q_follow=None,
        device: torch.device | None = None,
        level_rank: int = Rank.TWO,
    ):
        self.q_lead = q_lead
        self.q_follow = q_follow if q_follow is not None else q_lead
        self.device = device if device is not None else get_device()
        self.level_rank = level_rank

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        is_leading = env.current_trick is None
        q_net = self.q_lead if is_leading else self.q_follow

        state_enc = encode_state_tier1(env, player)
        hand = env.hands[player]
        action_encs = np.array(
            [encode_action(m, hand, self.level_rank) for m in legal]
        )
        history, hist_len = encode_history(env, player, self.level_rank)

        with torch.no_grad():
            B = len(legal)
            s = (
                torch.tensor(state_enc, device=self.device)
                .unsqueeze(0)
                .expand(B, -1)
            )
            a = torch.tensor(action_encs, device=self.device)
            h = (
                torch.tensor(history, device=self.device)
                .unsqueeze(0)
                .expand(B, -1, -1)
            )
            hl = torch.tensor(
                [hist_len], dtype=torch.long, device=self.device
            ).expand(B)
            idx = q_net(s, a, h, hl).argmax().item()

        return legal[idx]
