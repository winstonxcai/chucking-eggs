"""RLAgentLSTM — wraps trained Q-networks in the Agent interface.

Supports both single-network and lead/follow-split modes:
  - q_follow=None  → single-network mode (same net for lead and follow)
  - q_follow=<net> → split mode (separate nets for lead and follow)

Single-network mode is a cleaner ablation for measuring the LSTM contribution
in isolation. Split mode is the full production setup.
"""

from __future__ import annotations

import numpy as np
import torch

from ..cards import Rank
from ..encoding import encode_action, encode_history, encode_state
from ..q_network import get_device
from .base import Agent


class RLAgentLSTM(Agent):
    """Wraps q_lead (+ optional q_follow) Q-networks in the Agent interface.

    Parameters
    ----------
    q_lead:     Q-network for leading decisions (or the single network).
    q_follow:   Q-network for following decisions. If None, uses q_lead
                for both (single-network / ablation mode).
    device:     Torch device. Defaults to get_device().
    level_rank: Wild card rank for this game.
    """

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
        """Choose the highest-Q legal move given current game state."""
        legal = env.legal_moves(player)

        # Trivial: only one option
        if len(legal) == 1:
            return legal[0]

        is_leading = env.current_trick is None
        q_net = self.q_lead if is_leading else self.q_follow

        state_enc = encode_state(env, player)
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
