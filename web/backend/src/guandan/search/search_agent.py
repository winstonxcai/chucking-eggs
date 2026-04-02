"""SearchAgent — drop-in replacement for RLAgentLSTM with lookahead."""

from __future__ import annotations

import torch

from ..agents.base import Agent
from ..cards import Rank
from ..training.q_network import QNetworkLSTM, get_device
from .search import q_search


class SearchAgent(Agent):
    """Wraps Q-networks with determinized search for stronger play."""

    def __init__(
        self,
        q_lead: QNetworkLSTM,
        q_follow: QNetworkLSTM | None = None,
        device: torch.device | None = None,
        level_rank: int = Rank.TWO,
        n_worlds: int = 20,
        depth: int = 3,
        prune_top_k: int = 5,
        opp_bot: Agent | None = None,
    ):
        self.q_lead = q_lead
        self.q_follow = q_follow if q_follow is not None else q_lead
        self.device = device if device is not None else get_device()
        self.level_rank = level_rank
        self.n_worlds = n_worlds
        self.depth = depth
        self.prune_top_k = prune_top_k
        self.opp_bot = opp_bot

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        move, info = q_search(
            env, player, self.q_lead, self.q_follow,
            self.device, self.level_rank,
            self.n_worlds, self.depth, self.prune_top_k,
            opp_bot=self.opp_bot,
        )
        return move
