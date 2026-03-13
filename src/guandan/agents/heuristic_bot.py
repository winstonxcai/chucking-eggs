"""HeuristicBot — wraps existing HeuristicAgent with the Agent interface."""

from __future__ import annotations

from ..cards import Rank
from ..heuristic import HeuristicAgent
from .base import Agent


class HeuristicBot(Agent):
    """Rule-based agent with hand decomposition and partner awareness."""

    def __init__(self, level_rank: int = Rank.TWO):
        self._inner = HeuristicAgent(level_rank)
        self.level_rank = level_rank

    def act(self, env, player: int):
        return self._inner.choose_action(env, player)
