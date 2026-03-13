"""Agent hierarchy for Guan Dan.

Usage:
    from guandan.agents import make_agent, RandomBot, StrategicBot, MonteCarloBot

    agent = make_agent("strategic", level_rank=Rank.TWO)
    mc = make_agent("monte_carlo", level_rank=Rank.TWO, n_sims=50, n_workers=8)
    combo = agent.act(env, player)
"""

from __future__ import annotations

from ..cards import Rank
from .base import Agent
from .greedy_bot import GreedyBot
from .heuristic_bot import HeuristicBot
from .monte_carlo_bot import MonteCarloBot
from .random_bot import RandomBot
from .strategic_bot import StrategicBot

__all__ = [
    "Agent",
    "RandomBot",
    "GreedyBot",
    "HeuristicBot",
    "StrategicBot",
    "MonteCarloBot",
    "AGENT_REGISTRY",
    "make_agent",
]

AGENT_REGISTRY: dict[str, type[Agent]] = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "heuristic": HeuristicBot,
    "strategic": StrategicBot,
    "monte_carlo": MonteCarloBot,
}


def make_agent(name: str, level_rank: int = Rank.TWO, **kwargs) -> Agent:
    """Create an agent by name.

    Extra kwargs are forwarded to MonteCarloBot (n_sims, n_workers).
    """
    cls = AGENT_REGISTRY[name]
    if name == "random":
        return cls()
    if name == "monte_carlo":
        return cls(level_rank=level_rank, **kwargs)
    return cls(level_rank=level_rank)
