"""Agent hierarchy for Guan Dan.

Usage:
    from guandan.agents import make_agent, RandomBot, StrategicBot

    agent = make_agent("strategic", level_rank=Rank.TWO)
    combo = agent.act(env, player)
"""

from __future__ import annotations

from ..cards import Rank
from .base import Agent
from .greedy_bot import GreedyBot
from .heuristic_bot import HeuristicBot
from .random_bot import RandomBot
from .strategic_bot import StrategicBot

__all__ = [
    "Agent",
    "RandomBot",
    "GreedyBot",
    "HeuristicBot",
    "StrategicBot",
    "AGENT_REGISTRY",
    "make_agent",
]

AGENT_REGISTRY: dict[str, type[Agent]] = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "heuristic": HeuristicBot,
    "strategic": StrategicBot,
}


def make_agent(name: str, level_rank: int = Rank.TWO) -> Agent:
    """Create an agent by name."""
    cls = AGENT_REGISTRY[name]
    if name == "random":
        return cls()
    return cls(level_rank=level_rank)
