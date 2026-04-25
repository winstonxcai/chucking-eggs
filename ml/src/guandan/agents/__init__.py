"""Rule-based agent hierarchy for Guan Dan.

Usage:
    from guandan.agents import make_agent, RandomBot, StrategicBot

    agent = make_agent("strategic", level_rank=Rank.TWO)
    combo = agent.act(env, player)

All RL/ML/search agents are archived under ``ml/_archive/`` as of 2026-04-24.
See ``ml/LOGBOOK.md`` for the chronology of prior training attempts.
"""

from __future__ import annotations

from ..cards import Rank
from .base import Agent
from .ez_bot import EzBot
from .greedy_bot import GreedyBot
from .heuristic_bot import HeuristicBot
from .hulalala_bot import HulalalaBot
from .jidan_bot import JidanBot
from .lalala_bot import LalalaBot
from .liuzha_bot import LiuzhaBot
from .noai_bot import NoAIBot
from .random_bot import RandomBot
from .strategic_bot import StrategicBot
from .wjsd_bot import WjsdBot
from .xingdream_bot import XingDreamBot
from .yaoji_bot import YaojiBot

__all__ = [
    "Agent",
    "AGENT_REGISTRY",
    "make_agent",
    "RandomBot",
    "GreedyBot",
    "HeuristicBot",
    "StrategicBot",
    "EzBot",
    "XingDreamBot",
    "NoAIBot",
    "LalalaBot",
    "LiuzhaBot",
    "HulalalaBot",
    "YaojiBot",
    "JidanBot",
    "WjsdBot",
]

AGENT_REGISTRY: dict[str, type[Agent]] = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "heuristic": HeuristicBot,
    "strategic": StrategicBot,
    "xingdream": XingDreamBot,
    "noai": NoAIBot,
    "lalala": LalalaBot,
    "liuzha": LiuzhaBot,
    "hulalala": HulalalaBot,
    "yaoji": YaojiBot,
    "jidan": JidanBot,
    "ez": EzBot,
    "wjsd": WjsdBot,
}


def make_agent(name: str, level_rank: int = Rank.TWO, **kwargs) -> Agent:
    cls = AGENT_REGISTRY[name]
    if name == "random":
        return cls()
    return cls(level_rank=level_rank)
