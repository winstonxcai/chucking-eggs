"""Rule-based agent hierarchy for Guan Dan.

Usage:
    from guandan.agents import make_agent, RandomBot, StrategicBot

    agent = make_agent("strategic", level_rank=Rank.TWO)
    combo = agent.act(env, player)

Use ``make_agent("dart", checkpoint=<path>)`` to load a trained DART or
GuanZero comparison checkpoint. See ``ml/LOGBOOK.md`` for the chronology of
prior training attempts.
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
    "AGENT_META",
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

AGENT_META: dict[str, dict] = {
    name: {
        "label": cls.label,
        "description": cls.description,
        "source": cls.source,
        **({"award": cls.award} if cls.award is not None else {}),
    }
    for name, cls in AGENT_REGISTRY.items()
}


def make_agent(
    name: str,
    level_rank: int = Rank.TWO,
    checkpoint: str | None = None,
    **kwargs,
) -> Agent:
    if name == "dart":
        if checkpoint is None:
            raise ValueError("make_agent('dart') requires a checkpoint= path")
        from guandan.dart.agent import DartBot  # lazy: avoids torch import for rule-based runs
        return DartBot.load(checkpoint)
    if name == "llm":
        from .llm_bot import LLMBot  # lazy: avoids litellm import for rule-based runs
        if "model" not in kwargs:
            raise ValueError("make_agent('llm') requires a model= name (e.g. 'gpt-4o-mini')")
        return LLMBot(level_rank=level_rank, **kwargs)
    cls = AGENT_REGISTRY[name]
    if name == "random":
        return cls()
    return cls(level_rank=level_rank)
