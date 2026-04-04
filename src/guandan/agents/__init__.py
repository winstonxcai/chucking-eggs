"""Agent hierarchy for Guan Dan.

Usage:
    from guandan.agents import make_agent, RandomBot, StrategicBot, MonteCarloBot
    from guandan.agents import RLAgentLSTM

    agent = make_agent("strategic", level_rank=Rank.TWO)
    mc = make_agent("monte_carlo", level_rank=Rank.TWO, n_sims=50, n_workers=8)
    rl = RLAgentLSTM(q_lead, q_follow, device)
    combo = agent.act(env, player)
"""

from __future__ import annotations

from ..cards import Rank
from .base import Agent
from .greedy_bot import GreedyBot
from .heuristic_bot import HeuristicBot
from .monte_carlo_bot import MonteCarloBot
from .random_bot import RandomBot
try:
    from .rl_agent import RLAgentLSTM
except ImportError:
    RLAgentLSTM = None  # type: ignore[assignment,misc]
from .strategic_bot import StrategicBot
from .ez_bot import EzBot
try:
    from .llm_bot import LLMBot
except Exception:
    LLMBot = None  # type: ignore[assignment,misc]
from .hulalala_bot import HulalalaBot
from .jidan_bot import JidanBot
from .lalala_bot import LalalaBot
from .liuzha_bot import LiuzhaBot
from .noai_bot import NoAIBot
from .wjsd_bot import WjsdBot
from .xingdream_bot import XingDreamBot
from .yaoji_bot import YaojiBot

__all__ = [
    "Agent",
    "RandomBot",
    "GreedyBot",
    "HeuristicBot",
    "StrategicBot",
    "MonteCarloBot",
    "RLAgentLSTM",
    "AGENT_REGISTRY",
    "make_agent",
]

AGENT_REGISTRY: dict[str, type[Agent]] = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "heuristic": HeuristicBot,
    "strategic": StrategicBot,
    "monte_carlo": MonteCarloBot,
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

if LLMBot is not None:
    AGENT_REGISTRY["llm"] = LLMBot

try:
    from .guanzero_bot import GuanZeroBot
    AGENT_REGISTRY["guanzero"] = GuanZeroBot
except ImportError:
    GuanZeroBot = None  # type: ignore[assignment,misc]


def make_agent(name: str, level_rank: int = Rank.TWO, **kwargs) -> Agent:
    """Create an agent by name.

    Extra kwargs are forwarded to MonteCarloBot (n_sims, n_workers).
    """
    cls = AGENT_REGISTRY[name]
    if name == "random":
        return cls()
    if name in ("monte_carlo", "llm"):
        return cls(level_rank=level_rank, **kwargs)
    return cls(level_rank=level_rank)
