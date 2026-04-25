"""Load AI agents at startup and provide async inference."""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

from guandan.agents import Agent, make_agent
from guandan.cards import Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv


# Bot personalities per difficulty.
# ELOs are calibrated Glicko-2 ratings from a 13-bot round-robin WR matrix
# (200 games/matchup, 31,200 total). See ml/_archive/ for the analysis scripts.
BOT_POOLS = {
    "wjsd": [
        {"name": "Wjsd", "avatar": "dragon", "elo": 1212},
    ],
    "liuzha": [
        {"name": "Liuzha", "avatar": "dragon", "elo": 1260},
    ],
    "hulalala": [
        {"name": "Hulalala", "avatar": "dragon", "elo": 1264},
    ],
    "easy": [
        {"name": "Koala", "avatar": "koala", "elo": 1415},
        {"name": "Turtle", "avatar": "turtle", "elo": 1415},
        {"name": "Lamb", "avatar": "lamb", "elo": 1415},
    ],
    "competition": [
        {"name": "Lalala", "avatar": "dragon", "elo": 1464},
    ],
    "casual": [
        {"name": "Panda", "avatar": "panda", "elo": 1523},
        {"name": "Owl", "avatar": "owl", "elo": 1523},
        {"name": "Cat", "avatar": "cat", "elo": 1523},
    ],
    "hard": [
        {"name": "Tiger", "avatar": "tiger", "elo": 1621},
        {"name": "Falcon", "avatar": "falcon", "elo": 1621},
        {"name": "Leopard", "avatar": "leopard", "elo": 1621},
    ],
    "master": [
        {"name": "NoAI", "avatar": "dragon", "elo": 1726},
    ],
    "yaoji": [
        {"name": "Yaoji", "avatar": "dragon", "elo": 1772},
    ],
    "jidan": [
        {"name": "Jidan", "avatar": "dragon", "elo": 1779},
    ],
}

DIFFICULTY_TO_AGENT = {
    "easy": "greedy",
    "wjsd": "wjsd",           # SAU 3rd Prize (2020 NJUPT)
    "casual": "xingdream",
    "hard": "strategic",
    "competition": "lalala",  # SEU 1st Prize (Li Jing)
    "yaoji": "yaoji",         # NUAA 3rd Prize (2020 NJUPT)
    "jidan": "jidan",             # NUAA 2nd Prize (2020 NJUPT)
    "hulalala": "hulalala",       # SEU 3rd Prize (2020 NJUPT)
    "liuzha": "liuzha",           # SEU 2nd Prize (2020 NJUPT)
    "master": "noai",             # Fudan 2nd Prize (Chen Yuguan)
}


class AIService:
    def __init__(self) -> None:
        self.agents: dict[str, Agent] = {}
        self._load_agents()

    def _load_agents(self) -> None:
        for agent_name in ("greedy", "xingdream", "heuristic", "strategic",
                           "lalala", "noai", "wjsd", "yaoji", "jidan",
                           "hulalala", "liuzha"):
            self.agents[agent_name] = make_agent(agent_name, level_rank=Rank.TWO)


    def get_agent(self, difficulty: str) -> Agent:
        agent_name = DIFFICULTY_TO_AGENT[difficulty]
        if agent_name not in self.agents:
            agent_name = "strategic"
        return self.agents[agent_name]

    def get_agent_name(self, difficulty: str) -> str:
        return DIFFICULTY_TO_AGENT[difficulty]

    async def get_ai_move(self, agent: Agent, env: GuanDanEnv, player: int) -> Combo:
        """Run agent inference in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(agent.act, env, player))

    def pick_bots(self, difficulty: str) -> list[dict]:
        """Pick 3 random bot personalities for a game."""
        import random
        pool = BOT_POOLS[difficulty]
        if len(pool) >= 3:
            return random.sample(pool, 3)
        base = pool[0]
        return [dict(base, name=f"{base['name']} {i + 1}") for i in range(3)]
