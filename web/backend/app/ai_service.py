"""Load AI agents at startup and provide async inference."""

from __future__ import annotations

import asyncio
from functools import partial

from guandan.agents import Agent, make_agent
from guandan.cards import Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv


# Bot personalities per difficulty
BOT_POOLS = {
    "easy": [
        {"name": "Koala", "avatar": "koala", "elo": 600},
        {"name": "Turtle", "avatar": "turtle", "elo": 600},
        {"name": "Lamb", "avatar": "lamb", "elo": 600},
    ],
    "medium": [
        {"name": "Fox", "avatar": "fox", "elo": 1000},
        {"name": "Raccoon", "avatar": "raccoon", "elo": 1000},
        {"name": "Wolf", "avatar": "wolf", "elo": 1000},
    ],
    "hard": [
        {"name": "Tiger", "avatar": "tiger", "elo": 1350},
        {"name": "Falcon", "avatar": "falcon", "elo": 1350},
        {"name": "Leopard", "avatar": "leopard", "elo": 1350},
    ],
    "expert": [
        {"name": "Dragon", "avatar": "dragon", "elo": 1700},
    ],
}

DIFFICULTY_TO_AGENT = {
    "easy": "greedy",
    "medium": "heuristic",
    "hard": "strategic",
    "expert": "strategic",  # fallback if no checkpoint
}


class AIService:
    def __init__(self) -> None:
        self.agents: dict[str, Agent] = {}
        self._load_agents()

    def _load_agents(self) -> None:
        """Load rule-based agents. Neural agent loaded separately if checkpoint exists."""
        for agent_name in ("greedy", "heuristic", "strategic"):
            self.agents[agent_name] = make_agent(agent_name, level_rank=Rank.TWO)

    def get_agent(self, difficulty: str) -> Agent:
        agent_name = DIFFICULTY_TO_AGENT[difficulty]
        return self.agents[agent_name]

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
        # If pool is small (expert has 1), repeat
        return [random.choice(pool) for _ in range(3)]
