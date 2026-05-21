"""Load AI agents at startup and provide async inference."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from importlib import resources

from guandan.agents import AGENT_META, AGENT_REGISTRY, Agent, make_agent
from guandan.cards import Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv

_ELOS: dict[str, int] = json.loads(
    resources.files("guandan").joinpath("elos.json").read_text()
)

AGENT_INFO: dict[str, dict] = {
    name: {**meta, "elo": _ELOS.get(name, 1500)}
    for name, meta in AGENT_META.items()
}


class AIService:
    def __init__(self) -> None:
        self.agents: dict[str, Agent] = {}
        self._load_agents()

    def _load_agents(self) -> None:
        for agent_name in AGENT_REGISTRY:
            if agent_name == "dart":
                continue
            self.agents[agent_name] = make_agent(agent_name, level_rank=Rank.TWO)

    def get_agent(self, difficulty: str) -> Agent:
        return self.agents.get(difficulty, self.agents["strategic"])

    def get_agent_name(self, difficulty: str) -> str:
        return difficulty

    async def get_ai_move(self, agent: Agent, env: GuanDanEnv, player: int) -> Combo:
        """Run agent inference in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(agent.act, env, player))

    def pick_bots(self, difficulty: str) -> list[dict]:
        """Return 3 bot info dicts for the AI opponents in a solo game."""
        info = AGENT_INFO[difficulty]
        label, elo = info["label"], info["elo"]
        return [{"name": f"{label} {i + 1}", "elo": elo} for i in range(3)]
