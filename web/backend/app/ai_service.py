"""Load AI agents at startup and provide async inference."""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

import torch

from guandan.agents import Agent, RLAgentLSTM, make_agent
from guandan.cards import Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv
from guandan.training.q_network import QNetworkLSTM, get_device


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
    "expert": "rl",  # falls back to strategic if no checkpoint
}


class AIService:
    def __init__(self) -> None:
        self.agents: dict[str, Agent] = {}
        self.rl_checkpoint_name: str = "strategic_fallback"
        self._load_agents()

    def _load_agents(self) -> None:
        """Load rule-based agents, then try loading RL checkpoint."""
        for agent_name in ("greedy", "heuristic", "strategic"):
            self.agents[agent_name] = make_agent(agent_name, level_rank=Rank.TWO)
        self._try_load_rl_agent()

    def _try_load_rl_agent(self) -> None:
        checkpoint_path = Path(__file__).resolve().parents[3] / "checkpoints" / "selfplay_best.pt"
        if not checkpoint_path.exists():
            print(f"No RL checkpoint at {checkpoint_path}, expert uses strategic fallback")
            return
        device = get_device()
        q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
        q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        lead_key = "lead_state_dict" if "lead_state_dict" in ckpt else "lead"
        follow_key = "follow_state_dict" if "follow_state_dict" in ckpt else "follow"
        try:
            q_lead.load_state_dict(ckpt[lead_key], strict=False)
            q_follow.load_state_dict(ckpt[follow_key], strict=False)
        except RuntimeError as e:
            print(f"RL checkpoint incompatible ({e}), expert uses strategic fallback")
            return
        q_lead.eval()
        q_follow.eval()
        self.agents["rl"] = RLAgentLSTM(q_lead, q_follow, device)
        ep = ckpt.get("episode", "?")
        self.rl_checkpoint_name = f"selfplay_best_ep{ep}"
        print(f"Loaded RL agent from {checkpoint_path} (ep {ep})")

    def get_agent(self, difficulty: str) -> Agent:
        agent_name = DIFFICULTY_TO_AGENT[difficulty]
        if agent_name not in self.agents:
            agent_name = "strategic"
        return self.agents[agent_name]

    def get_agent_name(self, difficulty: str) -> str:
        agent_name = DIFFICULTY_TO_AGENT[difficulty]
        if agent_name == "rl":
            return self.rl_checkpoint_name
        return agent_name

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
