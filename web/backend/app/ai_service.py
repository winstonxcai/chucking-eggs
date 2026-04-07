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
# (200 games/matchup, 31,200 total). See runs/wr_matrix_v2/results.json.
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
    "impossible": [
        {"name": "Oracle", "avatar": "crystal_ball", "elo": 1850},
        {"name": "Seer", "avatar": "crystal_ball", "elo": 1850},
        {"name": "Prophet", "avatar": "crystal_ball", "elo": 1850},
    ],
}

DIFFICULTY_TO_AGENT = {
    "easy": "greedy",
    "wjsd": "wjsd",           # SAU 3rd Prize (2020 NJUPT)
    "casual": "xingdream",
    "hard": "strategic",
    "competition": "lalala",  # SEU 1st Prize (Li Jing)
    "yaoji": "yaoji",         # NUAA 3rd Prize (2020 NJUPT)
    "jidan": "jidan",         # NUAA 2nd Prize (2020 NJUPT)
    "hulalala": "hulalala",   # SEU 3rd Prize (2020 NJUPT)
    "liuzha": "liuzha",       # SEU 2nd Prize (2020 NJUPT)
    "master": "noai",         # Fudan 2nd Prize (Chen Yuguan)
    "impossible": "impossible",  # Tier 1: sees partner's hand
}


class AIService:
    def __init__(self) -> None:
        self.agents: dict[str, Agent] = {}
        self.rl_checkpoint_name: str = "strategic_fallback"
        self._load_agents()

    def _load_agents(self) -> None:
        """Load rule-based agents."""
        for agent_name in ("greedy", "xingdream", "heuristic", "strategic",
                           "lalala", "noai", "wjsd", "yaoji", "jidan",
                           "hulalala", "liuzha"):
            self.agents[agent_name] = make_agent(agent_name, level_rank=Rank.TWO)
        self._try_load_impossible_agent()

    def _try_load_impossible_agent(self) -> None:
        """Load Tier 1 (partner-visible) agent if checkpoint exists."""
        checkpoints_dir = Path(__file__).resolve().parents[3] / "ml" / "checkpoints"
        tier1_files = sorted(checkpoints_dir.glob("tier1_*.pt"))
        if not tier1_files:
            return
        checkpoint_path = tier1_files[-1]
        try:
            import torch
            from guandan.agents.impossible_bot import ImpossibleBot
            from guandan.training.q_network import QNetworkLSTM, get_device
            from guandan.training.visibility.encoding import STATE_DIM_TIER1
        except ImportError as e:
            print(f"Tier 1 dependencies unavailable ({e})")
            return
        device = get_device()
        q_lead = QNetworkLSTM(d_state=STATE_DIM_TIER1, lstm_hidden=256, hidden=1024).to(device)
        q_follow = QNetworkLSTM(d_state=STATE_DIM_TIER1, lstm_hidden=256, hidden=1024).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        lead_key = "lead_state_dict" if "lead_state_dict" in ckpt else "lead"
        follow_key = "follow_state_dict" if "follow_state_dict" in ckpt else "follow"
        try:
            q_lead.load_state_dict(ckpt[lead_key])
            q_follow.load_state_dict(ckpt[follow_key])
        except RuntimeError as e:
            print(f"Tier 1 checkpoint incompatible ({e})")
            return
        q_lead.eval()
        q_follow.eval()
        self.agents["impossible"] = ImpossibleBot(q_lead, q_follow, device)
        print(f"Loaded Tier 1 (impossible) agent from {checkpoint_path}")

    def _try_load_rl_agent(self) -> None:
        checkpoints_dir = Path(__file__).resolve().parents[3] / "ml" / "checkpoints"
        prod_files = sorted(checkpoints_dir.glob("prod_*.pt"))
        checkpoint_path = prod_files[-1] if prod_files else checkpoints_dir / "selfplay_best.pt"
        if not checkpoint_path.exists():
            print(f"No RL checkpoint at {checkpoint_path}, expert uses strategic fallback")
            return
        try:
            import torch
            from guandan.agents import RLAgentLSTM
            from guandan.training.q_network import QNetworkLSTM, get_device
        except ImportError as e:
            print(f"RL dependencies unavailable ({e}), expert uses strategic fallback")
            return
        device = get_device()
        q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024, use_gnn=True, gnn_out=128).to(device)
        q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024, use_gnn=True, gnn_out=128).to(device)
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
        self.rl_checkpoint_name = checkpoint_path.stem
        print(f"Loaded RL agent from {checkpoint_path}")

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
        # Single-entry pool: number each instance so they're distinguishable
        base = pool[0]
        return [dict(base, name=f"{base['name']} {i + 1}") for i in range(3)]
