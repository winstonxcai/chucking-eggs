"""PVGuanBot — deployable agent for the partner-visible PTIE experiment.

Uses only the actor head at inference. Critic is discarded.
Plug-in compatible with ml/scripts/eval/bots.py via Agent.act interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from ..agents.base import Agent
from ..agents.partner_oracle_bot import _reflect_env
from ..cards import Rank
from ..game import GuanDanEnv
from .actor_critic import ActorCriticNet, get_device, load_warmstart
from .encoders import ACTION_DIM, ACTOR_DIM, encode_action, encode_actor_pair_features
from .legal_utils import dedup_strategic


class PVGuanBot(Agent):
    """Partner-visible actor. Critic not used at inference.

    Args:
        checkpoint_path: path to a pvguan checkpoint (.pt).
        level_rank:       current game level (used for encoding).
        sample:           if True, sample from the policy; else argmax.
        temperature:      softmax temperature at eval time (default 1.0).
        device:           torch device override.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        level_rank: int = Rank.TWO,
        sample: bool = False,
        temperature: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.level_rank  = level_rank
        self.sample      = sample
        self.temperature = temperature
        self.device      = device or get_device()

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        meta = ckpt.get("metadata", {})

        d_state_actor  = meta.get("actor_dim",  ACTOR_DIM)
        d_action       = meta.get("action_dim", ACTION_DIM)
        hidden         = meta.get("hidden",     256)

        self.net = ActorCriticNet(
            d_state_actor=d_state_actor,
            d_action=d_action,
            hidden=hidden,
            temperature=temperature,
        ).to(self.device)
        self.net.actor_head.load_state_dict(ckpt["actor_state_dict"])
        self.net.eval()

    @classmethod
    def from_net(
        cls,
        net: ActorCriticNet,
        level_rank: int = Rank.TWO,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> "PVGuanBot":
        """Wrap a live training net (no checkpoint I/O). Net is used as-is —
        the caller is responsible for net.eval() / net.train() bookkeeping."""
        bot = cls.__new__(cls)
        bot.level_rank  = level_rank
        bot.sample      = sample
        bot.temperature = temperature
        bot.device      = next(net.parameters()).device
        bot.net         = net
        return bot

    def act(self, env: GuanDanEnv, player: int):
        """Choose a legal action. Returns a Combo."""
        # Reflect env so team {player, partner} appears as {0, 2}
        canonical_player = player
        if player in (1, 3):
            env = _reflect_env(env)
            canonical_player = player ^ 1  # 1→0, 3→2

        legal = dedup_strategic(env.legal_moves(canonical_player))
        if len(legal) == 1:
            return legal[0]  # no choice

        hand = list(env.hands[canonical_player])

        state_actor_list = []
        action_list = []
        for move in legal:
            sf = encode_actor_pair_features(env, canonical_player, move, legal)
            af = encode_action(move, hand, env.level_rank)
            state_actor_list.append(sf)
            action_list.append(af)

        sa = torch.tensor(
            np.array(state_actor_list, dtype=np.float32), device=self.device
        )  # [K, ACTOR_DIM]
        ac = torch.tensor(
            np.array(action_list, dtype=np.float32), device=self.device
        )  # [K, ACTION_DIM]
        mask = torch.ones(len(legal), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits = self.net.policy_logits(sa, ac, mask)  # [K]
            if self.sample:
                idx = torch.distributions.Categorical(logits=logits).sample().item()
            else:
                idx = logits.argmax().item()

        return legal[int(idx)]
