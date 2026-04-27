"""ActorCriticNet for the partner-visible PTIE experiment.

Actor head:  (state[764] ⊕ action[198]) → logit. One logit per candidate.
Critic head: state_critic[875] → V(s). Action-independent.

Both heads are 4-layer MLPs. Critic final layer is zero-initialised so
V(s) = 0 at iter 0 — gradients still flow (only the last layer is zeroed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .encoders import ACTION_DIM, ACTOR_DIM, CRITIC_DIM


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _MLP(nn.Module):
    """4-layer MLP: in → hidden → hidden → hidden → 1."""

    def __init__(self, d_in: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ActorCriticNet(nn.Module):
    """Shared parameter store for actor and critic heads.

    actor_head  : (state ⊕ action) [B, ACTOR_DIM + ACTION_DIM] → logit [B]
    critic_head : state_critic      [B, CRITIC_DIM]            → V     [B]

    The critic's final linear layer is zero-initialised so V=0 at cold-start,
    but hidden-layer weights are Xavier-normal so gradients flow from iter 1.
    """

    def __init__(
        self,
        d_state_actor: int = ACTOR_DIM,
        d_action: int = ACTION_DIM,
        d_state_critic: int = CRITIC_DIM,
        hidden: int = 256,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_state_actor  = d_state_actor
        self.d_action       = d_action
        self.d_state_critic = d_state_critic
        self.temperature    = temperature

        self.actor_head  = _MLP(d_state_actor + d_action, hidden)
        self.critic_head = _MLP(d_state_critic, hidden)

        # Zero-init only the final layer of the critic
        nn.init.zeros_(self.critic_head.net[-1].weight)
        nn.init.zeros_(self.critic_head.net[-1].bias)

    def score_actions(
        self,
        state_actor: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """state_actor: [B, d_state_actor], actions: [B, d_action] → logit [B]."""
        return self.actor_head(torch.cat([state_actor, actions], dim=-1))

    def value(self, state_critic: torch.Tensor) -> torch.Tensor:
        """state_critic: [B, d_state_critic] → V [B]."""
        return self.critic_head(state_critic)

    def policy_logits(
        self,
        state_actor: torch.Tensor,
        actions: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked logits (illegal actions → -inf). Temperature applied.

        state_actor: [K, d_state_actor] — K candidates for one decision.
        actions:     [K, d_action]
        legal_mask:  [K] bool
        Returns:     [K] logits with illegal actions set to -1e9.
        """
        logits = self.score_actions(state_actor, actions) / self.temperature
        logits = logits.masked_fill(~legal_mask, -1e9)
        return logits


def load_warmstart(
    net: ActorCriticNet,
    checkpoint_path: str | Path,
    map_location: Any = "cpu",
) -> dict:
    """Load actor head weights from a distillation checkpoint.

    Critic head is left at its zero-init state (trained from scratch during PPO).
    Returns the checkpoint metadata dict.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    net.actor_head.load_state_dict(ckpt["actor_state_dict"])
    return ckpt.get("metadata", {})
