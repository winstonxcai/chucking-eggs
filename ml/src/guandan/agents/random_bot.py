"""RandomBot — plays a uniformly random legal move."""

from __future__ import annotations

import random

from .base import Agent


class RandomBot(Agent):
    """Floor-level agent. Picks a random legal move every turn."""

    label = "Random"
    description = "Plays completely at random."
    source = "In-house"

    def act(self, env, player: int):
        return random.choice(env.legal_moves(player))
