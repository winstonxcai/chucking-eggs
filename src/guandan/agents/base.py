"""Base class for all Guan Dan agents."""

from __future__ import annotations


class Agent:
    """Base class for all Guan Dan agents."""

    def act(self, env, player: int):
        """Given game environment and player seat, return a Combo to play."""
        raise NotImplementedError
