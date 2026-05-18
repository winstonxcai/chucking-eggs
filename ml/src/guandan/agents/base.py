"""Base class for all Guan Dan agents."""

from __future__ import annotations


class Agent:
    """Base class for all Guan Dan agents."""

    label: str = ""
    description: str = ""
    source: str = ""
    color: str = "#777777"
    award: str | None = None
    sample_tag: int = 0       # opponent_id written to EpisodeTags; 0 = untagged
    coord_target: bool = False  # True = losing against this bot triggers coord training bucket

    def act(self, env, player: int):
        """Given game environment and player seat, return a Combo to play."""
        raise NotImplementedError
