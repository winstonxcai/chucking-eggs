"""Opponent pool for population-based self-play.

Stores past Q-network checkpoints and samples opponents from the pool
to prevent strategy collapse during self-play training.
"""

from __future__ import annotations

import random
from collections import deque


class OpponentPool:
    """Ring buffer of past Q-network state_dicts.

    Parameters:
        max_size: Maximum number of checkpoints to keep.
        self_play_prob: Probability of returning None (use current weights).
    """

    def __init__(self, max_size: int = 10, self_play_prob: float = 0.7):
        self.max_size = max_size
        self.self_play_prob = self_play_prob
        self._pool: deque[tuple[dict, dict]] = deque(maxlen=max_size)

    def add(self, lead_sd: dict, follow_sd: dict) -> None:
        """Add current weights to pool (deep copy, moved to CPU)."""
        cpu_lead = {k: v.cpu().clone() for k, v in lead_sd.items()}
        cpu_follow = {k: v.cpu().clone() for k, v in follow_sd.items()}
        self._pool.append((cpu_lead, cpu_follow))

    def sample(self) -> tuple[dict, dict] | None:
        """Sample opponent weights.

        Returns None for self-play (current weights),
        or (lead_sd, follow_sd) for a pool opponent.
        """
        if not self._pool or random.random() < self.self_play_prob:
            return None
        return random.choice(self._pool)

    def __len__(self) -> int:
        return len(self._pool)
