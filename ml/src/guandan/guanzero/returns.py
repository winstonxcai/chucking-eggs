"""Monte Carlo return computation for DMC training.

GuanZero is sparse-reward: the only signal is the team-signed terminal
reward at episode end. With ``gamma=1.0`` the MC return G_t is the
terminal reward for *every* timestep on that player's trajectory.

We still expose ``gamma`` so future ablations can experiment with
discounted returns (e.g. shaped intermediate rewards).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypedDict

import numpy as np


# Engine ``get_rewards()`` returns ±3 / ±2 / ±1 depending on team finish
# order. Normalize by 3 so MC targets fall in [-1, 1] — matches the scale
# the LSTM+MLP learns most stably.
TERMINAL_REWARD_SCALE = 3.0


class TrajectoryStep(TypedDict):
    """One timestep produced by the actor rollout, before MC return is assigned."""
    player: int
    encoded: dict[str, np.ndarray]


@dataclass(slots=True, frozen=True)
class TrainSample:
    player: int
    encoded: dict[str, np.ndarray]
    mc_return: float


def normalize_terminal_rewards(rewards: Mapping[int, float]) -> dict[int, float]:
    return {p: r / TERMINAL_REWARD_SCALE for p, r in rewards.items()}


def compute_mc_returns(
    trajectory: list[TrajectoryStep],
    terminal_rewards: Mapping[int, float],
    gamma: float = 1.0,
) -> list[TrainSample]:
    """Compute G_t per (player, timestep).

    ``trajectory`` is a list of ``{"player": int, "encoded": dict}`` ordered
    by play time. ``terminal_rewards`` is the engine output; it gets
    normalized to [-1, 1] internally.

    Per-player accumulation walks each player's own subsequence backward.
    With gamma=1.0 (paper default for sparse terminal reward) every step
    on a given player's trajectory takes their normalized terminal reward.
    """
    norm = normalize_terminal_rewards(terminal_rewards)

    by_player: dict[int, list[int]] = {p: [] for p in range(4)}
    for idx, step in enumerate(trajectory):
        by_player[step["player"]].append(idx)

    returns = [0.0] * len(trajectory)
    for p, idxs in by_player.items():
        terminal = norm[p]
        g = terminal  # last step on p's trajectory carries the full reward
        for i in reversed(idxs):
            returns[i] = g
            g = gamma * g  # zero intermediate rewards in DMC

    return [
        TrainSample(
            player=trajectory[i]["player"],
            encoded=trajectory[i]["encoded"],
            mc_return=returns[i],
        )
        for i in range(len(trajectory))
    ]


__all__ = [
    "TrajectoryStep",
    "TrainSample",
    "compute_mc_returns",
    "normalize_terminal_rewards",
    "TERMINAL_REWARD_SCALE",
]
