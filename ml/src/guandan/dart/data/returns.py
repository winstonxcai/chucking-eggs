"""Monte Carlo return computation for DMC training.

Dart is sparse-reward: the only signal is the team-signed terminal
reward at episode end. With ``gamma=1.0`` the MC return G_t is the
terminal reward for *every* timestep on that player's trajectory.

We still expose ``gamma`` so future ablations can experiment with
discounted returns (e.g. shaped intermediate rewards).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from ..constants import NUM_PLAYERS

# Engine ``get_rewards()`` returns ±3 / ±2 / ±1 depending on team finish
# order. Normalize by 3 so MC targets fall in [-1, 1] — matches the scale
# the LSTM+MLP learns most stably.
TERMINAL_REWARD_SCALE = 3.0


class TrajectoryStep(TypedDict, total=False):
    """One timestep produced by the actor rollout, before MC return is assigned."""
    player: int
    encoded: dict[str, np.ndarray]
    # Per-sample diagnostic tags computed at decision time in play_episode.
    phase_self: int
    trick_role: int
    phase_partner: int
    action_type: int
    is_pass: int
    is_bomb: int
    bomb_available: int
    num_legal_actions: int
    q_gap: float
    chosen_by_epsilon: int


@dataclass(slots=True, frozen=True)
class EpisodeTags:
    """Episode-level metadata broadcast verbatim to every emitted TrainSample.

    Computed once per episode in the worker (based on curriculum / opponent
    mixing) and applied to all steps by ``compute_mc_returns``.
    """
    mode:        int = 0  # EPISODE_MODE_* — see sample_tags.py
    opponent_id: int = 0  # OPPONENT_* — see sample_tags.py
    latest_team: int = 0  # 0 or 1 — which team is the "latest" policy


@dataclass(slots=True, frozen=True)
class TrainSample:
    player: int
    encoded: dict[str, np.ndarray]
    mc_return: float
    # Per-sample tags (carried from TrajectoryStep):
    phase_self: int = 0
    trick_role: int = 0
    phase_partner: int = 0
    action_type: int = 0
    is_pass: int = 0
    is_bomb: int = 0
    bomb_available: int = 0
    num_legal_actions: int = 0
    q_gap: float = float("nan")
    chosen_by_epsilon: int = 0
    # Per-episode / per-player tags (broadcast in compute_mc_returns):
    episode_mode: int = 0
    opponent_id: int = 0
    latest_team: int = 0
    terminal_reward: float = 0.0


def normalize_terminal_rewards(rewards: Mapping[int, float]) -> dict[int, float]:
    return {p: r / TERMINAL_REWARD_SCALE for p, r in rewards.items()}


def compute_mc_returns(
    trajectory: list[TrajectoryStep],
    terminal_rewards: Mapping[int, float],
    gamma: float = 1.0,
    *,
    tags: EpisodeTags | None = None,
) -> list[TrainSample]:
    """Compute G_t per (player, timestep).

    ``trajectory`` is a list of step dicts ordered by play time.
    ``terminal_rewards`` is the engine output; it gets normalized to
    [-1, 1] for ``mc_return`` but the raw value is also carried on each
    sample as ``terminal_reward`` for diagnostic bucketing.

    ``tags`` are episode-level — broadcast verbatim to every emitted sample.
    """
    tags = tags or EpisodeTags()
    norm = normalize_terminal_rewards(terminal_rewards)

    by_player: dict[int, list[int]] = {p: [] for p in range(NUM_PLAYERS)}
    for idx, step in enumerate(trajectory):
        by_player[step["player"]].append(idx)

    returns = [0.0] * len(trajectory)
    for p, idxs in by_player.items():
        terminal = norm[p]
        g = terminal
        for i in reversed(idxs):
            returns[i] = g
            g = gamma * g

    out: list[TrainSample] = []
    for i in range(len(trajectory)):
        step = trajectory[i]
        p = step["player"]
        out.append(TrainSample(
            player=p,
            encoded=step["encoded"],
            mc_return=returns[i],
            phase_self=int(step.get("phase_self", 0)),
            trick_role=int(step.get("trick_role", 0)),
            phase_partner=int(step.get("phase_partner", 0)),
            action_type=int(step.get("action_type", 0)),
            is_pass=int(step.get("is_pass", 0)),
            is_bomb=int(step.get("is_bomb", 0)),
            bomb_available=int(step.get("bomb_available", 0)),
            num_legal_actions=int(step.get("num_legal_actions", 0)),
            q_gap=float(step.get("q_gap", float("nan"))),
            chosen_by_epsilon=int(step.get("chosen_by_epsilon", 0)),
            episode_mode=int(tags.mode),
            opponent_id=int(tags.opponent_id),
            latest_team=int(tags.latest_team),
            terminal_reward=float(terminal_rewards[p]),
        ))
    return out


__all__ = [
    "EpisodeTags",
    "TrajectoryStep",
    "TrainSample",
    "compute_mc_returns",
    "normalize_terminal_rewards",
    "TERMINAL_REWARD_SCALE",
]
