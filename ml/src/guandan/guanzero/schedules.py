"""Exploration schedules for GuanZero training.

Each schedule function is a pure function of (episode, config) — no state.
Additional schedules (exponential, piecewise) can be added here as siblings.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class EpsilonConfig:
    """Parameters for a linear epsilon-greedy decay schedule.

    Defined here as a lightweight standalone; it will be folded into the
    unified ``TrainConfig`` during the Phase I nesting refactor.
    """
    start: float = 0.1
    final: float = 0.01
    decay_episodes: int = 15_000


def epsilon_linear(episode: int, cfg: "TrainConfig") -> float:  # type: ignore[name-defined]
    """Linear epsilon decay from ``cfg.epsilon_start`` to ``cfg.epsilon_final``.

    Accepts a full ``TrainConfig`` (flat form) so the caller doesn't need to
    construct an ``EpsilonConfig`` — the fields are read by name.
    """
    decay = cfg.epsilon_decay_episodes
    if decay <= 0:
        return cfg.epsilon_final
    frac = min(1.0, episode / decay)
    return cfg.epsilon_start + frac * (cfg.epsilon_final - cfg.epsilon_start)


__all__ = ["epsilon_linear", "EpsilonConfig"]
