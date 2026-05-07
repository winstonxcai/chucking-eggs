"""Exploration schedules for GuanZero training.

Each schedule function is a pure function of (episode, config) — no state.
Additional schedules (exponential, piecewise) can be added here as siblings.
"""

from __future__ import annotations

from .config import EpsilonConfig


def epsilon_linear(episode: int, cfg: EpsilonConfig) -> float:
    """Linear decay from ``cfg.start`` to ``cfg.final`` over ``cfg.decay_episodes``."""
    if cfg.decay_episodes <= 0:
        return cfg.final
    frac = min(1.0, episode / cfg.decay_episodes)
    return cfg.start + frac * (cfg.final - cfg.start)


__all__ = ["epsilon_linear"]
