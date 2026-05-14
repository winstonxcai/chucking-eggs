"""Distributed actor-learner orchestration.

Process modules:
- ``train``            — top-level orchestrator (entry point via ``main``)
- ``learner``          — unified ``learner_loop`` + ``LearnerProtocol``
- ``learner_seat``     — ``SeatLearner`` (per-seat Q-nets, paper §4.2)
- ``learner_shared``   — ``SharedHeadLearner`` (shared trunk, active variant)
- ``loss_buckets``     — per-batch metric stratification helpers
- ``weight_publish``   — atomic weight publish + read helpers
- ``worker``           — per-actor subprocess; ``actor_loop`` target
- ``actor``            — single-episode self-play helpers used by the worker
- ``inference_server`` — optional shared-GPU inference server
"""

from .actor import argmax_q, argmax_q_role, play_episode, select_legal
from .train import main, train

__all__ = [
    "argmax_q",
    "argmax_q_role",
    "main",
    "play_episode",
    "select_legal",
    "train",
]
