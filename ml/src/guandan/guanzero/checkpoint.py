"""Checkpoint I/O and weight-snapshot utilities.

All code that reads or writes model state to disk lives here. Both the
training entry point and the learner subprocess use ``save_checkpoint``;
actors and the inference server use ``WeightSnapshot`` (via
``load_latest_weights`` in learner.py, which returns one of these).
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import torch


def unwrap_compiled(net: torch.nn.Module) -> torch.nn.Module:
    """Return the original module, stripping any torch.compile wrapper."""
    return getattr(net, "_orig_mod", net)


def save_checkpoint(
    path: Path,
    q_nets: dict[int, Any],
    cfg: Any,
    episode_or_update: int,
) -> None:
    """Save Q-net state dicts + config to ``path``.

    ``cfg`` must be a dataclass instance (``TrainConfig`` or any sub-config).
    ``episode_or_update`` is stored under the ``"episode"`` key for backwards
    compatibility with existing checkpoint readers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode_or_update,
            "config": dataclasses.asdict(cfg),
            "q_nets": {p: unwrap_compiled(q_nets[p]).state_dict() for p in range(4)},
        },
        path,
    )


def load_checkpoint(path: Path | str) -> dict:
    """Load and return the raw checkpoint dict from ``path``.

    The returned dict contains ``"episode"``, ``"config"``, and ``"q_nets"``
    keys as written by ``save_checkpoint``.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


@dataclasses.dataclass
class WeightSnapshot:
    """A versioned snapshot of published Q-net weights.

    Produced by ``learner.py:publish_weights``, consumed by actors (via
    ``maybe_sync_weights``) and the inference server (disk-poll refresh thread).
    """
    version: int
    state_dicts: dict[int, dict]


__all__ = ["save_checkpoint", "load_checkpoint", "unwrap_compiled", "WeightSnapshot"]
