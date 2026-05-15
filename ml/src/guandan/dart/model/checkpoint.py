"""Checkpoint I/O and weight-snapshot utilities.

All code that reads or writes model state to disk lives here. Both the
training entry point and the learner subprocess use ``save_checkpoint_base``;
actors and the inference server use ``WeightSnapshot`` (via
``load_latest_weights`` in learner.py, which returns one of these).
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def unwrap_compiled(module: nn.Module) -> nn.Module:
    """Return the underlying module from a torch.compile() wrapper, or the module as-is.

    torch.compile() wraps the original module in an OptimizedModule and stores
    the original in ``._orig_mod``. State dicts must be extracted from the
    original to remain loadable without torch.compile.
    """
    return getattr(module, "_orig_mod", module)


def save_checkpoint_base(
    path: Path,
    q_nets: dict[int, nn.Module],
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
            "q_nets": {
                p: unwrap_compiled(q_nets[p]).state_dict()
                for p in range(4)
            },
        },
        path,
    )


save_checkpoint = save_checkpoint_base


def save_checkpoint_shared(
    path: Path,
    q_net: nn.Module,
    cfg: Any,
    episode_or_update: int,
) -> None:
    """Save shared-head Q-net state dict + config to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode_or_update,
            "config": dataclasses.asdict(cfg),
            "q_net": unwrap_compiled(q_net).state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path | str) -> dict:
    """Load and return the raw checkpoint dict from ``path``.

    The returned dict contains ``"episode"``, ``"config"``, and ``"q_nets"``
    keys as written by ``save_checkpoint``.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def load_frozen_shared_qnet(
    path: str | Path,
    qnet_cfg: Any,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build a SharedHeadQNet from ``qnet_cfg`` and load frozen weights from ``path``.

    Returns the network in ``eval()`` mode with all parameters frozen
    (``requires_grad=False``). Used by actors to instantiate population
    opponents whose weights never change during the run.
    """
    # Local import — q_network depends on torch but not on this module, so
    # keeping the import lazy avoids any future circular-import surprises.
    from .q_network import SharedHeadQNet

    net = SharedHeadQNet(qnet_cfg).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["q_net"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def load_frozen_trick_qnet(
    path: str | Path,
    qnet_cfg: Any,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build a SharedTrickHeadQNet from ``qnet_cfg`` and load frozen weights from ``path``.

    Sibling of ``load_frozen_shared_qnet`` for the ``shared_trick_heads`` model
    type. Returns the network in ``eval()`` mode with all parameters frozen.
    """
    from .q_network import SharedTrickHeadQNet

    net = SharedTrickHeadQNet(qnet_cfg).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["q_net"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@dataclasses.dataclass
class WeightSnapshot:
    """A versioned snapshot of published Q-net weights.

    Produced by ``learner.py:publish_weights``, consumed by actors (via
    ``maybe_sync_weights``) and the inference server (disk-poll refresh thread).
    """
    version: int
    state_dicts: dict[int | str, dict]
    updates: int = 0  # learner update count at publish time; used by actors for epsilon decay


__all__ = [
    "unwrap_compiled",
    "save_checkpoint_base",
    "save_checkpoint",
    "save_checkpoint_shared",
    "load_checkpoint",
    "load_frozen_shared_qnet",
    "load_frozen_trick_qnet",
    "WeightSnapshot",
]
