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
from typing import Any, Literal

import torch
import torch.nn as nn

CheckpointSaveType = Literal["weight", "full"]


def _validate_checkpoint_save_type(save_type: str) -> CheckpointSaveType:
    if save_type not in ("weight", "full"):
        raise ValueError(
            f"checkpoint_save_type must be 'weight' or 'full'; got {save_type!r}"
        )
    return save_type


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
    *,
    learner_state: dict | None = None,
    replay_state: dict | None = None,
    rng_state: dict | None = None,
    save_type: CheckpointSaveType = "full",
) -> None:
    """Save Q-net state dicts + config to ``path``.

    ``cfg`` must be a dataclass instance (``TrainConfig`` or any sub-config).
    ``episode_or_update`` is stored under the ``"episode"`` key for backwards
    compatibility with existing checkpoint readers.
    """
    save_type = _validate_checkpoint_save_type(save_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format_version": 2,
        "checkpoint_save_type": save_type,
        "episode": episode_or_update,
        "total_updates": episode_or_update,
        "config": dataclasses.asdict(cfg),
        "q_nets": {
            p: unwrap_compiled(q_nets[p]).state_dict()
            for p in range(4)
        },
    }
    if save_type == "weight":
        torch.save(payload, path)
        return
    if learner_state is not None:
        payload["learner_state"] = learner_state
    if replay_state is not None:
        payload["replay_state"] = replay_state
    if rng_state is not None:
        payload["rng_state"] = rng_state
    torch.save(payload, path)


save_checkpoint = save_checkpoint_base


def save_checkpoint_dart(
    path: Path,
    q_net: nn.Module,
    cfg: Any,
    episode_or_update: int,
    *,
    learner_state: dict | None = None,
    replay_state: dict | None = None,
    rng_state: dict | None = None,
    save_type: CheckpointSaveType = "full",
) -> None:
    """Save one Dart Q-net state dict plus config to ``path``."""
    save_type = _validate_checkpoint_save_type(save_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format_version": 2,
        "checkpoint_save_type": save_type,
        "episode": episode_or_update,
        "total_updates": episode_or_update,
        "config": dataclasses.asdict(cfg),
        "q_net": unwrap_compiled(q_net).state_dict(),
    }
    if save_type == "weight":
        torch.save(payload, path)
        return
    if learner_state is not None:
        payload["learner_state"] = learner_state
    if replay_state is not None:
        payload["replay_state"] = replay_state
    if rng_state is not None:
        payload["rng_state"] = rng_state
    torch.save(payload, path)


def load_checkpoint(path: Path | str) -> dict:
    """Load and return the raw checkpoint dict from ``path``.

    The returned dict contains ``"episode"``, ``"config"``, and ``"q_nets"``
    keys as written by ``save_checkpoint``.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def load_frozen_dart_qnet(
    path: str | Path,
    qnet_cfg: Any,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build a DartQNet from ``qnet_cfg`` and load frozen weights from ``path``.

    Returns the network in ``eval()`` mode with all parameters frozen
    (``requires_grad=False``). Used by actors to instantiate population
    opponents whose weights never change during the run.
    """
    from .q_network import DartQNet

    net = DartQNet(qnet_cfg).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
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
    "CheckpointSaveType",
    "unwrap_compiled",
    "save_checkpoint_base",
    "save_checkpoint",
    "save_checkpoint_dart",
    "load_checkpoint",
    "load_frozen_dart_qnet",
    "WeightSnapshot",
]
