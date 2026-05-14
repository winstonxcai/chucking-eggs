"""Atomic weight publishing and reading for the distributed actor-learner loop.

Actors poll ``weight_dir/latest.txt`` for the current version number, then
load ``weight_dir/weights_{version}.pt``.  Both writes are made atomic via
``os.replace`` (POSIX rename — no partial-read window).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import torch

from ..model.checkpoint import WeightSnapshot
from ..model.q_network import SharedHeadQNet, SharedTrickHeadQNet


def publish_weights(q_nets: dict, weight_dir: Path, version: int, updates: int = 0) -> None:
    """Atomically write per-seat Q-net weights to disk."""
    weight_dir.mkdir(parents=True, exist_ok=True)
    tmp   = weight_dir / f"weights_{version}.tmp"
    final = weight_dir / f"weights_{version}.pt"

    torch.save(
        {
            "version": version,
            "updates": updates,
            "state_dicts": {
                p: {k: v.detach().cpu()
                    for k, v in getattr(q_nets[p], "_orig_mod", q_nets[p]).state_dict().items()}
                for p in range(4)
            },
        },
        tmp,
    )
    os.replace(tmp, final)

    ver_tmp = weight_dir / "latest.tmp"
    ver_tmp.write_text(f"{version} {updates}")
    os.replace(ver_tmp, weight_dir / "latest.txt")

    for stale in weight_dir.glob("weights_*.pt"):
        if stale != final:
            stale.unlink(missing_ok=True)
    for stale in weight_dir.glob("weights_*.tmp"):
        stale.unlink(missing_ok=True)


def publish_weights_shared(
    q_net: Union[SharedHeadQNet, SharedTrickHeadQNet],
    weight_dir: Path,
    version: int,
    updates: int = 0,
) -> None:
    """Atomically write one shared-head Q-net snapshot to disk."""
    weight_dir.mkdir(parents=True, exist_ok=True)
    tmp   = weight_dir / f"weights_{version}.tmp"
    final = weight_dir / f"weights_{version}.pt"
    torch.save(
        {
            "version": version,
            "updates": updates,
            "state_dicts": {
                "shared": {
                    k: v.detach().cpu()
                    for k, v in getattr(q_net, "_orig_mod", q_net).state_dict().items()
                }
            },
        },
        tmp,
    )
    os.replace(tmp, final)

    ver_tmp = weight_dir / "latest.tmp"
    ver_tmp.write_text(f"{version} {updates}")
    os.replace(ver_tmp, weight_dir / "latest.txt")

    for stale in weight_dir.glob("weights_*.pt"):
        if stale != final:
            stale.unlink(missing_ok=True)
    for stale in weight_dir.glob("weights_*.tmp"):
        stale.unlink(missing_ok=True)


def read_latest_metadata(weight_dir: Path) -> tuple[int, int] | None:
    """Cheaply read (version, updates) from ``latest.txt`` without torch.load.

    Returns ``None`` if no published weights yet, or the file is unreadable.
    Format: ``"<version> <updates>"`` (post-2026-05). Falls back to bare int
    for backward compatibility with old runs that wrote just the version.
    """
    ver_path = weight_dir / "latest.txt"
    if not ver_path.exists():
        return None
    try:
        parts = ver_path.read_text().strip().split()
        version = int(parts[0])
        updates = int(parts[1]) if len(parts) > 1 else 0
        return version, updates
    except (ValueError, IndexError):
        return None


def load_latest_weights(weight_dir: Path) -> WeightSnapshot | None:
    """Read the latest published version and state dicts.

    Returns ``None`` if weights have not been published yet or the latest
    snapshot is temporarily unreadable during an atomic replacement.
    """
    meta = read_latest_metadata(weight_dir)
    if meta is None:
        return None
    version, _ = meta
    try:
        payload = torch.load(
            weight_dir / f"weights_{version}.pt",
            map_location="cpu",
            weights_only=True,
        )
        return WeightSnapshot(
            version=int(payload["version"]),
            state_dicts=payload["state_dicts"],
            updates=int(payload.get("updates", 0)),
        )
    except Exception:
        return None


__all__ = [
    "publish_weights",
    "publish_weights_shared",
    "read_latest_metadata",
    "load_latest_weights",
]
