"""Reproducibility helpers for Dart training processes."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


_NP_SEED_MOD = 2**32


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs in the current process."""
    random.seed(seed)
    np.random.seed(seed % _NP_SEED_MOD)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_torch_rng_state() -> dict[str, Any]:
    """Capture Torch RNG state for exact learner resume."""
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def capture_actor_rng_state(rng: random.Random) -> dict[str, Any]:
    """Capture all actor-side RNG state used by rollout selection."""
    return {
        "actor_random": rng.getstate(),
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": capture_torch_rng_state(),
    }


def restore_torch_rng_state(state: dict[str, Any] | None) -> None:
    """Restore a state produced by ``capture_torch_rng_state``."""
    if not state:
        return
    cpu = state.get("cpu")
    if cpu is not None:
        torch.set_rng_state(cpu)
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)


def restore_actor_rng_state(rng: random.Random, state: dict[str, Any] | None) -> None:
    """Restore a state produced by ``capture_actor_rng_state``."""
    if not state:
        return
    actor_random = state.get("actor_random")
    if actor_random is not None:
        rng.setstate(actor_random)
    python_random = state.get("python_random")
    if python_random is not None:
        random.setstate(python_random)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    restore_torch_rng_state(state.get("torch"))


__all__ = [
    "seed_everything",
    "capture_actor_rng_state",
    "capture_torch_rng_state",
    "restore_actor_rng_state",
    "restore_torch_rng_state",
]
