"""Per-player FIFO replay buffer of (encoded_dict, mc_return) samples.

DMC samples are correlated within an episode but not within a player's
own subsequence (each timestep only ever produces one Q target). We keep
four independent buffers — one per seat — so the learner can sample a
balanced minibatch even when a single episode contributed unevenly to
each seat.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

from .returns import TrainSample


class ReplayBuffer:
    def __init__(self, capacity_per_player: int = 50_000) -> None:
        self.buffers: dict[int, deque[TrainSample]] = {
            p: deque(maxlen=capacity_per_player) for p in range(4)
        }

    def push(self, samples: list[TrainSample]) -> None:
        for s in samples:
            self.buffers[s.player].append(s)

    def push_many(self, samples: list[TrainSample]) -> None:
        """Alias for push; used by the learner to ingest actor batches."""
        self.push(samples)

    def total_size(self) -> int:
        return sum(len(b) for b in self.buffers.values())

    def size(self, player: int) -> int:
        return len(self.buffers[player])

    def sample_for_player(self, player: int, batch_size: int) -> list[TrainSample]:
        buf = self.buffers[player]
        if not buf:
            return []
        n = min(batch_size, len(buf))
        return random.sample(buf, n)


def collate(samples: list[TrainSample], device: str | torch.device = "cpu") -> tuple[
    dict[str, torch.Tensor], torch.Tensor
]:
    """Stack a list of TrainSample dicts into batched torch tensors."""
    keys = samples[0].encoded.keys()
    batch: dict[str, torch.Tensor] = {}
    for k in keys:
        arr = np.stack([s.encoded[k] for s in samples], axis=0)
        batch[k] = torch.from_numpy(arr).to(device)
    targets = torch.tensor(
        [s.mc_return for s in samples], dtype=torch.float32, device=device
    )
    return batch, targets


def collate_encoded(
    encoded_list: list[dict[str, np.ndarray]],
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Same as ``collate`` but for raw encoded dicts (no MC return). Used
    by actor / eval to score legal actions in a single forward pass."""
    keys = encoded_list[0].keys()
    batch: dict[str, torch.Tensor] = {}
    for k in keys:
        arr = np.stack([e[k] for e in encoded_list], axis=0)
        batch[k] = torch.from_numpy(arr).to(device)
    return batch


__all__ = ["ReplayBuffer", "collate", "collate_encoded"]
