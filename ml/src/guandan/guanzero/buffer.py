"""Per-player FIFO replay buffer of (encoded_dict, mc_return) samples.

DMC samples are correlated within an episode but not within a player's
own subsequence (each timestep only ever produces one Q target). We keep
four independent buffers — one per seat — so the learner can sample a
balanced minibatch even when a single episode contributed unevenly to
each seat.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from .returns import TrainSample


class ReplayBuffer:
    """Per-player circular FIFO buffer with O(1) random access for sampling.

    Backed by a fixed-size Python list per player + write pointer, so
    ``random.sample`` reads are O(k) instead of O(k*n) (deque indexing is
    O(n) per access — was the dominant cost on the learner hot path).
    """

    def __init__(self, capacity_per_player: int = 50_000) -> None:
        self.capacity = capacity_per_player
        self.buffers: dict[int, list[TrainSample | None]] = {
            p: [None] * capacity_per_player for p in range(4)
        }
        self.write_idx: dict[int, int] = {p: 0 for p in range(4)}
        self.sizes: dict[int, int] = {p: 0 for p in range(4)}

    def _append(self, s: TrainSample) -> None:
        p = s.player
        idx = self.write_idx[p]
        self.buffers[p][idx] = s
        self.write_idx[p] = (idx + 1) % self.capacity
        if self.sizes[p] < self.capacity:
            self.sizes[p] += 1

    def push(self, samples: list[TrainSample]) -> None:
        for s in samples:
            self._append(s)

    def push_many(self, samples: list[TrainSample]) -> None:
        """Alias for push; used by the learner to ingest actor batches."""
        self.push(samples)

    def push_stacked(
        self,
        stacked: dict[str, np.ndarray],
        players: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        """Ingest a pre-stacked actor batch by splitting per player."""
        for p in range(4):
            mask = players == p
            if not mask.any():
                continue
            for i in np.flatnonzero(mask):
                enc = {k: stacked[k][i] for k in stacked}
                self._append(TrainSample(p, enc, float(returns[i])))

    def total_size(self) -> int:
        return sum(self.sizes.values())

    def size(self, player: int) -> int:
        return self.sizes[player]

    def sample_for_player(self, player: int, batch_size: int) -> list[TrainSample]:
        n = self.sizes[player]
        if n == 0:
            return []
        k = min(batch_size, n)
        # random.sample on a range is O(k); list indexing is O(1).
        indices = random.sample(range(n), k)
        buf = self.buffers[player]
        return [buf[i] for i in indices]


def collate(samples: list[TrainSample], device: str | torch.device = "cpu") -> tuple[
    dict[str, torch.Tensor], torch.Tensor
]:
    """Stack a list of TrainSample dicts into batched torch tensors.

    Note on pin_memory: when the learner runs the model under
    ``with torch.cuda.stream(s)`` the H2D copy and the compute serialize on
    the same stream, so non_blocking has no overlap to exploit. Pinning
    pages is pure overhead in that pattern, so we skip it.
    """
    keys = samples[0].encoded.keys()
    batch: dict[str, torch.Tensor] = {}
    for k in keys:
        arr = np.stack([s.encoded[k] for s in samples], axis=0)
        batch[k] = torch.from_numpy(arr).to(device)
    targets_arr = np.asarray([s.mc_return for s in samples], dtype=np.float32)
    targets = torch.from_numpy(targets_arr).to(device)
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
