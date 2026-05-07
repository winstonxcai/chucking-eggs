"""Per-player FIFO replay buffer with contiguous-array storage.

Storage layout (per player):
    own_hand:                  uint8  [capacity, 108]
    others_hand:               uint8  [capacity, 108]
    recent_action_each_player: uint8  [capacity, 4, 108]
    played_cards_others:       uint8  [capacity, 3, 108]
    remaining_counts_others:   uint8  [capacity, 3, 27]
    level:                     uint8  [capacity, 13]
    history:                   uint8  [capacity, 20, 108]
    behavior:                  uint8  [capacity, 9]
    candidate_action:          uint8  [capacity, 108]
    returns:                   float32 [capacity]

Why this shape:
    * The encoder produces 0/1 binary multi-hot/one-hot tensors. uint8 stores
      them exactly with 4× less memory + 4× less PCIe bandwidth on H2D vs
      float32. Cast back to float32 happens on GPU after H2D — cheap.
    * Sampling = one np.random.randint + fancy-indexing per key, then a single
      H2D copy per key. No per-sample numpy stacks (the previous bottleneck —
      profiling showed collate+H2D was 73% of update wall time without this).

DMC samples are correlated within an episode but not within a player's own
subsequence. We keep four independent buffers (one per seat) so the learner
can sample a balanced minibatch even when an episode contributed unevenly.
"""

from __future__ import annotations

import numpy as np
import torch

from .encoder import ENCODE_CHANNEL_KEYS, ENCODE_CHANNEL_SHAPES
from .returns import TrainSample

_KEY_SHAPES = ENCODE_CHANNEL_SHAPES
_KEYS = ENCODE_CHANNEL_KEYS

Batch = tuple[dict[str, torch.Tensor], torch.Tensor]


class ReplayBuffer:
    """Per-player circular FIFO buffer of contiguous uint8 arrays.

    Hot paths:
      * ``push_stacked`` — actor → buffer ingest, ~0 cost beyond the index
        write (no per-sample object construction).
      * ``sample_batch_for_player`` — buffer → GPU sampling. Single fancy-index
        per key, single H2D per key, uint8→float32 cast on GPU.
    """

    def __init__(self, capacity_per_player: int = 50_000) -> None:
        self.capacity = capacity_per_player
        # {player: {key: ndarray[capacity, *shape] uint8}}
        self.fields: dict[int, dict[str, np.ndarray]] = {
            p: {
                k: np.zeros((capacity_per_player, *shape), dtype=np.uint8)
                for k, shape in _KEY_SHAPES.items()
            }
            for p in range(4)
        }
        self.returns: dict[int, np.ndarray] = {
            p: np.zeros(capacity_per_player, dtype=np.float32) for p in range(4)
        }
        self.write_idx: dict[int, int] = {p: 0 for p in range(4)}
        self.sizes: dict[int, int] = {p: 0 for p in range(4)}

    # ─── introspection ───────────────────────────────────────

    def total_size(self) -> int:
        return sum(self.sizes.values())

    def size(self, player: int) -> int:
        return self.sizes[player]

    def clear(self) -> None:
        """Reset all write pointers and sizes without reallocating storage."""
        for p in range(4):
            self.write_idx[p] = 0
            self.sizes[p] = 0

    # ─── ingest ──────────────────────────────────────────────

    def push(self, samples: list[TrainSample]) -> None:
        """Legacy single-sample path. Routes through push_stacked.

        Tolerates partial encoded dicts (e.g. test fixtures): keys not present
        in any sample are left unchanged in the backing store.
        """
        if not samples:
            return
        encoded = [s.encoded for s in samples]
        present_keys = [k for k in _KEYS if k in encoded[0]]
        stacked = {k: np.stack([d[k] for d in encoded], axis=0) for k in present_keys}
        players = np.asarray([s.player for s in samples], dtype=np.int8)
        returns = np.asarray([s.mc_return for s in samples], dtype=np.float32)
        self.push_stacked(stacked, players, returns)

    def push_stacked(
        self,
        stacked: dict[str, np.ndarray],
        players: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        """Append a pre-stacked actor batch, splitting per player.

        ``stacked[k]`` has shape ``[N, *_KEY_SHAPES[k]]`` (any dtype convertible
        to uint8 — encoder gives float32 0/1, we cast at write time).
        ``players[N]`` and ``returns[N]`` align with the leading dim of each
        stacked array.
        """
        cap = self.capacity
        for p in range(4):
            mask = players == p
            n = int(mask.sum())
            if n == 0:
                continue
            src_idx = np.flatnonzero(mask)
            start = self.write_idx[p]
            # Circular destination indices
            dst_idx = (start + np.arange(n)) % cap
            for k in stacked:  # only update keys actually provided
                self.fields[p][k][dst_idx] = stacked[k][src_idx].astype(np.uint8, copy=False)
            self.returns[p][dst_idx] = returns[src_idx]
            self.write_idx[p] = (start + n) % cap
            self.sizes[p] = min(self.sizes[p] + n, cap)

    # ─── sampling (hot path) ─────────────────────────────────

    def sample_batch_for_player(
        self,
        player: int,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
        """Sample a batch and push to ``device`` as float32. None if cold.

        One ``np.random.randint`` + one fancy-index per key + one async H2D
        per key. uint8 → float32 cast happens on the GPU side, so the wire
        carries 4× less data than the previous float32 collate path.
        """
        n = self.sizes[player]
        if n < batch_size:
            return None
        # randint with replacement is ~3× faster than random.sample for our
        # batch sizes and the off-policy DMC objective is replacement-tolerant.
        idx = np.random.randint(0, n, size=batch_size)
        batch: dict[str, torch.Tensor] = {}
        for k in _KEYS:
            arr = self.fields[player][k][idx]  # uint8 [B, *shape]
            t = torch.from_numpy(arr).to(device, non_blocking=True)
            batch[k] = t.float()  # GPU-side cast, free relative to PCIe copy
        targets = torch.from_numpy(self.returns[player][idx]).to(
            device, non_blocking=True
        )
        return batch, targets

    # ─── sampling (legacy path for tests) ────────────────────

    def sample_for_player(self, player: int, batch_size: int) -> list[TrainSample]:
        """Legacy path: returns list[TrainSample] for tests/debugging.

        Reconstructs TrainSample objects from the contiguous backing store —
        slow, allocates per sample. The hot path is ``sample_batch_for_player``.
        """
        n = self.sizes[player]
        if n == 0:
            return []
        k = min(batch_size, n)
        idx = np.random.choice(n, size=k, replace=False)
        out: list[TrainSample] = []
        for i in idx:
            i_int = int(i)
            enc = {
                key: self.fields[player][key][i_int].astype(np.float32)
                for key in _KEYS
            }
            out.append(TrainSample(player, enc, float(self.returns[player][i_int])))
        return out


def collate_encoded(
    encoded_list: list[dict[str, np.ndarray]],
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Collate raw encoded dicts (no MC return). Used by the actor at decision
    time to score legal actions in a single forward pass — distinct from the
    learner's replay sampling path.

    ``non_blocking=True`` matters on MPS: the default ``.to(device)`` issues a
    per-call stream sync, which dominates actor wall time on the hot path
    (~1.7 ms/decision for 9 small H2D copies vs ~0.2 ms when async). The
    consuming forward pass is on the same stream, so ordering is preserved.
    """
    keys = encoded_list[0].keys()
    batch: dict[str, torch.Tensor] = {}
    for k in keys:
        arr = np.stack([e[k] for e in encoded_list], axis=0)
        batch[k] = torch.from_numpy(arr).to(device, non_blocking=True)
    return batch


__all__ = ["ReplayBuffer", "collate_encoded", "Batch"]
