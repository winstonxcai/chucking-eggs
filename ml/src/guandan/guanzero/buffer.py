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

from .encoder import (
    ENCODE_ACTION_KEYS,
    ENCODE_CHANNEL_KEYS,
    ENCODE_CHANNEL_SHAPES,
    ENCODE_STATE_KEYS,
)
from .encoding.role_encoder import (
    ROLE_ENCODE_ACTION_KEYS,
    ROLE_ENCODE_CHANNEL_KEYS,
    ROLE_ENCODE_CHANNEL_SHAPES,
    ROLE_ENCODE_STATE_KEYS,
)
from .returns import TrainSample

BUCKET_NAMES = ["general_self_play", "hard_bot_general", "coordination_endgame"]
BUCKET_IDS: dict[str, int] = {name: i for i, name in enumerate(BUCKET_NAMES)}
# 0 = general_self_play, 1 = hard_bot_general, 2 = coordination_endgame

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
        """Test/debug single-sample path. Routes through ``push_stacked``.

        Encoded samples must contain the full canonical encoder schema. Missing
        channels raise ``KeyError`` instead of silently leaving stale values in
        the backing store.
        """
        if not samples:
            return
        encoded = [s.encoded for s in samples]
        stacked = {k: np.stack([d[k] for d in encoded], axis=0) for k in _KEYS}
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


class RoleAwareReplayBuffer:
    """Single circular replay buffer for the shared-head role-aware model."""

    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = capacity
        self.fields: dict[str, np.ndarray] = {}
        for k, shape in ROLE_ENCODE_CHANNEL_SHAPES.items():
            dtype = np.int8 if k == "seat_id" else np.uint8
            self.fields[k] = np.zeros((capacity, *shape), dtype=dtype)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.bucket = np.zeros(capacity, dtype=np.int8)
        self.ptr = 0
        self.full = False

    def size(self) -> int:
        """Total samples currently stored."""
        return self.capacity if self.full else self.ptr

    def size_by_seat(self) -> dict[int, int]:
        """Return stored sample counts by absolute seat."""
        n = self.size()
        if n == 0:
            return {p: 0 for p in range(4)}
        seats = self.fields["seat_id"][:n].astype(np.int64, copy=False)
        return {p: int(np.sum(seats == p)) for p in range(4)}

    def push_stacked(
        self,
        stacked: dict[str, np.ndarray],
        returns: np.ndarray,
        buckets: np.ndarray | None = None,
    ) -> None:
        """Write a batch of N role-encoded samples."""
        n = int(len(returns))
        if n == 0:
            return
        buckets_to_write = (
            buckets.astype(np.int8, copy=False)
            if buckets is not None
            else np.zeros(n, dtype=np.int8)
        )
        if n >= self.capacity:
            start = n - self.capacity
            for k in ROLE_ENCODE_CHANNEL_KEYS:
                self.fields[k][:] = stacked[k][start:].astype(self.fields[k].dtype, copy=False)
            self.returns[:] = returns[start:].astype(np.float32, copy=False)
            self.bucket[:] = buckets_to_write[start:]
            self.ptr = 0
            self.full = True
            return

        start_ptr = self.ptr
        end = start_ptr + n
        if end <= self.capacity:
            dst = slice(start_ptr, end)
            src = slice(None)
            for k in ROLE_ENCODE_CHANNEL_KEYS:
                self.fields[k][dst] = stacked[k][src].astype(self.fields[k].dtype, copy=False)
            self.returns[dst] = returns[src].astype(np.float32, copy=False)
            self.bucket[dst] = buckets_to_write[src]
        else:
            first = self.capacity - start_ptr
            second = n - first
            for k in ROLE_ENCODE_CHANNEL_KEYS:
                arr = stacked[k].astype(self.fields[k].dtype, copy=False)
                self.fields[k][start_ptr:] = arr[:first]
                self.fields[k][:second] = arr[first:]
            ret = returns.astype(np.float32, copy=False)
            self.returns[start_ptr:] = ret[:first]
            self.returns[:second] = ret[first:]
            self.bucket[start_ptr:] = buckets_to_write[:first]
            self.bucket[:second] = buckets_to_write[first:]
        self.ptr = (self.ptr + n) % self.capacity
        self.full = self.full or end >= self.capacity

    def sample_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Uniformly sample a role-aware batch."""
        n = self.size()
        if n < batch_size:
            raise ValueError(f"buffer has {n} samples, need {batch_size}")
        idx = np.random.randint(0, n, size=batch_size)
        return self._sample_indices(idx, device)

    def sample_batch_balanced(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Stratified sample with approximately equal rows per seat."""
        n = self.size()
        seats = self.fields["seat_id"][:n].astype(np.int64, copy=False)
        counts = [batch_size // 4] * 4
        for p in range(batch_size % 4):
            counts[p] += 1
        idx_parts = []
        for p, count in enumerate(counts):
            seat_idx = np.flatnonzero(seats == p)
            if len(seat_idx) < count:
                raise ValueError(f"seat {p} has {len(seat_idx)} samples, need {count}")
            idx_parts.append(np.random.choice(seat_idx, size=count, replace=True))
        idx = np.concatenate(idx_parts)
        np.random.shuffle(idx)
        return self._sample_indices(idx, device)

    def sample_batch_stratified(
        self,
        batch_size: int,
        mix: dict[str, float],
        device: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Sample with overlapping inclusion-rule buckets.

        Mix keys map to inclusion rules over the bucket field:
          "general"               -> all samples
          "hard_bot_general"      -> bucket in {1, 2}
          "coordination_endgame"  -> bucket == 2 (backfill from {1,2} if empty)

        Empty inclusion sets fall back to uniform over all samples.
        Last mix entry takes the remainder to ensure total == batch_size.
        """
        n = self.size()
        if n < batch_size:
            raise ValueError(f"buffer has {n} samples, need {batch_size}")
        buckets_arr = self.bucket[:n]

        # Precompute index pools for each inclusion rule
        all_idx = np.arange(n)
        hbg_idx = np.flatnonzero(buckets_arr >= 1)
        coord_idx = np.flatnonzero(buckets_arr == 2)

        def _pool_for(key: str) -> np.ndarray:
            if key == "general":
                return all_idx
            if key == "hard_bot_general":
                return hbg_idx if len(hbg_idx) > 0 else all_idx
            if key == "coordination_endgame":
                if len(coord_idx) > 0:
                    return coord_idx
                # Backfill from hard_bot_general, not self-play
                return hbg_idx if len(hbg_idx) > 0 else all_idx
            raise ValueError(f"Unknown replay_mix key: {key!r}")

        idx_parts: list[np.ndarray] = []
        remaining = batch_size
        mix_items = list(mix.items())
        for i, (key, frac) in enumerate(mix_items):
            n_draw = remaining if i == len(mix_items) - 1 else max(1, round(batch_size * frac))
            n_draw = min(n_draw, remaining)
            pool = _pool_for(key)
            idx_parts.append(np.random.choice(pool, size=n_draw, replace=True))
            remaining -= n_draw
            if remaining <= 0:
                break

        idx = np.concatenate(idx_parts)
        np.random.shuffle(idx)
        return self._sample_indices(idx, device)

    def _sample_indices(
        self,
        idx: np.ndarray,
        device: str | torch.device,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}
        for k in ROLE_ENCODE_CHANNEL_KEYS:
            t = torch.from_numpy(self.fields[k][idx]).to(device, non_blocking=True)
            batch[k] = t.long() if k == "seat_id" else t.float()
        targets = torch.from_numpy(self.returns[idx]).to(device, non_blocking=True)
        return batch, targets


def collate_base_encoded(
    encoded_groups: list[list[dict[str, np.ndarray]]],
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Collate decisions as shared state rows plus flattened action rows.

    ``encoded_groups`` is one list of encoded candidate-action dicts per
    decision. The returned ``state_batch`` has one row per decision,
    ``action_batch`` has one row per candidate action, and ``repeats`` maps
    each state row to its number of action rows.
    """
    if not encoded_groups:
        raise ValueError("encoded_groups must be non-empty")
    if any(len(group) == 0 for group in encoded_groups):
        raise ValueError("encoded_groups cannot contain empty decisions")

    state_batch = {
        k: torch.from_numpy(np.stack([group[0][k] for group in encoded_groups], axis=0))
        .to(device, non_blocking=True)
        for k in ENCODE_STATE_KEYS
    }
    flat_actions = [row for group in encoded_groups for row in group]
    action_batch = {
        k: torch.from_numpy(np.stack([row[k] for row in flat_actions], axis=0))
        .to(device, non_blocking=True)
        for k in ENCODE_ACTION_KEYS
    }
    repeats = torch.tensor(
        [len(group) for group in encoded_groups],
        dtype=torch.long,
        device=device,
    )
    return state_batch, action_batch, repeats


collate_grouped_encoded = collate_base_encoded


def collate_role_encoded(
    encoded_groups: list[list[dict[str, np.ndarray]]],
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Collate role-encoded decisions for SharedHeadQNet.forward_grouped."""
    if not encoded_groups:
        raise ValueError("encoded_groups must be non-empty")
    if any(len(group) == 0 for group in encoded_groups):
        raise ValueError("encoded_groups cannot contain empty decisions")

    state_batch = {
        k: torch.from_numpy(np.stack([group[0][k] for group in encoded_groups], axis=0))
        .to(device, non_blocking=True)
        for k in ROLE_ENCODE_STATE_KEYS
    }
    flat_actions = [row for group in encoded_groups for row in group]
    action_batch = {
        k: torch.from_numpy(np.stack([row[k] for row in flat_actions], axis=0))
        .to(device, non_blocking=True)
        for k in ROLE_ENCODE_ACTION_KEYS
    }
    repeats = torch.tensor(
        [len(group) for group in encoded_groups],
        dtype=torch.long,
        device=device,
    )
    return state_batch, action_batch, repeats


__all__ = [
    "BUCKET_NAMES",
    "BUCKET_IDS",
    "ReplayBuffer",
    "RoleAwareReplayBuffer",
    "collate_base_encoded",
    "collate_grouped_encoded",
    "collate_role_encoded",
    "Batch",
]
