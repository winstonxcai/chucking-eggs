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

from ..model.encoding.base_encoder import (
    ENCODE_ACTION_KEYS,
    ENCODE_CHANNEL_KEYS,
    ENCODE_CHANNEL_SHAPES,
    ENCODE_STATE_KEYS,
)
from ..model.encoding.role_encoder import (
    ROLE_ENCODE_ACTION_KEYS,
    ROLE_ENCODE_CHANNEL_KEYS,
    ROLE_ENCODE_CHANNEL_SHAPES,
    ROLE_ENCODE_STATE_KEYS,
    ROLE_ENCODE_TRICK_ACTION_KEYS,
    ROLE_ENCODE_TRICK_CHANNEL_KEYS,
    ROLE_ENCODE_TRICK_CHANNEL_SHAPES,
)

# Integer-scalar fields stored as int8 (head-routing keys). All other fields
# are uint8 multi-hot / one-hot encodings.
_INT_FIELDS: frozenset[str] = frozenset({"seat_id", "trick_head_id"})
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
    """Single circular replay buffer for the shared-head role-aware model.

    Schema is mode-aware: pass ``head_scheme="trick_relative"`` for the
    256-wide player_blocks + ``trick_head_id`` schema (used by
    ``SharedTrickHeadQNet``). Default is the legacy 252-wide + ``seat_id``
    schema (used by ``SharedHeadQNet``).
    """

    def __init__(
        self,
        capacity: int = 200_000,
        head_scheme: str = "absolute_seat",
    ) -> None:
        if head_scheme not in ("absolute_seat", "trick_relative"):
            raise ValueError(
                f"head_scheme must be 'absolute_seat' or 'trick_relative'; "
                f"got {head_scheme!r}"
            )
        self.capacity = capacity
        self.head_scheme = head_scheme
        if head_scheme == "trick_relative":
            self.channel_shapes = ROLE_ENCODE_TRICK_CHANNEL_SHAPES
            self.channel_keys = ROLE_ENCODE_TRICK_CHANNEL_KEYS
            self.head_field = "trick_head_id"
        else:
            self.channel_shapes = ROLE_ENCODE_CHANNEL_SHAPES
            self.channel_keys = ROLE_ENCODE_CHANNEL_KEYS
            self.head_field = "seat_id"
        self.fields: dict[str, np.ndarray] = {}
        for k, shape in self.channel_shapes.items():
            dtype = np.int8 if k in _INT_FIELDS else np.uint8
            self.fields[k] = np.zeros((capacity, *shape), dtype=dtype)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.bucket = np.zeros(capacity, dtype=np.int8)
        self.phase_self        = np.zeros(capacity, dtype=np.int8)
        self.trick_role        = np.zeros(capacity, dtype=np.int8)
        self.phase_partner     = np.zeros(capacity, dtype=np.int8)
        self.action_type       = np.zeros(capacity, dtype=np.int8)
        self.is_pass           = np.zeros(capacity, dtype=np.int8)
        self.is_bomb           = np.zeros(capacity, dtype=np.int8)
        self.bomb_available    = np.zeros(capacity, dtype=np.int8)
        self.num_legal_actions = np.zeros(capacity, dtype=np.int16)
        self.q_gap             = np.full(capacity, np.nan, dtype=np.float32)
        self.chosen_by_epsilon = np.zeros(capacity, dtype=np.int8)
        self.episode_mode      = np.zeros(capacity, dtype=np.int8)
        self.opponent_id       = np.zeros(capacity, dtype=np.int8)
        self.latest_team       = np.zeros(capacity, dtype=np.int8)
        self.terminal_reward   = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.full = False

    def size(self) -> int:
        """Total samples currently stored."""
        return self.capacity if self.full else self.ptr

    def size_by_seat(self) -> dict[int, int]:
        """Return stored sample counts by head-routing key (0..3).

        For ``head_scheme="absolute_seat"`` the buckets are absolute seats.
        For ``head_scheme="trick_relative"`` they are trick-relative head ids.
        """
        n = self.size()
        if n == 0:
            return {p: 0 for p in range(4)}
        ids = self.fields[self.head_field][:n].astype(np.int64, copy=False)
        return {p: int(np.sum(ids == p)) for p in range(4)}

    # Per-sample diagnostic tag fields written/read alongside the encoded
    # buffers. Mirrors the queue-message keys from worker.py.
    _TAG_FIELDS: tuple[tuple[str, type], ...] = (
        ("phase_self",        np.int8),
        ("trick_role",        np.int8),
        ("phase_partner",     np.int8),
        ("action_type",       np.int8),
        ("is_pass",           np.int8),
        ("is_bomb",           np.int8),
        ("bomb_available",    np.int8),
        ("num_legal_actions", np.int16),
        ("q_gap",             np.float32),
        ("chosen_by_epsilon", np.int8),
        ("episode_mode",      np.int8),
        ("opponent_id",       np.int8),
        ("latest_team",       np.int8),
        ("terminal_reward",   np.float32),
    )

    def push_stacked(
        self,
        stacked: dict[str, np.ndarray],
        returns: np.ndarray,
        buckets: np.ndarray | None = None,
        tags: dict[str, np.ndarray] | None = None,
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
        tag_writes: dict[str, np.ndarray] = {}
        for name, dtype in self._TAG_FIELDS:
            if tags is not None and name in tags:
                tag_writes[name] = tags[name].astype(dtype, copy=False)
            else:
                fill = np.nan if dtype == np.float32 and name == "q_gap" else 0
                tag_writes[name] = np.full(n, fill, dtype=dtype)
        if n >= self.capacity:
            start = n - self.capacity
            for k in self.channel_keys:
                self.fields[k][:] = stacked[k][start:].astype(self.fields[k].dtype, copy=False)
            self.returns[:] = returns[start:].astype(np.float32, copy=False)
            self.bucket[:] = buckets_to_write[start:]
            for name, _ in self._TAG_FIELDS:
                getattr(self, name)[:] = tag_writes[name][start:]
            self.ptr = 0
            self.full = True
            return

        start_ptr = self.ptr
        end = start_ptr + n
        if end <= self.capacity:
            dst = slice(start_ptr, end)
            src = slice(None)
            for k in self.channel_keys:
                self.fields[k][dst] = stacked[k][src].astype(self.fields[k].dtype, copy=False)
            self.returns[dst] = returns[src].astype(np.float32, copy=False)
            self.bucket[dst] = buckets_to_write[src]
            for name, _ in self._TAG_FIELDS:
                getattr(self, name)[dst] = tag_writes[name][src]
        else:
            first = self.capacity - start_ptr
            second = n - first
            for k in self.channel_keys:
                arr = stacked[k].astype(self.fields[k].dtype, copy=False)
                self.fields[k][start_ptr:] = arr[:first]
                self.fields[k][:second] = arr[first:]
            ret = returns.astype(np.float32, copy=False)
            self.returns[start_ptr:] = ret[:first]
            self.returns[:second] = ret[first:]
            self.bucket[start_ptr:] = buckets_to_write[:first]
            self.bucket[:second] = buckets_to_write[first:]
            for name, _ in self._TAG_FIELDS:
                arr = tag_writes[name]
                getattr(self, name)[start_ptr:] = arr[:first]
                getattr(self, name)[:second] = arr[first:]
        self.ptr = (self.ptr + n) % self.capacity
        self.full = self.full or end >= self.capacity

    def sample_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        *,
        return_tags: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor] | tuple[
        dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]
    ]:
        """Uniformly sample a role-aware batch."""
        n = self.size()
        if n < batch_size:
            raise ValueError(f"buffer has {n} samples, need {batch_size}")
        idx = np.random.randint(0, n, size=batch_size)
        return self._sample_indices(idx, device, return_tags=return_tags)

    def sample_batch_balanced(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        *,
        return_tags: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor] | tuple[
        dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]
    ]:
        """Stratified sample with approximately equal rows per head bucket.

        Bucket interpretation depends on ``self.head_scheme``: absolute seats
        for ``absolute_seat``, trick-relative head ids for ``trick_relative``.
        """
        n = self.size()
        ids = self.fields[self.head_field][:n].astype(np.int64, copy=False)
        counts = [batch_size // 4] * 4
        for p in range(batch_size % 4):
            counts[p] += 1
        idx_parts = []
        for p, count in enumerate(counts):
            seat_idx = np.flatnonzero(ids == p)
            if len(seat_idx) < count:
                raise ValueError(
                    f"{self.head_field} bucket {p} has {len(seat_idx)} samples, need {count}"
                )
            idx_parts.append(np.random.choice(seat_idx, size=count, replace=True))
        idx = np.concatenate(idx_parts)
        np.random.shuffle(idx)
        return self._sample_indices(idx, device, return_tags=return_tags)

    def sample_batch_balanced_k1_capped(
        self,
        batch_size: int,
        max_k1_frac: float,
        device: str | torch.device = "cpu",
        *,
        return_tags: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor] | tuple[
        dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]
    ]:
        """Like sample_batch_balanced but caps K=1 samples at ``max_k1_frac``.

        K=1 portion: uniform from samples with ``num_legal_actions == 1``.
        K>1 portion: per-head balanced over samples with K>1.
        If the K=1 pool is empty, the K>1 portion absorbs the full batch.
        """
        n = self.size()
        k1_mask = self.num_legal_actions[:n] == 1
        n_k1_target = round(batch_size * max_k1_frac)
        n_free_target = batch_size - n_k1_target

        k1_idx_all = np.flatnonzero(k1_mask)
        if len(k1_idx_all) == 0 or n_k1_target == 0:
            n_free_target = batch_size
            k1_part = np.empty(0, dtype=np.int64)
        else:
            k1_part = np.random.choice(k1_idx_all, size=n_k1_target, replace=True)

        free_mask = ~k1_mask
        ids = self.fields[self.head_field][:n].astype(np.int64, copy=False)
        counts = [n_free_target // 4] * 4
        for p in range(n_free_target % 4):
            counts[p] += 1
        free_parts: list[np.ndarray] = []
        for p, count in enumerate(counts):
            head_free = np.flatnonzero(free_mask & (ids == p))
            if len(head_free) < count:
                raise ValueError(
                    f"head {p} has {len(head_free)} K>1 samples; need {count} "
                    f"(K=1 cap={max_k1_frac})"
                )
            free_parts.append(np.random.choice(head_free, size=count, replace=True))

        idx = np.concatenate([k1_part, *free_parts])
        np.random.shuffle(idx)
        return self._sample_indices(idx, device, return_tags=return_tags)

    def sample_batch_stratified(
        self,
        batch_size: int,
        mix: dict[str, float],
        device: str | torch.device = "cpu",
        *,
        return_tags: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor] | tuple[
        dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]
    ]:
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
        q_gap_arr = self.q_gap[:n]
        pivotal_idx = np.flatnonzero(~np.isnan(q_gap_arr) & (q_gap_arr < 0.05))
        partner_coord_idx = np.flatnonzero(self.trick_role[:n] == 1)
        bomb_idx = np.flatnonzero(
            (self.is_bomb[:n] == 1) | (self.bomb_available[:n] == 1)
        )

        def _pool_for(key: str) -> np.ndarray:
            if key == "general":
                return all_idx
            if key == "hard_bot_general":
                return hbg_idx if len(hbg_idx) > 0 else all_idx
            if key == "coordination_endgame":
                if len(coord_idx) > 0:
                    return coord_idx
                return hbg_idx if len(hbg_idx) > 0 else all_idx
            if key == "pivotal_qgap":
                return pivotal_idx if len(pivotal_idx) > 0 else all_idx
            if key == "partner_active_coordination":
                return partner_coord_idx if len(partner_coord_idx) > 0 else all_idx
            if key == "bomb_decision":
                return bomb_idx if len(bomb_idx) > 0 else all_idx
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
        return self._sample_indices(idx, device, return_tags=return_tags)

    def _sample_indices(
        self,
        idx: np.ndarray,
        device: str | torch.device,
        *,
        return_tags: bool = False,
    ):
        batch: dict[str, torch.Tensor] = {}
        for k in self.channel_keys:
            t = torch.from_numpy(self.fields[k][idx]).to(device, non_blocking=True)
            batch[k] = t.long() if k in _INT_FIELDS else t.float()
        targets = torch.from_numpy(self.returns[idx]).to(device, non_blocking=True)
        if not return_tags:
            return batch, targets
        tags: dict[str, torch.Tensor] = {}
        for name, _ in self._TAG_FIELDS:
            arr = getattr(self, name)[idx]
            tags[name] = torch.from_numpy(arr.copy()).to(device, non_blocking=True)
        return batch, targets, tags


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
    action_keys: tuple[str, ...] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Collate role-encoded decisions for shared-head forward_grouped.

    ``action_keys`` selects which fields land in the action batch. Defaults to
    ``ROLE_ENCODE_ACTION_KEYS`` (legacy absolute_seat schema). Pass
    ``ROLE_ENCODE_TRICK_ACTION_KEYS`` for the trick_relative schema.

    Auto-detection: if ``action_keys`` is None and the first encoded row has a
    ``trick_head_id`` key, the trick action keys are used.
    """
    if not encoded_groups:
        raise ValueError("encoded_groups must be non-empty")
    if any(len(group) == 0 for group in encoded_groups):
        raise ValueError("encoded_groups cannot contain empty decisions")

    if action_keys is None:
        first_row = encoded_groups[0][0]
        action_keys = (
            ROLE_ENCODE_TRICK_ACTION_KEYS
            if "trick_head_id" in first_row
            else ROLE_ENCODE_ACTION_KEYS
        )

    state_batch = {
        k: torch.from_numpy(np.stack([group[0][k] for group in encoded_groups], axis=0))
        .to(device, non_blocking=True)
        for k in ROLE_ENCODE_STATE_KEYS
    }
    flat_actions = [row for group in encoded_groups for row in group]
    action_batch = {
        k: torch.from_numpy(np.stack([row[k] for row in flat_actions], axis=0))
        .to(device, non_blocking=True)
        for k in action_keys
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
