"""Per-player replay buffer + collation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from guandan.dart.data.buffer import (
    ReplayBuffer,
    RoleAwareReplayBuffer,
    collate_base_encoded,
    collate_grouped_encoded,
    collate_role_encoded,
)
from guandan.dart.model.encoding.base_encoder import ENCODE_CHANNEL_SHAPES
from guandan.dart.model.encoding.role_encoder import (
    ROLE_ENCODE_CHANNEL_SHAPES,
)
from guandan.dart.data.returns import TrainSample


def _sample(player: int, val: float) -> TrainSample:
    enc = {
        key: np.zeros(shape, dtype=np.float32)
        for key, shape in ENCODE_CHANNEL_SHAPES.items()
    }
    enc["own_hand"].fill(val)
    return TrainSample(player=player, encoded=enc, mc_return=val)


def _role_stacked(n: int, offset: int = 0) -> dict[str, np.ndarray]:
    stacked: dict[str, np.ndarray] = {}
    for key, shape in ROLE_ENCODE_CHANNEL_SHAPES.items():
        if key == "trick_head_id":
            stacked[key] = np.asarray([(offset + i) % 4 for i in range(n)], dtype=np.int8)
        else:
            arr = np.zeros((n, *shape), dtype=np.uint8)
            if key == "candidate_action":
                arr[:, 0] = 1
            stacked[key] = arr
    return stacked


def _role_rows(n: int, offset: int = 0) -> list[dict[str, np.ndarray]]:
    stacked = _role_stacked(n, offset)
    return [{key: stacked[key][i] for key in stacked} for i in range(n)]


def test_per_player_capacity_enforced():
    buf = ReplayBuffer(capacity_per_player=3)
    for i in range(5):
        buf.push([_sample(0, float(i))])
    assert buf.size(0) == 3
    # FIFO: only the last 3 (vals 2,3,4) should remain
    samples = buf.sample_for_player(0, 3)
    vals = sorted(s.mc_return for s in samples)
    assert vals == [2.0, 3.0, 4.0]


def test_sampling_balances_across_players():
    buf = ReplayBuffer(capacity_per_player=10)
    for p in range(4):
        buf.push([_sample(p, float(p))])
    assert buf.total_size() == 4
    for p in range(4):
        s = buf.sample_for_player(p, 5)
        assert len(s) == 1
        assert s[0].player == p


def test_collate_base_encoded_splits_state_and_action_rows():
    groups = [
        [_sample(0, 1.0).encoded, _sample(0, 2.0).encoded],
        [_sample(1, 3.0).encoded],
    ]

    state_batch, action_batch, repeats = collate_base_encoded(groups)

    assert state_batch["history"].shape == (2, 20, 108)
    assert action_batch["candidate_action"].shape == (3, 108)
    assert repeats.tolist() == [2, 1]


def test_collate_base_encoded_alias_matches():
    groups = [[_sample(0, 1.0).encoded], [_sample(1, 2.0).encoded]]

    state_a, action_a, repeats_a = collate_base_encoded(groups)
    state_b, action_b, repeats_b = collate_grouped_encoded(groups)

    assert set(state_a) == set(state_b)
    assert set(action_a) == set(action_b)
    assert all(np.array_equal(state_a[k].numpy(), state_b[k].numpy()) for k in state_a)
    assert all(np.array_equal(action_a[k].numpy(), action_b[k].numpy()) for k in action_a)
    assert np.array_equal(repeats_a.numpy(), repeats_b.numpy())


def test_push_requires_full_canonical_schema():
    sample = _sample(0, 1.0)
    del sample.encoded["history"]

    with pytest.raises(KeyError):
        ReplayBuffer(capacity_per_player=3).push([sample])


def test_clear_resets_sizes_without_reallocating_storage():
    buf = ReplayBuffer(capacity_per_player=3)
    storage_before = buf.fields[0]["own_hand"]
    buf.push([_sample(0, 1.0), _sample(1, 2.0)])

    assert buf.total_size() == 2
    buf.clear()

    assert buf.total_size() == 0
    assert buf.size(0) == 0
    assert buf.fields[0]["own_hand"] is storage_before


def test_role_buffer_push_sample_and_size_by_seat():
    buf = RoleAwareReplayBuffer(capacity=16)
    stacked = _trick_stacked(8)
    returns = np.arange(8, dtype=np.float32)
    buf.push_stacked(stacked, returns)

    assert buf.size() == 8
    assert buf.size_by_seat() == {0: 2, 1: 2, 2: 2, 3: 2}
    batch, targets = buf.sample_batch(4)

    assert batch["player_blocks"].shape == (4, 4, 256)
    assert batch["global_features"].shape == (4, 13)
    assert batch["history_actions"].shape == (4, 20, 108)
    assert batch["history_roles"].shape == (4, 20, 4)
    assert batch["history_is_pass"].shape == (4, 20, 1)
    assert batch["behavior"].shape == (4, 9)
    assert batch["candidate_action"].shape == (4, 108)
    assert batch["trick_head_id"].shape == (4,)
    assert batch["trick_head_id"].dtype == torch.int64
    assert targets.shape == (4,)


def test_role_buffer_circular_overwrite_sets_full_on_partial_wrap():
    buf = RoleAwareReplayBuffer(capacity=5)
    buf.push_stacked(_trick_stacked(3), np.arange(3, dtype=np.float32))
    buf.push_stacked(_trick_stacked(3, offset=1), np.arange(3, 6, dtype=np.float32))

    assert buf.size() == 5
    assert sum(buf.size_by_seat().values()) == 5
    batch, targets = buf.sample_batch(5)
    assert batch["trick_head_id"].shape == (5,)
    assert targets.shape == (5,)


def test_role_buffer_state_dict_restores_samples_and_rng():
    buf = RoleAwareReplayBuffer(capacity=16, seed=123)
    stacked = _trick_stacked(12)
    returns = np.arange(12, dtype=np.float32)
    buf.push_stacked(stacked, returns)
    first_idx = buf.rng.integers(0, 12, size=4)

    state = buf.state_dict()
    restored = RoleAwareReplayBuffer(capacity=16, seed=999)
    restored.load_state_dict(state)

    assert restored.size() == buf.size()
    assert restored.ptr == buf.ptr
    assert np.array_equal(restored.returns[:12], buf.returns[:12])
    assert np.array_equal(restored.fields["trick_head_id"][:12], buf.fields["trick_head_id"][:12])
    # RNG state should continue from the checkpoint, not from the constructor seed.
    assert np.array_equal(restored.rng.integers(0, 12, size=4), buf.rng.integers(0, 12, size=4))
    assert not np.array_equal(first_idx, restored.rng.integers(0, 12, size=4))


def test_role_buffer_balanced_sample():
    buf = RoleAwareReplayBuffer(capacity=32)
    buf.push_stacked(_trick_stacked(16), np.arange(16, dtype=np.float32))

    batch, _targets = buf.sample_batch_balanced(8)
    counts = {p: int((batch["trick_head_id"] == p).sum().item()) for p in range(4)}
    assert counts == {0: 2, 1: 2, 2: 2, 3: 2}


def test_collate_role_encoded_shapes_repeats_and_head_ids():
    groups = [_role_rows(2, offset=0), _role_rows(3, offset=2)]

    state_batch, action_batch, repeats = collate_role_encoded(groups)

    assert state_batch["player_blocks"].shape == (2, 4, 256)
    assert action_batch["candidate_action"].shape == (5, 108)
    assert action_batch["trick_head_id"].tolist() == [0, 1, 2, 3, 0]
    assert repeats.tolist() == [2, 3]


# ─── Role-aware Dart buffer ───────────────────────────────────────────


def _trick_stacked(n: int, offset: int = 0) -> dict[str, np.ndarray]:
    """Build a stacked role-encoded Dart batch."""
    stacked: dict[str, np.ndarray] = {}
    for key, shape in ROLE_ENCODE_CHANNEL_SHAPES.items():
        if key == "trick_head_id":
            stacked[key] = np.asarray([(offset + i) % 4 for i in range(n)], dtype=np.int8)
        else:
            arr = np.zeros((n, *shape), dtype=np.uint8)
            if key == "candidate_action":
                arr[:, 0] = 1
            stacked[key] = arr
    return stacked


def test_trick_buffer_stores_256_wide_player_blocks():
    buf = RoleAwareReplayBuffer(capacity=8)
    assert buf.head_field == "trick_head_id"
    assert buf.fields["player_blocks"].shape == (8, 4, 256)


def test_trick_buffer_stratifies_on_trick_head_id():
    buf = RoleAwareReplayBuffer(capacity=32)
    buf.push_stacked(_trick_stacked(16), np.arange(16, dtype=np.float32))

    batch, _targets = buf.sample_batch_balanced(8)
    counts = {p: int((batch["trick_head_id"] == p).sum().item()) for p in range(4)}
    assert counts == {0: 2, 1: 2, 2: 2, 3: 2}


def test_trick_buffer_size_by_seat_uses_trick_head_id():
    buf = RoleAwareReplayBuffer(capacity=16)
    # 4 samples per head id 0..3
    buf.push_stacked(_trick_stacked(16), np.arange(16, dtype=np.float32))
    sizes = buf.size_by_seat()
    assert sizes == {0: 4, 1: 4, 2: 4, 3: 4}


def test_collate_role_encoded_autodetects_trick_scheme():
    n = 2
    base = _trick_stacked(n, offset=0)
    rows_a = [{k: base[k][i] for k in base} for i in range(n)]
    base_b = _trick_stacked(3, offset=2)
    rows_b = [{k: base_b[k][i] for k in base_b} for i in range(3)]
    groups = [rows_a, rows_b]

    state_batch, action_batch, repeats = collate_role_encoded(groups)

    assert state_batch["player_blocks"].shape == (2, 4, 256)
    assert action_batch["candidate_action"].shape == (5, 108)
    assert "trick_head_id" in action_batch
    assert action_batch["trick_head_id"].tolist() == [0, 1, 2, 3, 0]
    assert repeats.tolist() == [2, 3]


def test_role_buffer_defaults_to_dart_schema():
    buf = RoleAwareReplayBuffer(capacity=8)
    assert buf.head_field == "trick_head_id"
    assert buf.fields["player_blocks"].shape == (8, 4, 256)


def test_buffer_roundtrips_all_tags():
    """Push samples with known tag values and assert sample_batch returns them unchanged."""
    n = 16
    buf = RoleAwareReplayBuffer(capacity=n)
    stacked = _trick_stacked(n)
    returns = np.linspace(-1.0, 1.0, n, dtype=np.float32)

    tags = {
        "phase_self":        np.asarray([i % 3 for i in range(n)], dtype=np.int8),
        "trick_role":        np.asarray([i % 3 for i in range(n)], dtype=np.int8),
        "phase_partner":     np.asarray([i % 4 for i in range(n)], dtype=np.int8),
        "action_type":       np.asarray([i % 17 for i in range(n)], dtype=np.int8),
        "is_pass":           np.asarray([i % 2 for i in range(n)], dtype=np.int8),
        "is_bomb":           np.asarray([(i + 1) % 2 for i in range(n)], dtype=np.int8),
        "bomb_available":    np.asarray([i % 2 for i in range(n)], dtype=np.int8),
        "num_legal_actions": np.asarray([i + 1 for i in range(n)], dtype=np.int16),
        "q_gap":             np.asarray([np.nan if i == 0 else i * 0.01 for i in range(n)], dtype=np.float32),
        "chosen_by_epsilon": np.asarray([i % 2 for i in range(n)], dtype=np.int8),
        "episode_mode":      np.asarray([i % 3 for i in range(n)], dtype=np.int8),
        "opponent_id":       np.asarray([i % 6 for i in range(n)], dtype=np.int8),
        "latest_team":       np.asarray([i % 2 for i in range(n)], dtype=np.int8),
        "terminal_reward":   np.asarray([float(i - 8) for i in range(n)], dtype=np.float32),
    }
    buf.push_stacked(stacked, returns, tags=tags)

    # Verify direct backing-store values match
    for name, expected in tags.items():
        stored = getattr(buf, name)[:n]
        if expected.dtype == np.float32:
            # NaN-safe comparison
            mask = np.isnan(expected)
            assert np.all(np.isnan(stored[mask]))
            assert np.allclose(stored[~mask], expected[~mask])
        else:
            assert np.array_equal(stored, expected)

    # Verify sample_batch with return_tags=True returns torch tensors with same shape
    batch, targets, out_tags = buf.sample_batch(n, return_tags=True)
    assert set(out_tags.keys()) == set(tags.keys())
    for name in tags:
        assert out_tags[name].shape[0] == n


def test_k1_capped_sampler_hits_target_fraction():
    """sample_batch_balanced_k1_capped should produce batches with ~max_k1_frac K=1 samples."""
    buf = RoleAwareReplayBuffer(capacity=400)
    n = 400
    stacked = _trick_stacked(n)
    returns = np.zeros(n, dtype=np.float32)
    # First quarter are K=1, rest are K>1. Both groups span all 4 head buckets
    # (head_id = i % 4), so per-head balanced K>1 sampling has data in every head.
    num_legal = np.array([1 if i < n // 4 else 5 for i in range(n)], dtype=np.int16)
    tags = {"num_legal_actions": num_legal}
    buf.push_stacked(stacked, returns, tags=tags)

    batch_size = 200
    batch, _targets, out_tags = buf.sample_batch_balanced_k1_capped(
        batch_size, max_k1_frac=0.05, return_tags=True,
    )
    k1_count = int((out_tags["num_legal_actions"] == 1).sum().item())
    expected = round(batch_size * 0.05)
    assert k1_count == expected, f"expected {expected} K=1 samples, got {k1_count}"
    free_count = batch_size - k1_count
    assert int((out_tags["num_legal_actions"] > 1).sum().item()) == free_count
