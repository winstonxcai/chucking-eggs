"""Per-player replay buffer + collation."""

from __future__ import annotations

import numpy as np

from guandan.guanzero.buffer import ReplayBuffer, collate_encoded
from guandan.guanzero.returns import TrainSample


def _sample(player: int, val: float) -> TrainSample:
    enc = {
        "own_hand": np.full(108, val, dtype=np.float32),
        "history": np.zeros((20, 108), dtype=np.float32),
    }
    return TrainSample(player=player, encoded=enc, mc_return=val)


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


def test_collate_encoded_no_targets():
    encs = [
        {"own_hand": np.ones(108, dtype=np.float32),
         "history": np.zeros((20, 108), dtype=np.float32)},
        {"own_hand": np.zeros(108, dtype=np.float32),
         "history": np.zeros((20, 108), dtype=np.float32)},
    ]
    batch = collate_encoded(encs)
    assert batch["own_hand"].shape == (2, 108)
