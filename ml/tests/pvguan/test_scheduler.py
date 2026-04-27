"""Deterministic hand scheduler tests.

Asserts: same seed + same hand count → identical (idx, deal_seed, level_seed)
sequence regardless of worker timing variance.
"""

from __future__ import annotations

import random
import time

import pytest

from guandan.pvguan.rollout import HandScheduler, _hash64


# ─── hash64 properties ────────────────────────────────────────────────────────

def test_hash64_deterministic():
    assert _hash64(42, 0) == _hash64(42, 0)
    assert _hash64(42, 1) == _hash64(42, 1)


def test_hash64_different_indices():
    assert _hash64(42, 0) != _hash64(42, 1)


def test_hash64_different_seeds():
    assert _hash64(0, 5) != _hash64(1, 5)


def test_hash64_tag_differentiates():
    assert _hash64(0, 0, b"") != _hash64(0, 0, b"level")


def test_hash64_returns_int():
    v = _hash64(99, 3, b"x")
    assert isinstance(v, int)
    assert v >= 0


# ─── HandScheduler determinism ───────────────────────────────────────────────

def test_scheduler_same_seed_same_sequence():
    """Identical seeds → identical (idx, deal_seed, level_seed) tuples."""
    N = 100
    sched1 = HandScheduler(global_run_seed=42)
    sched2 = HandScheduler(global_run_seed=42)
    seq1 = [sched1.next_hand() for _ in range(N)]
    seq2 = [sched2.next_hand() for _ in range(N)]
    assert seq1 == seq2


def test_scheduler_different_seeds_differ():
    """Different run seeds → different sequences."""
    N = 20
    sched1 = HandScheduler(global_run_seed=0)
    sched2 = HandScheduler(global_run_seed=1)
    seq1 = [sched1.next_hand() for _ in range(N)]
    seq2 = [sched2.next_hand() for _ in range(N)]
    assert seq1 != seq2


def test_scheduler_increments_global_index():
    """Global hand index must increment by 1 each call."""
    sched = HandScheduler(global_run_seed=7)
    indices = [sched.next_hand()[0] for _ in range(10)]
    assert indices == list(range(10))


def test_scheduler_deal_seed_in_range():
    """deal_seed must be a non-negative 31-bit integer."""
    sched = HandScheduler(global_run_seed=1)
    for _ in range(50):
        _, deal_seed, _ = sched.next_hand()
        assert 0 <= deal_seed < 2**31


def test_scheduler_level_seed_in_range():
    """level_seed must be in [0, 12] (indices into 13 ranks)."""
    sched = HandScheduler(global_run_seed=2)
    for _ in range(200):
        _, _, level_seed = sched.next_hand()
        assert 0 <= level_seed < 13


def test_scheduler_all_ranks_covered():
    """Over enough hands, all 13 level-rank indices should appear."""
    sched = HandScheduler(global_run_seed=3)
    seen = set()
    for _ in range(500):
        _, _, level_seed = sched.next_hand()
        seen.add(level_seed)
    assert seen == set(range(13)), f"Missing ranks: {set(range(13)) - seen}"


def test_scheduler_robust_to_timing_variance():
    """Simulated worker delays must not affect the generated sequence."""
    N = 50
    rng = random.Random(999)

    sched1 = HandScheduler(global_run_seed=42)
    seq1 = [sched1.next_hand() for _ in range(N)]

    sched2 = HandScheduler(global_run_seed=42)
    seq2 = []
    for i in range(N):
        if i % 7 == 0:
            # Tiny sleep simulating variable worker latency
            time.sleep(rng.uniform(0, 0.001))
        seq2.append(sched2.next_hand())

    assert seq1 == seq2


def test_scheduler_no_duplicates():
    """Each (deal_seed, level_seed) pair must be unique for distinct indices."""
    sched = HandScheduler(global_run_seed=5)
    pairs = {(d, l) for _, d, l in (sched.next_hand() for _ in range(200))}
    # With 200 draws and a good hash, collisions should be nil
    assert len(pairs) == 200


def test_two_schedulers_same_prefix_same_values():
    """Second scheduler picks up the same values at the same indices
    even if it didn't call next_hand for earlier indices."""
    sched_full = HandScheduler(global_run_seed=99)
    seq_full = [sched_full.next_hand() for _ in range(30)]

    # A scheduler starting from counter=20 should match the full one at [20:]
    sched_partial = HandScheduler(global_run_seed=99)
    sched_partial._counter = 20
    seq_partial = [sched_partial.next_hand() for _ in range(10)]

    assert seq_partial == seq_full[20:30]
